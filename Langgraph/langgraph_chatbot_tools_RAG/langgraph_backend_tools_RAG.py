import os
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from pypdf import PdfReader
from dotenv import load_dotenv
from docx import Document

import asyncio
import threading
import io
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.tools import tool, BaseTool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient
from binance.client import Client
import aiosqlite

load_dotenv()

model = ChatGroq(model='openai/gpt-oss-120b')

# Your API credentials
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_SECRET_KEY")
client = Client(api_key, api_secret)


PERSIST_DIR = r'D:\intern preparation\Langgraph\langgraph_chatbot_tools_RAG\chroma_db'


def _extract_text_from_file(file_source, filename: str) -> str:
    """Extract raw text from a .docx or .pdf file.
    `file_source` can be a path (str) or a file-like object (e.g. io.BytesIO
    from an uploaded file) - both python-docx's Document() and pypdf's
    PdfReader() accept either.
    """
    name_lower = filename.lower()
    if name_lower.endswith('.docx'):
        doc = Document(file_source)
        return '\n'.join(p.text for p in doc.paragraphs)
    elif name_lower.endswith('.pdf'):
        reader = PdfReader(file_source)
        return '\n\n'.join((page.extract_text() or '') for page in reader.pages)
    else:
        raise TypeError(f"Unsupported file type: {filename}")


def _chunk_text(full_text: str):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents([LCDocument(page_content=full_text)])
    print('total chunks are : ', len(chunks))
    return chunks


def preprocessing_for_rag():
    """Build (or load) the persistent Chroma store, seeded with the default document."""
    print('--' * 20, 'chunking and embedding started ...')
    embed_model = VoyageAIEmbeddings(model="voyage-law-2")

    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        local_db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embed_model)
    else:
        local_db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embed_model)

    return local_db


local_db = preprocessing_for_rag()


def add_uploaded_document(file_bytes: bytes, filename: str) -> int:
    """Extract text from an uploaded file's raw bytes, chunk it, and add it
    into the existing Chroma vector store so retrieve_format can find it
    right away. Returns the number of chunks added.
    """

    file_stream = io.BytesIO(file_bytes)
    full_text = _extract_text_from_file(file_stream, filename)
    if not full_text.strip():
        raise ValueError(f"No extractable text found in '{filename}'.")
    chunks = _chunk_text(full_text)
    local_db.add_documents(chunks)
    return len(chunks)

# Force torch/sentence-transformers to fully complete its lazy native
# thread-pool / MKL initialization HERE, single-threaded, synchronously,
# before we ever start a background thread. Doing the first real model
# inference later (concurrently with aiosqlite/MCP threads) is what was
# causing the silent crash.
try:
    print('--' * 20, 'warming up embedding model ...')
    local_db.as_retriever(search_kwargs={'k': 1}).invoke('warmup query')
    print('--' * 20, 'embedding model warmup complete')
except Exception as _warmup_exc:
    print('RAG warmup failed (continuing anyway):', repr(_warmup_exc))


# -------------------------------------------------------------
# Dedicated background event loop (needed because MCP tool
# loading, the checkpointer, and streaming are all async).
# Started ONLY after all heavy native-library initialization above
# (torch/transformers/chroma) has fully completed on the main thread.
# -------------------------------------------------------------
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop (used by the frontend for streaming)."""
    return _submit_async(coro)


@tool
def retrieve_format(user_query: str):
    """Retrieve relevant documents from the local Chroma database based on the user's query."""
    try:
        retrieval = local_db.as_retriever(search_kwargs={'k': 3})
        similar_chunks = retrieval.invoke(user_query)
        formatted_doc = '\n\n'.join(doc.page_content for doc in similar_chunks)
        return formatted_doc
    except Exception as exc:
        print('RAG retrieval error:', repr(exc))
        return f"RAG retrieval failed: {exc}"



@tool
def get_price(symbol: str):
    """Fetch latest price of cryptocurrency or stocks listed on Binance (spot or futures).
    For crypto (BTC, ETH, DOGE, MYX, etc.), the tool auto-converts to trading pairs (BTCUSDT, ETHUSDT, MYXUSDT).
    For stocks (IBM, GOOGLE, AAPL, NETFLIX, etc.), add 'B' suffix (IBMB, APPLEB, NFLXB) for spot trading.
    Works with both spot and futures markets."""

    original_symbol = symbol.upper()

    # Strategy: Try multiple symbol formats to find the price
    symbols_to_try = [
        original_symbol + 'USDT',  # Crypto format (BTCUSDT, MYXUSDT, ETHUSDT)
        original_symbol + 'B',      # Stock spot format (IBMB, APPLEB, NFLXB)
        original_symbol,             # Raw symbol fallback
    ]

    for symbol_attempt in symbols_to_try:
        try:
            ticker = client.get_symbol_ticker(symbol=symbol_attempt)
            price = float(ticker['price'])
            return f"Symbol: {symbol_attempt}, Price: {price}"
        except Exception:
            continue

    # If spot market fails, suggest the correct formats
    return f"Error: Symbol '{original_symbol}' not found on Binance spot market. \nTry these formats:\n- Crypto: {original_symbol}USDT (e.g., BTCUSDT, MYXUSDT, ETHUSDT)\n- Stocks: {original_symbol}B (e.g., IBMB, APPLEB, NFLXB)"


# Remote MCP tool (your fastmcp expense tracker server)
mcp_client = MultiServerMCPClient(
    {
        "expense_tracker": {
            "transport": "streamable_http",  # if this fails, try "sse"
            "url": "https://test-server-very-silver-koala.fastmcp.app/mcp"
        }
    }
)


def load_mcp_tools() -> list[BaseTool]:
    try:
        return run_async(mcp_client.get_tools())
    except Exception:
        return []


mcp_tools = load_mcp_tools()

tools = [get_price, *mcp_tools, retrieve_format]

llm_with_tool = model.bind_tools(tools)


# State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Nodes
# -------------------
async def chat_node(state: ChatState):
    messages = state['messages']

    system_msg = SystemMessage(content="""You are a helpful assistant with access to tools for fetching cryptocurrency/stock prices , for tracking expenses and for RAG based queries .

IMPORTANT INSTRUCTION:
- When the user asks about the price of ANY coin or stock (e.g., "What's the price of Bitcoin?", "Tell me SPCX stock price", "How much is ETH?"), you MUST use the get_price tool.
- The get_price tool can fetch prices from Binance. Pass the symbol (like BTC, ETH, DOGE, IBM, GOOGLE etc.) and the tool will handle the USDT conversion.
- When the user asks to add, list, or summarize expenses, use the expense tracker tool(s).
- For questions about prices, ALWAYS use the tool - don't try to guess or use outdated knowledge.
- And use retrieved document to answer RAG based query.
- For other questions (general knowledge, help, etc.), answer directly without tools.
- Be conversational and helpful.""")

    response = await llm_with_tool.ainvoke([system_msg] + messages)

    return {'messages': [response]}


graph = StateGraph(ChatState)

tool_node = ToolNode(tools)

graph.add_node('chatbot', chat_node)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'chatbot')
graph.add_conditional_edges('chatbot', tools_condition)
graph.add_edge('tools', 'chatbot')


# Checkpointer (SQLite instead of Postgres)
# -------------------
async def _init_checkpointer():
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chatbot.db")
    conn = await aiosqlite.connect(database=db_path)
    return AsyncSqliteSaver(conn)


checkpointer = run_async(_init_checkpointer())

chatbot = graph.compile(checkpointer=checkpointer)



# Helper
# -------------------
async def _alist_threads():
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)


def retrieve_all_threads():
    return run_async(_alist_threads())

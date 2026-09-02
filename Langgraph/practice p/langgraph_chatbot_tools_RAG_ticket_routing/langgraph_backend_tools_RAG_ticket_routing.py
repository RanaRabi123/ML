import os
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from pypdf import PdfReader
from dotenv import load_dotenv
from docx import Document

import operator
import asyncio
import threading
import io
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from typing_extensions import TypedDict, NotRequired
from typing import Annotated, Literal
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.tools import tool, BaseTool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode, tools_condition, InjectedState
from langgraph.types import Command
import aiosqlite

load_dotenv()

model_simple = ChatGroq(model='openai/gpt-oss-120b')

PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chroma_db')


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




# State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    # These don't exist yet on a brand-new thread (no ticket created / no title generated yet), so they must be optional - InjectedState
    # validates the live state against this TypedDict, and would otherwise reject the very first create_ticket call with a "field required" error.
    complain_number: NotRequired[int]
    complain: NotRequired[Literal['resolved', 'not-resolved']]
    ticket_message: Annotated[list[str], operator.add]
    chat_title: NotRequired[str]


def chat_title_func(first_message: str) -> str:
    """Generate a short title from the user's first message."""
    if not first_message:
        return "New Conversation"
    response = model_simple.invoke([
        SystemMessage(content="Generate a concise 4-6 word title for this chat based on the user's message. Return only the title, no quotes."),
        HumanMessage(content=first_message)
    ])
    return response.content.strip()



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
def create_ticket(issue_summary: str, state: Annotated[ChatState, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId]):
    """Escalate the user's issue to real human support by creating a ticket.
    Only call this when the FAQ/RAG documents cannot resolve the user's issue,
    and only after the issue is clear and specific (ask a clarifying question
    first if it's vague, e.g. "not working" -> ask what specifically isn't working).

    Args:
        issue_summary: A clear, one- or two-sentence summary of the user's
            issue, written from what they've told you in the conversation so far.
    """
    # `state` and `tool_call_id` are injected by LangGraph, not filled in by
    # the LLM - see InjectedState/InjectedToolCallId above. `complain_number`
    # may not exist yet on a brand-new thread, hence the .get default.
    complain_number = state.get('complain_number', 0) + 1

    ticket_notice = (
        f"Ticket #{complain_number} has been created. "
        f"Our support team will follow up shortly regarding: {issue_summary}"
    )

    # A plain `return {...}` from a tool only becomes the ToolMessage's text -
    # it does NOT update graph state. Command(update=...) is what actually
    # writes complain_number/complain/ticket_message back into the checkpoint.
    return Command(update={
        'complain_number': complain_number,
        'complain': 'not-resolved',
        'ticket_message': [issue_summary],
        'messages': [ToolMessage(content=ticket_notice, tool_call_id=tool_call_id)],
    })



tools = [retrieve_format, create_ticket]
model_with_tool = ChatGroq(model='openai/gpt-oss-120b').bind_tools(tools)


def count_words(messages: list[BaseMessage]) -> int:
    return sum(len(msg.content.split()) for msg in messages)


def summarize_chat(messages):

    summarized_messages = []
    to_summarize = messages[:-6]
    to_keep = messages[-6:]
    summarized_response =model_simple.invoke([
                SystemMessage(content="You are a chat summarizer, you need to summarize the given chat history so that it's context can be kept , while makeing it brief, "), 
                HumanMessage(content=to_summarize)
            ]).content
    summarized_messages.append(summarized_response, to_keep)
    return summarized_messages

# Nodes
# -------------------
async def chat_node(state: ChatState):
    messages = state['messages']

    token_count = count_words(messages)
    if token_count > 7500:
        messages = summarize_chat(messages)


    system_msg = SystemMessage(content="""You are a helpful customer support assistant with access to tools for RAG based queries and you can move them to real human assistant when you cannot satisfy the user's query/problem and if user problem/question is ambigious , ask him to give a clear one.

    IMPORTANT INSTRUCTION:
    - Use retrieved document to answer RAG/FAQ based query.
    - If you cannot solve the user problem from RAG documents , then you can direct him to human/create_ticket and you can ask him to for clear question.
    - if you think that retrieved chunks are not good matching with user query/problem , then you redriect him to create_ticket (such as : if user has billing, product, account and otehr issue that retrieved/FAQ document cannot resolved).
    - if it's question is ambigious and not clear then before redriecting it to create_ticket ask him a clear query/question/problem (such as : if user say it is not working , ask him what is not workign ? ) . 
    - if even after redriecting to create_ticket tool , if user still unsatisfied/unhappy then ask him to give a clear question for more reliability. 
    - No need to respond anything irrelevant and no need to give too much explanation and depth. 
    - Be conversational and helpful.""")

    response = await model_with_tool.ainvoke([system_msg] + messages)

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
    """Walk every checkpoint in the SQLite DB and collect, per thread_id, the
    chat_title that was persisted into that thread's state (if any).
    Returns {thread_id: title_or_None}.
    """
    threads: dict = {}
    async for checkpoint in checkpointer.alist(None):
        tid = checkpoint.config['configurable']['thread_id']
        if tid not in threads:
            threads[tid] = None
        title = checkpoint.checkpoint.get('channel_values', {}).get('chat_title')
        if title and not threads[tid]:
            threads[tid] = title
    return threads


def retrieve_all_threads():
    """Returns {thread_id: chat_title_or_None} for every thread found in chatbot.db."""
    return run_async(_alist_threads())
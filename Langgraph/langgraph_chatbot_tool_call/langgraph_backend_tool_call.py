import os 
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection
from psycopg.rows import dict_row
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from binance.client import Client


load_dotenv()
model = ChatGroq(
    model  ='llama-3.3-70b-versatile'
)

# Your API credentials
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_SECRET_KEY")
client = Client(api_key, api_secret)

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
        except Exception as e:
            continue
    
    # If spot market fails, suggest the correct formats
    return f"Error: Symbol '{original_symbol}' not found on Binance spot market. \nTry these formats:\n- Crypto: {original_symbol}USDT (e.g., BTCUSDT, MYXUSDT, ETHUSDT)\n- Stocks: {original_symbol}B (e.g., IBMB, APPLEB, NFLXB)"

tools = [get_price]

llm_with_tool = model.bind_tools(tools)


class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]

graph = StateGraph(ChatState)


def chat_node(state:ChatState):
    messages = state['messages'] 

    system_msg = SystemMessage(content="""You are a helpful assistant with access to tools for fetching cryptocurrency and stock prices.

IMPORTANT INSTRUCTION:
- When the user asks about the price of ANY coin or stock (e.g., "What's the price of Bitcoin?", "Tell me SPCX stock price", "How much is ETH?"), you MUST use the get_price tool.
- The get_price tool can fetch prices from Binance. Pass the symbol (like BTC, ETH, DOGE,IBM, GOOGLE etc.) and the tool will handle the USDT conversion.
- For questions about prices, ALWAYS use the tool - don't try to guess or use outdated knowledge.
- For other questions (general knowledge, help, etc.), answer directly without tools.
- Be conversational and helpful.""")

    response = llm_with_tool.invoke([system_msg]+messages)

    return {'messages':[response]}

tool_node = ToolNode(tools)

graph.add_node('chatbot', chat_node)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'chatbot')
graph.add_conditional_edges('chatbot', tools_condition)
graph.add_edge('tools', 'chatbot')

graph.compile()

DB_URI = os.getenv['POSTGRES_DB_URI']

conn = Connection.connect(DB_URI, autocommit=True, row_factory=dict_row)
checkpointer = PostgresSaver(conn)
checkpointer.setup()

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for thread in checkpointer.list(None):
        all_threads.add(thread.config['configurable']['thread_id'])
    return list(all_threads)
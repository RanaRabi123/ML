import os 
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from binance.client import Client
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langgraph.types import interrupt, Command
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model = 'openai/gpt-oss-120b')

api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_SECRET_KEY")
client = Client(api_key, api_secret)


@tool
def get_crypto_stock_price(symbol: str):
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


@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing a given quantity of a stock symbol.

    HUMAN-IN-THE-LOOP:
    Before confirming the purchase, this tool will interrupt
    and wait for a human decision ("yes" / anything else).
    """
    # This pauses the graph and returns control to the caller
    decision = interrupt(f"Approve buying {quantity} shares of {symbol}? (yes/no)")

    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": f"Purchase order placed for {quantity} shares of {symbol}.",
            "symbol": symbol,
            "quantity": quantity,
        }
    
    else:
        return {
            "status": "cancelled",
            "message": f"Purchase of {quantity} shares of {symbol} was declined by human.",
            "symbol": symbol,
            "quantity": quantity,
        }


tools = [get_crypto_stock_price, purchase_stock]
llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)


memory = MemorySaver()


graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=memory)




if __name__ == "__main__":

    thread_id = "demo-thread"

    while True:
        user_input = input("You: ")
        if user_input.lower().strip() in {"exit", "quit"}:
            print("Goodbye!")
            break

        state = {"messages": [HumanMessage(content=user_input)]}

        result = chatbot.invoke(
            state,
            config={"configurable": {"thread_id": thread_id}},
        )

        # Keep resolving interrupts until none are left for this turn
        while result.get("__interrupt__"):
            interrupts = result["__interrupt__"]
            resume_map = {}

            for i in interrupts:
                print(f"HITL: {i.value}")
                decision = input("Your decision: ").strip().lower()
                resume_map[i.id] = decision

            result = chatbot.invoke(
                Command(resume=resume_map),
                config={"configurable": {"thread_id": thread_id}},
            )

        messages = result["messages"]
        last_msg = messages[-1]
        print(f"Bot: {last_msg.content}\n")
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection
from psycopg.rows import dict_row

load_dotenv()
model = ChatGroq(
    model  ='llama-3.3-70b-versatile'
)

class ChatState(TypedDict):
    message : Annotated[list[BaseMessage], add_messages]

graph = StateGraph(ChatState)


def chat_node(state:ChatState):
    messgaes = state['message'] 
    response = model.invoke(messgaes)

    return {'message':[response]}

graph.add_node('chatbot', chat_node)

graph.add_edge(START, 'chatbot')
graph.add_edge('chatbot', END)



DB_URI = "postgresql://postgres:password_postgres@localhost:5432/chatbot_db?sslmode=disable"

conn = Connection.connect(DB_URI, autocommit=True, row_factory=dict_row)
checkpointer = PostgresSaver(conn)
checkpointer.setup()

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for thread in checkpointer.list(None):
        all_threads.add(thread.config['configurable']['thread_id'])
    return list(all_threads)
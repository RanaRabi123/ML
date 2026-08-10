from langgraph.graph import StateGraph , START , END
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated
from langgraph.graph.message import BaseMessage, add_messages
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()
model = ChatGroq(
    model  ='llama-3.3-70b-versatile'
)

class ChatState(TypedDict):
    message : Annotated[list[BaseMessage], add_messages]

graph = StateGraph(ChatState)


def chat_node(state:ChatState):
    messages = state['message'] 
    response = model.invoke(messages)

    return {'message':[response]}

graph.add_node('chatbot', chat_node)

graph.add_edge(START, 'chatbot')
graph.add_edge('chatbot', END)

checkpoint = InMemorySaver()

chatbot = graph.compile(checkpointer=checkpoint)
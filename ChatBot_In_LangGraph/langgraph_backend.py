from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
import google.generativeai as genai
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

import os
load_dotenv()

# ✅ Configure GenAI with your API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # ✅ Use a valid model name from your list
# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-pro-latest",  # FIXED: Replaced "gemini-pro"
#     temperature=0.7
# )

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

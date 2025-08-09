from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import google.generativeai as genai
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import sqlite3

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

conn = sqlite3.connect('chatbot.db',check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrive_all_threads():
    """Retrieve all threads from the database."""
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)
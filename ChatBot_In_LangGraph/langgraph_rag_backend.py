from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Dict, Any, TypedDict, Optional

from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import requests

load_dotenv()

# ==========================================================
# 1️⃣ Gemini Model + Embeddings
# ==========================================================
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    api_key=os.getenv("GOOGLE_API_KEY"),
)

# ==========================================================
# 2️⃣ Thread-based RAG Stores
# ==========================================================
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}
_CURRENT_THREAD_ID: Optional[str] = None  # used by rag_tool


def _get_retriever(thread_id: str):
    return _THREAD_RETRIEVERS.get(thread_id)


def ingest_pdf(
    file_bytes: bytes, thread_id: str, filename: Optional[str] = None
) -> dict:
    """Load PDF → chunk → embed → store a retriever per thread."""
    if not file_bytes:
        return {"error": "PDF upload failed: empty file."}

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
        temp.write(file_bytes)
        pdf_path = temp.name

    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=900, chunk_overlap=120
        )
        chunks = splitter.split_documents(docs)

        vectordb = FAISS.from_documents(chunks, embeddings)
        retriever = vectordb.as_retriever(search_kwargs={"k": 4})

        _THREAD_RETRIEVERS[thread_id] = retriever
        _THREAD_METADATA[thread_id] = {
            "filename": filename or os.path.basename(pdf_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {"success": True, "metadata": _THREAD_METADATA[thread_id]}

    finally:
        try:
            os.remove(pdf_path)
        except Exception:
            pass


# ==========================================================
# 3️⃣ Tools: Search + Calculator + Stock + RAG
# ==========================================================
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(a: float, b: float, op: str) -> dict:
    """Perform basic math: a (+ - * /) b"""
    try:
        result = eval(f"{a} {op} {b}")
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch stock price via AlphaVantage API."""
    key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not key:
        return {"error": "Missing ALPHAVANTAGE_API_KEY in .env"}

    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={key}"
    )
    resp = requests.get(url)
    return resp.json()


@tool
def rag_tool(query: str) -> dict:
    """Retrieve PDF context from the currently active chat thread."""
    if _CURRENT_THREAD_ID is None:
        return {"error": "No active thread. Cannot use RAG."}

    retriever = _get_retriever(_CURRENT_THREAD_ID)
    if not retriever:
        return {
            "error": "No PDF uploaded for this chat yet. Please upload a PDF first."
        }

    docs = retriever.invoke(query)
    return {
        "query": query,
        "context": [d.page_content for d in docs],
        "source": _THREAD_METADATA.get(_CURRENT_THREAD_ID),
    }


tools = [search_tool, calculator, get_stock_price, rag_tool]
llm_with_tools = llm.bind_tools(tools)

# ==========================================================
# 4️⃣ Chat State
# ==========================================================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ==========================================================
# 5️⃣ Chat Node
# ==========================================================
def chat_node(state: ChatState, config=None):
    """Main LLM node that can decide to call tools."""
    global _CURRENT_THREAD_ID

    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    if thread_id is not None:
        thread_id = str(thread_id)
    _CURRENT_THREAD_ID = thread_id  # used inside rag_tool

    system_msg = SystemMessage(
        content=(
            "You are a helpful RAG assistant.\n"
            "- If the user asks about the uploaded PDF, call `rag_tool` with the query.\n"
            "- Use web search, calculator, and stock tools when helpful.\n"
            "- If no document exists yet for this chat, politely ask the user to upload a PDF.\n"
        )
    )

    messages = [system_msg] + state["messages"]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}


# ==========================================================
# 6️⃣ SQLite Memory Persistence
# ==========================================================
conn = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# ==========================================================
# 7️⃣ LangGraph Wiring
# ==========================================================
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition, {"tools": "tools", "__end__": END})
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# ==========================================================
# 8️⃣ Helper Functions
# ==========================================================
def retrieve_all_threads():
    """Return list of all stored thread IDs from the checkpoint DB."""
    return list({
        cp.config["configurable"]["thread_id"]
        for cp in checkpointer.list(None)
    })


def thread_document_metadata(thread_id: str):
    """Return metadata of the PDF attached to a thread, if any."""
    return _THREAD_METADATA.get(str(thread_id), {})

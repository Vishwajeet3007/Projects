from typing import List, TypedDict
from pydantic import BaseModel
import re
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

# -----------------------------
# Models
# -----------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

UPPER_TH = 0.7
LOWER_TH = 0.3

# -----------------------------
# STATE
# -----------------------------
class State(TypedDict):
    question: str
    docs: List[Document]
    good_docs: List[Document]
    verdict: str
    reason: str
    strips: List[str]
    kept_strips: List[str]
    refined_context: str
    web_query: str
    web_docs: List[Document]
    answer: str


# -----------------------------
# VECTOR STORE BUILDER
# -----------------------------
def build_vector_store(pdf_paths: List[str]):
    docs = []
    for path in pdf_paths:
        docs.extend(PyPDFLoader(path).load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


# -----------------------------
# RETRIEVE
# -----------------------------
def retrieve_node(state: State):
    retriever = state["vector_store"].as_retriever(search_kwargs={"k": 4})
    return {"docs": retriever.invoke(state["question"])}


# -----------------------------
# DOC EVALUATOR
# -----------------------------
class DocEvalScore(BaseModel):
    score: float
    reason: str


doc_eval_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict retrieval evaluator.\n"
     "Return relevance score in [0.0,1.0]. Output JSON."),
    ("human", "Question: {question}\n\nChunk:\n{chunk}")
])

doc_eval_chain = doc_eval_prompt | llm.with_structured_output(DocEvalScore)


def eval_each_doc_node(state: State):
    scores = []
    good = []

    for d in state["docs"]:
        out = doc_eval_chain.invoke({
            "question": state["question"],
            "chunk": d.page_content
        })
        scores.append(out.score)
        if out.score > LOWER_TH:
            good.append(d)

    if any(s > UPPER_TH for s in scores):
        return {"good_docs": good, "verdict": "CORRECT", "reason": "High relevance found."}

    if len(scores) > 0 and all(s < LOWER_TH for s in scores):
        return {"good_docs": [], "verdict": "INCORRECT", "reason": "All chunks irrelevant."}

    return {"good_docs": good, "verdict": "AMBIGUOUS", "reason": "Partially relevant."}


# -----------------------------
# SENTENCE DECOMPOSE
# -----------------------------
def decompose(text):
    text = re.sub(r"\s+", " ", text)
    return re.split(r"(?<=[.!?])\s+", text)


class KeepOrDrop(BaseModel):
    keep: bool


filter_prompt = ChatPromptTemplate.from_messages([
    ("system", "Return keep=true only if sentence helps answer question. Output JSON."),
    ("human", "Question: {question}\nSentence:\n{sentence}")
])

filter_chain = filter_prompt | llm.with_structured_output(KeepOrDrop)


# -----------------------------
# WEB SEARCH
# -----------------------------
tavily = TavilySearchResults(max_results=5)


class WebQuery(BaseModel):
    query: str


rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rewrite into short keyword web query. Output JSON with key: query"),
    ("human", "Question: {question}")
])

rewrite_chain = rewrite_prompt | llm.with_structured_output(WebQuery)


def rewrite_query_node(state: State):
    out = rewrite_chain.invoke({"question": state["question"]})
    return {"web_query": out.query}


def web_search_node(state: State):
    results = tavily.invoke({"query": state["web_query"]})
    web_docs = []

    for r in results:
        text = f"{r.get('title')}\n{r.get('content','')}"
        web_docs.append(Document(page_content=text))

    return {"web_docs": web_docs}


# -----------------------------
# REFINE
# -----------------------------
def refine(state: State):
    if state["verdict"] == "CORRECT":
        docs = state["good_docs"]
    elif state["verdict"] == "INCORRECT":
        docs = state["web_docs"]
    else:
        docs = state["good_docs"] + state["web_docs"]

    context = "\n\n".join(d.page_content for d in docs)
    sentences = decompose(context)

    kept = []
    for s in sentences:
        if len(s) > 20:
            if filter_chain.invoke({"question": state["question"], "sentence": s}).keep:
                kept.append(s)

    return {"refined_context": "\n".join(kept)}


# -----------------------------
# GENERATE
# -----------------------------
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer only using context. If insufficient say: I don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])


def generate(state: State):
    out = (answer_prompt | llm).invoke({
        "question": state["question"],
        "context": state["refined_context"]
    })
    return {"answer": out.content}


# -----------------------------
# GRAPH BUILDER
# -----------------------------
def build_crag_app(vector_store):
    g = StateGraph(State)

    g.add_node("retrieve", retrieve_node)
    g.add_node("eval", eval_each_doc_node)
    g.add_node("rewrite", rewrite_query_node)
    g.add_node("web", web_search_node)
    g.add_node("refine", refine)
    g.add_node("generate", generate)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "eval")

    def route(state):
        if state["verdict"] == "CORRECT":
            return "refine"
        return "rewrite"

    g.add_conditional_edges("eval", route, {
        "refine": "refine",
        "rewrite": "rewrite"
    })

    g.add_edge("rewrite", "web")
    g.add_edge("web", "refine")
    g.add_edge("refine", "generate")
    g.add_edge("generate", END)

    return g.compile()

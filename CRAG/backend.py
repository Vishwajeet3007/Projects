from typing import List, TypedDict, Any
from pydantic import BaseModel
import re
import os
import json
import hashlib
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.callbacks.manager import get_openai_callback
from langgraph.graph import StateGraph, START, END
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
tavily = TavilySearchResults(max_results=5)

UPPER_TH = 0.75
LOWER_TH = 0.35
MAX_TOKEN_BUDGET = 6000

PDF_STORE_DIR = Path("data") / "pdfs"
INDEX_DIR = Path("faiss_index")
MANIFEST_PATH = INDEX_DIR / "manifest.json"


def discover_pdf_paths() -> List[str]:
    PDF_STORE_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(str(p.resolve()) for p in PDF_STORE_DIR.glob("*.pdf") if p.is_file())


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_manifest(pdf_paths: List[str]) -> dict:
    return {Path(p).name: _sha256_file(p) for p in sorted(pdf_paths)}


def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_manifest(manifest: dict) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


# =========================
# VECTOR STORE BUILDER
# =========================
def build_or_load_vector_store(pdf_paths: List[str] | None = None):
    paths = [p for p in (pdf_paths or discover_pdf_paths()) if os.path.exists(p)]

    if not paths:
        raise ValueError("No PDF files found. Upload at least one PDF.")

    current_manifest = _build_manifest(paths)
    index_ready = (INDEX_DIR / "index.faiss").exists() and (INDEX_DIR / "index.pkl").exists()

    if index_ready and _load_manifest() == current_manifest:
        return FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    docs = []
    for path in paths:
        docs.extend(PyPDFLoader(path).load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(str(INDEX_DIR))
    _save_manifest(current_manifest)

    return vector_store


# =========================
# STATE
# =========================
class State(TypedDict):
    question: str
    vector_store: Any
    expanded_queries: List[str]
    docs: List[Document]
    good_docs: List[Document]
    scores: List[float]
    verdict: str
    reason: str
    refined_context: str
    web_query: str
    web_docs: List[Document]
    answer: str
    verified: bool
    citations: List[str]
    confidence: float
    token_usage: int
    cost_warning: str


# =========================
# MULTI QUERY EXPANSION
# =========================
class QueryExpansion(BaseModel):
    queries: List[str]


expand_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Generate 3 semantically different reformulations."),
        ("human", "Question: {question}"),
    ]
)

expand_chain = expand_prompt | llm.with_structured_output(QueryExpansion)


def expand_query_node(state: State):
    out = expand_chain.invoke({"question": state["question"]})
    return {"expanded_queries": [state["question"]] + out.queries}


# =========================
# RETRIEVE
# =========================
def retrieve_node(state: State):
    retriever = state["vector_store"].as_retriever(search_kwargs={"k": 3})
    all_docs = []
    for q in state["expanded_queries"]:
        all_docs.extend(retriever.invoke(q))
    return {"docs": all_docs}


# =========================
# DOC EVALUATION
# =========================
class DocEvalScore(BaseModel):
    score: float


eval_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Return relevance score 0-1 JSON."),
        ("human", "Question: {question}\\nChunk:\\n{chunk}"),
    ]
)

eval_chain = eval_prompt | llm.with_structured_output(DocEvalScore)


def eval_node(state: State):
    scores = []
    good = []

    for d in state["docs"]:
        s = eval_chain.invoke(
            {"question": state["question"], "chunk": d.page_content}
        ).score
        scores.append(s)
        if s > LOWER_TH:
            good.append(d)

    if not scores:
        return {
            "good_docs": [],
            "verdict": "INCORRECT",
            "scores": [],
            "reason": "No chunks retrieved",
        }

    max_s = max(scores)
    avg_s = sum(scores) / len(scores)

    if max_s > UPPER_TH and avg_s > 0.5:
        verdict = "CORRECT"
    elif all(s < LOWER_TH for s in scores):
        verdict = "INCORRECT"
    else:
        verdict = "AMBIGUOUS"

    return {
        "good_docs": good,
        "scores": scores,
        "verdict": verdict,
        "reason": f"max={max_s:.2f}, avg={avg_s:.2f}",
    }


# =========================
# WEB SEARCH
# =========================
class WebQuery(BaseModel):
    query: str


rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Rewrite into keyword web query JSON."),
        ("human", "Question: {question}"),
    ]
)

rewrite_chain = rewrite_prompt | llm.with_structured_output(WebQuery)


def rewrite_node(state: State):
    return {"web_query": rewrite_chain.invoke({"question": state["question"]}).query}


def web_node(state: State):
    results = tavily.invoke({"query": state["web_query"]})
    web_docs = [Document(page_content=r.get("content", "")) for r in results]
    return {"web_docs": web_docs}


# =========================
# REFINE
# =========================
def refine_node(state: State):
    if state["verdict"] == "CORRECT":
        # High confidence in local retrieval: stay grounded in local docs.
        docs = state["good_docs"]
    elif state["verdict"] == "AMBIGUOUS":
        # Partial confidence: blend local evidence with web correction.
        docs = state["good_docs"] + state.get("web_docs", [])
    else:
        # Low confidence in local retrieval: rely on corrected web evidence.
        docs = state.get("web_docs", [])

    context = "\\n\\n".join(d.page_content for d in docs)
    return {"refined_context": context[:5000]}


# =========================
# GENERATE
# =========================
gen_prompt = ChatPromptTemplate.from_messages(
    [("system", "Answer only using context."), ("human", "Question: {question}\\nContext:\\n{context}")]
)


def generate_node(state: State):
    if not state["refined_context"].strip():
        return {"answer": "I don't know.", "token_usage": 0, "cost_warning": ""}

    with get_openai_callback() as cb:
        ans = (gen_prompt | llm).invoke(
            {"question": state["question"], "context": state["refined_context"]}
        ).content
        tokens = cb.total_tokens

    cost_warning = ""
    if tokens > MAX_TOKEN_BUDGET:
        cost_warning = "High token usage"

    return {"answer": ans, "token_usage": tokens, "cost_warning": cost_warning}


# =========================
# VERIFY
# =========================
verify_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Is answer fully supported by context? YES or NO."),
        ("human", "Answer:\\n{answer}\\nContext:\\n{context}"),
    ]
)


def verify_node(state: State):
    out = (verify_prompt | llm).invoke(
        {"answer": state["answer"], "context": state["refined_context"]}
    ).content

    return {"verified": "YES" in out.upper()}


# =========================
# CITATION
# =========================
class Citation(BaseModel):
    citations: List[str]


citation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Extract 2-4 key citation sentences from context JSON."),
        ("human", "Context:\\n{context}"),
    ]
)

citation_chain = citation_prompt | llm.with_structured_output(Citation)


def citation_node(state: State):
    if not state["refined_context"]:
        return {"citations": []}
    return {"citations": citation_chain.invoke({"context": state["refined_context"]}).citations}


# =========================
# CONFIDENCE
# =========================
confidence_prompt = ChatPromptTemplate.from_messages(
    [("system", "Return confidence 0-100."), ("human", "Answer:\\n{answer}")]
)


def uncertainty_node(state: State):
    conf = (confidence_prompt | llm).invoke({"answer": state["answer"]}).content
    num = re.findall(r"\d+", conf)
    return {"confidence": float(num[0]) if num else 50}


# =========================
# GRAPH
# =========================
def build_crag_app(vector_store):
    g = StateGraph(State)

    g.add_node("expand", expand_query_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("eval", eval_node)
    g.add_node("rewrite", rewrite_node)
    g.add_node("web", web_node)
    g.add_node("refine", refine_node)
    g.add_node("generate", generate_node)
    g.add_node("verify", verify_node)
    g.add_node("citation", citation_node)
    g.add_node("uncertainty", uncertainty_node)

    g.add_edge(START, "expand")
    g.add_edge("expand", "retrieve")
    g.add_edge("retrieve", "eval")

    def route_eval(state):
        if state["verdict"] == "CORRECT":
            return "refine"
        if state["verdict"] == "AMBIGUOUS":
            return "rewrite_ambiguous"
        return "rewrite_incorrect"

    g.add_conditional_edges(
        "eval",
        route_eval,
        {
            "refine": "refine",
            "rewrite_ambiguous": "rewrite",
            "rewrite_incorrect": "rewrite",
        },
    )

    g.add_edge("rewrite", "web")
    g.add_edge("web", "refine")
    g.add_edge("refine", "generate")
    g.add_edge("generate", "verify")
    g.add_edge("verify", "citation")
    g.add_edge("citation", "uncertainty")
    g.add_edge("uncertainty", END)

    return g.compile()

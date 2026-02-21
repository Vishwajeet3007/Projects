from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4")

# ---------------------
# STATE
# ---------------------

class StudyState(TypedDict):
    user_input: str
    mode: str
    answer: str
    score: float
    feedback: str

# ---------------------
# ROUTER NODE
# ---------------------

def router_node(state: StudyState):
    text = state["user_input"].lower()

    if "explain" in text:
        mode = "explanation"
    elif "error" in text or "debug" in text:
        mode = "code"
    elif "interview" in text:
        mode = "interview"
    else:
        mode = "explanation"

    return {"mode": mode}

# ---------------------
# EXPLANATION NODE
# ---------------------

def explanation_node(state: StudyState):
    prompt = f"""
    Explain clearly with:
    - Definition
    - Example
    - Simple analogy
    - If coding, include code example

    Question: {state['user_input']}
    """

    response = llm([HumanMessage(content=prompt)])
    return {"answer": response.content}

# ---------------------
# CODE NODE
# ---------------------

def code_node(state: StudyState):
    prompt = f"""
    Debug or improve this code/question:

    {state['user_input']}

    Give corrected version and explanation.
    """

    response = llm([HumanMessage(content=prompt)])
    return {"answer": response.content}

# ---------------------
# INTERVIEW NODE
# ---------------------

def interview_node(state: StudyState):
    prompt = f"""
    Act as a technical interviewer.
    Evaluate the following answer:

    {state['user_input']}

    Provide:
    - Score out of 10
    - Strengths
    - Weaknesses
    - Improvement suggestions
    """

    response = llm([HumanMessage(content=prompt)])
    return {"answer": response.content}

# ---------------------
# EVALUATION NODE
# ---------------------

def evaluation_node(state: StudyState):
    # Can later add structured scoring logic
    return {"feedback": "Evaluation complete."}

# ---------------------
# BUILD GRAPH
# ---------------------

builder = StateGraph(StudyState)

builder.add_node("router", router_node)
builder.add_node("explanation", explanation_node)
builder.add_node("code", code_node)
builder.add_node("interview", interview_node)
builder.add_node("evaluation", evaluation_node)

builder.set_entry_point("router")

builder.add_conditional_edges(
    "router",
    lambda state: state["mode"],
    {
        "explanation": "explanation",
        "code": "code",
        "interview": "interview",
    },
)

builder.add_edge("explanation", "evaluation")
builder.add_edge("code", "evaluation")
builder.add_edge("interview", "evaluation")

builder.add_edge("evaluation", END)

graph = builder.compile()
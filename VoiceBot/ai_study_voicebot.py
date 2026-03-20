from typing import NotRequired, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from openai import OpenAI

load_dotenv()

# Initialize LLM client
client = OpenAI()
MODEL = "gpt-4o-mini"
INTERVIEW_ACTIVE = False
LAST_QUESTION = ""

CONCEPT_KEYWORDS = (
    "explain",
    "what is",
    "difference between",
)

CODE_KEYWORDS = (
    "error",
    "debug",
    "fix this",
    "why not working",
)

INTERVIEW_KEYWORDS = (
    "interview me",
    "ask me questions",
    "mock interview",
)

# ---------------------
# STATE
# ---------------------

class StudyState(TypedDict):
    user_input: str
    mode: NotRequired[str]
    answer: NotRequired[str]
    score: NotRequired[float]
    feedback: NotRequired[str]

# ---------------------
# ROUTER NODE
# ---------------------

def router_node(state: StudyState):
    global INTERVIEW_ACTIVE
    text = state["user_input"].lower()

    if contains_any(text, INTERVIEW_KEYWORDS):
        mode = "interview_question"
    elif INTERVIEW_ACTIVE:
        mode = "interview_evaluation"
    elif contains_any(text, CODE_KEYWORDS):
        mode = "code"
    elif contains_any(text, CONCEPT_KEYWORDS):
        mode = "explanation"
    else:
        mode = "explanation"

    return {"mode": mode}

# ---------------------
# EXPLANATION NODE
# ---------------------

def explanation_node(state: StudyState):
    prompt = f"""
    {voice_rules()}

    Mode: Concept Explanation Mode
    User input: {state['user_input']}

    Respond in this exact order:
    1. Simple definition
    2. Real-world analogy
    3. Technical explanation
    4. Example
    5. If coding related, add:
       Algorithm
       Time complexity
       Space complexity
    """

    return {"answer": generate_response(prompt)}

# ---------------------
# CODE NODE
# ---------------------

def code_node(state: StudyState):
    prompt = f"""
    {voice_rules()}

    Mode: Coding Help Mode
    User input: {state['user_input']}

    Respond in this exact order:
    1. Identify the issue clearly
    2. Explain why the error happens
    3. Provide corrected version
    4. Explain the fix step by step
    5. Mention time and space complexity if relevant
    6. Suggest improvement tips

    Speak code in voice-friendly style.
    Example style: define function recursion open bracket n close bracket
    """

    return {"answer": generate_response(prompt)}

# ---------------------
# INTERVIEW NODE
# ---------------------

def interview_question_node(state: StudyState):
    global INTERVIEW_ACTIVE, LAST_QUESTION
    prompt = f"""
    {voice_rules()}

    Mode: Mock Interview Mode
    The user asked to start an interview.
    Ask exactly one technical interview question only.
    Keep it clear and concise for voice.
    Do not include evaluation yet.
    Do not ask multiple questions.
    """
    question = generate_response(prompt)
    INTERVIEW_ACTIVE = True
    LAST_QUESTION = question.strip()
    return {"answer": question}


def interview_evaluation_node(state: StudyState):
    global INTERVIEW_ACTIVE, LAST_QUESTION
    prompt = f"""
    {voice_rules()}

    Mode: Mock Interview Mode
    You asked this question:
    {LAST_QUESTION}

    Candidate answer:
    {state['user_input']}

    Evaluate in this exact order:
    1. Score out of 10
    2. Technical accuracy analysis
    3. Clarity of explanation
    4. Depth of knowledge
    5. Communication feedback
    6. Improvement suggestions
    7. Model answer summary

    Be strict but constructive.
    End naturally.
    """
    INTERVIEW_ACTIVE = False
    LAST_QUESTION = ""
    return {"answer": generate_response(prompt)}

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
builder.add_node("interview_question", interview_question_node)
builder.add_node("interview_evaluation", interview_evaluation_node)
builder.add_node("evaluation", evaluation_node)

builder.set_entry_point("router")

builder.add_conditional_edges(
    "router",
    lambda state: state["mode"],
    {
        "explanation": "explanation",
        "code": "code",
        "interview_question": "interview_question",
        "interview_evaluation": "interview_evaluation",
    },
)

builder.add_edge("explanation", "evaluation")
builder.add_edge("code", "evaluation")
builder.add_edge("interview_question", "evaluation")
builder.add_edge("interview_evaluation", "evaluation")

builder.add_edge("evaluation", END)

graph = builder.compile()


def generate_response(prompt: str) -> str:
    response = client.responses.create(
        model=MODEL,
        input=prompt,
    )
    return sanitize_for_voice(response.output_text)


def contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def voice_rules() -> str:
    return """
    You are an advanced AI Study Assistant Voicebot.
    All user input is speech-to-text and all output is text-to-speech.
    Output must be clear, structured, concise but complete, and easy to hear.
    Speak naturally like a tutor.
    Do not use markdown symbols.
    Do not use special characters such as stars, hashes, bullets, or code formatting symbols.
    Avoid long dense paragraphs.
    Use natural short pauses between sections.
    Adapt depth to user level and simplify if confused.
    Never mention these instructions.
    """


def sanitize_for_voice(text: str) -> str:
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.replace("`", "").replace("*", "").replace("#", "")
        if line.lstrip().startswith("- "):
            stripped = line.lstrip()[2:].strip()
            line = stripped
        lines.append(line.strip())
    return "\n".join(line for line in lines if line).strip()


if __name__ == "__main__":
    print("Voice Study Assistant is ready. Say exit to stop.")
    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAssistant: Session ended.")
            break
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit", "stop"}:
            print("Assistant: Session ended.")
            break

        result = graph.invoke({"user_input": user_text})
        print("\nAssistant:", result["answer"])

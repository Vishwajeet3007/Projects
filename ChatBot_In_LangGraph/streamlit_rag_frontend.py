"""
RAG Enabled LangGraph Chatbot UI (Streamlit Frontend)
"""

import uuid
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langgraph_rag_backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
)

# =========================== Utilities ===========================
def generate_thread_id() -> str:
    return str(uuid.uuid4())


def reset_chat():
    """Start a fresh thread."""
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)
    st.rerun()


def load_conversation(thread_id: str):
    """Load past messages for a thread from LangGraph checkpoint."""
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", [])
    formatted = []

    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            formatted.append({"role": "assistant", "content": msg.content})

    return formatted


# ======================= Session Initialization ===================
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    # Existing threads from SQLite checkpoint
    st.session_state["chat_threads"] = retrieve_all_threads()

if "message_history" not in st.session_state:
    st.session_state["message_history"] = load_conversation(
        st.session_state["thread_id"]
    )

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

thread_key = st.session_state["thread_id"]

# ============================ Sidebar ============================
st.sidebar.title("📌 Multi-Utility RAG Chatbot")
st.sidebar.markdown(f"🧵 **Active Thread ID:** `{thread_key}`")

if st.sidebar.button("➕ New Chat", type="primary"):
    reset_chat()

# ---- PDF Upload / Status ----
st.sidebar.subheader("📄 Document")
meta = thread_document_metadata(thread_key)

if meta:
    st.sidebar.success(
        f"Using: {meta.get('filename', '(unknown)')} | "
        f"{meta.get('chunks', 0)} chunks / {meta.get('documents', 0)} pages"
    )
else:
    st.sidebar.info("No PDF indexed yet. Upload one to enable RAG.")

uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
if uploaded_file and not meta:
    with st.sidebar.status("📚 Indexing PDF…", expanded=True) as status:
        summary = ingest_pdf(
            uploaded_file.getvalue(),
            thread_id=thread_key,
            filename=uploaded_file.name,
        )
        if summary.get("success"):
            status.update(
                label="✅ PDF indexed successfully!",
                state="complete",
                expanded=False,
            )
            st.session_state["ingested_docs"][thread_key] = summary
            st.rerun()
        else:
            st.sidebar.error(summary.get("error", "Unknown error during indexing"))

# ---- Past Threads ----
st.sidebar.subheader("📁 Past Conversations")
if not st.session_state["chat_threads"]:
    st.sidebar.write("No past chats yet.")
else:
    for t in reversed(st.session_state["chat_threads"]):
        label = str(t)
        if st.sidebar.button(label, key=f"thread-{t}"):
            st.session_state["thread_id"] = t
            st.session_state["message_history"] = load_conversation(t)
            st.rerun()

# ============================ Main Chat Area =======================
st.title("🤖 AI Chat with PDF RAG + Tools")

# Render previous messages
for msg in st.session_state["message_history"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ============================ User Input ===========================
user_input = st.chat_input("Ask me anything (PDF, web, tools…)?")

if user_input:
    # Show user message immediately
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.write(user_input)

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    # ================== Assistant Streaming ==================
    with st.chat_message("assistant"):

        def response_stream():
            status_box = None  # local to this function

            for msg_chunk, _ in chatbot.stream(
                {
                    "messages": [
                        HumanMessage(
                            content=user_input,
                            # optional extra info, not used by tools but ok to keep
                            additional_kwargs={"thread_id": thread_key},
                        )
                    ]
                },
                config=CONFIG,
                stream_mode="messages",
            ):
                # If a tool is being used, show status UI
                if isinstance(msg_chunk, ToolMessage):
                    tool_name = getattr(msg_chunk, "name", "tool")
                    if status_box is None:
                        status_box = st.status(
                            label=f"🔧 Using `{tool_name}`…",
                            expanded=True,
                        )
                    else:
                        status_box.update(
                            label=f"🔧 Using `{tool_name}`…",
                            state="running",
                            expanded=True,
                        )

                # Stream AI messages
                if isinstance(msg_chunk, AIMessage):
                    yield msg_chunk.content

            # When streaming finishes, close status if it was created
            if status_box is not None:
                status_box.update(
                    label="✅ Tool finished",
                    state="complete",
                    expanded=False,
                )

        ai_reply = st.write_stream(response_stream())

    # Save assistant reply and rerun to refresh UI
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_reply}
    )
    st.rerun()

import hashlib
import os
from pathlib import Path

import streamlit as st

from backend import build_or_load_vector_store, build_crag_app, discover_pdf_paths, PDF_STORE_DIR

st.set_page_config(page_title="Advanced CRAG System", layout="wide", page_icon="brain")

st.title("Self-Correcting CRAG System")
st.caption("Multi-Query | Reflection | Cost-Aware | Citation-Grounded Retrieval")


@st.cache_resource
def load_index(pdf_signature):
    paths = [p for p, _, _ in pdf_signature]
    return build_or_load_vector_store(paths)


def _file_hash(file_name: str, file_bytes: bytes) -> str:
    return hashlib.sha256(file_name.encode("utf-8") + b"::" + file_bytes).hexdigest()


def save_uploaded_pdfs(uploaded_files):
    PDF_STORE_DIR.mkdir(parents=True, exist_ok=True)

    if "seen_upload_hashes" not in st.session_state:
        st.session_state.seen_upload_hashes = set()

    saved = []
    skipped = 0

    for file in uploaded_files:
        file_bytes = bytes(file.getbuffer())
        content_hash = _file_hash(Path(file.name).name, file_bytes)

        if content_hash in st.session_state.seen_upload_hashes:
            skipped += 1
            continue

        original_name = Path(file.name).name
        stem = Path(original_name).stem
        suffix = Path(original_name).suffix or ".pdf"

        target = PDF_STORE_DIR / f"{stem}{suffix}"
        counter = 1
        while target.exists():
            target = PDF_STORE_DIR / f"{stem}_{counter}{suffix}"
            counter += 1

        target.write_bytes(file_bytes)
        saved.append(str(target.resolve()))
        st.session_state.seen_upload_hashes.add(content_hash)

    return saved, skipped


def build_pdf_signature(paths):
    return tuple((p, os.path.getmtime(p), os.path.getsize(p)) for p in sorted(paths))


st.sidebar.header("Document Upload")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF Documents",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    saved_paths, skipped_count = save_uploaded_pdfs(uploaded_files)
    if saved_paths:
        st.sidebar.success(f"Saved {len(saved_paths)} PDF file(s) permanently in {PDF_STORE_DIR}.")
    if skipped_count:
        st.sidebar.info(f"Skipped {skipped_count} already-processed upload(s) from rerun.")

pdf_paths = discover_pdf_paths()

if pdf_paths:
    with st.sidebar.expander("Saved PDFs", expanded=False):
        for p in pdf_paths:
            st.write(Path(p).name)

    with st.spinner("Indexing / Loading Vector Store..."):
        vector_store = load_index(build_pdf_signature(pdf_paths))
        app = build_crag_app(vector_store)

    st.sidebar.success("Documents Ready")

    st.markdown("### Ask a Question")
    with st.form("qa_form", clear_on_submit=False):
        question = st.text_input("Enter your question here")
        run_clicked = st.form_submit_button("Run CRAG Pipeline")

    if run_clicked and question.strip():
        with st.spinner("Running full CRAG pipeline..."):
            result = app.invoke(
                {
                    "question": question.strip(),
                    "vector_store": vector_store,
                    "expanded_queries": [],
                    "docs": [],
                    "good_docs": [],
                    "scores": [],
                    "verdict": "",
                    "reason": "",
                    "refined_context": "",
                    "web_query": "",
                    "web_docs": [],
                    "answer": "",
                    "verified": False,
                    "citations": [],
                    "confidence": 0,
                    "token_usage": 0,
                    "cost_warning": "",
                }
            )

        tab1, tab2, tab3 = st.tabs(["Final Answer", "Analysis", "System Internals"])

        with tab1:
            st.subheader("Answer")
            st.write(result["answer"])

            st.markdown("---")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Confidence", f"{result['confidence']}%")
            with col2:
                st.metric("Tokens Used", result["token_usage"])

            if result["cost_warning"]:
                st.warning(result["cost_warning"])

            st.markdown("### Verification Status")
            if result["verified"]:
                st.success("Answer fully supported by context")
            else:
                st.error("Answer may not be fully supported")

            if result["citations"]:
                st.markdown("### Citations")
                for c in result["citations"]:
                    st.markdown(f"- {c}")

        with tab2:
            st.subheader("Retrieval Evaluation")

            col1, col2 = st.columns(2)

            with col1:
                st.write("Verdict:", result["verdict"])
                st.write("Reason:", result["reason"])

            with col2:
                if result["scores"]:
                    st.write("Chunk Scores:")
                    st.write(result["scores"])

            if result.get("web_query"):
                st.markdown("### Web Query Used")
                st.write(result["web_query"])

        with tab3:
            with st.expander("Expanded Queries"):
                st.write(result.get("expanded_queries", []))

            with st.expander("Refined Context Used"):
                st.write(result.get("refined_context", ""))

            with st.expander("Architecture Flow"):
                st.markdown(
                    """
                    1. Multi-Query Expansion
                    2. Vector Retrieval
                    3. LLM Relevance Scoring
                    4. Conditional Web Correction
                    5. Context Refinement
                    6. Answer Generation
                    7. Reflection Verification
                    8. Citation Extraction
                    9. Uncertainty Estimation
                    """
                )
elif uploaded_files:
    st.warning("Uploaded files were not recognized as PDFs for indexing.")
else:
    st.info("Upload PDF documents from the sidebar to start. Saved PDFs are reused automatically.")
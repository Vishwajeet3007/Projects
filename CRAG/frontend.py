import streamlit as st
import tempfile
from backend import build_vector_store, build_crag_app

st.set_page_config(page_title="CRAG System", layout="wide")

st.title("🔎 Corrective RAG (CRAG) System")

# Upload PDFs
uploaded_files = st.file_uploader(
    "Upload PDF Documents",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Processing PDFs..."):
        pdf_paths = []
        for file in uploaded_files:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(file.read())
            pdf_paths.append(tmp.name)

        vector_store = build_vector_store(pdf_paths)
        app = build_crag_app(vector_store)

    st.success("Documents Indexed Successfully ✅")

    question = st.text_input("Ask a Question")

    if st.button("Run CRAG") and question:
        with st.spinner("Running CRAG pipeline..."):
            result = app.invoke({
                "question": question,
                "docs": [],
                "good_docs": [],
                "verdict": "",
                "reason": "",
                "strips": [],
                "kept_strips": [],
                "refined_context": "",
                "web_query": "",
                "web_docs": [],
                "answer": "",
                "vector_store": vector_store
            })

        st.subheader("Verdict")
        st.info(result["verdict"])

        st.subheader("Answer")
        st.write(result["answer"])

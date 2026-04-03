from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


import streamlit as st
from streamlit_agraph import Config as AGraphConfig
from streamlit_agraph import Edge, Node, agraph

from graph_rag.bootstrap import ApplicationServices
from graph_rag.config import get_config
from graph_rag.events import GraphUpdateEvent
from graph_rag.ui.graph_state import apply_graph_event, default_graph_state, mark_retrieved_nodes, snapshot_graph_state


st.set_page_config(page_title="Hierarchical Graph RAG", layout="wide")

config = get_config()
services = ApplicationServices(config)


def initialize_state() -> None:
    if "graph_state" not in st.session_state:
        st.session_state.graph_state = default_graph_state()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "graph_render_version" not in st.session_state:
        st.session_state.graph_render_version = 0


initialize_state()


def save_uploaded_files(uploaded_files) -> list[Path]:
    saved_paths: list[Path] = []
    for uploaded_file in uploaded_files:
        destination = config.pdf_dir / uploaded_file.name
        destination.write_bytes(uploaded_file.read())
        saved_paths.append(destination)
    return saved_paths


def clear_similarity_edges_from_state() -> None:
    st.session_state.graph_state["edges"] = {
        edge_id: edge
        for edge_id, edge in st.session_state.graph_state["edges"].items()
        if not edge_id.endswith(":SIMILAR")
    }


def render_graph(graph_placeholder) -> None:
    snapshot = snapshot_graph_state(st.session_state.graph_state)
    with graph_placeholder.container():
        st.subheader("Graph Visualization")
        if not snapshot["nodes"]:
            st.info("Upload PDFs and start ingestion to watch the graph grow in real time.")
            return

        nodes = [
            Node(
                id=node["id"],
                label=node["label"],
                size=node["size"],
                color=node["color"],
                title=node["title"],
            )
            for node in snapshot["nodes"].values()
        ]
        edges = [
            Edge(
                source=edge["source"],
                target=edge["target"],
                label=edge["label"],
                color=edge["color"],
            )
            for edge in snapshot["edges"].values()
        ]
        graph_config = AGraphConfig(
            width="100%",
            height=520,
            directed=True,
            physics=True,
            hierarchical=False,
            nodeHighlightBehavior=True,
            highlightColor="#FACC15",
            collapsible=True,
        )
        st.session_state.graph_render_version += 1
        agraph(
            nodes=nodes,
            edges=edges,
            config=graph_config,
        )


def render_logs(log_placeholder) -> None:
    logs = st.session_state.graph_state["logs"]
    with log_placeholder.container():
        st.subheader("Logs")
        st.text_area(
            "Processing Logs",
            value="\n".join(logs) if logs else "No processing events yet.",
            height=220,
            disabled=True,
            label_visibility="collapsed",
        )


def update_live_views(graph_placeholder, log_placeholder) -> None:
    render_graph(graph_placeholder)
    render_logs(log_placeholder)


st.title("Hierarchical Graph RAG")
st.caption("Live graph construction during ingestion, summary-first retrieval, and adaptive context compression.")

left_col, right_col = st.columns([1.35, 1.0], gap="large")

with left_col:
    st.subheader("PDF Ingestion")
    uploads = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    ingest_col, rebuild_col = st.columns(2)
    ingest_clicked = ingest_col.button("Ingest Uploaded PDFs", use_container_width=True)
    rebuild_clicked = rebuild_col.button("Rebuild Similarity Edges", use_container_width=True)
    graph_placeholder = st.empty()
    logs_placeholder = st.empty()
    update_live_views(graph_placeholder, logs_placeholder)


def handle_graph_event(event) -> None:
    apply_graph_event(st.session_state.graph_state, event)
    update_live_views(graph_placeholder, logs_placeholder)


if ingest_clicked:
    if not uploads:
        st.warning("Upload at least one PDF first.")
    else:
        st.session_state.graph_state = default_graph_state()
        update_live_views(graph_placeholder, logs_placeholder)
        saved = save_uploaded_files(uploads)
        summaries = []
        with st.spinner("Building graph in real time..."):
            for path in saved:
                summaries.append(services.graph_builder.store_in_graph(path, event_callback=handle_graph_event))
            edge_count = services.edge_creator.create_similarity_edges(event_callback=handle_graph_event)
        st.success(f"Ingested {len(summaries)} PDF(s) and created {edge_count} similarity edges.")
        st.json(summaries)

if rebuild_clicked:
    clear_similarity_edges_from_state()
    update_live_views(graph_placeholder, logs_placeholder)
    with st.spinner("Recomputing summary similarity graph..."):
        edge_count = services.edge_creator.create_similarity_edges(event_callback=handle_graph_event)
    st.success(f"Created {edge_count} similarity edges.")

with right_col:
    st.subheader("Chat Interface")
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("sections"):
                st.caption(message["sections"])
            if message.get("usage"):
                st.caption(message["usage"])

    with st.form("chat_form", clear_on_submit=True):
        query = st.text_input("Ask about your ingested documents")
        submit_query = st.form_submit_button("Send", use_container_width=True)

    if submit_query:
        if not query.strip():
            st.warning("Enter a question first.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": query})
            with st.spinner("Retrieving graph context and generating answer..."):
                response = services.query_engine.generate_answer(query)
            mark_retrieved_nodes(st.session_state.graph_state, response.used_sections)
            apply_graph_event(
                st.session_state.graph_state,
                GraphUpdateEvent(
                    event_type="log",
                    message=(
                        "Query token usage: "
                        f"query={response.usage.query_tokens}, "
                        f"context={response.usage.context_tokens}, "
                        f"prompt={response.usage.prompt_tokens}, "
                        f"answer={response.usage.answer_tokens}, "
                        f"total={response.usage.total_tokens}"
                    ),
                ),
            )
            update_live_views(graph_placeholder, logs_placeholder)
            section_summary = ", ".join(section.section_id for section in response.used_sections) or "No sections used"
            usage_summary = (
                f"Tokens used: query={response.usage.query_tokens}, "
                f"context={response.usage.context_tokens}, "
                f"prompt={response.usage.prompt_tokens}, "
                f"answer={response.usage.answer_tokens}, "
                f"total={response.usage.total_tokens}, "
                f"budget={response.usage.available_context_budget}/{response.usage.model_context_limit}"
            )
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": response.answer,
                    "sections": f"Used nodes: {section_summary}",
                    "usage": usage_summary,
                }
            )
            st.rerun()


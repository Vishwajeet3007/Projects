from __future__ import annotations

from copy import deepcopy

from graph_rag.events import GraphUpdateEvent
from graph_rag.models import RetrievalSection


SECTION_COLOR = "#3B82F6"
SUMMARY_COLOR = "#22C55E"
CURRENT_COLOR = "#FACC15"
RETRIEVED_COLOR = "#EF4444"
SIMILAR_EDGE_COLOR = "#9CA3AF"
HAS_SUMMARY_EDGE_COLOR = "#86EFAC"
HIGHLIGHT_EVENTS = {
    "section_created",
    "summary_generated",
    "similarity_edge_created",
    "processing_highlight",
    "query_highlight",
}


def default_graph_state() -> dict:
    return {
        "nodes": {},
        "edges": {},
        "logs": [],
        "current_node_id": None,
    }


def snapshot_graph_state(graph_state: dict) -> dict:
    return {
        "nodes": deepcopy(graph_state["nodes"]),
        "edges": deepcopy(graph_state["edges"]),
        "logs": list(graph_state["logs"]),
        "current_node_id": graph_state.get("current_node_id"),
    }


def _set_node_color(node: dict, color: str) -> None:
    node["color"] = color


def _reset_current_highlight(graph_state: dict) -> None:
    current_node_id = graph_state.get("current_node_id")
    if current_node_id and current_node_id in graph_state["nodes"]:
        node = graph_state["nodes"][current_node_id]
        if node["type"] == "summary":
            _set_node_color(node, SUMMARY_COLOR)
        elif node["type"] == "section":
            _set_node_color(node, SECTION_COLOR)


def apply_graph_event(graph_state: dict, event: GraphUpdateEvent) -> dict:
    payload = event.payload
    if event.event_type in HIGHLIGHT_EVENTS:
        _reset_current_highlight(graph_state)

    if event.event_type == "section_created":
        node_id = payload["node_id"]
        graph_state["nodes"][node_id] = {
            "id": node_id,
            "label": payload["label"],
            "type": "section",
            "color": CURRENT_COLOR,
            "size": 24,
            "title": payload.get("title", payload["label"]),
        }
        graph_state["current_node_id"] = node_id
    elif event.event_type == "summary_generated":
        section_id = payload["section_id"]
        summary_id = payload["summary_id"]
        if section_id in graph_state["nodes"]:
            graph_state["nodes"][section_id]["label"] = payload["section_label"]
            graph_state["nodes"][section_id]["title"] = payload["section_title"]
        graph_state["nodes"][summary_id] = {
            "id": summary_id,
            "label": payload["summary_label"],
            "type": "summary",
            "color": CURRENT_COLOR,
            "size": 18,
            "title": payload.get("summary_title", payload["summary_label"]),
        }
        edge_id = f"{section_id}->{summary_id}:HAS_SUMMARY"
        graph_state["edges"][edge_id] = {
            "source": section_id,
            "target": summary_id,
            "label": "HAS_SUMMARY",
            "color": HAS_SUMMARY_EDGE_COLOR,
        }
        graph_state["current_node_id"] = summary_id
    elif event.event_type == "similarity_edge_created":
        edge_id = f'{payload["source"]}->{payload["target"]}:SIMILAR'
        graph_state["edges"][edge_id] = {
            "source": payload["source"],
            "target": payload["target"],
            "label": f'SIMILAR ({payload["score"]:.2f})',
            "color": SIMILAR_EDGE_COLOR,
        }
        graph_state["current_node_id"] = payload["target"]
    elif event.event_type == "processing_highlight":
        node_id = payload["node_id"]
        if node_id in graph_state["nodes"]:
            _set_node_color(graph_state["nodes"][node_id], CURRENT_COLOR)
            graph_state["current_node_id"] = node_id
    elif event.event_type == "query_highlight":
        for section in payload["sections"]:
            section_id = section["section_id"]
            summary_id = section["summary_id"]
            if section_id in graph_state["nodes"]:
                _set_node_color(graph_state["nodes"][section_id], RETRIEVED_COLOR)
            if summary_id in graph_state["nodes"]:
                _set_node_color(graph_state["nodes"][summary_id], RETRIEVED_COLOR)
        graph_state["current_node_id"] = None

    graph_state["logs"].append(event.message)
    graph_state["logs"] = graph_state["logs"][-200:]
    return graph_state


def mark_retrieved_nodes(graph_state: dict, sections: list[RetrievalSection]) -> dict:
    payload = {
        "sections": [
            {
                "section_id": section.section_id,
                "summary_id": section.summary_id,
            }
            for section in sections
        ]
    }
    return apply_graph_event(
        graph_state,
        GraphUpdateEvent(
            event_type="query_highlight",
            message="Retrieved nodes highlighted",
            payload=payload,
        ),
    )

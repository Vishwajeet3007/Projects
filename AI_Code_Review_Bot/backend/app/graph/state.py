"""State definitions for the LangGraph workflow."""

from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict


class ReviewGraphState(TypedDict, total=False):
    """Mutable state passed through the LangGraph workflow."""

    source_type: str
    source_identifier: str
    files: list[dict[str, Any]]
    metadata: dict[str, Any]
    static_analysis: dict[str, Any]
    rag_context: list[str]
    analyzer_output: dict[str, Any]
    bug_report: dict[str, Any]
    complexity_report: dict[str, Any]
    security_report: dict[str, Any]
    optimization_report: dict[str, Any]
    documentation_report: dict[str, Any]
    test_report: dict[str, Any]
    scoring_report: dict[str, Any]
    final_report: dict[str, Any]

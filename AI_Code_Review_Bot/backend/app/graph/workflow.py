"""LangGraph workflow for the multi-agent code review pipeline."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.factory import AgentFactory
from app.agents.prompts import (
    ANALYZER_PROMPT,
    BUG_FINDER_PROMPT,
    COMPLEXITY_PROMPT,
    DOCUMENTATION_PROMPT,
    OPTIMIZER_PROMPT,
    REVIEWER_PROMPT,
    SCORING_PROMPT,
    SECURITY_PROMPT,
    TEST_GENERATOR_PROMPT,
)
from app.graph.state import ReviewGraphState



def build_review_workflow(agent_factory: AgentFactory):
    """Construct and compile the LangGraph review workflow."""

    graph = StateGraph(ReviewGraphState)

    async def analyzer_node(state: ReviewGraphState) -> dict[str, Any]:
        payload = {
            "files": state["files"],
            "source_type": state["source_type"],
            "static_analysis": state["static_analysis"],
            "rag_context": state["rag_context"],
        }
        return {
            "analyzer_output": await agent_factory.run_json_agent(
                system_prompt=ANALYZER_PROMPT,
                task_payload=payload,
            )
        }

    async def bug_node(state: ReviewGraphState) -> dict[str, Any]:
        return {
            "bug_report": await agent_factory.run_json_agent(
                system_prompt=BUG_FINDER_PROMPT,
                task_payload=_specialist_payload(state),
            )
        }

    async def complexity_node(state: ReviewGraphState) -> dict[str, Any]:
        return {
            "complexity_report": await agent_factory.run_json_agent(
                system_prompt=COMPLEXITY_PROMPT,
                task_payload=_specialist_payload(state),
            )
        }

    async def security_node(state: ReviewGraphState) -> dict[str, Any]:
        return {
            "security_report": await agent_factory.run_json_agent(
                system_prompt=SECURITY_PROMPT,
                task_payload=_specialist_payload(state),
            )
        }

    async def optimizer_node(state: ReviewGraphState) -> dict[str, Any]:
        return {
            "optimization_report": await agent_factory.run_json_agent(
                system_prompt=OPTIMIZER_PROMPT,
                task_payload=_specialist_payload(state),
            )
        }

    async def documentation_node(state: ReviewGraphState) -> dict[str, Any]:
        return {
            "documentation_report": await agent_factory.run_json_agent(
                system_prompt=DOCUMENTATION_PROMPT,
                task_payload=_specialist_payload(state),
            )
        }

    async def test_node(state: ReviewGraphState) -> dict[str, Any]:
        return {
            "test_report": await agent_factory.run_json_agent(
                system_prompt=TEST_GENERATOR_PROMPT,
                task_payload=_specialist_payload(state),
            )
        }

    async def scoring_node(state: ReviewGraphState) -> dict[str, Any]:
        return {
            "scoring_report": await agent_factory.run_json_agent(
                system_prompt=SCORING_PROMPT,
                task_payload=_specialist_payload(state),
            )
        }

    async def reviewer_node(state: ReviewGraphState) -> dict[str, Any]:
        reviewer_payload = {
            "files": state["files"],
            "static_analysis": state["static_analysis"],
            "analyzer_output": state.get("analyzer_output", {}),
            "bug_report": state.get("bug_report", {}),
            "complexity_report": state.get("complexity_report", {}),
            "security_report": state.get("security_report", {}),
            "optimization_report": state.get("optimization_report", {}),
            "documentation_report": state.get("documentation_report", {}),
            "test_report": state.get("test_report", {}),
            "scoring_report": state.get("scoring_report", {}),
            "rag_context": state.get("rag_context", []),
        }
        return {
            "final_report": await agent_factory.run_json_agent(
                system_prompt=REVIEWER_PROMPT,
                task_payload=reviewer_payload,
            )
        }

    graph.add_node("analyzer", analyzer_node)
    graph.add_node("bug_finder", bug_node)
    graph.add_node("complexity", complexity_node)
    graph.add_node("security", security_node)
    graph.add_node("optimizer", optimizer_node)
    graph.add_node("documentation", documentation_node)
    graph.add_node("test_generator", test_node)
    graph.add_node("code_scoring", scoring_node)
    graph.add_node("reviewer", reviewer_node)

    graph.add_edge(START, "analyzer")
    for node_name in (
        "bug_finder",
        "complexity",
        "security",
        "optimizer",
        "documentation",
        "test_generator",
        "code_scoring",
    ):
        graph.add_edge("analyzer", node_name)
        graph.add_edge(node_name, "reviewer")
    graph.add_edge("reviewer", END)

    return graph.compile()



def _specialist_payload(state: ReviewGraphState) -> dict[str, Any]:
    """Build the common payload sent to specialist agents."""

    return {
        "files": state["files"],
        "static_analysis": state["static_analysis"],
        "analyzer_output": state.get("analyzer_output", {}),
        "rag_context": state.get("rag_context", []),
    }

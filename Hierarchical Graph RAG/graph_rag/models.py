from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PageRecord:
    page_id: str
    page_number: int
    text: str


@dataclass(slots=True)
class SectionRecord:
    doc_id: str
    doc_name: str
    page_id: str
    page_number: int
    section_id: str
    text: str
    token_count: int


@dataclass(slots=True)
class SummaryRecord:
    summary_id: str
    section_id: str
    text: str
    embedding: list[float]


@dataclass(slots=True)
class RetrievalSection:
    section_id: str
    page_id: str
    page_number: int
    text: str
    token_count: int
    summary_id: str
    summary_text: str
    score: float
    doc_id: str
    doc_name: str
    compression_applied: bool = False
    context_text: str = ""
    context_token_count: int = 0


@dataclass(slots=True)
class QueryUsage:
    query_tokens: int
    context_tokens: int
    answer_tokens: int
    total_tokens: int
    prompt_tokens: int = 0
    available_context_budget: int = 0
    model_context_limit: int = 0


@dataclass(slots=True)
class QueryResponse:
    answer: str
    used_sections: list[RetrievalSection]
    usage: QueryUsage
    metadata: dict[str, Any] = field(default_factory=dict)

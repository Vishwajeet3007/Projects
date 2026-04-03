from __future__ import annotations

from graph_rag.config import AppConfig
from graph_rag.ingestion.summarizer import SectionSummarizer
from graph_rag.models import RetrievalSection
from graph_rag.rag.compression import compress_section_if_needed
from graph_rag.token_utils import count_tokens


SYSTEM_PROMPT = (
    "Answer strictly from the supplied context. "
    "Be precise, cite the supporting section ids inline, "
    "and say when the evidence is incomplete."
)


def calculate_context_budget(query: str, config: AppConfig) -> dict[str, int]:
    query_tokens = count_tokens(query, config.answer_model)
    header = f"User Query: {query}\n\nRetrieved Context:\n"
    header_tokens = count_tokens(header, config.answer_model)
    system_tokens = count_tokens(SYSTEM_PROMPT, config.answer_model)
    user_suffix = f"\n\nAnswer the question: {query}"
    user_suffix_tokens = count_tokens(user_suffix, config.answer_model)
    hard_limit_budget = (
        config.answer_model_context_limit
        - config.reserved_generation_tokens
        - config.token_guard_margin
        - system_tokens
        - header_tokens
        - user_suffix_tokens
    )
    available_context_budget = max(0, min(config.max_context_tokens, hard_limit_budget))
    return {
        "query_tokens": query_tokens,
        "header_tokens": header_tokens,
        "system_prompt_tokens": system_tokens,
        "user_suffix_tokens": user_suffix_tokens,
        "available_context_budget": available_context_budget,
        "model_context_limit": config.answer_model_context_limit,
        "reserved_generation_tokens": config.reserved_generation_tokens,
        "token_guard_margin": config.token_guard_margin,
    }


def build_context(
    query: str,
    retrieved_sections: list[RetrievalSection],
    summarizer: SectionSummarizer,
    config: AppConfig,
) -> tuple[str, list[RetrievalSection], int, dict[str, int]]:
    budget_info = calculate_context_budget(query, config)
    budget = budget_info["available_context_budget"]
    context_chunks: list[str] = []
    used_sections: list[RetrievalSection] = []
    running_tokens = 0

    for section in retrieved_sections:
        updated = compress_section_if_needed(section, summarizer, config)
        chunk = (
            f"Document: {updated.doc_name}\n"
            f"Page: {updated.page_number}\n"
            f"Section ID: {updated.section_id}\n"
            f"Summary ID: {updated.summary_id}\n"
            f"Relevance Score: {updated.score:.4f}\n"
            f"Content:\n{updated.context_text}"
        )
        chunk_tokens = count_tokens(chunk, config.answer_model)
        if running_tokens + chunk_tokens > budget:
            continue
        context_chunks.append(chunk)
        used_sections.append(updated)
        running_tokens += chunk_tokens

    header = f"User Query: {query}\n\nRetrieved Context:\n"
    context = header + "\n\n---\n\n".join(context_chunks)
    total_tokens = count_tokens(context, config.answer_model)
    return context, used_sections, total_tokens, budget_info

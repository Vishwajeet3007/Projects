from __future__ import annotations

from graph_rag.config import AppConfig
from graph_rag.ingestion.summarizer import SectionSummarizer
from graph_rag.models import RetrievalSection
from graph_rag.token_utils import count_tokens


def compress_section_if_needed(
    section: RetrievalSection,
    summarizer: SectionSummarizer,
    config: AppConfig,
) -> RetrievalSection:
    if section.token_count <= config.compression_threshold_tokens:
        section.context_text = section.text
        section.context_token_count = section.token_count
        section.compression_applied = False
        return section

    summary = summarizer.summarize_section(section.text)
    section.context_text = f"[Compressed summary]\n{summary}"
    section.context_token_count = count_tokens(section.context_text, config.answer_model)
    section.compression_applied = True
    return section

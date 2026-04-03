from __future__ import annotations

import logging

from openai import OpenAI

from graph_rag.cache import PersistentCache
from graph_rag.config import AppConfig


logger = logging.getLogger(__name__)


class SectionSummarizer:
    def __init__(self, config: AppConfig, cache: PersistentCache) -> None:
        self.config = config
        self.cache = cache
        self.client = OpenAI(api_key=config.openai_api_key) if config.has_openai else None

    def summarize_section(self, section_text: str) -> str:
        cached = self.cache.get_summary(section_text, self.config.summarizer_model)
        if cached:
            logger.debug("Summary cache hit for model=%s", self.config.summarizer_model)
            return cached

        if not self.client:
            summary = self._fallback_summary(section_text)
            self.cache.set_summary(section_text, self.config.summarizer_model, summary)
            return summary

        response = self.client.responses.create(
            model=self.config.summarizer_model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "Summarize the provided section in 2 concise sentences. "
                        "Preserve entities, numbers, and technical relationships. "
                        "Optimize for retrieval, not style."
                    ),
                },
                {"role": "user", "content": section_text},
            ],
        )
        summary = (response.output_text or "").strip()
        if not summary:
            summary = self._fallback_summary(section_text)
        self.cache.set_summary(section_text, self.config.summarizer_model, summary)
        return summary

    @staticmethod
    def _fallback_summary(section_text: str) -> str:
        sentences = [part.strip() for part in section_text.replace("\n", " ").split(".") if part.strip()]
        if not sentences:
            return section_text[:240]
        return ". ".join(sentences[:2]).strip()[:320]

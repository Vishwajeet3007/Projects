from __future__ import annotations

import json
import logging
from pathlib import Path

from openai import OpenAI

from graph_rag.cache import PersistentCache
from graph_rag.config import AppConfig
from graph_rag.models import QueryResponse, QueryUsage
from graph_rag.rag.context_builder import SYSTEM_PROMPT, build_context
from graph_rag.rag.hierarchical_retriever import HierarchicalRetriever
from graph_rag.token_utils import count_tokens


logger = logging.getLogger(__name__)


class QueryEngine:
    def __init__(
        self,
        config: AppConfig,
        retriever: HierarchicalRetriever,
        summarizer,
        cache: PersistentCache,
    ) -> None:
        self.config = config
        self.retriever = retriever
        self.summarizer = summarizer
        self.cache = cache
        self.client = OpenAI(api_key=config.openai_api_key) if config.has_openai else None
        self.usage_log_path = Path(config.log_dir) / "query_usage.jsonl"

    def generate_answer(self, query: str) -> QueryResponse:
        retrieved_sections, retrieval_metadata = self.retriever.retrieve(query)
        query_tokens = count_tokens(query, self.config.answer_model)
        if not retrieved_sections:
            usage = QueryUsage(
                query_tokens=query_tokens,
                context_tokens=0,
                answer_tokens=0,
                total_tokens=query_tokens,
                prompt_tokens=query_tokens,
                available_context_budget=0,
                model_context_limit=self.config.answer_model_context_limit,
            )
            self._log_usage(query, usage, [], retrieval_metadata)
            return QueryResponse(
                answer="No graph context is available yet. Ingest one or more PDFs first.",
                used_sections=[],
                usage=usage,
                metadata=retrieval_metadata,
            )

        context, used_sections, context_tokens, budget_info = build_context(
            query=query,
            retrieved_sections=retrieved_sections,
            summarizer=self.summarizer,
            config=self.config,
        )
        user_prompt = f"{context}\n\nAnswer the question: {query}"
        answer = self._answer_with_llm(user_prompt, used_sections)
        answer_tokens = count_tokens(answer, self.config.answer_model)
        user_prompt_tokens = count_tokens(user_prompt, self.config.answer_model)
        prompt_tokens = budget_info["system_prompt_tokens"] + user_prompt_tokens
        usage = QueryUsage(
            query_tokens=query_tokens,
            context_tokens=context_tokens,
            answer_tokens=answer_tokens,
            total_tokens=prompt_tokens + answer_tokens,
            prompt_tokens=prompt_tokens,
            available_context_budget=budget_info["available_context_budget"],
            model_context_limit=budget_info["model_context_limit"],
        )
        retrieval_metadata.update(
            {
                "token_budget": budget_info,
                "context_within_limit": prompt_tokens + self.config.reserved_generation_tokens <= self.config.answer_model_context_limit,
                "used_section_count": len(used_sections),
            }
        )
        self._log_usage(query, usage, used_sections, retrieval_metadata)
        return QueryResponse(
            answer=answer,
            used_sections=used_sections,
            usage=usage,
            metadata=retrieval_metadata,
        )

    def _answer_with_llm(self, user_prompt: str, used_sections) -> str:
        if not self.client:
            context_preview = "\n".join(
                f"- {section.doc_name} p.{section.page_number}: {section.summary_text}"
                for section in used_sections
            )
            return (
                "OPENAI_API_KEY is not configured, so this is a retrieval-only response.\n\n"
                f"Most relevant evidence:\n{context_preview}"
            )

        response = self.client.responses.create(
            model=self.config.answer_model,
            input=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        )
        return (response.output_text or "").strip()

    def _log_usage(self, query: str, usage: QueryUsage, used_sections, retrieval_metadata: dict) -> None:
        payload = {
            "query": query,
            "usage": {
                "query_tokens": usage.query_tokens,
                "context_tokens": usage.context_tokens,
                "prompt_tokens": usage.prompt_tokens,
                "answer_tokens": usage.answer_tokens,
                "available_context_budget": usage.available_context_budget,
                "model_context_limit": usage.model_context_limit,
                "total_tokens": usage.total_tokens,
            },
            "used_sections": [
                {
                    "section_id": section.section_id,
                    "summary_id": section.summary_id,
                    "page_number": section.page_number,
                    "doc_name": section.doc_name,
                    "score": section.score,
                    "compressed": section.compression_applied,
                }
                for section in used_sections
            ],
            "retrieval_metadata": retrieval_metadata,
        }
        with self.usage_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

        logger.info(
            "Token usage | query=%s | query_tokens=%s | context_tokens=%s | prompt_tokens=%s | answer_tokens=%s | total_tokens=%s | budget=%s/%s",
            query[:80],
            usage.query_tokens,
            usage.context_tokens,
            usage.prompt_tokens,
            usage.answer_tokens,
            usage.total_tokens,
            usage.available_context_budget,
            usage.model_context_limit,
        )

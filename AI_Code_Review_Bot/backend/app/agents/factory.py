"""Shared LLM helpers for agent execution."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.core.config import Settings

logger = logging.getLogger(__name__)


class AgentFactory:
    """Factory for invoking specialist review agents."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required to run AI review agents.")

        self._llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.1,
            timeout=max(180, settings.request_timeout_seconds),
        )

    async def run_json_agent(
        self,
        *,
        system_prompt: str,
        task_payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Invoke an LLM agent and parse its JSON response."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(task_payload, indent=2)),
        ]
        response = await self._llm.ainvoke(messages)
        content = response.content if isinstance(response.content, str) else str(response.content)

        try:
            return self._parse_json_content(content)
        except json.JSONDecodeError as exc:
            logger.warning("Agent returned non-JSON output: %s", content)
            raise ValueError("Agent returned invalid JSON output.") from exc

    def _parse_json_content(self, content: str) -> dict[str, Any]:
        """Parse JSON with a few safe fallbacks for common LLM formatting drift."""

        candidates = [self._extract_json(content)]
        object_match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if object_match:
            candidates.append(object_match.group(0).strip())

        for candidate in candidates:
            if not candidate:
                continue
            normalized = self._sanitize_json(candidate)
            try:
                parsed = json.loads(normalized)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed

        raise json.JSONDecodeError("Unable to parse JSON", content, 0)

    @staticmethod
    def _extract_json(content: str) -> str:
        """Extract raw JSON from plain or fenced model output."""

        fenced_match = re.search(r"```json\s*(.*?)\s*```", content, flags=re.DOTALL)
        if fenced_match:
            return fenced_match.group(1)
        return content.strip()

    @staticmethod
    def _sanitize_json(content: str) -> str:
        """Remove a few common JSON formatting issues from model output."""

        sanitized = content.strip()
        sanitized = re.sub(r",\s*([}\]])", r"\1", sanitized)
        return sanitized

"""Request and response schemas for code reviews."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator


class ReviewIssue(BaseModel):
    """Represents a specific review finding."""

    title: str
    description: str
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    file_path: str | None = None
    line: int | None = None
    recommendation: str | None = None


class ReviewReport(BaseModel):
    """Structured review output returned by the reviewer agent."""

    bugs: list[ReviewIssue] = Field(default_factory=list)
    security_issues: list[ReviewIssue] = Field(default_factory=list)
    time_complexity: str = ""
    space_complexity: str = ""
    optimizations: list[str] = Field(default_factory=list)
    refactored_code: str = ""
    documentation: str = ""
    unit_tests: str = ""
    code_quality_score: float = Field(default=0, ge=0, le=10)
    final_summary: str = ""

    @field_validator(
        "time_complexity",
        "space_complexity",
        "refactored_code",
        "documentation",
        "unit_tests",
        "final_summary",
        mode="before",
    )
    @classmethod
    def normalize_text_fields(cls, value: Any) -> str:
        """Normalize model output into plain text fields."""

        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "\n".join(str(item) for item in value)
        if isinstance(value, dict):
            return json.dumps(value, indent=2)
        return str(value)

    @field_validator("optimizations", mode="before")
    @classmethod
    def normalize_optimizations(cls, value: Any) -> list[str]:
        """Normalize optimization suggestions into strings."""

        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if not isinstance(value, list):
            return [str(value)]

        normalized: list[str] = []
        for item in value:
            if isinstance(item, str):
                normalized.append(item)
                continue

            if isinstance(item, dict):
                parts = []
                for key in (
                    "title",
                    "type",
                    "category",
                    "issue",
                    "recommendation",
                    "suggestion",
                    "description",
                    "reason",
                    "impact",
                ):
                    field_value = item.get(key)
                    if field_value:
                        label = key.replace("_", " ").capitalize()
                        parts.append(f"{label}: {field_value}")

                normalized.append("; ".join(parts) if parts else json.dumps(item))
                continue

            normalized.append(str(item))

        return normalized

    @field_validator("code_quality_score", mode="before")
    @classmethod
    def normalize_score(cls, value: Any) -> float:
        """Normalize score values produced by the LLM."""

        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return 0
        if isinstance(value, dict):
            for key in ("score", "code_quality_score", "rating"):
                nested_value = value.get(key)
                if isinstance(nested_value, (int, float)):
                    return float(nested_value)
                if isinstance(nested_value, str):
                    try:
                        return float(nested_value.strip())
                    except ValueError:
                        continue
        return 0


class CodeFile(BaseModel):
    """A source file supplied for review."""

    path: str
    content: str
    language: str | None = None


class SourceBundle(BaseModel):
    """Normalized source bundle used by the workflow."""

    source_type: Literal["raw_code", "repository", "pull_request"]
    identifier: str
    files: list[CodeFile]
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReviewMetadata(BaseModel):
    """Execution metadata stored with a review result."""

    source_type: Literal["raw_code", "repository", "pull_request"]
    source_identifier: str
    reviewed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent_versions: dict[str, str] = Field(default_factory=dict)
    static_analysis: dict[str, Any] = Field(default_factory=dict)
    rag_context: list[str] = Field(default_factory=list)
    execution_notes: list[str] = Field(default_factory=list)


class StoredReview(BaseModel):
    """Database representation of a stored review."""

    metadata: ReviewMetadata
    report: ReviewReport


class ReviewCodeRequest(BaseModel):
    """Request body for raw code reviews."""

    code: str
    filename: str = "snippet.py"
    language: str = "python"
    repository_context: str | None = None


class ReviewRepoRequest(BaseModel):
    """Request body for GitHub repository reviews."""

    repo_url: HttpUrl
    branch: str | None = None


class ReviewPRRequest(BaseModel):
    """Request body for GitHub pull request reviews."""

    pr_url: HttpUrl


class ReviewResponse(BaseModel):
    """API response wrapper."""

    metadata: ReviewMetadata
    report: ReviewReport

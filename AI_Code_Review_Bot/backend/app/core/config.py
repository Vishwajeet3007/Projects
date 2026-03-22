"""Application configuration."""

from functools import lru_cache
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed settings for the application."""

    app_name: str = "AI Code Review Bot"
    app_env: Literal["development", "staging", "production"] = "development"
    debug: bool = True
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        alias="CORS_ORIGINS",
    )

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        alias="OPENAI_EMBEDDING_MODEL",
    )

    github_token: str = Field(default="", alias="GITHUB_TOKEN")
    github_api_base_url: str = Field(
        default="https://api.github.com",
        alias="GITHUB_API_BASE_URL",
    )

    mongodb_uri: str = Field(
        default="mongodb://mongodb:27017",
        alias="MONGODB_URI",
    )
    mongodb_database: str = Field(default="ai_code_review_bot", alias="MONGODB_DATABASE")

    vector_store_path: str = Field(default="data/faiss_index", alias="VECTOR_STORE_PATH")
    knowledge_base_path: str = Field(
        default="docs/knowledge",
        alias="KNOWLEDGE_BASE_PATH",
    )
    max_repo_files: int = Field(default=40, alias="MAX_REPO_FILES")
    max_file_chars: int = Field(default=16000, alias="MAX_FILE_CHARS")
    prompt_file_limit: int = Field(default=12, alias="PROMPT_FILE_LIMIT")
    prompt_file_chars: int = Field(default=3000, alias="PROMPT_FILE_CHARS")
    prompt_total_chars: int = Field(default=18000, alias="PROMPT_TOTAL_CHARS")
    request_timeout_seconds: int = Field(default=60, alias="REQUEST_TIMEOUT_SECONDS")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("debug", mode="before")
    @classmethod
    def normalize_debug(cls, value: Any) -> bool:
        """Handle common non-boolean DEBUG environment values safely."""

        if isinstance(value, bool):
            return value
        if value is None:
            return True

        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "on", "debug", "development"}:
            return True
        if normalized in {"0", "false", "no", "off", "release", "prod", "production"}:
            return False
        return True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings object."""

    return Settings()

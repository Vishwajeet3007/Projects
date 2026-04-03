from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


ROOT_DIR = Path(__file__).resolve().parent.parent
PACKAGE_DIR = ROOT_DIR / "graph_rag"
DATA_DIR = PACKAGE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
CACHE_DIR = ROOT_DIR / ".cache"
LOG_DIR = ROOT_DIR / "logs"


@dataclass(slots=True)
class AppConfig:
    neo4j_uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", "password"))
    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    embedding_model: str = field(default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    summarizer_model: str = field(default_factory=lambda: os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4.1-mini"))
    answer_model: str = field(default_factory=lambda: os.getenv("OPENAI_ANSWER_MODEL", "gpt-4.1-mini"))
    answer_model_context_limit: int = field(default_factory=lambda: int(os.getenv("ANSWER_MODEL_CONTEXT_LIMIT", "128000")))
    reserved_generation_tokens: int = field(default_factory=lambda: int(os.getenv("RESERVED_GENERATION_TOKENS", "2000")))
    token_guard_margin: int = field(default_factory=lambda: int(os.getenv("TOKEN_GUARD_MARGIN", "500")))
    similarity_threshold: float = field(default_factory=lambda: float(os.getenv("SIMILARITY_THRESHOLD", "0.75")))
    similarity_top_k: int = field(default_factory=lambda: int(os.getenv("SIMILARITY_TOP_K", "3")))
    traversal_hops: int = field(default_factory=lambda: int(os.getenv("SIMILAR_TRAVERSAL_HOPS", "2")))
    compression_threshold_tokens: int = field(default_factory=lambda: int(os.getenv("COMPRESSION_THRESHOLD_TOKENS", "300")))
    max_context_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_CONTEXT_TOKENS", "1800")))
    section_soft_limit: int = field(default_factory=lambda: int(os.getenv("SECTION_SOFT_LIMIT_TOKENS", "450")))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    pdf_dir: Path = field(default_factory=lambda: Path(os.getenv("PDF_DIR", str(PDF_DIR))))
    cache_dir: Path = field(default_factory=lambda: Path(os.getenv("CACHE_DIR", str(CACHE_DIR))))
    log_dir: Path = field(default_factory=lambda: Path(os.getenv("LOG_DIR", str(LOG_DIR))))
    namespace: str = field(default_factory=lambda: os.getenv("GRAPH_NAMESPACE", "default"))

    def ensure_directories(self) -> None:
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key)


def get_config() -> AppConfig:
    config = AppConfig()
    config.ensure_directories()
    return config

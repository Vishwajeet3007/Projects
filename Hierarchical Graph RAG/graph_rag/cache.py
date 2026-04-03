from __future__ import annotations

import hashlib
from typing import Any

from diskcache import Cache


class PersistentCache:
    def __init__(self, cache_dir: str) -> None:
        self.cache = Cache(cache_dir)

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join((text or "").split())

    @classmethod
    def _hash_text(cls, prefix: str, text: str) -> str:
        payload = f"{prefix}:{cls._normalize_text(text)}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def get_summary(self, text: str, model: str) -> str | None:
        return self.cache.get(self._hash_text(f"summary:{model}", text))

    def set_summary(self, text: str, model: str, summary: str) -> None:
        self.cache.set(self._hash_text(f"summary:{model}", text), summary)

    def get_embedding(self, text: str, model: str) -> list[float] | None:
        return self.cache.get(self._hash_text(f"embedding:{model}", text))

    def set_embedding(self, text: str, model: str, embedding: list[float]) -> None:
        self.cache.set(self._hash_text(f"embedding:{model}", text), embedding)

    def get_value(self, namespace: str, key: str) -> Any:
        return self.cache.get(self._hash_text(namespace, key))

    def set_value(self, namespace: str, key: str, value: Any) -> None:
        self.cache.set(self._hash_text(namespace, key), value)

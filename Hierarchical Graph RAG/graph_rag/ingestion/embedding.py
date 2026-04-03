from __future__ import annotations

import hashlib
import logging
import math
import re

from openai import OpenAI

from graph_rag.cache import PersistentCache
from graph_rag.config import AppConfig


logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, config: AppConfig, cache: PersistentCache) -> None:
        self.config = config
        self.cache = cache
        self.client = OpenAI(api_key=config.openai_api_key) if config.has_openai else None

    def generate_embedding(self, text: str) -> list[float]:
        cached = self.cache.get_embedding(text, self.config.embedding_model)
        if cached is not None:
            logger.debug("Embedding cache hit for model=%s", self.config.embedding_model)
            return cached

        if self.client:
            response = self.client.embeddings.create(
                model=self.config.embedding_model,
                input=text,
            )
            embedding = list(response.data[0].embedding)
        else:
            embedding = self._fallback_embedding(text)

        self.cache.set_embedding(text, self.config.embedding_model, embedding)
        return embedding

    @staticmethod
    def _fallback_embedding(text: str, dimensions: int = 256) -> list[float]:
        vector = [0.0] * dimensions
        tokens = re.findall(r"\w+", text.lower())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % dimensions
            sign = 1 if digest[4] % 2 == 0 else -1
            vector[index] += float(sign)

        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

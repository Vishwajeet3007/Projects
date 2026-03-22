"""FAISS-backed retrieval helpers for coding best practices."""

from __future__ import annotations

from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from app.core.config import Settings


class KnowledgeRetriever:
    """Builds and queries a FAISS index of best-practice documents."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._vector_store: FAISS | None = None

    def load(self) -> None:
        """Load or build the FAISS index."""

        if self._vector_store is not None or not self._settings.openai_api_key:
            return

        vector_store_path = Path(self._settings.vector_store_path)
        embeddings = OpenAIEmbeddings(
            model=self._settings.openai_embedding_model,
            api_key=self._settings.openai_api_key,
        )

        if vector_store_path.exists():
            self._vector_store = FAISS.load_local(
                folder_path=str(vector_store_path),
                embeddings=embeddings,
                allow_dangerous_deserialization=True,
            )
            return

        knowledge_dir = Path(self._settings.knowledge_base_path)
        documents = []
        for path in knowledge_dir.glob("*.md"):
            documents.extend(TextLoader(str(path), encoding="utf-8").load())

        if not documents:
            return

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=120)
        split_documents = splitter.split_documents(documents)
        self._vector_store = FAISS.from_documents(split_documents, embeddings)
        vector_store_path.mkdir(parents=True, exist_ok=True)
        self._vector_store.save_local(str(vector_store_path))

    def retrieve_context(self, query: str, *, top_k: int = 4) -> list[str]:
        """Retrieve relevant best-practice snippets."""

        self.load()
        if self._vector_store is None:
            return []
        docs = self._vector_store.similarity_search(query, k=top_k)
        return [doc.page_content for doc in docs]

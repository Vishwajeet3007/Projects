from __future__ import annotations

from graph_rag.cache import PersistentCache
from graph_rag.config import AppConfig, get_config
from graph_rag.graph.edge_creator import EdgeCreator
from graph_rag.graph.graph_builder import GraphBuilder
from graph_rag.graph.graph_retriever import GraphRetriever
from graph_rag.graph.neo4j_connection import Neo4jConnection
from graph_rag.ingestion.embedding import EmbeddingService
from graph_rag.ingestion.summarizer import SectionSummarizer
from graph_rag.logging_utils import configure_logging
from graph_rag.rag.hierarchical_retriever import HierarchicalRetriever
from graph_rag.rag.query_engine import QueryEngine


class ApplicationServices:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        configure_logging(self.config.log_dir, self.config.log_level)
        self.cache = PersistentCache(str(self.config.cache_dir))
        self.connection = Neo4jConnection(self.config)
        self.summarizer = SectionSummarizer(self.config, self.cache)
        self.embedder = EmbeddingService(self.config, self.cache)
        self.graph_builder = GraphBuilder(
            config=self.config,
            connection=self.connection,
            summarizer=self.summarizer,
            embedder=self.embedder,
        )
        self.edge_creator = EdgeCreator(self.config, self.connection)
        self.graph_retriever = GraphRetriever(self.config, self.connection)
        self.hierarchical_retriever = HierarchicalRetriever(self.graph_retriever, self.embedder)
        self.query_engine = QueryEngine(
            config=self.config,
            retriever=self.hierarchical_retriever,
            summarizer=self.summarizer,
            cache=self.cache,
        )

    def close(self) -> None:
        self.connection.close()

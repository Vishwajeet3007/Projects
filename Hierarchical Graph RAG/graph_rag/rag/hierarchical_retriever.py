from __future__ import annotations

from graph_rag.graph.graph_retriever import GraphRetriever
from graph_rag.ingestion.embedding import EmbeddingService
from graph_rag.models import RetrievalSection


class HierarchicalRetriever:
    def __init__(self, graph_retriever: GraphRetriever, embedder: EmbeddingService) -> None:
        self.graph_retriever = graph_retriever
        self.embedder = embedder

    def retrieve(self, query: str) -> tuple[list[RetrievalSection], dict]:
        query_embedding = self.embedder.generate_embedding(query)
        best_summary = self.graph_retriever.find_best_summary_node(query_embedding)
        if not best_summary:
            return [], {"query_embedding_dim": len(query_embedding)}

        neighbor_nodes = self.graph_retriever.traverse_similar_nodes(
            best_summary["summary_id"],
            limit=2,
        )
        summary_ids = [best_summary["summary_id"], *[item["summary_id"] for item in neighbor_nodes]]
        sections = self.graph_retriever.retrieve_sections(summary_ids, query_embedding)

        metadata = {
            "best_summary_id": best_summary["summary_id"],
            "neighbor_summary_ids": [item["summary_id"] for item in neighbor_nodes],
            "query_embedding_dim": len(query_embedding),
        }
        return sections[:3], metadata

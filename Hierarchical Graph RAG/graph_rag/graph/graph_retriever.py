from __future__ import annotations

from graph_rag.config import AppConfig
from graph_rag.graph.edge_creator import cosine_similarity
from graph_rag.graph.neo4j_connection import Neo4jConnection
from graph_rag.models import RetrievalSection


class GraphRetriever:
    def __init__(self, config: AppConfig, connection: Neo4jConnection) -> None:
        self.config = config
        self.connection = connection

    def find_best_summary_node(self, query_embedding: list[float]) -> dict | None:
        summaries = self.connection.run_query(
            """
            MATCH (m:Summary)
            RETURN m.summary_id AS summary_id,
                   m.text AS text,
                   m.embedding AS embedding,
                   m.section_id AS section_id,
                   m.doc_id AS doc_id,
                   m.doc_name AS doc_name
            """
        )
        best: dict | None = None
        best_score = -1.0
        for record in summaries:
            embedding = record.get("embedding") or []
            if not embedding:
                continue
            score = cosine_similarity(query_embedding, embedding)
            if score > best_score:
                best = {**record, "score": score}
                best_score = score
        return best

    def traverse_similar_nodes(self, summary_id: str, limit: int = 2) -> list[dict]:
        return self.connection.run_query(
            """
            MATCH (:Summary {summary_id: $summary_id})-[r:SIMILAR]->(neighbor:Summary)
            RETURN neighbor.summary_id AS summary_id,
                   neighbor.text AS text,
                   neighbor.section_id AS section_id,
                   neighbor.doc_id AS doc_id,
                   neighbor.doc_name AS doc_name,
                   r.score AS score
            ORDER BY r.score DESC
            LIMIT $limit
            """,
            {"summary_id": summary_id, "limit": limit},
        )

    def retrieve_sections(self, summary_ids: list[str], query_embedding: list[float]) -> list[RetrievalSection]:
        results = self.connection.run_query(
            """
            MATCH (section:Section)-[:HAS_SUMMARY]->(summary:Summary)
            WHERE summary.summary_id IN $summary_ids
            RETURN section.section_id AS section_id,
                   section.page_id AS page_id,
                   section.text AS text,
                   section.token_count AS token_count,
                   section.page_number AS page_number,
                   section.doc_id AS doc_id,
                   section.doc_name AS doc_name,
                   summary.summary_id AS summary_id,
                   summary.text AS summary_text,
                   summary.embedding AS embedding
            """,
            {"summary_ids": summary_ids},
        )

        sections: list[RetrievalSection] = []
        for record in results:
            score = cosine_similarity(query_embedding, record.get("embedding") or [])
            sections.append(
                RetrievalSection(
                    section_id=record["section_id"],
                    page_id=record["page_id"],
                    page_number=record["page_number"],
                    text=record["text"],
                    token_count=record["token_count"],
                    summary_id=record["summary_id"],
                    summary_text=record["summary_text"],
                    score=score,
                    doc_id=record["doc_id"],
                    doc_name=record["doc_name"],
                )
            )

        sections.sort(key=lambda item: item.score, reverse=True)
        return sections

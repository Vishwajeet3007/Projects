from __future__ import annotations

import logging
import math

from graph_rag.config import AppConfig
from graph_rag.events import GraphEventCallback, GraphUpdateEvent
from graph_rag.graph.neo4j_connection import Neo4jConnection


logger = logging.getLogger(__name__)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left)) or 1.0
    right_norm = math.sqrt(sum(b * b for b in right)) or 1.0
    return numerator / (left_norm * right_norm)


class EdgeCreator:
    def __init__(self, config: AppConfig, connection: Neo4jConnection) -> None:
        self.config = config
        self.connection = connection

    def create_similarity_edges(self, event_callback: GraphEventCallback | None = None) -> int:
        summaries = self.connection.run_query(
            """
            MATCH (s:Summary)
            RETURN s.summary_id AS summary_id,
                   s.embedding AS embedding,
                   s.section_id AS section_id
            """
        )
        edges_created = 0

        self.connection.execute_write("MATCH ()-[r:SIMILAR]->() DELETE r")

        for source in summaries:
            candidates: list[tuple[str, float]] = []
            source_embedding = source.get("embedding") or []
            source_id = source["summary_id"]
            source_section = source["section_id"]
            self._emit(
                event_callback,
                "processing_highlight",
                f"Processing similarity links for {source_id}",
                {"node_id": source_id},
            )

            for target in summaries:
                if target["summary_id"] == source_id or target["section_id"] == source_section:
                    continue
                target_embedding = target.get("embedding") or []
                if not source_embedding or not target_embedding:
                    continue
                score = cosine_similarity(source_embedding, target_embedding)
                if score >= self.config.similarity_threshold:
                    candidates.append((target["summary_id"], score))

            candidates.sort(key=lambda item: item[1], reverse=True)
            for neighbor_id, score in candidates[: self.config.similarity_top_k]:
                self.connection.execute_write(
                    """
                    MATCH (a:Summary {summary_id: $source_id})
                    MATCH (b:Summary {summary_id: $target_id})
                    MERGE (a)-[r:SIMILAR]->(b)
                    SET r.score = $score
                    """,
                    {"source_id": source_id, "target_id": neighbor_id, "score": score},
                )
                edges_created += 1
                self._emit(
                    event_callback,
                    "similarity_edge_created",
                    f"Edge created: {source_id} -> {neighbor_id}",
                    {"source": source_id, "target": neighbor_id, "score": score},
                )

        logger.info("Created %s similarity edges", edges_created)
        return edges_created

    @staticmethod
    def _emit(
        event_callback: GraphEventCallback | None,
        event_type: str,
        message: str,
        payload: dict | None = None,
    ) -> None:
        if event_callback is None:
            return
        event_callback(GraphUpdateEvent(event_type=event_type, message=message, payload=payload or {}))

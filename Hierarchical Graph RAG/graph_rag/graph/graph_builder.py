from __future__ import annotations

import logging
from pathlib import Path

from graph_rag.config import AppConfig
from graph_rag.events import GraphEventCallback, GraphUpdateEvent
from graph_rag.graph.neo4j_connection import Neo4jConnection
from graph_rag.ingestion.embedding import EmbeddingService
from graph_rag.ingestion.pdf_loader import load_pdf
from graph_rag.ingestion.section_splitter import split_into_sections
from graph_rag.ingestion.summarizer import SectionSummarizer
from graph_rag.models import SectionRecord, SummaryRecord
from graph_rag.token_utils import count_tokens


logger = logging.getLogger(__name__)


class GraphBuilder:
    def __init__(
        self,
        config: AppConfig,
        connection: Neo4jConnection,
        summarizer: SectionSummarizer,
        embedder: EmbeddingService,
    ) -> None:
        self.config = config
        self.connection = connection
        self.summarizer = summarizer
        self.embedder = embedder

    def store_in_graph(
        self,
        pdf_path: str | Path,
        event_callback: GraphEventCallback | None = None,
    ) -> dict[str, int | str]:
        doc_id, doc_name, pages = load_pdf(pdf_path)
        self.connection.ensure_schema()
        self._clear_document_subgraph(doc_id)
        self.connection.execute_write(
            """
            MERGE (d:Document {doc_id: $doc_id})
            SET d.name = $name, d.namespace = $namespace, d.updated_at = datetime()
            """,
            {"doc_id": doc_id, "name": doc_name, "namespace": self.config.namespace},
        )
        self._emit(
            event_callback,
            "log",
            f"Document queued: {doc_name}",
            {"doc_id": doc_id},
        )

        section_count = 0
        summary_count = 0

        for page in pages:
            self.connection.execute_write(
                """
                MATCH (d:Document {doc_id: $doc_id})
                MERGE (p:Page {page_id: $page_id})
                SET p.page_number = $page_number, p.text = $text, p.doc_id = $doc_id
                MERGE (d)-[:HAS_PAGE]->(p)
                """,
                {
                    "doc_id": doc_id,
                    "page_id": page.page_id,
                    "page_number": page.page_number,
                    "text": page.text,
                },
            )

            for index, section_text in enumerate(split_into_sections(page.text), start=1):
                section_id = f"{page.page_id}-section-{index}"
                summary_id = f"{section_id}-summary"
                token_count = count_tokens(section_text, self.config.answer_model)
                section = SectionRecord(
                    doc_id=doc_id,
                    doc_name=doc_name,
                    page_id=page.page_id,
                    page_number=page.page_number,
                    section_id=section_id,
                    text=section_text,
                    token_count=token_count,
                )

                self._write_section_node(section)
                self._emit(
                    event_callback,
                    "section_created",
                    f"Section created: {section.section_id}",
                    {
                        "node_id": section.section_id,
                        "label": f"Section {index}\nP{page.page_number}",
                        "title": self._truncate(section.text, 240),
                    },
                )

                summary_text = self.summarizer.summarize_section(section.text)
                embedding = self.embedder.generate_embedding(summary_text)
                summary = SummaryRecord(
                    summary_id=summary_id,
                    section_id=section_id,
                    text=summary_text,
                    embedding=embedding,
                )
                self._write_summary_node(section, summary)
                self._emit(
                    event_callback,
                    "summary_generated",
                    f"Summary generated: {summary.summary_id}",
                    {
                        "section_id": section.section_id,
                        "summary_id": summary.summary_id,
                        "section_label": f"Section {index}\nP{page.page_number}",
                        "section_title": self._truncate(section.text, 240),
                        "summary_label": self._truncate(summary.text, 54),
                        "summary_title": summary.text,
                    },
                )
                self._emit(
                    event_callback,
                    "log",
                    f"Embedding created: {summary.summary_id}",
                    {
                        "summary_id": summary.summary_id,
                        "dimension": len(summary.embedding),
                    },
                )
                section_count += 1
                summary_count += 1

        logger.info(
            "Stored document %s with %s sections and %s summaries",
            doc_name,
            section_count,
            summary_count,
        )
        return {
            "doc_id": doc_id,
            "doc_name": doc_name,
            "pages": len(pages),
            "sections": section_count,
            "summaries": summary_count,
        }

    def _clear_document_subgraph(self, doc_id: str) -> None:
        self.connection.execute_write(
            """
            MATCH (d:Document {doc_id: $doc_id})-[:HAS_PAGE]->(p:Page)
            OPTIONAL MATCH (p)-[:HAS_SECTION]->(s:Section)
            OPTIONAL MATCH (s)-[:HAS_SUMMARY]->(m:Summary)
            WITH collect(DISTINCT m) AS summaries,
                 collect(DISTINCT s) AS sections,
                 collect(DISTINCT p) AS pages
            FOREACH (node IN summaries | DETACH DELETE node)
            FOREACH (node IN sections | DETACH DELETE node)
            FOREACH (node IN pages | DETACH DELETE node)
            """,
            {"doc_id": doc_id},
        )

    def _write_section_node(self, section: SectionRecord) -> None:
        self.connection.execute_write(
            """
            MATCH (p:Page {page_id: $page_id})
            MERGE (s:Section {section_id: $section_id})
            SET s.text = $text,
                s.token_count = $token_count,
                s.page_number = $page_number,
                s.page_id = $page_id,
                s.doc_id = $doc_id,
                s.doc_name = $doc_name
            MERGE (p)-[:HAS_SECTION]->(s)
            """,
            {
                "page_id": section.page_id,
                "section_id": section.section_id,
                "text": section.text,
                "token_count": section.token_count,
                "page_number": section.page_number,
                "doc_id": section.doc_id,
                "doc_name": section.doc_name,
            },
        )

    def _write_summary_node(self, section: SectionRecord, summary: SummaryRecord) -> None:
        self.connection.execute_write(
            """
            MATCH (s:Section {section_id: $section_id})
            MERGE (m:Summary {summary_id: $summary_id})
            SET m.text = $summary_text,
                m.embedding = $embedding,
                m.section_id = $section_id,
                m.doc_id = $doc_id,
                m.doc_name = $doc_name
            MERGE (s)-[:HAS_SUMMARY]->(m)
            """,
            {
                "section_id": section.section_id,
                "doc_id": section.doc_id,
                "doc_name": section.doc_name,
                "summary_id": summary.summary_id,
                "summary_text": summary.text,
                "embedding": summary.embedding,
            },
        )

    def _emit(
        self,
        event_callback: GraphEventCallback | None,
        event_type: str,
        message: str,
        payload: dict | None = None,
    ) -> None:
        if event_callback is None:
            return
        event_callback(GraphUpdateEvent(event_type=event_type, message=message, payload=payload or {}))

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3] + "..."

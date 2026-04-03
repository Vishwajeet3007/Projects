from __future__ import annotations

from neo4j import GraphDatabase

from graph_rag.config import AppConfig


class Neo4jConnection:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password),
        )

    def close(self) -> None:
        self.driver.close()

    def verify(self) -> None:
        self.driver.verify_connectivity()

    def run_query(self, query: str, parameters: dict | None = None) -> list[dict]:
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def execute_write(self, query: str, parameters: dict | None = None) -> None:
        with self.driver.session() as session:
            session.run(query, parameters or {})

    def ensure_schema(self) -> None:
        queries = [
            "CREATE CONSTRAINT document_doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
            "CREATE CONSTRAINT page_page_id IF NOT EXISTS FOR (p:Page) REQUIRE p.page_id IS UNIQUE",
            "CREATE CONSTRAINT section_section_id IF NOT EXISTS FOR (s:Section) REQUIRE s.section_id IS UNIQUE",
            "CREATE CONSTRAINT summary_summary_id IF NOT EXISTS FOR (s:Summary) REQUIRE s.summary_id IS UNIQUE",
            "CREATE INDEX section_token_count IF NOT EXISTS FOR (s:Section) ON (s.token_count)",
            "CREATE INDEX summary_text IF NOT EXISTS FOR (s:Summary) ON (s.text)",
        ]
        for query in queries:
            self.execute_write(query)

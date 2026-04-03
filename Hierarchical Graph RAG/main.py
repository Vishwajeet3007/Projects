from __future__ import annotations

import argparse

from graph_rag.bootstrap import ApplicationServices
from graph_rag.config import get_config


def ingest_all_pdfs(services: ApplicationServices) -> None:
    pdf_paths = sorted(services.config.pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {services.config.pdf_dir}")
        return

    for path in pdf_paths:
        summary = services.graph_builder.store_in_graph(path)
        print(f"Ingested {summary['doc_name']} with {summary['sections']} sections")

    edge_count = services.edge_creator.create_similarity_edges()
    print(f"Created {edge_count} similarity edges")


def query_once(services: ApplicationServices, query: str) -> None:
    response = services.query_engine.generate_answer(query)
    print("\nAnswer:\n")
    print(response.answer)
    print("\nUsed sections:")
    for section in response.used_sections:
        print(
            f"- {section.doc_name} | page {section.page_number} | "
            f"section {section.section_id} | score={section.score:.4f} | "
            f"compressed={section.compression_applied}"
        )
    print(
        "\nToken usage:"
        f" query={response.usage.query_tokens},"
        f" context={response.usage.context_tokens},"
        f" answer={response.usage.answer_tokens},"
        f" total={response.usage.total_tokens}"
    )


def interactive_shell(services: ApplicationServices) -> None:
    print("Hierarchical Graph RAG interactive shell")
    print("Type 'exit' to quit.")
    while True:
        query = input("\nQuery> ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if query:
            query_once(services, query)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hierarchical Graph RAG")
    parser.add_argument("--ingest", action="store_true", help="Ingest all PDFs in graph_rag/data/pdfs")
    parser.add_argument("--query", type=str, help="Run a single query")
    parser.add_argument("--shell", action="store_true", help="Start the interactive query shell")
    args = parser.parse_args()

    config = get_config()
    services = ApplicationServices(config)

    try:
        services.connection.verify()
        if args.ingest:
            ingest_all_pdfs(services)
        elif args.query:
            query_once(services, args.query)
        elif args.shell:
            interactive_shell(services)
        else:
            print("Configuration loaded successfully.")
            print(f"PDF folder: {config.pdf_dir}")
            print("Use one of the following:")
            print("  python main.py --ingest")
            print("  python main.py --query \"your question\"")
            print("  python main.py --shell")
            print("  streamlit run graph_rag/ui/streamlit_app.py")
    except Exception as exc:
        print("Unable to connect to Neo4j or start the application.")
        print(f"Reason: {exc}")
        print("Update your .env values and make sure Neo4j is running, then try again.")
    finally:
        services.close()


if __name__ == "__main__":
    main()

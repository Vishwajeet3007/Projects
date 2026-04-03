# Hierarchical Graph RAG

Production-style Retrieval-Augmented Generation system that converts PDF documents into a hierarchical knowledge graph and retrieves only the most relevant summaries and sections before answering. The design minimizes token usage with summary-first retrieval, graph traversal, adaptive context compression, and live graph visualization during ingestion.

## Features

- Multi-PDF ingestion pipeline
- Hierarchical graph schema: `Document -> Page -> Section -> Summary`
- Summary embeddings stored in Neo4j for cheap retrieval
- `SIMILAR` edges between summary nodes for graph traversal
- Adaptive compression for long sections
- Token usage logging per query
- Persistent caching for summaries and embeddings
- Streamlit chat UI with live graph visualization and logs
- Real-time updates while sections, summaries, embeddings, and edges are created

## Project Structure

```text
graph_rag/
├── data/pdfs/
├── ingestion/
│   ├── pdf_loader.py
│   ├── section_splitter.py
│   ├── summarizer.py
│   └── embedding.py
├── graph/
│   ├── neo4j_connection.py
│   ├── graph_builder.py
│   ├── edge_creator.py
│   └── graph_retriever.py
├── rag/
│   ├── hierarchical_retriever.py
│   ├── context_builder.py
│   ├── compression.py
│   └── query_engine.py
├── ui/
│   ├── graph_state.py
│   └── streamlit_app.py
├── bootstrap.py
├── cache.py
├── config.py
├── events.py
├── logging_utils.py
├── models.py
└── token_utils.py
main.py
requirements.txt
README.md
```

## Live Visualization

During ingestion the Streamlit UI updates in real time:

- Section nodes appear in blue when created.
- Summary nodes appear in green after summary generation.
- The current node being processed is highlighted in yellow.
- Similarity edges are drawn in gray as they are created.
- Nodes used during query answering are highlighted in red.
- The logs panel records `Section created`, `Summary generated`, `Embedding created`, and `Edge created` events.

## Neo4j Schema

### Nodes

- `(:Document {doc_id, name, namespace})`
- `(:Page {page_id, page_number, text})`
- `(:Section {section_id, text, token_count, page_number, doc_id, doc_name})`
- `(:Summary {summary_id, text, embedding, section_id, doc_id, doc_name})`

### Relationships

- `(Document)-[:HAS_PAGE]->(Page)`
- `(Page)-[:HAS_SECTION]->(Section)`
- `(Section)-[:HAS_SUMMARY]->(Summary)`
- `(Summary)-[:SIMILAR {score}]->(Summary)`

## Retrieval Flow

1. Convert the user query into an embedding.
2. Find the best matching `Summary` node.
3. Traverse the top 2 similar summary neighbors.
4. Retrieve the connected `Section` nodes.
5. If a section is larger than `COMPRESSION_THRESHOLD_TOKENS`, replace it with a short summary.
6. Build a token-bounded context and send it to the answer model.
7. Return the answer plus the section and summary ids used.

## Environment Variables

Create a `.env` file in the project root:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
OPENAI_API_KEY=your_openai_api_key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_SUMMARY_MODEL=gpt-4.1-mini
OPENAI_ANSWER_MODEL=gpt-4.1-mini
SIMILARITY_THRESHOLD=0.75
SIMILARITY_TOP_K=3
SIMILAR_TRAVERSAL_HOPS=2
COMPRESSION_THRESHOLD_TOKENS=300
MAX_CONTEXT_TOKENS=1800
```

If `OPENAI_API_KEY` is missing, the app falls back to deterministic local embeddings and extractive summaries for development. Final answer generation then returns a retrieval-only response until the key is configured.

## Setup

1. Create the virtual environment:

```powershell
python -m venv venv
```

2. Activate it:

```powershell
venv\Scripts\activate
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Place PDFs into `graph_rag/data/pdfs/`.

5. Run the CLI:

```powershell
python main.py --ingest
python main.py --query "What are the key findings?"
```

6. Or run the Streamlit UI:

```powershell
streamlit run graph_rag/ui/streamlit_app.py
```

## Token Usage Logging

Every query appends usage metadata to:

```text
logs/query_usage.jsonl
```

Each record includes query tokens, context tokens, answer tokens, total tokens, and the document/section nodes used.

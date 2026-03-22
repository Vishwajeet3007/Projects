# Architecture Diagram

```mermaid
flowchart TD
    A[Client Inputs\nRaw Code | Repo URL | PR URL] --> B[FastAPI Backend]
    B --> C[Input Normalizer]
    C --> D[AST Static Analysis]
    C --> E[GitHub API Ingestion]
    C --> F[RAG Retrieval\nFAISS + Best Practices Docs]
    D --> G[LangGraph Workflow]
    E --> G
    F --> G

    G --> H[Analyzer Agent]
    H --> I[Bug Finder Agent]
    H --> J[Complexity Agent]
    H --> K[Security Agent]
    H --> L[Optimizer Agent]
    H --> M[Documentation Agent]
    H --> N[Test Generator Agent]
    H --> O[Code Scoring Agent]

    I --> P[Reviewer Agent]
    J --> P
    K --> P
    L --> P
    M --> P
    N --> P
    O --> P

    P --> Q[Structured Review JSON]
    Q --> R[MongoDB Review History]
    Q --> S[React Frontend]
    Q --> T[VS Code Extension]
    Q --> U[CI or PR Automation]
```

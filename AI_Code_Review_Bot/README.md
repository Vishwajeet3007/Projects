# AI Code Review Bot

Production-ready multi-agent AI code review system built with Python, FastAPI, LangChain, LangGraph, FAISS, MongoDB, React, Docker, GitHub Actions, and a VS Code extension.

## Features

- Multi-agent LangGraph workflow with 9 specialized agents
- Review raw code, GitHub repositories, and pull requests
- Bug detection, complexity analysis, security review, optimization advice
- Docstring generation, refactored code suggestions, and unit test generation
- Code quality scoring from 0 to 10
- RAG with FAISS over coding best-practice documents
- AST-based static analysis for Python before LLM review
- MongoDB storage for review history
- React frontend and VS Code extension clients
- Docker and CI support

## Folder Structure

```text
AI_Code_Review_Bot/
|-- backend/
|   |-- app/
|   |   |-- agents/
|   |   |-- analysis/
|   |   |-- api/routes/
|   |   |-- core/
|   |   |-- db/
|   |   |-- graph/
|   |   |-- integrations/
|   |   |-- rag/
|   |   |-- repositories/
|   |   |-- schemas/
|   |   |-- services/
|   |   `-- utils/
|   `-- tests/
|-- docs/
|   |-- architecture.md
|   |-- mongodb-schema.md
|   `-- knowledge/
|-- examples/
|-- frontend/
|-- vscode-extension/
|-- .github/workflows/
|-- Dockerfile
|-- docker-compose.yml
|-- requirements.txt
`-- README.md
```

## Multi-Agent Workflow

```text
START -> Analyzer ->
  |- Bug Finder
  |- Complexity
  |- Security
  |- Optimizer
  |- Documentation
  |- Test Generator
  |- Code Scoring
Reviewer -> END
```

See [architecture.md](docs/architecture.md) for the detailed diagram.

## API Endpoints

- `POST /api/v1/review-code`
- `POST /api/v1/review-code/upload`
- `POST /api/v1/review-repo`
- `POST /api/v1/review-pr`
- `GET /health`

## Output Format

```json
{
  "bugs": [],
  "security_issues": [],
  "time_complexity": "",
  "space_complexity": "",
  "optimizations": [],
  "refactored_code": "",
  "documentation": "",
  "unit_tests": "",
  "code_quality_score": 0,
  "final_summary": ""
}
```

## Local Setup

### 1. Backend

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
copy .env.example .env
```

Set at minimum:

- `OPENAI_API_KEY`
- `MONGODB_URI`
- `GITHUB_TOKEN` for private repositories or higher API rate limits

Run the API:

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

### 3. VS Code Extension

```bash
cd vscode-extension
npm install
npm run compile
```

Then open the `vscode-extension` folder in VS Code and press `F5` to launch the Extension Development Host.

## Docker

Run the API, MongoDB, and frontend together:

```bash
docker-compose up --build
```

## CI/CD

- `.github/workflows/ci.yml` runs backend tests, frontend build, and VS Code extension compile.
- `.github/workflows/pr-review.yml` can trigger automatic PR review against a deployed backend using `CODE_REVIEW_API_URL`.

## Testing

```bash
set PYTHONPATH=backend
pytest backend/tests
```

## Example Requests

### Review raw code

```bash
curl -X POST http://localhost:8000/api/v1/review-code \
  -H "Content-Type: application/json" \
  -d @examples/review_code_input.json
```

See [review_output.json](examples/review_output.json) for a sample response.

## MongoDB Schema

See [mongodb-schema.md](docs/mongodb-schema.md).

## Implementation Notes

- FastAPI is async end to end for API and GitHub ingestion.
- LangGraph coordinates the analyzer, specialist agents, and final reviewer.
- FAISS retrieval enriches prompts with local best-practice documents.
- AST static analysis runs before the LLM workflow to ground the review.
- MongoDB stores review metadata and final reports for history and analytics.
- The React frontend and VS Code extension both use the same backend API.

## Suggested Next Steps

- Add authentication for multi-tenant usage.
- Add background job processing for large repository reviews.
- Add webhook handling for GitHub PR automation.
- Add richer language parsing with Tree-sitter for non-Python languages.

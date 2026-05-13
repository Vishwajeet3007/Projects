# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
# Activate virtual environment
source venv/bin/activate  # or on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## Testing

```bash
# Run all tests
pytest

# Run a specific test
pytest tests/test_filename.py::test_function_name
```

## Architecture

- **Flask web application** with SQLite database backend
- **Single-file app** (`app.py`) containing all routes
- **Database layer** in `database/db.py` - students implement `get_db()`, `init_db()`, `seed_db()`
- **Templates** use Jinja2 with a base template (`templates/base.html`) for consistent layout
- **Static assets** in `static/css/` and `static/js/`

## Key Patterns

- Routes render templates or return placeholder strings for future implementation
- Database uses SQLite with row_factory enabled and foreign keys support
- Session-based authentication (to be implemented)

## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- For cross-module "how does X relate to Y" questions, prefer `graphify query "<question>"`, `graphify path "<A>" "<B>"`, or `graphify explain "<concept>"` over grep — these traverse the graph's EXTRACTED + INFERRED edges instead of scanning files
- After modifying code files in this session, run `graphify update .` to keep the graph current (AST-only, no API cost)

# MongoDB Schema

Collection: `reviews`

```json
{
  "_id": "ObjectId",
  "metadata": {
    "source_type": "raw_code | repository | pull_request",
    "source_identifier": "string",
    "reviewed_at": "ISODate",
    "agent_versions": {
      "llm": "string",
      "workflow": "string"
    },
    "static_analysis": {
      "languages": {},
      "file_count": 0,
      "python": {
        "functions": 0,
        "classes": 0,
        "imports": [],
        "issues": []
      }
    },
    "rag_context": ["string"],
    "execution_notes": ["string"]
  },
  "report": {
    "bugs": [],
    "security_issues": [],
    "time_complexity": "string",
    "space_complexity": "string",
    "optimizations": ["string"],
    "refactored_code": "string",
    "documentation": "string",
    "unit_tests": "string",
    "code_quality_score": 0,
    "final_summary": "string"
  }
}
```

Recommended indexes:

- `metadata.source_type`
- `metadata.source_identifier`
- `metadata.reviewed_at`

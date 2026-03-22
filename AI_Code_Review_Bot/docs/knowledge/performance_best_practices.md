# Performance Review Checklist

- Look for repeated work inside loops and repeated network calls.
- Cache immutable or frequently reused lookups where appropriate.
- Use streaming or pagination for large repositories and pull requests.
- Limit file size and repository breadth before passing context to LLMs.
- Prefer targeted refactors that improve readability and asymptotic cost together.

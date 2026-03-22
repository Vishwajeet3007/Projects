# Python Review Best Practices

- Validate untrusted input close to the boundary.
- Avoid broad exception handlers that hide operational failures.
- Prefer small composable functions over large procedural blocks.
- Ensure async endpoints avoid blocking file or network operations.
- Add explicit tests for edge cases, failure modes, and malformed payloads.

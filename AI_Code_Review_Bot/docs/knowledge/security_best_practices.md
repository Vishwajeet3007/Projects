# Security Review Checklist

- Watch for eval, exec, unsafe deserialization, and shell injection.
- Do not log secrets, tokens, or raw credentials.
- Enforce authentication and authorization separately.
- Sanitize user-controlled paths, commands, and templates.
- Use least privilege for database and GitHub API tokens.

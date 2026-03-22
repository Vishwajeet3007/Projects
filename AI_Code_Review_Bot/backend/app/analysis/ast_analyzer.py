"""AST and syntax-tree based static analysis helpers."""

from __future__ import annotations

import ast
from collections import Counter
from typing import Any


class StaticAnalysisService:
    """Performs lightweight static analysis before LLM review."""

    def analyze(self, files: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze source files and return normalized findings."""

        summary: dict[str, Any] = {
            "languages": Counter(),
            "file_count": len(files),
            "python": {
                "functions": 0,
                "classes": 0,
                "imports": [],
                "issues": [],
            },
        }

        for file_item in files:
            path = file_item["path"]
            content = file_item["content"]
            language = (file_item.get("language") or self._guess_language(path)).lower()
            summary["languages"][language] += 1

            if language == "python":
                self._analyze_python_file(path, content, summary["python"])

        summary["languages"] = dict(summary["languages"])
        return summary

    def _analyze_python_file(self, path: str, content: str, bucket: dict[str, Any]) -> None:
        """Analyze a Python file using the built-in AST parser."""

        try:
            tree = ast.parse(content)
        except SyntaxError as exc:
            bucket["issues"].append(
                {
                    "title": "Syntax error",
                    "description": str(exc),
                    "severity": "high",
                    "file_path": path,
                    "line": exc.lineno,
                    "recommendation": "Fix the syntax error before executing automated review flows.",
                }
            )
            return

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                bucket["functions"] += 1
                for default_value in node.args.defaults:
                    if isinstance(default_value, (ast.List, ast.Dict, ast.Set)):
                        bucket["issues"].append(
                            {
                                "title": "Mutable default argument",
                                "description": (
                                    f"Function '{node.name}' uses a mutable default argument."
                                ),
                                "severity": "medium",
                                "file_path": path,
                                "line": node.lineno,
                                "recommendation": "Use None as the default and create a new object inside the function.",
                            }
                        )

            if isinstance(node, ast.ClassDef):
                bucket["classes"] += 1

            if isinstance(node, (ast.Import, ast.ImportFrom)):
                bucket["imports"].append(ast.unparse(node))

            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {"eval", "exec"}:
                    bucket["issues"].append(
                        {
                            "title": f"Use of {node.func.id}",
                            "description": f"Potentially unsafe call to {node.func.id} detected.",
                            "severity": "high",
                            "file_path": path,
                            "line": getattr(node, "lineno", None),
                            "recommendation": f"Avoid {node.func.id} on untrusted input.",
                        }
                    )

            if isinstance(node, ast.ExceptHandler) and node.type is None:
                bucket["issues"].append(
                    {
                        "title": "Bare except",
                        "description": "Bare except clause can hide unexpected failures.",
                        "severity": "medium",
                        "file_path": path,
                        "line": getattr(node, "lineno", None),
                        "recommendation": "Catch specific exception types and log failures clearly.",
                    }
                )

    @staticmethod
    def _guess_language(path: str) -> str:
        """Infer a language from file extension."""

        if path.endswith(".py"):
            return "python"
        if path.endswith((".js", ".jsx")):
            return "javascript"
        if path.endswith((".ts", ".tsx")):
            return "typescript"
        if path.endswith(".java"):
            return "java"
        if path.endswith(".go"):
            return "go"
        return "text"

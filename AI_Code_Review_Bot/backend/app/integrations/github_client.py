"""GitHub API integration utilities."""

from __future__ import annotations

import base64
import re

import httpx

from app.core.config import Settings
from app.schemas.review import CodeFile, SourceBundle


class GitHubClient:
    """Fetches repository and pull request content from GitHub."""

    REPO_PATTERN = re.compile(r"github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?/?$")
    PR_PATTERN = re.compile(
        r"github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<number>\d+)"
    )

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def fetch_repository_bundle(self, repo_url: str, branch: str | None = None) -> SourceBundle:
        """Fetch a repository tree and selected file contents."""

        owner, repo = self._parse_repo_url(repo_url)
        headers = self._headers()

        async with httpx.AsyncClient(
            base_url=self._settings.github_api_base_url,
            headers=headers,
            timeout=self._settings.request_timeout_seconds,
        ) as client:
            repo_response = await client.get(f"/repos/{owner}/{repo}")
            repo_response.raise_for_status()
            repo_payload = repo_response.json()
            resolved_branch = branch or repo_payload["default_branch"]

            branch_response = await client.get(f"/repos/{owner}/{repo}/branches/{resolved_branch}")
            branch_response.raise_for_status()
            tree_sha = branch_response.json()["commit"]["commit"]["tree"]["sha"]

            tree_response = await client.get(
                f"/repos/{owner}/{repo}/git/trees/{tree_sha}",
                params={"recursive": "1"},
            )
            tree_response.raise_for_status()
            tree_payload = tree_response.json()

            selected_paths = [
                item["path"]
                for item in tree_payload.get("tree", [])
                if item.get("type") == "blob" and self._is_reviewable_file(item["path"])
            ][: self._settings.max_repo_files]

            files = []
            for path in selected_paths:
                files.append(await self._fetch_file(client, owner, repo, path, resolved_branch))

        return SourceBundle(
            source_type="repository",
            identifier=f"{owner}/{repo}@{resolved_branch}",
            files=files,
            metadata={"repo_url": repo_url, "branch": resolved_branch},
        )

    async def fetch_pull_request_bundle(self, pr_url: str) -> SourceBundle:
        """Fetch changed files from a pull request."""

        owner, repo, number = self._parse_pr_url(pr_url)
        headers = self._headers()

        async with httpx.AsyncClient(
            base_url=self._settings.github_api_base_url,
            headers=headers,
            timeout=self._settings.request_timeout_seconds,
        ) as client:
            pr_response = await client.get(f"/repos/{owner}/{repo}/pulls/{number}")
            pr_response.raise_for_status()
            pr_payload = pr_response.json()

            files_response = await client.get(f"/repos/{owner}/{repo}/pulls/{number}/files")
            files_response.raise_for_status()
            changed_files = files_response.json()

            files: list[CodeFile] = []
            for file_item in changed_files[: self._settings.max_repo_files]:
                if self._is_reviewable_file(file_item["filename"]):
                    files.append(
                        await self._fetch_file(
                            client,
                            owner,
                            repo,
                            file_item["filename"],
                            pr_payload["head"]["ref"],
                        )
                    )

        return SourceBundle(
            source_type="pull_request",
            identifier=f"{owner}/{repo}#PR-{number}",
            files=files,
            metadata={"pr_url": pr_url, "number": number},
        )

    async def _fetch_file(
        self,
        client: httpx.AsyncClient,
        owner: str,
        repo: str,
        path: str,
        ref: str,
    ) -> CodeFile:
        """Fetch file contents from GitHub contents API."""

        response = await client.get(
            f"/repos/{owner}/{repo}/contents/{path}",
            params={"ref": ref},
        )
        response.raise_for_status()
        payload = response.json()
        raw_content = base64.b64decode(payload["content"]).decode("utf-8", errors="ignore")
        return CodeFile(
            path=path,
            content=raw_content[: self._settings.max_file_chars],
            language=self._guess_language(path),
        )

    def _headers(self) -> dict[str, str]:
        """Build GitHub API headers."""

        headers = {"Accept": "application/vnd.github+json"}
        if self._settings.github_token:
            headers["Authorization"] = f"Bearer {self._settings.github_token}"
        return headers

    def _parse_repo_url(self, repo_url: str) -> tuple[str, str]:
        """Parse a GitHub repository URL."""

        match = self.REPO_PATTERN.search(repo_url)
        if not match:
            raise ValueError("Invalid GitHub repository URL.")
        return match.group("owner"), match.group("repo")

    def _parse_pr_url(self, pr_url: str) -> tuple[str, str, int]:
        """Parse a GitHub pull request URL."""

        match = self.PR_PATTERN.search(pr_url)
        if not match:
            raise ValueError("Invalid GitHub pull request URL.")
        return match.group("owner"), match.group("repo"), int(match.group("number"))

    @staticmethod
    def _is_reviewable_file(path: str) -> bool:
        """Filter files to a reviewable code-centric subset."""

        ignored_segments = {".git", "node_modules", "dist", "build", "__pycache__"}
        if any(segment in path.split("/") for segment in ignored_segments):
            return False

        allowed_suffixes = {
            ".py",
            ".js",
            ".jsx",
            ".ts",
            ".tsx",
            ".java",
            ".go",
            ".rb",
            ".rs",
            ".md",
            ".yml",
            ".yaml",
        }
        return any(path.endswith(suffix) for suffix in allowed_suffixes)

    @staticmethod
    def _guess_language(path: str) -> str:
        """Infer language from filename."""

        suffix_map: dict[str, str] = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".java": "java",
            ".go": "go",
            ".rb": "ruby",
            ".rs": "rust",
        }
        for suffix, language in suffix_map.items():
            if path.endswith(suffix):
                return language
        return "text"

"""Application service for end-to-end code review orchestration."""

from __future__ import annotations

from app.agents.factory import AgentFactory
from app.analysis.ast_analyzer import StaticAnalysisService
from app.core.config import Settings
from app.graph.workflow import build_review_workflow
from app.integrations.github_client import GitHubClient
from app.rag.faiss_store import KnowledgeRetriever
from app.repositories.review_repository import ReviewRepository
from app.schemas.review import (
    ReviewMetadata,
    ReviewReport,
    ReviewResponse,
    SourceBundle,
    StoredReview,
)
from app.utils.code_loader import bundle_from_raw_code


class ReviewService:
    """Coordinates the review workflow across analyzers, agents, and persistence."""

    def __init__(
        self,
        settings: Settings,
        repository: ReviewRepository,
        github_client: GitHubClient,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._github_client = github_client
        self._static_analysis = StaticAnalysisService()
        self._knowledge_retriever = KnowledgeRetriever(settings)
        self._workflow = None

    async def review_raw_code(
        self,
        *,
        code: str,
        filename: str,
        language: str,
        repository_context: str | None = None,
    ) -> ReviewResponse:
        """Review a raw code snippet."""

        bundle = bundle_from_raw_code(code, filename, language)
        if repository_context:
            bundle.metadata["repository_context"] = repository_context
        return await self._run_review(bundle)

    async def review_repository(self, repo_url: str, branch: str | None = None) -> ReviewResponse:
        """Review a GitHub repository."""

        bundle = await self._github_client.fetch_repository_bundle(repo_url, branch)
        return await self._run_review(bundle)

    async def review_pull_request(self, pr_url: str) -> ReviewResponse:
        """Review a GitHub pull request."""

        bundle = await self._github_client.fetch_pull_request_bundle(pr_url)
        return await self._run_review(bundle)

    async def run_bundle_review(self, bundle: SourceBundle) -> ReviewResponse:
        """Run a review for a pre-built source bundle."""

        return await self._run_review(bundle)

    async def _run_review(self, bundle: SourceBundle) -> ReviewResponse:
        """Execute the full multi-agent review pipeline and persist the result."""

        files = [file.model_dump() for file in bundle.files]
        static_analysis = self._static_analysis.analyze(files)
        rag_query = f"{bundle.source_type} review best practices for {bundle.identifier}"
        rag_context = self._knowledge_retriever.retrieve_context(rag_query)
        prompt_files, prompt_notes = self._build_prompt_files(files)

        workflow = self._get_workflow()
        final_state = await workflow.ainvoke(
            {
                "source_type": bundle.source_type,
                "source_identifier": bundle.identifier,
                "files": prompt_files,
                "metadata": bundle.metadata,
                "static_analysis": static_analysis,
                "rag_context": rag_context,
            }
        )

        report = ReviewReport.model_validate(final_state["final_report"])
        metadata = ReviewMetadata(
            source_type=bundle.source_type,
            source_identifier=bundle.identifier,
            static_analysis=static_analysis,
            rag_context=rag_context,
            execution_notes=[
                "Static analysis completed before agent execution.",
                "RAG context retrieved from local best-practice knowledge base.",
                *prompt_notes,
            ],
            agent_versions={
                "llm": self._settings.openai_model,
                "workflow": "langgraph-v1",
            },
        )
        response = ReviewResponse(metadata=metadata, report=report)

        await self._repository.save_review(
            StoredReview(
                metadata=metadata,
                report=report,
            )
        )
        return response

    def _build_prompt_files(self, files: list[dict]) -> tuple[list[dict], list[str]]:
        """Reduce large bundles to a prompt-sized subset for the LLM workflow."""

        prioritized_files = sorted(files, key=self._file_priority)
        prompt_files: list[dict] = []
        remaining_chars = self._settings.prompt_total_chars
        truncation_suffix = "\n# ... truncated for prompt budget ..."

        for file_item in prioritized_files:
            if len(prompt_files) >= self._settings.prompt_file_limit or remaining_chars <= 0:
                break

            content = file_item.get("content", "")
            if not content:
                continue

            clip_size = min(self._settings.prompt_file_chars, remaining_chars)
            clipped_content = content[:clip_size]
            if len(content) > clip_size and clip_size > len(truncation_suffix):
                clipped_content = content[: clip_size - len(truncation_suffix)] + truncation_suffix

            prompt_files.append(
                {
                    **file_item,
                    "content": clipped_content,
                }
            )
            remaining_chars -= len(clipped_content)

        notes = [
            f"Workflow prompt context reduced from {len(files)} files to {len(prompt_files)} prioritized files.",
            (
                "Prompt budget used "
                f"{self._settings.prompt_total_chars - remaining_chars}/{self._settings.prompt_total_chars} characters."
            ),
        ]
        return prompt_files, notes

    def _file_priority(self, file_item: dict) -> tuple[int, str]:
        """Rank files so the most important context reaches the LLM first."""

        path = file_item.get("path", "").lower()
        high_priority_names = {
            "readme.md",
            "dockerfile",
            "docker-compose.yml",
            "requirements.txt",
            "package.json",
            "pyproject.toml",
            ".env.example",
        }
        if any(path.endswith(name) for name in high_priority_names):
            return (0, path)
        if path.endswith(("main.py", "app.py", "settings.py", "config.py", "routes.py")):
            return (1, path)
        if path.endswith((".py", ".ts", ".tsx", ".js", ".jsx")):
            return (2, path)
        return (3, path)

    def _get_workflow(self):
        """Lazily initialize the LangGraph workflow."""

        if self._workflow is None:
            agent_factory = AgentFactory(self._settings)
            self._workflow = build_review_workflow(agent_factory)
        return self._workflow

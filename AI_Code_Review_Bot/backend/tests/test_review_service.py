"""Tests for review service prompt budgeting."""

from app.core.config import Settings
from app.integrations.github_client import GitHubClient
from app.repositories.review_repository import ReviewRepository
from app.services.review_service import ReviewService


class DummyMongoManager:
    pass



def test_build_prompt_files_reduces_large_context() -> None:
    settings = Settings(
        PROMPT_FILE_LIMIT=2,
        PROMPT_FILE_CHARS=20,
        PROMPT_TOTAL_CHARS=35,
    )
    service = ReviewService(
        settings=settings,
        repository=ReviewRepository(DummyMongoManager()),
        github_client=GitHubClient(settings),
    )

    prompt_files, notes = service._build_prompt_files(
        [
            {"path": "README.md", "content": "a" * 50, "language": "text"},
            {"path": "main.py", "content": "b" * 50, "language": "python"},
            {"path": "extra.py", "content": "c" * 50, "language": "python"},
        ]
    )

    assert len(prompt_files) == 2
    assert prompt_files[0]["path"] == "README.md"
    assert "Prompt budget used" in notes[1]

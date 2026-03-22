"""FastAPI application entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.review import get_review_service, router as review_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.mongo import MongoManager
from app.integrations.github_client import GitHubClient
from app.repositories.review_repository import ReviewRepository
from app.services.review_service import ReviewService

settings = get_settings()
configure_logging()

mongo_manager = MongoManager(settings)
github_client = GitHubClient(settings)
review_repository = ReviewRepository(mongo_manager)
review_service = ReviewService(settings, review_repository, github_client)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Manage application startup and shutdown tasks."""

    await mongo_manager.connect()
    yield
    await mongo_manager.disconnect()


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _review_service_dependency() -> ReviewService:
    """Return the application review service instance."""

    return review_service


app.dependency_overrides[get_review_service] = _review_service_dependency
app.include_router(review_router, prefix=settings.api_prefix)


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    """Simple health endpoint."""

    return {"status": "ok"}

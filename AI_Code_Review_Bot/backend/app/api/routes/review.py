"""Review API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.schemas.review import ReviewCodeRequest, ReviewPRRequest, ReviewRepoRequest, ReviewResponse
from app.services.review_service import ReviewService
from app.utils.code_loader import bundle_from_upload

router = APIRouter(tags=["reviews"])



def get_review_service() -> ReviewService:
    """Dependency placeholder overridden during app startup."""

    raise RuntimeError("Review service dependency has not been configured.")


@router.post("/review-code", response_model=ReviewResponse)
async def review_code(
    request: ReviewCodeRequest,
    review_service: ReviewService = Depends(get_review_service),
) -> ReviewResponse:
    """Review raw code submitted in JSON format."""

    try:
        return await review_service.review_raw_code(
            code=request.code,
            filename=request.filename,
            language=request.language,
            repository_context=request.repository_context,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@router.post("/review-code/upload", response_model=ReviewResponse)
async def review_uploaded_code(
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
    review_service: ReviewService = Depends(get_review_service),
) -> ReviewResponse:
    """Review an uploaded source file."""

    try:
        bundle = await bundle_from_upload(file, language)
        return await review_service.run_bundle_review(bundle)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@router.post("/review-repo", response_model=ReviewResponse)
async def review_repo(
    request: ReviewRepoRequest,
    review_service: ReviewService = Depends(get_review_service),
) -> ReviewResponse:
    """Review a GitHub repository URL."""

    try:
        return await review_service.review_repository(str(request.repo_url), request.branch)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@router.post("/review-pr", response_model=ReviewResponse)
async def review_pr(
    request: ReviewPRRequest,
    review_service: ReviewService = Depends(get_review_service),
) -> ReviewResponse:
    """Review a GitHub pull request URL."""

    try:
        return await review_service.review_pull_request(str(request.pr_url))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

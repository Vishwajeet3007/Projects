"""Tests for the review API."""

from fastapi.testclient import TestClient

from app.api.routes.review import get_review_service
from app.main import app
from app.schemas.review import ReviewMetadata, ReviewReport, ReviewResponse


class StubReviewService:
    async def review_raw_code(self, **_: str) -> ReviewResponse:
        return ReviewResponse(
            metadata=ReviewMetadata(
                source_type="raw_code",
                source_identifier="snippet.py",
            ),
            report=ReviewReport(
                final_summary="Stub review completed.",
                code_quality_score=8,
            ),
        )


app.dependency_overrides[get_review_service] = lambda: StubReviewService()
client = TestClient(app)



def test_review_code_endpoint_returns_payload() -> None:
    response = client.post(
        "/api/v1/review-code",
        json={
            "code": "print('hello')",
            "filename": "snippet.py",
            "language": "python",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["report"]["code_quality_score"] == 8
    assert payload["report"]["final_summary"] == "Stub review completed."

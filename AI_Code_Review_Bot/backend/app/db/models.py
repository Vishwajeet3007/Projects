"""Database model helpers."""

from __future__ import annotations

from app.schemas.review import StoredReview



def serialize_review(review: StoredReview) -> dict:
    """Convert a review model to a MongoDB-friendly dictionary."""

    return review.model_dump(mode="json")

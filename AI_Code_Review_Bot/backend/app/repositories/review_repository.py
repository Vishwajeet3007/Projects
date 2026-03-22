"""Persistence helpers for storing review history."""

from __future__ import annotations

from app.db.models import serialize_review
from app.db.mongo import MongoManager
from app.schemas.review import StoredReview


class ReviewRepository:
    """Stores and retrieves code review reports."""

    def __init__(self, mongo_manager: MongoManager) -> None:
        self._mongo_manager = mongo_manager

    async def save_review(self, review: StoredReview) -> str:
        """Persist a review report in MongoDB."""

        collection = self._mongo_manager.database["reviews"]
        result = await collection.insert_one(serialize_review(review))
        return str(result.inserted_id)

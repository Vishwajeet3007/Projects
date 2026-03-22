"""MongoDB client helpers."""

from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.core.config import Settings


class MongoManager:
    """Manages MongoDB connections for the application lifecycle."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.client: AsyncIOMotorClient | None = None

    async def connect(self) -> None:
        """Initialize the database client."""

        if self.client is None:
            self.client = AsyncIOMotorClient(self._settings.mongodb_uri)

    async def disconnect(self) -> None:
        """Close the database client."""

        if self.client is not None:
            self.client.close()
            self.client = None

    @property
    def database(self) -> AsyncIOMotorDatabase:
        """Return the configured database."""

        if self.client is None:
            raise RuntimeError("MongoDB client is not connected.")
        return self.client[self._settings.mongodb_database]

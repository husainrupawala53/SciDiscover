"""
db/mongodb.py — Async MongoDB connection using Motor.
"""

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.config import get_settings
import logging

logger = logging.getLogger(__name__)

_client: AsyncIOMotorClient = None


async def connect_mongodb() -> None:
    """Initialize the MongoDB Motor client."""
    global _client
    settings = get_settings()
    _client = AsyncIOMotorClient(settings.MONGODB_URL)
    # Ping to verify connection
    await _client.admin.command("ping")
    logger.info("✅ MongoDB connected successfully")


async def close_mongodb() -> None:
    """Close the MongoDB Motor client."""
    global _client
    if _client:
        _client.close()
        logger.info("MongoDB connection closed")


def get_database() -> AsyncIOMotorDatabase:
    """Return the application database handle."""
    settings = get_settings()
    return _client[settings.MONGODB_DB]


def get_papers_collection():
    """Shortcut to the 'papers' collection."""
    return get_database()["papers"]

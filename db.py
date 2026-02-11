# db.py

from functools import lru_cache

from pymongo import MongoClient

from config import MONGO_URI, MONGO_DB_NAME


@lru_cache(maxsize=1)
def _get_client() -> MongoClient | None:
    """Create and cache a MongoDB client (or return None if not configured)."""
    if not MONGO_URI:
        return None
    try:
        client = MongoClient(MONGO_URI)
        return client
    except Exception as e:
        print(f"(db) Failed to create Mongo client: {e}")
        return None


def get_db():
    """Return the configured database object, or None if unavailable."""
    client = _get_client()
    if client is None or not MONGO_DB_NAME:
        return None
    try:
        return client[MONGO_DB_NAME]
    except Exception as e:
        print(f"(db) Failed to get DB: {e}")
        return None


"""
MongoDB client — singleton pattern with graceful degradation.

If MONGODB_URI is not set or the connection fails, all database operations
fall back silently to the in-memory store so the API keeps working locally.

For a single-user setup this is fine; multi-user persistence requires MongoDB.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ConfigurationError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.logger import get_logger

log = get_logger(__name__)

_client: Optional[MongoClient] = None
_db: Optional[Database] = None


def get_db() -> Optional[Database]:
    """
    Return the MongoDB database instance, or None if unavailable.
    Callers must handle None and fall back to in-memory storage.
    """
    global _client, _db

    if _db is not None:
        return _db

    try:
        from config.settings import get_settings
        cfg = get_settings()
        uri = cfg.mongodb_uri
        db_name = cfg.mongodb_db_name

        if not uri:
            log.info("MONGODB_URI not set — running with in-memory conversation store")
            return None

        log.info("Connecting to MongoDB: db=%s", db_name)
        _client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        # Verify connection is live.
        _client.admin.command("ping")
        _db = _client[db_name]
        log.info("MongoDB connected: db=%s", db_name)
        return _db

    except (ConnectionFailure, ConfigurationError) as exc:
        log.warning("MongoDB unavailable — falling back to in-memory store: %s", exc)
        return None
    except Exception as exc:
        log.warning("MongoDB setup error — falling back to in-memory store: %s", exc)
        return None


def close_db() -> None:
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db = None
        log.info("MongoDB connection closed")

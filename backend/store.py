"""
Conversation store — MongoDB-backed with in-memory fallback.

Priority:
  1. MongoDB (if MONGODB_URI is set and reachable)  ← persistent across restarts
  2. In-memory dict (if MongoDB is unavailable)     ← lost on process restart

This design means:
  - Single-user local dev works without any database setup.
  - Pointing at MongoDB Atlas gives full persistence automatically.
"""
from __future__ import annotations

from typing import Dict, Optional

from backend.models import ConversationEntry
from database import repository as db
from utils.logger import get_logger

log = get_logger(__name__)

# In-memory fallback (used when MongoDB is not available).
_mem: Dict[str, ConversationEntry] = {}

_WARNED_FALLBACK = False


def _warn_fallback() -> None:
    global _WARNED_FALLBACK
    if not _WARNED_FALLBACK:
        log.warning(
            "MongoDB unavailable — conversations stored in memory only "
            "(lost on restart). Set MONGODB_URI to enable persistence."
        )
        _WARNED_FALLBACK = True


# ── Public API (mirrors old store.py interface exactly) ──────────────────────

def save(entry: ConversationEntry) -> None:
    # Always write to memory for fast in-process reads.
    _mem[entry.conversation_id] = entry

    # Also persist to MongoDB when available.
    if db.is_available():
        db.save_conversation(entry.model_dump())
        # Append the user's prompt as a message if it's a new conversation.
        # (message history is bonus — the core state is in the document itself)
    else:
        _warn_fallback()


def get(conversation_id: str) -> Optional[ConversationEntry]:
    # Hot path: check memory first.
    if conversation_id in _mem:
        return _mem[conversation_id]

    # Cold path: check MongoDB (handles server restarts / different processes).
    if db.is_available():
        doc = db.get_conversation(conversation_id)
        if doc:
            try:
                entry = ConversationEntry(**doc)
                _mem[entry.conversation_id] = entry   # warm the cache
                return entry
            except Exception as exc:
                log.warning("Failed to deserialize conversation from MongoDB: %s", exc)
    return None


def delete(conversation_id: str) -> None:
    _mem.pop(conversation_id, None)
    if db.is_available():
        db.delete_conversation(conversation_id)


def list_all(limit: int = 50) -> list[ConversationEntry]:
    """Return recent conversations (from MongoDB or memory)."""
    if db.is_available():
        docs = db.list_conversations(limit=limit)
        result = []
        for doc in docs:
            try:
                result.append(ConversationEntry(**doc))
            except Exception:
                pass
        return result
    return list(_mem.values())[-limit:]


def backend_status() -> str:
    """'mongodb' | 'memory'"""
    return "mongodb" if db.is_available() else "memory"

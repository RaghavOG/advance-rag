"""
Conversation repository — MongoDB-backed with in-memory fallback.

Document schema (MongoDB collection: "conversations"):
{
  _id:                         conversation_id  (string UUID)
  original_prompt:             str
  created_at:                  datetime (UTC)
  updated_at:                  datetime (UTC)
  sub_queries:                 List[str]
  completed_sub_answers:       List[{question, answer}]
  pending_clarification_index: int | null
  pending_clarification_question: str | null

  messages: [                  # full message history
    {role: "user"|"assistant", content: str, timestamp: datetime}
  ]
}
"""
from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

from database.client import get_db
from utils.logger import get_logger

log = get_logger(__name__)

_COLLECTION = "conversations"


# ── helpers ───────────────────────────────────────────────────────────────────

def _now() -> datetime.datetime:
    return datetime.datetime.utcnow()


def _collection():
    db = get_db()
    return db[_COLLECTION] if db is not None else None


# ── public API ────────────────────────────────────────────────────────────────

def save_conversation(data: Dict[str, Any]) -> None:
    """
    Upsert a conversation document.
    `data` must contain `conversation_id` as its primary key.
    """
    col = _collection()
    if col is None:
        return   # no-op; caller uses in-memory store as primary

    cid = data["conversation_id"]
    data["updated_at"] = _now()
    data.setdefault("created_at", _now())

    col.replace_one(
        {"_id": cid},
        {**data, "_id": cid},
        upsert=True,
    )
    log.debug("Conversation upserted: %s", cid)


def get_conversation(conversation_id: str) -> Optional[Dict[str, Any]]:
    """Return the conversation document, or None if not found."""
    col = _collection()
    if col is None:
        return None
    doc = col.find_one({"_id": conversation_id})
    if doc:
        doc.pop("_id", None)   # remove MongoDB internal _id before returning
    return doc


def delete_conversation(conversation_id: str) -> None:
    col = _collection()
    if col is None:
        return
    col.delete_one({"_id": conversation_id})
    log.debug("Conversation deleted: %s", conversation_id)


def append_message(
    conversation_id: str,
    role: str,
    content: str,
) -> None:
    """
    Push a message into the messages array for a conversation.
    Creates the conversation document if it doesn't exist yet.
    """
    col = _collection()
    if col is None:
        return

    msg = {"role": role, "content": content, "timestamp": _now()}
    col.update_one(
        {"_id": conversation_id},
        {
            "$push": {"messages": msg},
            "$set": {"updated_at": _now()},
            "$setOnInsert": {"created_at": _now(), "_id": conversation_id},
        },
        upsert=True,
    )
    log.debug("Message appended: conv=%s role=%s", conversation_id, role)


def list_conversations(limit: int = 50) -> List[Dict[str, Any]]:
    """Return the most recent conversations (newest first)."""
    col = _collection()
    if col is None:
        return []
    docs = list(
        col.find({}, {"messages": 0})   # exclude message arrays for list view
           .sort("updated_at", -1)
           .limit(limit)
    )
    for d in docs:
        d.pop("_id", None)
    return docs


def is_available() -> bool:
    """True if MongoDB is reachable."""
    return get_db() is not None

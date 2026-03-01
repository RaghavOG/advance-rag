"""
Query routes.

POST /api/query        — run RAG pipeline for a user prompt
POST /api/clarify      — provide clarification for a pending sub-query
GET  /api/conversation/{id} — retrieve conversation state
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

# Allow importing root-level modules (embeddings/, graph/, etc.)
ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.models import (
    ClarificationRequest,
    ConversationEntry,
    QueryRequest,
    QueryResponse,
)
from backend.pipeline_adapter import graph_state_to_response
from backend import store

router = APIRouter(tags=["query"])

# Default PDF for demo when no upload path provided.
_DEMO_PDF = os.getenv("DEMO_PDF_PATH", "")


def _run_graph(prompt: str, extra_state: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Import and invoke rag_graph lazily to avoid heavy startup cost."""
    from graph.graph import rag_graph  # noqa: PLC0415

    initial: Dict[str, Any] = {"raw_prompt": prompt}
    if extra_state:
        initial.update(extra_state)
    return rag_graph.invoke(initial)


def _maybe_ingest(pdf_path: str | None) -> None:
    if not pdf_path:
        return
    p = Path(pdf_path)
    if not p.exists():
        raise HTTPException(status_code=400, detail=f"PDF not found: {pdf_path}")
    from ingestion.ingest import ingest_pdf  # noqa: PLC0415
    ingest_pdf(p)


# ── POST /api/query ───────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    conversation_id = req.conversation_id or str(uuid.uuid4())
    pdf = req.pdf_path or _DEMO_PDF

    try:
        _maybe_ingest(pdf)
        state = _run_graph(req.prompt)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    response = graph_state_to_response(state, conversation_id)

    # Persist conversation for clarification resumption.
    entry = ConversationEntry(
        conversation_id=conversation_id,
        original_prompt=req.prompt,
        sub_queries=state.get("sub_queries") or [],
        completed_sub_answers=[sa for sa in response.sub_answers if sa.status == "answered"],
    )
    if response.status == "clarification_required":
        cq = next(
            (sa.clarification_question for sa in response.sub_answers if sa.clarification_question),
            None,
        )
        idx = next(
            (i for i, sa in enumerate(response.sub_answers) if sa.status == "clarification_required"),
            0,
        )
        entry.pending_clarification_index = idx
        entry.pending_clarification_question = cq

    store.save(entry)
    return response


# ── POST /api/clarify ─────────────────────────────────────────────────────────

@router.post("/clarify", response_model=QueryResponse)
async def clarify(req: ClarificationRequest) -> QueryResponse:
    entry = store.get(req.conversation_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if entry.pending_clarification_index is None:
        raise HTTPException(status_code=400, detail="No pending clarification for this conversation")

    # Build a combined prompt: original sub-query + user's clarification answer.
    sub_queries = entry.sub_queries
    original_q = (
        sub_queries[entry.pending_clarification_index]
        if sub_queries and req.clarification_for < len(sub_queries)
        else entry.original_prompt
    )
    combined = f"{original_q} ({req.answer})"

    try:
        # Re-invoke with clarification_used=True so graph skips ambiguity check.
        state = _run_graph(combined, {"clarification_used": True, "clarified_query": combined})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    response = graph_state_to_response(state, req.conversation_id)
    entry.pending_clarification_index = None
    entry.pending_clarification_question = None
    store.save(entry)
    return response


# ── GET /api/conversation/{id} ────────────────────────────────────────────────

@router.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    entry = store.get(conversation_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Not found")
    return entry

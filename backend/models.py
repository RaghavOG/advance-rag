"""
Pydantic request/response models for the RAG API.

Every response is structured — never a raw string.
All fields are Optional where absence is meaningful (e.g. no timings on failure).
"""
from __future__ import annotations

import uuid
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# ── Requests ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    prompt: str = Field(..., description="User's question or multi-question prompt")
    conversation_id: Optional[str] = Field(
        None, description="Resume an existing conversation"
    )
    pdf_path: Optional[str] = Field(
        None, description="Path to PDF (server-side); upload handled separately"
    )


class ClarificationRequest(BaseModel):
    conversation_id: str = Field(..., description="The conversation requiring clarification")
    clarification_for: int = Field(..., description="Index of sub-query that needs clarification")
    answer: str = Field(..., description="User's clarification answer")


# ── Sub-answer (one per independent question in the prompt) ───────────────────

class Citation(BaseModel):
    doc_id: Optional[str] = None
    source: Optional[str] = None
    page: Optional[int] = None
    chunk_id: Optional[str] = None
    snippet: Optional[str] = None
    confidence: Optional[float] = None   # normalised retrieval confidence for this chunk
    raw_score: Optional[float] = None    # raw value from the vector store API
    backend: Optional[str] = None        # "chroma" | "faiss" | "pinecone"


class SubAnswer(BaseModel):
    query: str
    status: Literal["answered", "failed", "clarification_required", "processing"]
    answer: Optional[str] = None
    reasoning: Optional[str] = None
    citations: List[Citation] = []
    confidence: Optional[float] = None   # max retrieval confidence across citations
    # Present only when status == "clarification_required". Contains the question
    # the user needs to answer so the system can disambiguate their intent.
    clarification_question: Optional[str] = None


class QueryResponse(BaseModel):
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: Literal["answered", "clarification_required", "failure", "partial"]
    sub_answers: List[SubAnswer] = []
    error_message: Optional[str] = None
    # Per-node elapsed times in milliseconds — present when at least one node
    # measured timing.  Keys match the timing keys added by graph/nodes.py.
    # Example: {"rewrite_ms": 820, "retrieve_ms": 2341, "compress_ms": 910,
    #            "generate_ms": 3120, "ambiguity_ms": 640}
    timings: Optional[Dict[str, float]] = None


# ── Conversation store entry (in-memory / MongoDB) ───────────────────────────

class ConversationEntry(BaseModel):
    conversation_id: str
    original_prompt: str
    pending_clarification_index: Optional[int] = None
    pending_clarification_question: Optional[str] = None
    sub_queries: List[str] = []
    completed_sub_answers: List[SubAnswer] = []

"""
Pydantic request/response models for the RAG API.

Every response is structured — never a raw string.
"""
from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field
import uuid


# ── Requests ─────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    prompt: str = Field(..., description="User's question or multi-question prompt")
    conversation_id: Optional[str] = Field(None, description="Resume an existing conversation")
    pdf_path: Optional[str] = Field(None, description="Path to PDF (server-side); upload handled separately")


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


class SubAnswer(BaseModel):
    query: str
    status: Literal["answered", "failed", "clarification_required", "processing"]
    answer: Optional[str] = None
    reasoning: Optional[str] = None          # split from answer where possible
    citations: List[Citation] = []
    confidence: Optional[float] = None       # derived from retrieval scores
    clarification_question: Optional[str] = None


# ── Top-level response ────────────────────────────────────────────────────────

class QueryResponse(BaseModel):
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: Literal["answered", "clarification_required", "failure", "partial"]
    sub_answers: List[SubAnswer] = []
    error_message: Optional[str] = None


# ── Conversation store entry (in-memory) ─────────────────────────────────────

class ConversationEntry(BaseModel):
    conversation_id: str
    original_prompt: str
    pending_clarification_index: Optional[int] = None
    pending_clarification_question: Optional[str] = None
    sub_queries: List[str] = []
    completed_sub_answers: List[SubAnswer] = []

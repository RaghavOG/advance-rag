"""
LangGraph state definition.

A single TypedDict flows through every node in the graph.
All fields are Optional so nodes only update what they own.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TypedDict

from langchain_core.documents import Document


class RAGState(TypedDict, total=False):
    # ── LAYER 0 ──────────────────────────────────────────────────────────────
    # Raw prompt from the caller.
    raw_prompt: str
    # Trimmed, punctuation-normalized version.
    normalized_prompt: str

    # ── LAYER 1 ──────────────────────────────────────────────────────────────
    # Independent sub-questions found in the prompt.
    sub_queries: List[str]
    # "single" | "multi"
    query_route: str

    # ── LAYER 2 — per-query sub-graph ────────────────────────────────────────
    # The question currently being processed (one sub-query at a time).
    current_query: str

    # Ambiguity check
    is_ambiguous: bool
    clarification_question: Optional[str]
    # Guard: only one clarification turn is allowed.
    clarification_used: bool
    # The query after the user has answered the clarification (if any).
    clarified_query: str

    # Query rewriting / expansion
    rewritten_queries: List[str]

    # Adaptive top-k decisions
    top_k_text: int
    top_k_image: int
    top_k_audio: int

    # Retrieved docs with scores: List[Tuple[Document, float]]
    # Stored as a list of dicts so LangGraph can serialize them.
    retrieved_docs_with_scores: List[Dict[str, Any]]

    # Merged, deduplicated final doc set (Document list, serialized)
    final_retrieved_docs: List[Dict[str, Any]]

    # ── LAYER 3 ──────────────────────────────────────────────────────────────
    compressed_context: str
    answer_text: str

    # Retry counter for answer generation
    generation_retries: int

    # ── LAYER 4 ──────────────────────────────────────────────────────────────
    # Accumulator: List of {"question": str, "answer": str}
    sub_answers: List[Dict[str, str]]
    # Final formatted output
    final_answer: str

    # ── CONTROL FLAGS ────────────────────────────────────────────────────────
    # Human-readable error message surfaced to the caller when something fails.
    error_message: Optional[str]

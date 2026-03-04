"""
LangGraph state definition.

A single TypedDict flows through every node in the graph.
All fields are Optional so nodes only update what they own.

Per-query invariants
--------------------
Each sub-query independently tracks:
  - sub_query_index   : its position in sub_queries
  - query_status      : lifecycle ("pending" | "answered" | "failed" | "clarification_needed")
  - timings           : ms spent per node (accumulated across the whole graph run)
  - last_error        : the most recent per-query error detail (distinct from error_message)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class RAGState(TypedDict, total=False):
    # ── LAYER 0 ──────────────────────────────────────────────────────────────
    raw_prompt: str
    normalized_prompt: str

    # ── SAFETY (runs right after normalize) ──────────────────────────────────
    safety_flagged: bool           # True → pipeline short-circuits to safety_blocked_node
    safety_reason: Optional[str]   # Human-readable reason for the block

    # ── LAYER 1 ──────────────────────────────────────────────────────────────
    sub_queries: List[str]
    query_route: str               # "single" | "multi"

    # ── LAYER 2 — per-query sub-graph ────────────────────────────────────────
    current_query: str
    sub_query_index: int           # which element of sub_queries we are processing

    # Lifecycle
    query_status: str              # "pending" | "answered" | "failed" | "clarification_needed"

    # Ambiguity / clarification
    is_ambiguous: bool
    clarification_question: Optional[str]
    clarification_used: bool       # guard: only one clarification turn allowed
    clarified_query: str           # the query after user answered the clarification

    # Query rewriting
    rewritten_queries: List[str]

    # Adaptive top-k
    top_k_text: int
    top_k_image: int
    top_k_audio: int

    # HyDE intermediate results (retrieval-only, ephemeral — text is NEVER returned to user)
    hyde_doc_text: Optional[str]   # LLM-generated hypothetical document text
    # (Results from the hyde vector search are merged into retrieved_docs_with_scores)

    # Retrieved docs — each dict: {page_content, metadata, score, raw_score, backend}
    # score     = normalized confidence in [0,1] (higher is better), after score_normalizer_node
    # raw_score = the value returned directly by the vector store API
    # backend   = "chroma" | "faiss" | "pinecone"
    retrieved_docs_with_scores: List[Dict[str, Any]]

    # After dedup/merge; same schema as retrieved_docs_with_scores
    final_retrieved_docs: List[Dict[str, Any]]

    # ── LAYER 3 ──────────────────────────────────────────────────────────────
    compressed_context: str
    answer_text: str
    generation_retries: int

    # ── LAYER 4 ──────────────────────────────────────────────────────────────
    sub_answers: List[Dict[str, str]]   # List of {"question": str, "answer": str}
    final_answer: str

    # ── CONTROL FLAGS ────────────────────────────────────────────────────────
    error_message: Optional[str]   # Human-readable error surfaced to caller
    last_error: Optional[str]      # Per-query internal error detail (for resume logic)

    # ── TIMINGS (node_key → elapsed_ms) ──────────────────────────────────────
    # Accumulated across the whole graph run.  Each node that does meaningful
    # work appends its own entry without clobbering previous ones.
    timings: Dict[str, float]

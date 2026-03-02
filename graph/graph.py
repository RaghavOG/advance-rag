"""
LangGraph wiring: all 15 nodes connected with typed state and conditional edges.

Graph topology
--------------

  [normalize_user_prompt]
          │
  [detect_multi_query]
          │
  ┌───────┴───────────────────────────────┐
  │ single                                │ multi
  │                                       │   (loops per sub-query via
  │                                       │    multi_query_router)
  └─────────────┐                         │
                ▼                         ▼
          [ambiguity_check]        ──────────────────
                │                 │  (same sub-graph  │
         ┌──────┴──────┐          │   per sub-query)  │
         │ clear       │ ambiguous│                   │
         ▼             ▼          └──────────────────┘
  [query_rewrite] [clarification_node]
         │               │
         ▼               ▼ (resume with clarified_query)
  [adaptive_top_k_decision]
         │
  [retrieve_documents]
         │
  [merge_retrieval_results]
         │
  ┌──────┴──────────────────────────┐
  │ has docs                        │ no docs
  ▼                                 ▼
  [compress_context_node]  [retrieval_failure_node] → END
         │
  ┌──────┴─────────────────┐
  │ ok                     │ error
  ▼                        ▼
  [generate_answer_node]  [compression_failure_node]
         │                        │
  ┌──────┴───────┐                │
  │ ok           │ retry needed   │
  ▼              ▼                ▼
  [collect_sub_answers] ← [llm_timeout_failure_node]
         │
  [merge_final_answers]
         │
         END
"""
from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import END, StateGraph

from config.settings import get_settings
from graph.nodes import (
    adaptive_top_k_decision,
    ambiguity_check,
    clarification_node,
    collect_sub_answers,
    compress_context_node,
    compression_failure_node,
    detect_multi_query,
    generate_answer_node,
    llm_timeout_failure_node,
    merge_final_answers,
    merge_retrieval_results,
    normalize_user_prompt,
    query_rewrite_expand,
    retrieval_failure_node,
    retrieve_documents,
)
from graph.state import RAGState

# ── Conditional routing functions ────────────────────────────────────────────

def _route_query(state: Dict[str, Any]) -> str:
    """After detect_multi_query: go to ambiguity_check for both routes."""
    return "ambiguity_check"


def _route_ambiguity(state: Dict[str, Any]) -> str:
    """
    After ambiguity_check:
    - If ambiguous AND clarification not yet used → clarification_node
    - Otherwise → query_rewrite_expand
    """
    if state.get("is_ambiguous") and not state.get("clarification_used"):
        return "clarification_node"
    return "query_rewrite_expand"


def _route_post_clarification(state: Dict[str, Any]) -> str:
    """After clarification_node we short-circuit to END (caller must resume)."""
    return END


def _route_retrieval(state: Dict[str, Any]) -> str:
    """
    After merge_retrieval_results:
    - No docs retrieved → retrieval_failure_node
    - Low-confidence docs → retrieval_failure_node
    - Otherwise → compress_context_node
    """
    docs = state.get("final_retrieved_docs") or []
    if not docs:
        return "retrieval_failure_node"

    cfg = get_settings()
    threshold = cfg.retrieval_confidence_threshold
    # Scores are normalized confidences in [0, 1] — higher is BETTER.
    # best_confidence is the highest confidence among the retrieved docs.
    best_confidence = max((d.get("score", 0.0) for d in docs), default=0.0)

    from utils.logger import get_logger  # local import to avoid cycles
    log = get_logger(__name__)
    log.info(
        "  Retrieval confidence check: best_confidence=%.4f  threshold=%.4f"
        "  (%s)",
        best_confidence,
        threshold,
        "PASS" if threshold == 0 or best_confidence >= threshold else "FAIL",
    )

    if threshold > 0 and best_confidence < threshold:
        log.info(
            "  Best confidence %.4f < threshold %.4f → treating as retrieval failure",
            best_confidence,
            threshold,
        )
        return "retrieval_failure_node"

    return "compress_context_node"


def _route_compression(state: Dict[str, Any]) -> str:
    """
    After compress_context_node:
    - Error → compression_failure_node
    - OK → generate_answer_node
    """
    error = state.get("error_message", "")
    if error and error.startswith("compression_error"):
        return "compression_failure_node"
    return "generate_answer_node"


def _route_generation(state: Dict[str, Any]) -> str:
    """
    After generate_answer_node:
    - Needs retry AND retries available → generate_answer_node (self-loop)
    - Retries exhausted → llm_timeout_failure_node
    - OK → collect_sub_answers
    """
    error = state.get("error_message", "")
    cfg = get_settings()

    if error and error.startswith("generation_retry"):
        retries = state.get("generation_retries", 0)
        if retries <= cfg.graph_max_retries:
            return "generate_answer_node"
        return "llm_timeout_failure_node"

    if error and error.startswith("generation_failed"):
        return "llm_timeout_failure_node"

    return "collect_sub_answers"


def _route_collect(state: Dict[str, Any]) -> str:
    """
    After collect_sub_answers:
    - More sub-queries to process → back to ambiguity_check with next query
    - Done → merge_final_answers
    """
    sub_queries = state.get("sub_queries") or []
    sub_answers = state.get("sub_answers") or []
    answered_count = len(sub_answers)

    if answered_count < len(sub_queries):
        return "advance_to_next_query"
    return "merge_final_answers"


# ── Utility node: advance to the next sub-query ──────────────────────────────

def _advance_to_next_query(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pseudo-node: pull the next unanswered sub-query into current_query.
    Resets per-query fields.
    """
    sub_queries = state.get("sub_queries") or []
    answered_count = len(state.get("sub_answers") or [])
    next_query = sub_queries[answered_count] if answered_count < len(sub_queries) else ""
    return {
        "current_query": next_query,
        "is_ambiguous": False,
        "clarification_question": None,
        "clarified_query": "",
        "rewritten_queries": [],
        "retrieved_docs_with_scores": [],
        "final_retrieved_docs": [],
        "compressed_context": "",
        "answer_text": "",
        "generation_retries": 0,
        "error_message": None,
    }


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_rag_graph() -> StateGraph:
    """
    Assemble and compile the full RAG LangGraph.
    Returns a compiled graph ready to invoke.
    """
    sg = StateGraph(RAGState)

    # Register all nodes
    sg.add_node("normalize_user_prompt", normalize_user_prompt)
    sg.add_node("detect_multi_query", detect_multi_query)
    sg.add_node("ambiguity_check", ambiguity_check)
    sg.add_node("clarification_node", clarification_node)
    sg.add_node("query_rewrite_expand", query_rewrite_expand)
    sg.add_node("adaptive_top_k_decision", adaptive_top_k_decision)
    sg.add_node("retrieve_documents", retrieve_documents)
    sg.add_node("merge_retrieval_results", merge_retrieval_results)
    sg.add_node("retrieval_failure_node", retrieval_failure_node)
    sg.add_node("compress_context_node", compress_context_node)
    sg.add_node("compression_failure_node", compression_failure_node)
    sg.add_node("generate_answer_node", generate_answer_node)
    sg.add_node("llm_timeout_failure_node", llm_timeout_failure_node)
    sg.add_node("collect_sub_answers", collect_sub_answers)
    sg.add_node("advance_to_next_query", _advance_to_next_query)
    sg.add_node("merge_final_answers", merge_final_answers)

    # Entry point
    sg.set_entry_point("normalize_user_prompt")

    # ── Fixed edges ───────────────────────────────────────────────────────────
    sg.add_edge("normalize_user_prompt", "detect_multi_query")
    sg.add_edge("detect_multi_query", "ambiguity_check")

    # ambiguity → conditional
    sg.add_conditional_edges(
        "ambiguity_check",
        _route_ambiguity,
        {
            "clarification_node": "clarification_node",
            "query_rewrite_expand": "query_rewrite_expand",
        },
    )

    # clarification → END (caller must inject clarified_query and resume)
    sg.add_conditional_edges(
        "clarification_node",
        _route_post_clarification,
        {END: END},
    )

    # Linear sub-graph flow
    sg.add_edge("query_rewrite_expand", "adaptive_top_k_decision")
    sg.add_edge("adaptive_top_k_decision", "retrieve_documents")
    sg.add_edge("retrieve_documents", "merge_retrieval_results")

    # Retrieval → conditional
    sg.add_conditional_edges(
        "merge_retrieval_results",
        _route_retrieval,
        {
            "retrieval_failure_node": "retrieval_failure_node",
            "compress_context_node": "compress_context_node",
        },
    )

    # Retrieval failure → collect (so multi-query flow can continue)
    sg.add_edge("retrieval_failure_node", "collect_sub_answers")

    # Compression → conditional
    sg.add_conditional_edges(
        "compress_context_node",
        _route_compression,
        {
            "compression_failure_node": "compression_failure_node",
            "generate_answer_node": "generate_answer_node",
        },
    )

    # Compression fallback feeds straight into generation
    sg.add_edge("compression_failure_node", "generate_answer_node")

    # Generation → conditional (retry or proceed)
    sg.add_conditional_edges(
        "generate_answer_node",
        _route_generation,
        {
            "generate_answer_node": "generate_answer_node",
            "llm_timeout_failure_node": "llm_timeout_failure_node",
            "collect_sub_answers": "collect_sub_answers",
        },
    )

    # LLM timeout feeds collect so multi-query flow does not break
    sg.add_edge("llm_timeout_failure_node", "collect_sub_answers")

    # Collect → conditional (more queries or final merge)
    sg.add_conditional_edges(
        "collect_sub_answers",
        _route_collect,
        {
            "advance_to_next_query": "advance_to_next_query",
            "merge_final_answers": "merge_final_answers",
        },
    )

    # Loop back to process next sub-query
    sg.add_edge("advance_to_next_query", "ambiguity_check")

    # Terminal
    sg.add_edge("merge_final_answers", END)

    return sg.compile()


# Module-level compiled graph (import this in pipeline/run.py)
rag_graph = build_rag_graph()

__all__ = ["rag_graph", "build_rag_graph"]

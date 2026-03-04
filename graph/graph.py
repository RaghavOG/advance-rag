"""
LangGraph wiring — full RAG pipeline topology.

Node execution order
--------------------
  normalize_user_prompt
      │
  safety_filter_node         ← rule-based, fast
      ├─ blocked → safety_blocked_node → END
      │
  detect_multi_query
      ├─ multi + ENABLE_PARALLEL_MULTI_QUERY → parallel_multi_query_node → merge_final_answers
      │
  ambiguity_check
      ├─ ambiguous & !clarification_used → clarification_node → END
      │
  query_rewrite_expand
      │
  adaptive_top_k_decision
      │
  retrieve_documents          ← first-pass multi-rewrite search (no HyDE)
      │
  score_normalizer_node       ← raw_score → confidence in [0,1]
      │
  merge_retrieval_results     ← dedup, sort desc, top-k
      │
  hyde_augmentation_node      ← confidence-gated HyDE (triggers only when
      │                          best_confidence < HYDE_CONFIDENCE_THRESHOLD)
      │
  reranker_node               ← LLM re-scores (ENABLE_RERANKER, passthrough if off)
      ├─ no docs / low conf → retrieval_failure_node → collect_sub_answers
      │
  compress_context_node       ← with expansion guard
      ├─ error → compression_failure_node → generate_answer_node
      │
  generate_answer_node
      ├─ retry → generate_answer_node (self-loop)
      ├─ exhausted → llm_timeout_failure_node → collect_sub_answers
      │
  faithfulness_check_node     ← grounding verification (ENABLE_FAITHFULNESS_CHECK)
      │
  collect_sub_answers
      ├─ more sub-queries → process_next_subquery → ambiguity_check
      │
  merge_final_answers
      │
      END

Clarification resume (POST /api/clarify)
-----------------------------------------
The backend re-invokes the graph with:
  {"raw_prompt": combined, "clarification_used": True, "clarified_query": combined}

detect_multi_query preserves clarification_used=True from the initial state.
_route_ambiguity detects clarification_used=True or clarified_query being set
and routes directly to query_rewrite_expand, skipping the LLM ambiguity call.
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
    faithfulness_check_node,
    generate_answer_node,
    hyde_augmentation_node,
    llm_timeout_failure_node,
    merge_final_answers,
    merge_retrieval_results,
    normalize_user_prompt,
    parallel_multi_query_node,
    query_rewrite_expand,
    reranker_node,
    retrieval_failure_node,
    retrieve_documents,
    safety_blocked_node,
    safety_filter_node,
    score_normalizer_node,
)
from graph.state import RAGState
from utils.logger import get_logger

log = get_logger(__name__)


# ── Conditional routing functions ────────────────────────────────────────────

def _route_safety(state: Dict[str, Any]) -> str:
    if state.get("safety_flagged"):
        return "safety_blocked_node"
    return "detect_multi_query"


def _route_after_detect(state: Dict[str, Any]) -> str:
    """
    After detect_multi_query: choose sequential vs parallel path.

    - multi + ENABLE_PARALLEL_MULTI_QUERY → parallel executor (each sub-query
      runs the full pipeline via pipeline.single_query.run_single_query)
    - anything else → ambiguity_check (sequential per-query loop)
    """
    cfg = get_settings()
    if state.get("query_route") == "multi" and cfg.enable_parallel_multi_query:
        return "parallel_multi_query_node"
    return "ambiguity_check"


def _route_ambiguity(state: Dict[str, Any]) -> str:
    """
    After ambiguity_check:
    - clarified_query is set (user already answered clarification) → rewrite directly
    - clarification_used=True (injected by /api/clarify) → rewrite directly
    - truly ambiguous and no clarification used yet → clarification_node
    - otherwise → query_rewrite_expand
    """
    if state.get("clarified_query") or state.get("clarification_used"):
        return "query_rewrite_expand"
    if state.get("is_ambiguous") and not state.get("clarification_used"):
        return "clarification_node"
    return "query_rewrite_expand"


def _route_post_clarification(state: Dict[str, Any]) -> str:
    """After clarification_node we short-circuit to END (caller must resume)."""
    return END


def _route_retrieval(state: Dict[str, Any]) -> str:
    """
    After reranker_node (which follows hyde_augmentation_node):
    - No docs → retrieval_failure_node
    - All docs below confidence threshold → retrieval_failure_node
    - Otherwise → compress_context_node
    """
    docs = state.get("final_retrieved_docs") or []
    if not docs:
        log.info("  Retrieval gate: no docs → failure")
        return "retrieval_failure_node"

    cfg = get_settings()
    threshold = cfg.retrieval_confidence_threshold
    best_confidence = max((d.get("score", 0.0) for d in docs), default=0.0)

    gate_result = "PASS" if threshold == 0 or best_confidence >= threshold else "FAIL"
    log.info(
        "  Retrieval confidence gate: best_confidence=%.4f  threshold=%.4f  %s",
        best_confidence,
        threshold,
        gate_result,
    )

    if threshold > 0 and best_confidence < threshold:
        return "retrieval_failure_node"
    return "compress_context_node"


def _route_compression(state: Dict[str, Any]) -> str:
    error = state.get("error_message", "")
    if error and error.startswith("compression_error"):
        return "compression_failure_node"
    return "generate_answer_node"


def _route_generation(state: Dict[str, Any]) -> str:
    """
    After generate_answer_node:
    - generation_retry: loop back to generate_answer_node if retries remain
    - generation_failed / retries exhausted: llm_timeout_failure_node
    - success: faithfulness_check_node (which then proceeds to collect_sub_answers)
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

    return "faithfulness_check_node"


def _route_collect(state: Dict[str, Any]) -> str:
    """After collect_sub_answers: more sub-queries → loop, else → final merge."""
    sub_queries = state.get("sub_queries") or []
    sub_answers = state.get("sub_answers") or []
    if len(sub_answers) < len(sub_queries):
        return "process_next_subquery"
    return "merge_final_answers"


# ── Utility node ─────────────────────────────────────────────────────────────

def _process_next_subquery(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pseudo-node: advance current_query to the next unanswered sub-query.

    Resets all per-query state fields so the next sub-query gets a fresh run
    through the full pipeline (ambiguity → rewrite → retrieve → compress →
    generate → faithfulness).  clarification_used is intentionally NOT reset:
    a user cannot receive a second clarification prompt for remaining sub-queries.
    """
    sub_queries = state.get("sub_queries") or []
    answered_count = len(state.get("sub_answers") or [])
    next_query = sub_queries[answered_count] if answered_count < len(sub_queries) else ""
    return {
        "current_query": next_query,
        "sub_query_index": answered_count,
        "is_ambiguous": False,
        "clarification_question": None,
        "clarified_query": "",
        "rewritten_queries": [],
        "hyde_doc_text": "",
        "retrieved_docs_with_scores": [],
        "final_retrieved_docs": [],
        "compressed_context": "",
        "answer_text": "",
        "generation_retries": 0,
        "query_status": "pending",
        "last_error": None,
        "error_message": None,
        # clarification_used intentionally NOT reset — a user cannot get a second
        # clarification prompt for the remaining sub-queries.
    }


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_rag_graph() -> StateGraph:
    """Assemble and compile the full RAG LangGraph."""
    sg = StateGraph(RAGState)

    # Register all nodes
    sg.add_node("normalize_user_prompt", normalize_user_prompt)
    sg.add_node("safety_filter_node", safety_filter_node)
    sg.add_node("safety_blocked_node", safety_blocked_node)
    sg.add_node("detect_multi_query", detect_multi_query)
    sg.add_node("parallel_multi_query_node", parallel_multi_query_node)
    sg.add_node("ambiguity_check", ambiguity_check)
    sg.add_node("clarification_node", clarification_node)
    sg.add_node("query_rewrite_expand", query_rewrite_expand)
    sg.add_node("adaptive_top_k_decision", adaptive_top_k_decision)
    sg.add_node("retrieve_documents", retrieve_documents)
    sg.add_node("score_normalizer_node", score_normalizer_node)
    sg.add_node("merge_retrieval_results", merge_retrieval_results)
    sg.add_node("hyde_augmentation_node", hyde_augmentation_node)
    sg.add_node("reranker_node", reranker_node)
    sg.add_node("retrieval_failure_node", retrieval_failure_node)
    sg.add_node("compress_context_node", compress_context_node)
    sg.add_node("compression_failure_node", compression_failure_node)
    sg.add_node("generate_answer_node", generate_answer_node)
    sg.add_node("faithfulness_check_node", faithfulness_check_node)
    sg.add_node("llm_timeout_failure_node", llm_timeout_failure_node)
    sg.add_node("collect_sub_answers", collect_sub_answers)
    sg.add_node("process_next_subquery", _process_next_subquery)
    sg.add_node("merge_final_answers", merge_final_answers)

    # ── Entry point ───────────────────────────────────────────────────────────
    sg.set_entry_point("normalize_user_prompt")

    # ── Fixed edges ───────────────────────────────────────────────────────────
    sg.add_edge("normalize_user_prompt", "safety_filter_node")

    sg.add_conditional_edges(
        "safety_filter_node",
        _route_safety,
        {
            "safety_blocked_node": "safety_blocked_node",
            "detect_multi_query": "detect_multi_query",
        },
    )

    # Safety block → terminal
    sg.add_edge("safety_blocked_node", END)

    # Detect → conditional (parallel or sequential)
    sg.add_conditional_edges(
        "detect_multi_query",
        _route_after_detect,
        {
            "parallel_multi_query_node": "parallel_multi_query_node",
            "ambiguity_check": "ambiguity_check",
        },
    )

    # Parallel executor → merge (each sub-query already fully answered internally)
    sg.add_edge("parallel_multi_query_node", "merge_final_answers")

    # Ambiguity → conditional
    sg.add_conditional_edges(
        "ambiguity_check",
        _route_ambiguity,
        {
            "clarification_node": "clarification_node",
            "query_rewrite_expand": "query_rewrite_expand",
        },
    )

    # Clarification → END (caller resumes via POST /api/clarify)
    sg.add_conditional_edges("clarification_node", _route_post_clarification, {END: END})

    # Linear retrieval pipeline (HyDE now AFTER first-pass merge)
    sg.add_edge("query_rewrite_expand", "adaptive_top_k_decision")
    sg.add_edge("adaptive_top_k_decision", "retrieve_documents")
    sg.add_edge("retrieve_documents", "score_normalizer_node")
    sg.add_edge("score_normalizer_node", "merge_retrieval_results")
    sg.add_edge("merge_retrieval_results", "hyde_augmentation_node")
    sg.add_edge("hyde_augmentation_node", "reranker_node")

    # Reranker → confidence gate
    sg.add_conditional_edges(
        "reranker_node",
        _route_retrieval,
        {
            "retrieval_failure_node": "retrieval_failure_node",
            "compress_context_node": "compress_context_node",
        },
    )

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

    sg.add_edge("compression_failure_node", "generate_answer_node")

    # Generation → conditional (retry / timeout / faithfulness check)
    sg.add_conditional_edges(
        "generate_answer_node",
        _route_generation,
        {
            "generate_answer_node": "generate_answer_node",
            "llm_timeout_failure_node": "llm_timeout_failure_node",
            "faithfulness_check_node": "faithfulness_check_node",
        },
    )

    # Faithfulness check always proceeds to collect (it only annotates, never blocks)
    sg.add_edge("faithfulness_check_node", "collect_sub_answers")

    sg.add_edge("llm_timeout_failure_node", "collect_sub_answers")

    # Collect → conditional (loop or final merge)
    sg.add_conditional_edges(
        "collect_sub_answers",
        _route_collect,
        {
            "process_next_subquery": "process_next_subquery",
            "merge_final_answers": "merge_final_answers",
        },
    )

    sg.add_edge("process_next_subquery", "ambiguity_check")
    sg.add_edge("merge_final_answers", END)

    return sg.compile()


# Module-level compiled graph (import this in pipeline/run.py and tests)
rag_graph = build_rag_graph()

__all__ = ["rag_graph", "build_rag_graph"]

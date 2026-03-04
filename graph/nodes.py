"""
All LangGraph nodes.

New topology vs. previous version
-----------------------------------
  [normalize_user_prompt]
      │
  [safety_filter_node]          ← rule-based, fast, no LLM
      │ clean / blocked
      ├─ blocked → [safety_blocked_node] → END
      │
  [detect_multi_query]
      │ single / multi (sequential) / multi (parallel)
      ├─ parallel → [parallel_multi_query_node] → [merge_final_answers]
      │
  [ambiguity_check]
      ├─ ambiguous & !clarification_used → [clarification_node] → END
      │
  [query_rewrite_expand]
      │
  [adaptive_top_k_decision]
      │
  [retrieve_documents]          (multi-rewrite search only; raw scores)
      │
  [score_normalizer_node]       (raw_score → confidence, adds backend label)
      │
  [merge_retrieval_results]     (dedup, sort by confidence desc, top-k)
      │
  [hyde_augmentation_node]      ← NEW  (confidence-gated HyDE; runs AFTER first-pass)
      │                             Only triggers if best_confidence < HYDE_CONFIDENCE_THRESHOLD
      │
  [reranker_node]               (LLM re-scores, ENABLE_RERANKER)
      │ has docs / no docs
      ├─ no docs / low conf → [retrieval_failure_node] → [collect_sub_answers]
      │
  [compress_context_node]       (with expansion guard)
      │
  [generate_answer_node]
      │
  [faithfulness_check_node]     ← NEW  (grounding verification, ENABLE_FAITHFULNESS_CHECK)
      │
  [collect_sub_answers]
      │
  [merge_final_answers]
"""
from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from openai import OpenAI

from compression.compressor import compress_context
from config.settings import get_settings
from embeddings.factory import get_embedder
from generation.answer import generate_answer
from graph.reranker import rerank_documents
from graph.safety import check_safety
from query.decompose import split_queries
from query.rewrite import generate_hyde_document, rewrite_queries
from retrieval.retriever import _adaptive_top_k
from utils.logger import get_logger
from vectorstores.factory import create_vectorstore, normalize_score

log = get_logger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _openai(timeout: Optional[int] = None) -> OpenAI:
    cfg = get_settings()
    return OpenAI(api_key=cfg.openai_api_key, timeout=timeout or cfg.openai_timeout)


def _doc_to_dict(doc: Document, score: float = 0.0, raw_score: Optional[float] = None) -> Dict[str, Any]:
    return {
        "page_content": doc.page_content,
        "metadata": doc.metadata or {},
        "score": score,
        "raw_score": raw_score if raw_score is not None else score,
    }


def _dict_to_doc(d: Dict[str, Any]) -> Document:
    return Document(page_content=d["page_content"], metadata=d.get("metadata", {}))


def _add_timing(state: Dict[str, Any], key: str, elapsed_ms: float) -> Dict[str, float]:
    """
    Return an updated timings dict — reads existing entries so callers do not
    accidentally clobber sibling timings written by other nodes.
    """
    t = dict(state.get("timings") or {})
    t[key] = round(elapsed_ms, 1)
    return t


def _dedup_merge(
    docs: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Deduplicate by (source, chunk_id), sort by score desc, return top_k."""
    def _key(d: Dict[str, Any]) -> Tuple[Any, Any]:
        m = d.get("metadata", {})
        return (m.get("doc_id") or m.get("source"), m.get("chunk_id"))

    seen: set = set()
    result: List[Dict[str, Any]] = []
    for d in sorted(docs, key=lambda x: x.get("score", 0.0), reverse=True):
        k = _key(d)
        if k not in seen:
            seen.add(k)
            result.append(d)
            if len(result) >= top_k:
                break
    return result


# ── LAYER 0 ───────────────────────────────────────────────────────────────────

def normalize_user_prompt(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 0 — trim and normalize the raw prompt."""
    raw = state.get("raw_prompt", "")
    normalized = re.sub(r"\s+", " ", raw).strip()
    log.info("◉ NODE  normalize_user_prompt  prompt=%r", normalized[:80])
    return {
        "normalized_prompt": normalized,
        "query_status": "pending",
    }


# ── SAFETY ────────────────────────────────────────────────────────────────────

def safety_filter_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node S1 — fast rule-based safety check.
    Sets safety_flagged=True when prompt injection or policy violations are detected.
    No LLM calls; runs in <1 ms.
    """
    cfg = get_settings()
    prompt = state.get("normalized_prompt", "")
    log.info("◉ NODE  safety_filter_node")

    if not cfg.enable_safety_filter:
        log.debug("  Safety filter disabled (ENABLE_SAFETY_FILTER=false)")
        return {"safety_flagged": False, "safety_reason": None}

    is_safe, reason = check_safety(prompt, max_chars=cfg.safety_max_input_chars)
    if not is_safe:
        log.warning("  BLOCKED  reason=%r  prompt_preview=%r", reason, prompt[:60])
    else:
        log.debug("  CLEAN")

    return {
        "safety_flagged": not is_safe,
        "safety_reason": reason if not is_safe else None,
    }


def safety_blocked_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node S2 — terminal node when safety filter fires."""
    reason = state.get("safety_reason") or "Your request was blocked by the content policy."
    log.warning("◉ NODE  safety_blocked_node  reason=%r", reason)
    return {
        "final_answer": f"I'm unable to process that request. {reason}",
        "query_status": "failed",
        "error_message": f"safety_blocked: {reason}",
    }


# ── LAYER 1 ───────────────────────────────────────────────────────────────────

def detect_multi_query(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 1 — detect multiple independent questions."""
    prompt = state.get("normalized_prompt", "")
    cfg = get_settings()
    log.info("◉ NODE  detect_multi_query")
    sub_queries = split_queries(prompt)

    if not sub_queries:
        sub_queries = [prompt]

    log.info(
        "  sub_queries=%d  route=%s  queries=%s",
        len(sub_queries),
        "multi" if len(sub_queries) > 1 else "single",
        sub_queries,
    )

    if len(sub_queries) > cfg.max_sub_queries:
        sub_queries = sub_queries[: cfg.max_sub_queries]

    route = "multi" if len(sub_queries) > 1 else "single"
    current_query = sub_queries[0]

    # CRITICAL: preserve clarification_used if it was set by the caller
    # (e.g. POST /api/clarify passes clarification_used=True so the ambiguity
    # check is bypassed on the re-invocation after the user answers).
    incoming_clarification_used = state.get("clarification_used", False)
    incoming_clarified_query = state.get("clarified_query", "")

    return {
        "sub_queries": sub_queries,
        "query_route": route,
        "current_query": current_query,
        "sub_query_index": 0,
        # Preserve clarification state from the initial invoke payload
        "clarification_used": incoming_clarification_used,
        "clarified_query": incoming_clarified_query,
        "generation_retries": 0,
        "sub_answers": [],
        "timings": {},
    }


# ── PARALLEL MULTI-QUERY (optional fast path) ─────────────────────────────────

def parallel_multi_query_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node P — parallel sub-query executor.

    Runs each sub-query through the full pipeline (rewriting → retrieval →
    confidence-gated HyDE → reranking → compression → faithfulness check →
    generation) concurrently via a thread pool.  A semaphore caps concurrent
    OpenAI calls to avoid rate-limit errors.

    Delegates to pipeline.single_query.run_single_query() to avoid the circular
    import that would result from calling graph.graph.rag_graph inside
    graph/nodes.py.  That function mirrors every stage of the LangGraph path.

    Only reached when ENABLE_PARALLEL_MULTI_QUERY=true.

    Inputs:  state["sub_queries"], state["timings"]
    Outputs: state["sub_answers"], state["final_answer"], state["query_status"],
             state["timings"]
    """
    # Lazy import to avoid circular: nodes → graph.graph → nodes
    from pipeline.single_query import run_single_query  # noqa: PLC0415

    sub_queries = state.get("sub_queries") or []
    cfg = get_settings()
    max_workers = min(len(sub_queries), cfg.parallel_max_workers)
    log.info("◉ NODE  parallel_multi_query_node  sub_queries=%d  workers=%d", len(sub_queries), max_workers)

    semaphore = threading.Semaphore(max_workers)
    t_start = time.perf_counter()

    def _run(query: str) -> Dict[str, Any]:
        with semaphore:
            return run_single_query(query)

    sub_answers: List[Dict[str, str]] = []
    merged_timings = dict(state.get("timings") or {})

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_query = {pool.submit(_run, q): q for q in sub_queries}
        for future in as_completed(future_to_query, timeout=120):
            query = future_to_query[future]
            try:
                result = future.result()
                sub_answers.append({"question": query, "answer": result.get("answer", "")})
                for k, v in (result.get("timings") or {}).items():
                    merged_timings[f"{k}"] = max(merged_timings.get(k, 0.0), v)
            except Exception as exc:
                log.warning("  parallel sub-query failed for %r: %s", query[:40], exc)
                sub_answers.append({"question": query, "answer": "I couldn't find relevant information."})

    elapsed = round((time.perf_counter() - t_start) * 1000, 1)
    merged_timings["parallel_total_ms"] = elapsed
    log.info("  parallel_multi_query done  elapsed=%.1fms  answers=%d", elapsed, len(sub_answers))

    # Build final_answer from sub_answers so merge_final_answers works correctly
    lines: List[str] = []
    for i, item in enumerate(sub_answers, 1):
        lines.append(f"Question {i}: {item['question']}")
        lines.append("Answer:")
        lines.append(item["answer"])
        lines.append("")

    return {
        "sub_answers": sub_answers,
        "final_answer": "\n".join(lines).rstrip(),
        "query_status": "answered",
        "timings": merged_timings,
    }


# ── LAYER 2 ───────────────────────────────────────────────────────────────────

def ambiguity_check(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 2 — LLM-based ambiguity detection."""
    query = state.get("current_query", "")
    log.info("◉ NODE  ambiguity_check  query=%r", query[:80])
    cfg = get_settings()

    if not cfg.enable_clarification:
        log.info("  Ambiguity detection disabled by config")
        return {"is_ambiguous": False, "clarification_question": None}

    client = _openai(timeout=cfg.ambiguity_timeout)
    system = (
        "You decide whether a user's question is ambiguous.\n"
        "Ambiguous means: missing subject, vague references (this/that), or scope is unclear.\n"
        "Reply ONLY with valid JSON, no extra text:\n"
        '{"is_ambiguous": true|false, "clarification_question": "<string or null>"}'
    )

    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=cfg.ambiguity_model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": query}],
        temperature=0,
        max_tokens=120,
    )
    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    log.info("  ambiguity LLM: %.1fms", elapsed)

    content = resp.choices[0].message.content or "{}"
    if "```" in content:
        content = content.split("```")[1].replace("json", "").strip()
    try:
        data = json.loads(content)
    except Exception:
        data = {"is_ambiguous": False, "clarification_question": None}

    is_ambiguous = bool(data.get("is_ambiguous", False))
    cq = data.get("clarification_question")
    log.info("  is_ambiguous=%s  clarification_question=%r", is_ambiguous, cq)

    return {
        "is_ambiguous": is_ambiguous,
        "clarification_question": cq,
        "timings": _add_timing(state, "ambiguity_ms", elapsed),
    }


def clarification_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 3 — surface clarification question to caller."""
    cq = state.get("clarification_question") or "Could you please clarify your question?"
    log.info("◉ NODE  clarification_node  question=%r", cq)
    return {
        "clarification_used": True,
        "clarified_query": "",
        "query_status": "clarification_needed",
        "error_message": f"CLARIFICATION_NEEDED: {cq}",
    }


def query_rewrite_expand(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 4 — LLM-based query expansion."""
    query = state.get("clarified_query") or state.get("current_query", "")
    log.info("◉ NODE  query_rewrite_expand  query=%r", query[:80])
    t0 = time.perf_counter()
    rewrites = rewrite_queries(query)
    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    log.info("  rewrites=%d  rewrite_elapsed=%.1fms", len(rewrites), elapsed)
    return {
        "rewritten_queries": rewrites,
        "timings": _add_timing(state, "rewrite_ms", elapsed),
    }


def adaptive_top_k_decision(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 5 — heuristic adaptive top-k selection."""
    query = state.get("current_query", "")
    rewrites = state.get("rewritten_queries") or [query]
    cfg = get_settings()
    per_k = _adaptive_top_k(query, len(rewrites), base_k=cfg.top_k_text)
    log.info(
        "◉ NODE  adaptive_top_k_decision  top_k_text=%d  top_k_image=%d  top_k_audio=%d",
        per_k, cfg.top_k_image, cfg.top_k_audio,
    )
    return {"top_k_text": per_k, "top_k_image": cfg.top_k_image, "top_k_audio": cfg.top_k_audio}


def retrieve_documents(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 6 — multi-rewrite vector retrieval (first pass only; no HyDE here).

    Returns raw (un-normalised) scores.  score_normalizer_node converts them
    to [0,1] confidences in the very next step.

    HyDE is handled AFTER the first-pass merge in hyde_augmentation_node so
    that it can be confidence-gated (only triggered when first-pass results
    are weak).

    Each doc dict includes:
      raw_score : value from vs.similarity_search_with_score (semantics depend on backend)
      score     : set to raw_score here; overwritten by score_normalizer_node
      backend   : cfg.vector_store.value (for logging and debugging)

    Inputs:  state["rewritten_queries"], state["top_k_text"]
    Outputs: state["retrieved_docs_with_scores"], state["timings"]
    """
    rewrites = state.get("rewritten_queries") or [state.get("current_query", "")]
    top_k = state.get("top_k_text") or get_settings().top_k_text
    cfg = get_settings()
    log.info("◉ NODE  retrieve_documents  rewrites=%d  top_k=%d", len(rewrites), top_k)

    vs = create_vectorstore(collection_name=cfg.chroma_collection_text)
    t_start = time.perf_counter()
    results: List[Dict[str, Any]] = []

    for rewrite_id, rq in enumerate(rewrites):
        t_rw = time.perf_counter()
        try:
            scored = vs.similarity_search_with_score(rq, k=top_k)
            log.debug(
                "  [rewrite %d] %d result(s)  raw_scores=%s  backend=%s  elapsed=%.1fms",
                rewrite_id,
                len(scored),
                [round(float(s), 4) for _, s in scored],
                cfg.vector_store.value,
                (time.perf_counter() - t_rw) * 1000,
            )
        except Exception as exc:
            log.warning("  similarity_search_with_score failed rewrite %d (%s) — unscored fallback", rewrite_id, exc)
            raw_docs = vs.similarity_search(rq, k=top_k)
            scored = [(d, 0.0) for d in raw_docs]

        for doc, raw_score in scored:
            doc.metadata = doc.metadata or {}
            doc.metadata.setdefault("rewrite_id", rewrite_id)
            results.append({
                "page_content": doc.page_content,
                "metadata": {**doc.metadata, "backend": cfg.vector_store.value},
                "score": float(raw_score),       # normalizer overwrites this
                "raw_score": float(raw_score),
            })

    elapsed = round((time.perf_counter() - t_start) * 1000, 1)
    log.info("  Total raw results (pre-normalise): %d  retrieve_elapsed=%.1fms", len(results), elapsed)
    return {
        "retrieved_docs_with_scores": results,
        "timings": _add_timing(state, "retrieve_ms", elapsed),
    }


def score_normalizer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 7 — convert raw backend scores to normalised confidence in [0, 1].

    Emits per-doc fields:
      raw_score  : the original value from the vector store API
      score      : normalised confidence (higher = more relevant)
      backend    : vector store backend name

    Inputs:  state["retrieved_docs_with_scores"]
    Outputs: state["retrieved_docs_with_scores"] (updated with normalised scores)
    """
    log.info("◉ NODE  score_normalizer_node")
    cfg = get_settings()
    raw_list = state.get("retrieved_docs_with_scores") or []

    normalised: List[Dict[str, Any]] = []
    for d in raw_list:
        raw = d.get("raw_score", d.get("score", 0.0))
        confidence, norm_desc = normalize_score(float(raw))

        log.debug(
            "  raw_score=%.4f  confidence=%.4f  backend=%s  (%s)",
            raw,
            confidence,
            cfg.vector_store.value,
            norm_desc,
        )
        updated = dict(d)
        updated["score"] = confidence
        updated["raw_score"] = raw
        # Store backend and confidence in metadata too so they survive serialisation
        updated["metadata"] = {
            **updated.get("metadata", {}),
            "confidence": confidence,
            "backend": cfg.vector_store.value,
        }
        normalised.append(updated)

    return {"retrieved_docs_with_scores": normalised}


def merge_retrieval_results(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 8 — deduplicate chunks and select top-k by normalised confidence."""
    log.info("◉ NODE  merge_retrieval_results")
    raw = state.get("retrieved_docs_with_scores") or []
    top_k = state.get("top_k_text") or get_settings().top_k_text

    final = _dedup_merge(raw, top_k)

    confidence_values = [round(d.get("score", 0.0), 4) for d in final]
    log.info(
        "  After dedup/merge: %d docs  (from %d raw)  confidences=%s",
        len(final), len(raw), confidence_values,
    )
    return {"final_retrieved_docs": final}


# ── HyDE AUGMENTATION (confidence-gated, runs AFTER first-pass merge) ─────────

def hyde_augmentation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 9 — confidence-gated HyDE augmentation (runs after merge_retrieval_results).

    Activation logic:
      - ENABLE_HYDE must be true
      - best_confidence of first-pass results < HYDE_CONFIDENCE_THRESHOLD (default 0.5)
    If both conditions hold:
      1. Generate a hypothetical answer document via LLM (ephemeral, never shown to user)
      2. Embed the hypothetical document
      3. Search the vector store using that embedding
      4. Assign each HyDE-retrieved doc a fixed fallback confidence of 0.5
      5. Merge HyDE results into final_retrieved_docs (dedup, re-sort, top-k)

    If conditions are NOT met (high-confidence first pass or HyDE disabled):
      passthrough — final_retrieved_docs unchanged

    Inputs:  state["final_retrieved_docs"], state["top_k_text"],
             state["clarified_query"] / state["current_query"]
    Outputs: state["final_retrieved_docs"] (possibly augmented),
             state["hyde_doc_text"] (for audit), state["timings"]
    """
    cfg = get_settings()
    docs = state.get("final_retrieved_docs") or []
    query = state.get("clarified_query") or state.get("current_query", "")
    log.info("◉ NODE  hyde_augmentation_node  docs=%d  hyde_enabled=%s", len(docs), cfg.enable_hyde)

    if not cfg.enable_hyde:
        log.debug("  HyDE disabled (ENABLE_HYDE=false) — passthrough")
        return {}

    if not docs:
        log.debug("  No first-pass docs — skipping HyDE (nothing to augment)")
        return {}

    best_confidence = max((d.get("score", 0.0) for d in docs), default=0.0)
    if best_confidence >= cfg.hyde_confidence_threshold:
        log.info(
            "  First-pass best_confidence=%.4f >= threshold=%.4f → HyDE not needed",
            best_confidence,
            cfg.hyde_confidence_threshold,
        )
        return {}  # fast path: confidence already high

    log.info(
        "  Low first-pass confidence %.4f < threshold %.4f → triggering HyDE",
        best_confidence,
        cfg.hyde_confidence_threshold,
    )

    t0 = time.perf_counter()
    try:
        hyde_text = generate_hyde_document(query)
    except Exception as exc:
        log.warning("  HyDE document generation failed (%s) — passthrough", exc)
        return {}

    if not hyde_text:
        log.warning("  HyDE document generation returned empty — passthrough")
        return {}

    log.debug("  HyDE doc generated: %d chars", len(hyde_text))

    top_k = state.get("top_k_text") or cfg.top_k_text
    vs = create_vectorstore(collection_name=cfg.chroma_collection_text)

    try:
        embedder = get_embedder()
        vec = embedder.embed_query(hyde_text)
        hyde_raw_docs = vs.similarity_search_by_vector(vec, k=top_k)
        log.debug("  HyDE vector search: %d doc(s)", len(hyde_raw_docs))
    except Exception as exc:
        log.warning("  HyDE vector search failed (%s) — passthrough", exc)
        return {}

    # Assign fixed fallback confidence; -1.0 raw_score marks these as HyDE docs
    hyde_fallback = 0.5
    hyde_results: List[Dict[str, Any]] = []
    for doc in hyde_raw_docs:
        doc.metadata = doc.metadata or {}
        doc.metadata["hyde"] = True
        hyde_results.append({
            "page_content": doc.page_content,
            "metadata": {**doc.metadata, "confidence": hyde_fallback, "backend": cfg.vector_store.value},
            "score": hyde_fallback,
            "raw_score": -1.0,
        })

    # Merge first-pass + HyDE results: dedup, re-sort by confidence, top-k
    merged = _dedup_merge(docs + hyde_results, top_k)

    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    log.info(
        "  HyDE augmentation done: %d original + %d hyde → %d merged  elapsed=%.1fms",
        len(docs), len(hyde_results), len(merged), elapsed,
    )

    return {
        "final_retrieved_docs": merged,
        "hyde_doc_text": hyde_text,   # stored for audit/debugging; never shown to user
        "timings": _add_timing(state, "hyde_ms", elapsed),
    }


# ── RERANKER ──────────────────────────────────────────────────────────────────

def reranker_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 10 — optional LLM-based reranker.

    Re-scores the merged candidate docs against the query using a single LLM
    call (all docs in one shot).  Disabled passthrough if ENABLE_RERANKER=false.

    Inputs:  state["final_retrieved_docs"], state["clarified_query"] / current_query
    Outputs: state["final_retrieved_docs"] (re-ordered), state["timings"]
    """
    cfg = get_settings()
    docs = state.get("final_retrieved_docs") or []
    log.info("◉ NODE  reranker_node  docs=%d  enabled=%s", len(docs), cfg.enable_reranker)

    if not cfg.enable_reranker or not docs:
        return {}  # passthrough — final_retrieved_docs unchanged

    query = state.get("clarified_query") or state.get("current_query", "")
    t0 = time.perf_counter()
    reranked = rerank_documents(docs, query, top_n=cfg.reranker_top_n)
    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    log.info("  reranker elapsed=%.1fms  top_n=%d", elapsed, cfg.reranker_top_n)

    return {
        "final_retrieved_docs": reranked,
        "timings": _add_timing(state, "rerank_ms", elapsed),
    }


# ── LAYER 3 ───────────────────────────────────────────────────────────────────

def retrieval_failure_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 11 — surfaced when no docs were retrieved or confidence is too low."""
    query = state.get("current_query", "")[:60]
    log.warning("◉ NODE  retrieval_failure_node  query=%r", query)
    message = "I couldn't find relevant information in the documents."
    return {
        "answer_text": message,
        "final_answer": message,
        "query_status": "failed",
        "error_message": "retrieval_failure",
    }


def compress_context_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 12 — LLM-based context compression with expansion guard.

    After compression, the output length is validated against the source
    length.  If the compressor expanded the context (compressed inner content
    longer than raw source), the node falls back to extractive truncation of
    the raw chunks.  This prevents hallucinated content from being injected into
    the generation prompt via an over-creative compressor.

    Inputs:  state["final_retrieved_docs"], state["clarified_query"] / current_query
    Outputs: state["compressed_context"], state["timings"]
    """
    log.info("◉ NODE  compress_context_node")
    docs_raw = state.get("final_retrieved_docs") or []
    docs = [_dict_to_doc(d) for d in docs_raw]
    query = state.get("clarified_query") or state.get("current_query", "")
    cfg = get_settings()

    # Build source content string for expansion comparison
    raw_source_content = "\n\n".join(d.get("page_content", "") for d in docs_raw)

    t0 = time.perf_counter()
    try:
        compressed = compress_context(docs, query)
    except Exception as exc:
        return {"compressed_context": "", "error_message": f"compression_error:{exc}"}
    elapsed = round((time.perf_counter() - t0) * 1000, 1)

    # ── Expansion guard ───────────────────────────────────────────────────────
    # Strip <<BEGIN/END>> tags to get comparable inner content length.
    inner_match = re.search(
        r"<<BEGIN COMPRESSED CONTEXT>>(.*?)<<END COMPRESSED CONTEXT>>",
        compressed,
        re.DOTALL,
    )
    inner_content = inner_match.group(1).strip() if inner_match else compressed

    if len(inner_content) > len(raw_source_content):
        log.warning(
            "  Compressor EXPANDED context (%d chars > source %d chars) — extractive fallback",
            len(inner_content),
            len(raw_source_content),
        )
        # Rough char budget: cfg.max_context_tokens * 4 chars/token
        max_chars = getattr(cfg, "max_context_tokens", 2000) * 4
        truncated = raw_source_content[:max_chars]
        compressed = f"<<BEGIN COMPRESSED CONTEXT>>\n{truncated}\n<<END COMPRESSED CONTEXT>>"
    else:
        log.info(
            "  compress_context elapsed=%.1fms  %d→%d chars",
            elapsed,
            len(raw_source_content),
            len(inner_content),
        )

    return {
        "compressed_context": compressed,
        "timings": _add_timing(state, "compress_ms", elapsed),
    }


def compression_failure_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 13 — compression failure; falls back to extractive summary."""
    log.warning("◉ NODE  compression_failure_node  — extractive fallback")
    docs_raw = state.get("final_retrieved_docs") or []
    parts = []
    for i, d in enumerate(docs_raw[:3], 1):
        parts.append(f"[{i}] {d.get('page_content', '')[:400]}")
    extractive = "\n\n".join(parts) or "(no context)"
    compressed = f"<<BEGIN COMPRESSED CONTEXT>>\n{extractive}\n<<END COMPRESSED CONTEXT>>"
    return {"compressed_context": compressed, "error_message": "compression_fallback_used"}


def generate_answer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 14 — LLM answer generation."""
    retries = state.get("generation_retries", 0)
    log.info("◉ NODE  generate_answer_node  retry=%d", retries)
    compressed = state.get("compressed_context", "")
    query = state.get("clarified_query") or state.get("current_query", "")
    cfg = get_settings()

    t0 = time.perf_counter()
    try:
        answer = generate_answer(compressed, query)
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        log.info("  generate_answer elapsed=%.1fms", elapsed)
        return {
            "answer_text": answer,
            "query_status": "answered",
            "generation_retries": retries,
            "timings": _add_timing(state, "generate_ms", elapsed),
        }
    except Exception as exc:
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        log.info("  generate_answer failed after %.1fms: %s", elapsed, exc)
        if retries < cfg.graph_max_retries:
            return {
                "answer_text": "",
                "generation_retries": retries + 1,
                "last_error": str(exc),
                "error_message": f"generation_retry:{exc}",
            }
        return {
            "answer_text": "",
            "query_status": "failed",
            "generation_retries": retries,
            "last_error": str(exc),
            "error_message": f"generation_failed:{exc}",
        }


def faithfulness_check_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 15 — post-generation faithfulness / grounding check.

    Verifies that claims in the generated answer are supported by the retrieved
    context.  Uses a single LLM call (low token count) to identify unsupported
    claims.

    Behaviour:
      - ENABLE_FAITHFULNESS_CHECK=false (default): passthrough with no LLM call
      - faithful=true  : answer passes through unchanged
      - faithful=false : unsupported claims are logged at WARNING level and a
                         ⚠ warning prefix is prepended to the answer so the user
                         is informed that some claims may not be grounded

    The node never discards the answer — it only annotates it.

    Inputs:  state["answer_text"], state["compressed_context"]
    Outputs: state["answer_text"] (possibly with warning prefix), state["timings"]
    """
    cfg = get_settings()
    log.info("◉ NODE  faithfulness_check_node  enabled=%s", cfg.enable_faithfulness_check)

    if not cfg.enable_faithfulness_check:
        log.debug("  Faithfulness check disabled (ENABLE_FAITHFULNESS_CHECK=false) — passthrough")
        return {}

    answer = state.get("answer_text", "")
    if not answer or answer.startswith("I couldn't find"):
        log.debug("  No substantive answer to check — passthrough")
        return {}

    context = state.get("compressed_context", "")
    if not context:
        log.debug("  No context to check against — passthrough")
        return {}

    client = _openai(timeout=cfg.openai_timeout)
    system = (
        "You are a faithfulness checker for a RAG system.\n"
        "Given a CONTEXT and an ANSWER, identify any claims in the ANSWER\n"
        "that are NOT explicitly supported by the CONTEXT.\n"
        "Reply ONLY with valid JSON, no extra text:\n"
        '{"faithful": true|false, "unsupported_claims": ["claim1", "claim2"]}\n'
        "If the answer is fully grounded return faithful=true and unsupported_claims=[]."
    )
    user_content = f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"

    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=cfg.faithfulness_check_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
            max_tokens=300,
        )
        elapsed = round((time.perf_counter() - t0) * 1000, 1)

        content = (resp.choices[0].message.content or "{}").strip()
        if "```" in content:
            content = content.split("```")[1].replace("json", "").strip()
        data = json.loads(content)

        faithful = bool(data.get("faithful", True))
        unsupported: List[str] = data.get("unsupported_claims", [])

        if not faithful and unsupported:
            log.warning(
                "  Faithfulness FAILED  elapsed=%.1fms  unsupported_claims=%d: %s",
                elapsed,
                len(unsupported),
                unsupported[:3],
            )
            warning_prefix = (
                "\u26a0\ufe0f Note: Some claims in this answer may not be directly "
                "supported by the retrieved documents.\n\n"
            )
            return {
                "answer_text": warning_prefix + answer,
                "timings": _add_timing(state, "faithfulness_ms", elapsed),
            }

        log.info("  Faithfulness OK  elapsed=%.1fms", elapsed)
        return {"timings": _add_timing(state, "faithfulness_ms", elapsed)}

    except Exception as exc:
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        log.warning("  Faithfulness check failed (%s) — passthrough  elapsed=%.1fms", exc, elapsed)
        return {}


def llm_timeout_failure_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 16 — all generation retries exhausted."""
    log.error("◉ NODE  llm_timeout_failure_node  retries=%d", state.get("generation_retries", 0))
    return {
        "final_answer": "The system is temporarily unable to generate a response. Please try again.",
        "query_status": "failed",
        "error_message": "llm_timeout_failure",
    }


# ── LAYER 4 ───────────────────────────────────────────────────────────────────

def collect_sub_answers(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 17 — accumulate per-query answers in multi-query flow."""
    log.info("◉ NODE  collect_sub_answers")
    sub_answers = list(state.get("sub_answers") or [])
    question = state.get("current_query", "")
    answer = state.get("answer_text", "")
    if question and answer:
        sub_answers.append({"question": question, "answer": answer})
    return {"sub_answers": sub_answers}


def merge_final_answers(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 18 — format all sub-answers into one structured output."""
    sub = state.get("sub_answers") or []
    log.info("◉ NODE  merge_final_answers  sub_answers=%d", len(sub))
    if not sub:
        single = state.get("answer_text", "")
        return {"final_answer": single, "query_status": "answered"}

    lines: List[str] = []
    for i, item in enumerate(sub, 1):
        lines.append(f"Question {i}: {item['question']}")
        lines.append("Answer:")
        lines.append(item["answer"])
        lines.append("")
    return {"final_answer": "\n".join(lines).rstrip(), "query_status": "answered"}

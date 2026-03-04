"""
Direct single-query pipeline (no LangGraph).

Used exclusively by `parallel_multi_query_node` so that independent sub-queries
can be answered concurrently using a thread pool without circular imports
(nodes.py → graph.graph → nodes.py would be circular).

Pipeline stages mirror the LangGraph sequential path exactly:
  1.  Query rewriting            (ENABLE_QUERY_REWRITE)
  2.  Adaptive top-k             (heuristic)
  3.  First-pass retrieval       (multi-rewrite vector search, raw scores)
  4.  Score normalisation        (raw_score → confidence in [0,1])
  5.  Dedup / merge              (top-k by confidence)
  6.  Confidence-gated HyDE      (ENABLE_HYDE + HYDE_CONFIDENCE_THRESHOLD)
  7.  Reranking                  (ENABLE_RERANKER)
  8.  Context compression        (with expansion guard)
  9.  Answer generation
  10. Faithfulness check         (ENABLE_FAITHFULNESS_CHECK)

Returns a plain dict:
  {
    "answer"  : str,
    "docs"    : List[Dict],          # final_retrieved_docs after all stages
    "timings" : Dict[str, float],    # stage_key → elapsed_ms
    "error"   : Optional[str],
  }
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from openai import OpenAI

from compression.compressor import compress_context
from config.settings import get_settings
from embeddings.factory import get_embedder
from generation.answer import generate_answer
from graph.reranker import rerank_documents
from query.rewrite import generate_hyde_document, rewrite_queries
from retrieval.retriever import _adaptive_top_k
from utils.logger import get_logger
from vectorstores.factory import create_vectorstore, normalize_score

log = get_logger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dedup_merge(docs: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """Deduplicate by (source, chunk_id), sort confidence desc, return top_k."""
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


def _openai_client(cfg) -> OpenAI:
    return OpenAI(api_key=cfg.openai_api_key, timeout=cfg.openai_timeout)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_single_query(query: str) -> Dict[str, Any]:
    """
    Run the full single-query RAG pipeline and return a structured result dict.

    Mirrors every stage of the LangGraph sequential path so that the parallel
    multi-query path produces functionally identical results to the sequential
    path.  Thread-safe: each call creates its own vector-store client instance.
    """
    cfg = get_settings()
    timings: Dict[str, float] = {}
    error: Optional[str] = None

    # ── 1. Query rewriting ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        rewrites = rewrite_queries(query)
    except Exception as exc:
        log.warning("[single_query] rewrite failed: %s", exc)
        rewrites = [query]
    timings["rewrite_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # ── 2. Adaptive top-k ─────────────────────────────────────────────────────
    per_k = _adaptive_top_k(query, len(rewrites), cfg.top_k_text)

    # ── 3. First-pass retrieval (no HyDE here) ────────────────────────────────
    t0 = time.perf_counter()
    vs = create_vectorstore(collection_name=cfg.chroma_collection_text)
    raw_results: List[Dict[str, Any]] = []

    for rewrite_id, rq in enumerate(rewrites):
        try:
            scored = vs.similarity_search_with_score(rq, k=per_k)
            for doc, raw_score in scored:
                doc.metadata = doc.metadata or {}
                raw_results.append({
                    "page_content": doc.page_content,
                    "metadata": {**doc.metadata, "backend": cfg.vector_store.value},
                    "score": float(raw_score),      # normalised in step 4
                    "raw_score": float(raw_score),
                })
        except Exception as exc:
            log.warning("[single_query] retrieval failed for rewrite %d: %s", rewrite_id, exc)

    timings["retrieve_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # ── 4. Score normalisation ────────────────────────────────────────────────
    normalised_results: List[Dict[str, Any]] = []
    for d in raw_results:
        raw = d["raw_score"]
        confidence, _ = normalize_score(float(raw))
        updated = dict(d)
        updated["score"] = confidence
        updated["metadata"] = {**updated.get("metadata", {}), "confidence": confidence}
        normalised_results.append(updated)

    # ── 5. Dedup / merge ──────────────────────────────────────────────────────
    final_docs = _dedup_merge(normalised_results, per_k)

    # ── 6. Confidence-gated HyDE ──────────────────────────────────────────────
    # Mirrors hyde_augmentation_node: only triggers when ENABLE_HYDE=true AND
    # the best first-pass confidence is below HYDE_CONFIDENCE_THRESHOLD.
    if cfg.enable_hyde and final_docs:
        t0 = time.perf_counter()
        best_confidence = max((d.get("score", 0.0) for d in final_docs), default=0.0)

        if best_confidence < cfg.hyde_confidence_threshold:
            log.debug(
                "[single_query] HyDE triggered (best_conf=%.4f < threshold=%.4f)",
                best_confidence,
                cfg.hyde_confidence_threshold,
            )
            try:
                hyde_text = generate_hyde_document(query)
                if hyde_text:
                    embedder = get_embedder()
                    vec = embedder.embed_query(hyde_text)
                    hyde_raw_docs = vs.similarity_search_by_vector(vec, k=per_k)
                    hyde_fallback = 0.5
                    hyde_results: List[Dict[str, Any]] = []
                    for doc in hyde_raw_docs:
                        doc.metadata = doc.metadata or {}
                        doc.metadata["hyde"] = True
                        hyde_results.append({
                            "page_content": doc.page_content,
                            "metadata": {
                                **doc.metadata,
                                "confidence": hyde_fallback,
                                "backend": cfg.vector_store.value,
                            },
                            "score": hyde_fallback,
                            "raw_score": -1.0,
                        })
                    final_docs = _dedup_merge(final_docs + hyde_results, per_k)
                    log.debug("[single_query] HyDE merged → %d docs", len(final_docs))
            except Exception as exc:
                log.warning("[single_query] HyDE augmentation failed: %s", exc)

            timings["hyde_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        else:
            log.debug(
                "[single_query] HyDE skipped (best_conf=%.4f >= threshold=%.4f)",
                best_confidence,
                cfg.hyde_confidence_threshold,
            )

    # ── 7. Reranking (optional) ───────────────────────────────────────────────
    if cfg.enable_reranker and final_docs:
        t0 = time.perf_counter()
        final_docs = rerank_documents(final_docs, query, top_n=cfg.reranker_top_n)
        timings["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # ── 8. Context compression (with expansion guard) ─────────────────────────
    raw_source_content = "\n\n".join(d.get("page_content", "") for d in final_docs)
    t0 = time.perf_counter()
    context = ""
    answer = ""
    try:
        lc_docs = [
            Document(page_content=d["page_content"], metadata=d.get("metadata", {}))
            for d in final_docs
        ]
        if lc_docs:
            context = compress_context(lc_docs, query)
        timings["compress_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # Expansion guard: if compressor expanded the context fall back to extractive
        inner_match = re.search(
            r"<<BEGIN COMPRESSED CONTEXT>>(.*?)<<END COMPRESSED CONTEXT>>",
            context,
            re.DOTALL,
        )
        inner_content = inner_match.group(1).strip() if inner_match else context
        if raw_source_content and len(inner_content) > len(raw_source_content):
            log.warning("[single_query] Compressor expanded context — extractive fallback")
            max_chars = getattr(cfg, "max_context_tokens", 2000) * 4
            truncated = raw_source_content[:max_chars]
            context = f"<<BEGIN COMPRESSED CONTEXT>>\n{truncated}\n<<END COMPRESSED CONTEXT>>"

        # ── 9. Answer generation ──────────────────────────────────────────────
        t0 = time.perf_counter()
        if context:
            answer = generate_answer(context, query)
        timings["generate_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    except Exception as exc:
        error = str(exc)
        log.warning("[single_query] compress/generate failed: %s", exc)
        timings.setdefault("compress_ms", 0.0)
        timings.setdefault("generate_ms", 0.0)

    # ── 10. Faithfulness check (optional) ────────────────────────────────────
    if cfg.enable_faithfulness_check and answer and context:
        t0 = time.perf_counter()
        try:
            client = _openai_client(cfg)
            system = (
                "You are a faithfulness checker for a RAG system.\n"
                "Given a CONTEXT and an ANSWER, identify any claims in the ANSWER\n"
                "that are NOT explicitly supported by the CONTEXT.\n"
                "Reply ONLY with valid JSON, no extra text:\n"
                '{"faithful": true|false, "unsupported_claims": ["claim1", "claim2"]}\n'
                "If the answer is fully grounded return faithful=true and unsupported_claims=[]."
            )
            resp = client.chat.completions.create(
                model=cfg.faithfulness_check_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"},
                ],
                temperature=0,
                max_tokens=300,
            )
            content_raw = (resp.choices[0].message.content or "{}").strip()
            if "```" in content_raw:
                content_raw = content_raw.split("```")[1].replace("json", "").strip()
            data = json.loads(content_raw)
            faithful = bool(data.get("faithful", True))
            unsupported: List[str] = data.get("unsupported_claims", [])
            if not faithful and unsupported:
                log.warning(
                    "[single_query] Faithfulness FAILED  unsupported=%d: %s",
                    len(unsupported), unsupported[:3],
                )
                answer = (
                    "\u26a0\ufe0f Note: Some claims in this answer may not be directly "
                    "supported by the retrieved documents.\n\n" + answer
                )
        except Exception as exc:
            log.warning("[single_query] faithfulness check failed: %s", exc)
        timings["faithfulness_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "answer": answer or "I couldn't find relevant information for this question.",
        "docs": final_docs,
        "timings": timings,
        "error": error,
    }


__all__ = ["run_single_query"]

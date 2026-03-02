"""
All 15 LangGraph nodes.

Each node receives the full RAGState and returns a partial dict with only
the keys it owns.  LangGraph merges the returned dict into state.

Layer map
---------
0  normalize_user_prompt
1  detect_multi_query
2  ambiguity_check, clarification_node, query_rewrite_expand,
   adaptive_top_k_decision, retrieve_documents, merge_retrieval_results
3  retrieval_failure_node, compress_context_node, compression_failure_node,
   generate_answer_node, llm_timeout_failure_node
4  collect_sub_answers, merge_final_answers
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
from query.decompose import split_queries
from query.rewrite import generate_hyde_document, rewrite_queries
from retrieval.retriever import _adaptive_top_k
from utils.logger import get_logger
from vectorstores.factory import create_vectorstore, normalize_score

log = get_logger(__name__)


# ── helpers ──────────────────────────────────────────────────────────────────

def _openai(timeout: Optional[int] = None) -> OpenAI:
    cfg = get_settings()
    return OpenAI(api_key=cfg.openai_api_key, timeout=timeout or cfg.openai_timeout)


def _doc_to_dict(doc: Document, score: float = 0.0) -> Dict[str, Any]:
    return {"page_content": doc.page_content, "metadata": doc.metadata or {}, "score": score}


def _dict_to_doc(d: Dict[str, Any]) -> Document:
    return Document(page_content=d["page_content"], metadata=d.get("metadata", {}))


# ── LAYER 0 ──────────────────────────────────────────────────────────────────

def normalize_user_prompt(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 1 — trim and normalize the raw prompt."""
    raw = state.get("raw_prompt", "")
    normalized = re.sub(r"\s+", " ", raw).strip()
    log.info("◉ NODE  normalize_user_prompt  prompt=%r", normalized[:80])
    return {"normalized_prompt": normalized}


# ── LAYER 1 ──────────────────────────────────────────────────────────────────

def detect_multi_query(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 2 — detect multiple independent questions."""
    prompt = state.get("normalized_prompt", "")
    cfg = get_settings()
    log.info("◉ NODE  detect_multi_query")
    sub_queries = split_queries(prompt)

    if not sub_queries:
        sub_queries = [prompt]

    log.info("  sub_queries=%d  route=%s  queries=%s",
             len(sub_queries), "multi" if len(sub_queries) > 1 else "single", sub_queries)

    if len(sub_queries) > cfg.max_sub_queries:
        return {
            "sub_queries": sub_queries[:cfg.max_sub_queries],
            "query_route": "multi",
            "error_message": (
                f"Too many questions. Processing the first {cfg.max_sub_queries}."
            ),
        }

    route = "multi" if len(sub_queries) > 1 else "single"
    current_query = sub_queries[0]
    return {
        "sub_queries": sub_queries,
        "query_route": route,
        "current_query": current_query,
        "clarification_used": False,
        "generation_retries": 0,
        "sub_answers": [],
    }


# ── LAYER 2 ──────────────────────────────────────────────────────────────────

def ambiguity_check(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 3 — LLM-based ambiguity detection."""
    query = state.get("current_query", "")
    log.info("◉ NODE  ambiguity_check  query=%r", query[:80])
    cfg = get_settings()
    if not cfg.enable_clarification:
        log.info("  Ambiguity detection disabled by config — treating as unambiguous")
        return {"is_ambiguous": False, "clarification_question": None}

    client = _openai(timeout=cfg.ambiguity_timeout)

    system = (
        "You decide whether a user's question is ambiguous.\n"
        "Ambiguous means: missing subject, vague references (this/that), "
        "or scope is unclear.\n"
        "Reply ONLY with valid JSON, no extra text:\n"
        '{"is_ambiguous": true|false, "clarification_question": "<string or null>"}'
    )
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=cfg.ambiguity_model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": query}],
        temperature=0,
        max_tokens=120,
    )
    log.info("  ambiguity_check LLM call: %.3fs", time.perf_counter() - t0)

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
    return {"is_ambiguous": is_ambiguous, "clarification_question": cq}


def clarification_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 4 — surface clarification question to caller.
    This node does not call an LLM; it surfaces the text already produced
    by ambiguity_check.  In a chat-UI you would return this message to the user
    and resume the graph with the answer as new current_query.
    """
    cq = state.get("clarification_question") or "Could you please clarify your question?"
    log.info("◉ NODE  clarification_node  question=%r", cq)
    return {
        "clarification_used": True,
        "clarified_query": "",
        "error_message": f"CLARIFICATION_NEEDED: {cq}",
    }


def query_rewrite_expand(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 5 — LLM-based query expansion."""
    query = state.get("clarified_query") or state.get("current_query", "")
    log.info("◉ NODE  query_rewrite_expand  query=%r", query[:80])
    rewrites = rewrite_queries(query)
    log.info("  rewrites=%d", len(rewrites))
    return {"rewritten_queries": rewrites}


def adaptive_top_k_decision(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 6 — heuristic adaptive top-k selection."""
    query = state.get("current_query", "")
    rewrites = state.get("rewritten_queries") or [query]
    cfg = get_settings()
    per_k = _adaptive_top_k(query, len(rewrites), base_k=cfg.top_k_text)
    log.info("◉ NODE  adaptive_top_k_decision  top_k_text=%d  top_k_image=%d  top_k_audio=%d",
             per_k, cfg.top_k_image, cfg.top_k_audio)
    return {"top_k_text": per_k, "top_k_image": cfg.top_k_image, "top_k_audio": cfg.top_k_audio}


def retrieve_documents(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 7 — run vector retrieval per rewritten query."""
    rewrites = state.get("rewritten_queries") or [state.get("current_query", "")]
    log.info("◉ NODE  retrieve_documents  rewrites=%d", len(rewrites))
    top_k = state.get("top_k_text") or get_settings().top_k_text
    cfg = get_settings()

    # Always use the configured backend — passing collection_name for Chroma;
    # Pinecone and FAISS ignore it.  The factory logs the actual backend selected.
    vs = create_vectorstore(collection_name=cfg.chroma_collection_text)

    t_retrieve_start = time.perf_counter()
    results: List[Dict[str, Any]] = []
    for rewrite_id, rq in enumerate(rewrites):
        t_rw = time.perf_counter()
        try:
            scored = vs.similarity_search_with_score(rq, k=top_k)
            log.debug(
                "  [rewrite %d] %d result(s)  raw_scores=%s  backend=%s  elapsed=%.3fs",
                rewrite_id,
                len(scored),
                [round(float(s), 4) for _, s in scored],
                cfg.vector_store.value,
                time.perf_counter() - t_rw,
            )
        except Exception as exc:
            log.warning(
                "  similarity_search_with_score failed for rewrite %d (%s) — "
                "falling back to unscored search",
                rewrite_id,
                exc,
            )
            raw_docs = vs.similarity_search(rq, k=top_k)
            scored = [(d, 0.0) for d in raw_docs]

        for doc, raw_score in scored:
            confidence, norm_desc = normalize_score(float(raw_score))
            log.debug(
                "    raw_score=%.4f  confidence=%.4f  (%s)",
                raw_score,
                confidence,
                norm_desc,
            )
            doc.metadata = doc.metadata or {}
            doc.metadata.setdefault("rewrite_id", rewrite_id)
            # Store the normalized confidence — all downstream logic uses this.
            results.append(_doc_to_dict(doc, confidence))

    # ── HyDE pass ────────────────────────────────────────────────────────────
    # HyDE (Hypothetical Document Embeddings) is a RETRIEVAL-ONLY technique:
    #   1. Ask the LLM to write a hypothetical doc that would answer the query.
    #   2. Embed that text to get a richer semantic vector.
    #   3. Use that vector to search the real index.
    #   4. The hypothetical text is immediately discarded; only the retrieved
    #      real documents are kept.
    # Docs retrieved via HyDE do not have a raw similarity score because
    # similarity_search_by_vector does not return one.  We assign a mid-level
    # fallback confidence (0.5) so they participate in the merge and gating
    # rather than being silently zeroed out.
    hyde_fallback_confidence = 0.5

    query = state.get("clarified_query") or state.get("current_query", "")
    t_hyde = time.perf_counter()
    hyde_text = generate_hyde_document(query)
    if hyde_text:
        log.debug("  HyDE doc generated in %.3fs  (%d chars)", time.perf_counter() - t_hyde, len(hyde_text))
        try:
            t_hyde_search = time.perf_counter()
            embedder = get_embedder()
            vec = embedder.embed_query(hyde_text)
            hyde_docs = vs.similarity_search_by_vector(vec, k=top_k)
            log.debug(
                "  HyDE vector search: %d doc(s) retrieved  elapsed=%.3fs"
                "  (confidence assigned: %.2f — unscored fallback)",
                len(hyde_docs),
                time.perf_counter() - t_hyde_search,
                hyde_fallback_confidence,
            )
            for doc in hyde_docs:
                doc.metadata = doc.metadata or {}
                doc.metadata["hyde"] = True
                results.append(_doc_to_dict(doc, hyde_fallback_confidence))
        except Exception as exc:
            log.warning("  HyDE vector search failed (%s) — skipped", exc)

    log.info(
        "  Total raw results (pre-merge): %d  retrieve_elapsed=%.3fs",
        len(results),
        time.perf_counter() - t_retrieve_start,
    )
    return {"retrieved_docs_with_scores": results}


def merge_retrieval_results(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 8 — deduplicate chunks and select top-k by normalized confidence."""
    log.info("◉ NODE  merge_retrieval_results")
    raw = state.get("retrieved_docs_with_scores") or []
    top_k = state.get("top_k_text") or get_settings().top_k_text

    def _key(d: Dict[str, Any]) -> Tuple[Any, Any]:
        m = d.get("metadata", {})
        return (m.get("doc_id") or m.get("source"), m.get("chunk_id"))

    # Scores are normalized confidences in [0, 1] — sort DESCENDING so the
    # most confident (highest confidence) docs come first.
    sorted_raw = sorted(raw, key=lambda x: x.get("score", 0.0), reverse=True)
    seen: set = set()
    final: List[Dict[str, Any]] = []
    for d in sorted_raw:
        key = _key(d)
        if key in seen:
            continue
        seen.add(key)
        final.append(d)
        if len(final) >= top_k:
            break

    confidence_values = [round(d.get("score", 0.0), 4) for d in final]
    log.info(
        "  After dedup/merge: %d docs  (from %d raw)  confidences=%s",
        len(final),
        len(raw),
        confidence_values,
    )
    return {"final_retrieved_docs": final}


# ── LAYER 3 ──────────────────────────────────────────────────────────────────

def retrieval_failure_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 9 — surfaced when no docs were retrieved or confidence is too low."""
    query = state.get("current_query", "")[:60]
    log.warning("◉ NODE  retrieval_failure_node  query=%r", query)
    message = "I couldn't find relevant information in the documents."
    return {
        # Treat this like a completed per-query answer so the multi-query
        # loop can advance instead of spinning forever on the same question.
        "answer_text": message,
        "final_answer": message,
        "error_message": "retrieval_failure",
    }


def compress_context_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 10 — LLM-based context compression."""
    log.info("◉ NODE  compress_context_node")
    docs_raw = state.get("final_retrieved_docs") or []
    docs = [_dict_to_doc(d) for d in docs_raw]
    query = state.get("clarified_query") or state.get("current_query", "")
    t0 = time.perf_counter()
    try:
        compressed = compress_context(docs, query)
    except Exception as exc:
        return {
            "compressed_context": "",
            "error_message": f"compression_error:{exc}",
        }
    log.info("  compress_context elapsed=%.3fs", time.perf_counter() - t0)
    return {"compressed_context": compressed}


def compression_failure_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 11 — compression failure; falls back to extractive summary."""
    log.warning("◉ NODE  compression_failure_node  — using extractive fallback")
    docs_raw = state.get("final_retrieved_docs") or []
    fallback_parts = []
    for i, d in enumerate(docs_raw[:3], start=1):
        text = d.get("page_content", "")[:400]
        fallback_parts.append(f"[{i}] {text}")
    extractive = "\n\n".join(fallback_parts) if fallback_parts else "(no context)"
    compressed = f"<<BEGIN COMPRESSED CONTEXT>>\n{extractive}\n<<END COMPRESSED CONTEXT>>"
    return {
        "compressed_context": compressed,
        "error_message": "compression_fallback_used",
    }


def generate_answer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 12 — LLM answer generation."""
    retries = state.get("generation_retries", 0)
    log.info("◉ NODE  generate_answer_node  retry=%d", retries)
    compressed = state.get("compressed_context", "")
    query = state.get("clarified_query") or state.get("current_query", "")
    retries = state.get("generation_retries", 0)
    cfg = get_settings()

    t0 = time.perf_counter()
    try:
        answer = generate_answer(compressed, query)
        log.info("  generate_answer elapsed=%.3fs", time.perf_counter() - t0)
        return {"answer_text": answer, "generation_retries": retries}
    except Exception as exc:
        log.info("  generate_answer failed after %.3fs", time.perf_counter() - t0)
        if retries < cfg.graph_max_retries:
            return {
                "answer_text": "",
                "generation_retries": retries + 1,
                "error_message": f"generation_retry:{exc}",
            }
        return {
            "answer_text": "",
            "generation_retries": retries,
            "error_message": f"generation_failed:{exc}",
        }


def llm_timeout_failure_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 13 — all generation retries exhausted."""
    log.error("◉ NODE  llm_timeout_failure_node  retries=%d", state.get("generation_retries", 0))
    return {
        "final_answer": "The system is temporarily unable to generate a response. Please try again.",
        "error_message": "llm_timeout_failure",
    }


# ── LAYER 4 ──────────────────────────────────────────────────────────────────

def collect_sub_answers(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 14 — accumulate per-query answers in multi-query flow."""
    log.info("◉ NODE  collect_sub_answers")
    sub_answers = list(state.get("sub_answers") or [])
    question = state.get("current_query", "")
    answer = state.get("answer_text", "")
    if question and answer:
        sub_answers.append({"question": question, "answer": answer})
    return {"sub_answers": sub_answers}


def merge_final_answers(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 15 — format all sub-answers into one structured output."""
    sub = state.get("sub_answers") or []
    log.info("◉ NODE  merge_final_answers  sub_answers=%d", len(sub))
    sub_answers = state.get("sub_answers") or []
    if not sub_answers:
        single = state.get("answer_text", "")
        return {"final_answer": single}

    lines: List[str] = []
    for i, item in enumerate(sub_answers, start=1):
        lines.append(f"Question {i}: {item['question']}")
        lines.append("Answer:")
        lines.append(item["answer"])
        lines.append("")
    return {"final_answer": "\n".join(lines).rstrip()}

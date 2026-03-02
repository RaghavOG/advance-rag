"""
Retrieval layer for the minimal text-first pipeline.

Responsibilities:
- Accept a user query.
- Call the configured vector store.
- Apply top-k and optional metadata filters.
- Return LangChain Document objects.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from config.settings import get_settings
from embeddings.factory import get_embedder
from query.rewrite import generate_hyde_document, rewrite_queries
from utils.logger import get_logger
from vectorstores.factory import create_vectorstore, normalize_score

log = get_logger(__name__)


def _adaptive_top_k(query: str, num_rewrites: int, base_k: int) -> int:
    """
    Heuristic adaptive top-k.
    """
    text = query.lower()
    words = text.split()
    num_words = len(words)

    # Broad / explanatory queries should look at more context.
    long_or_explanatory = any(
        kw in text
        for kw in [
            "explain",
            "overview",
            "why",
            "how",
            "compare",
            "list",
            "failure",
            "modes",
            "best practices",
        ]
    ) or num_words >= 15

    if num_rewrites <= 1:
        # Short, factual-style question → keep k small for precision.
        if not long_or_explanatory and num_words <= 10 and base_k >= 3:
            return 3
        # Broader / longer question → use configured base_k.
        return base_k

    # Multiple rewrites: keep per-rewrite k modest so total retrieved ≈ base_k.
    per_rewrite = max(2, min(4, max(1, base_k // max(1, num_rewrites))))
    return per_rewrite


def retrieve_text(
    query: str,
    *,
    k: Optional[int] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """
    Retrieve top-k text chunks for a query using:
    - LLM-based query rewriting.
    - Optional HyDE document retrieval.
    - Adaptive per-rewrite top-k.
    """
    cfg = get_settings()
    log.info("━━━ RETRIEVAL  query=%r", query[:80])

    vs = create_vectorstore(collection_name=cfg.chroma_collection_text)

    # Rewrites (always include original query as first element).
    rewrites = rewrite_queries(query)
    per_rewrite_k = _adaptive_top_k(query, len(rewrites), base_k=cfg.top_k_text)
    log.info("  rewrites=%d  per_rewrite_k=%d  base_k=%d", len(rewrites), per_rewrite_k, cfg.top_k_text)

    # Collect (Document, score) pairs from similarity_search_with_score.
    results: List[Tuple[Document, float]] = []
    for rewrite_id, rq in enumerate(rewrites):
        log.debug("  [rewrite %d] %r", rewrite_id, rq[:80])
        try:
            scored = vs.similarity_search_with_score(rq, k=per_rewrite_k, filter=metadata_filter)
            log.debug(
                "    → %d result(s)  raw_scores=%s  backend=%s",
                len(scored),
                [round(float(s), 4) for _, s in scored],
                cfg.vector_store.value,
            )
        except Exception as exc:
            log.warning("  similarity_search_with_score failed (%s) — falling back to unscored", exc)
            docs_only = vs.similarity_search(rq, k=per_rewrite_k, filter=metadata_filter)
            scored = [(d, 0.0) for d in docs_only]

        for doc, raw_score in scored:
            confidence, _ = normalize_score(float(raw_score))
            doc.metadata = doc.metadata or {}
            if "rewrite_id" not in doc.metadata:
                doc.metadata["rewrite_id"] = rewrite_id
            results.append((doc, confidence))

    log.info("  Raw results before dedup: %d", len(results))

    # HyDE: use a hypothetical document as a vector query.
    hyde_doc = generate_hyde_document(query)
    hyde_docs: List[Document] = []
    if hyde_doc:
        log.info("  Running HyDE vector search")
        try:
            embedder = get_embedder()
            vec = embedder.embed_query(hyde_doc)
            hyde_docs = vs.similarity_search_by_vector(vec, k=per_rewrite_k, filter=metadata_filter)
            for doc in hyde_docs:
                doc.metadata = doc.metadata or {}
                doc.metadata["hyde"] = True
            log.info("  HyDE returned %d doc(s)", len(hyde_docs))
        except Exception as exc:
            log.warning("  HyDE search failed (%s) — skipped", exc)
            hyde_docs = []

    # Merge & deduplicate.
    def _key(d: Document) -> Tuple[Any, Any]:
        m = d.metadata or {}
        doc_id = m.get("doc_id") or m.get("source")
        chunk_id = m.get("chunk_id")
        return (doc_id, chunk_id)

    seen: set = set()
    final_docs: List[Document] = []

    # Scores are now normalized confidences in [0, 1] — sort DESCENDING so the
    # most confident docs come first.
    for doc, score in sorted(results, key=lambda x: x[1], reverse=True):
        key = _key(doc)
        if key in seen:
            continue
        seen.add(key)
        final_docs.append(doc)
        if k is not None and len(final_docs) >= k:
            break
        if k is None and len(final_docs) >= cfg.top_k_text:
            break

    for doc in hyde_docs:
        key = _key(doc)
        if key in seen:
            continue
        seen.add(key)
        final_docs.append(doc)
        if k is not None and len(final_docs) >= k:
            break
        if k is None and len(final_docs) >= cfg.top_k_text:
            break

    log.info("  Final retrieved docs: %d  sources=%s",
             len(final_docs),
             list({d.metadata.get("source", "?") for d in final_docs}))

    for i, doc in enumerate(final_docs):
        m = doc.metadata
        log.debug("  [doc %d] src=%s  page=%s  chunk=%s  hyde=%s  preview=%r",
                  i, m.get("source", "?"), m.get("page"), m.get("chunk_id"),
                  m.get("hyde", False), doc.page_content[:60])

    return final_docs


__all__ = ["retrieve_text"]

# Responsibilities:
# Accept query
# Call vectorstore
# Return Document[]
# Apply:
    # top-k
    # metadata filters

"""
LLM-based document reranker.

Architecture
------------
Sends all candidate documents + the query to a single fast LLM call.
Asks the model to return a JSON array of relevance scores (integers 0–10),
one per document, in the same order as supplied.

Scores are normalised to [0, 1] and written back into each doc dict as
"rerank_score".  The "score" field (used by downstream gating) is also
updated so the highest-confidence documents surface first.

Fallback
--------
On any failure (API error, JSON parse error, wrong array length) the
original ordering is preserved and the function returns docs[:top_n] unchanged.

This is a pure function — it does NOT modify the graph state itself.
The calling node (`reranker_node` in nodes.py) is responsible for
updating state.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from openai import OpenAI

from config.settings import get_settings
from utils.logger import get_logger

log = get_logger(__name__)

_MAX_SNIPPET_CHARS = 350  # chars per doc sent to reranker (keep prompt short)


def rerank_documents(
    docs: List[Dict[str, Any]],
    query: str,
    top_n: int,
) -> List[Dict[str, Any]]:
    """
    Re-score `docs` for relevance to `query` using an LLM call.

    Returns a new list of at most `top_n` dicts, each enriched with:
      - "rerank_score" : float in [0, 1]
      - "score"        : same as rerank_score (replaces prior confidence)

    On failure returns docs[:top_n] unchanged (passthrough).
    """
    if not docs:
        return docs

    cfg = get_settings()
    snippets: List[str] = []
    for i, d in enumerate(docs):
        text = d.get("page_content", "")[:_MAX_SNIPPET_CHARS].replace("\n", " ")
        snippets.append(f"[{i}] {text}")

    prompt = (
        f"Question: {query}\n\n"
        f"Passages:\n" + "\n\n".join(snippets) + "\n\n"
        f"Rate each passage's relevance to the question on a scale of 0 to 10, "
        f"where 10 = perfectly answers the question and 0 = completely irrelevant.\n"
        f"Return ONLY a JSON array of {len(docs)} integers in the same order as the passages. "
        f"Example for 3 passages: [8, 2, 6]"
    )

    try:
        client = OpenAI(api_key=cfg.openai_api_key, timeout=cfg.openai_timeout)
        resp = client.chat.completions.create(
            model=cfg.reranker_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=60,
        )
        content = (resp.choices[0].message.content or "").strip()
        # Strip accidental code fences
        if "```" in content:
            content = content.split("```")[1].replace("json", "").strip()

        scores = json.loads(content)

        if not isinstance(scores, list) or len(scores) != len(docs):
            log.warning(
                "Reranker returned unexpected result (expected list of %d, got %r) — skipping",
                len(docs),
                scores,
            )
            return docs[:top_n]

        # Pair docs with scores, sort descending, take top_n
        paired = sorted(
            zip(docs, [float(s) for s in scores]),
            key=lambda x: x[1],
            reverse=True,
        )

        result: List[Dict[str, Any]] = []
        for doc, raw_score in paired[:top_n]:
            d = dict(doc)
            normalised = round(max(0.0, min(10.0, raw_score)) / 10.0, 4)
            d["rerank_score"] = normalised
            d["score"] = normalised  # downstream gating uses "score"
            result.append(d)

        log.info(
            "Reranker: %d → %d docs  top_scores=%s",
            len(docs),
            len(result),
            [d["rerank_score"] for d in result],
        )
        return result

    except Exception as exc:
        log.warning("Reranker failed (%s) — returning original order (top %d)", exc, top_n)
        return docs[:top_n]


__all__ = ["rerank_documents"]

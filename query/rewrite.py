"""
LLM-based query rewriting and HyDE document generation.

This module:
- Generates multiple retrieval-only rewrites for a single user question.
- Optionally generates a HyDE (hypothetical) document to improve recall.
"""
from __future__ import annotations

from typing import List

from openai import OpenAI

from config.settings import get_settings
from utils.logger import get_logger

log = get_logger(__name__)


def _get_client() -> OpenAI:
    cfg = get_settings()
    return OpenAI(
        api_key=cfg.openai_api_key,
        timeout=cfg.hyde_timeout,
    )


def rewrite_queries(query: str) -> List[str]:
    """
    Generate 2–4 alternative phrasings of the same question to improve retrieval.
    """
    cfg = get_settings()
    if not cfg.enable_hyde:
        log.debug("HyDE/rewriting disabled — returning original query only")
        return [query]

    log.info("Query rewriting: model=%s  query=%r", cfg.hyde_model, query[:80])
    client = _get_client()
    system_prompt = (
        "You rewrite a user's question into up to 4 alternative search queries.\n"
        "- Do NOT change the intent.\n"
        "- Do NOT introduce new entities.\n"
        "- Focus on different phrasings and keyword combinations.\n"
        "Return ONLY a JSON array of strings, no extra text."
    )
    user_prompt = f"Original question:\n{query}"

    resp = client.chat.completions.create(
        model=cfg.hyde_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=min(cfg.hyde_max_tokens, 256),
        temperature=0.3,
    )
    content = resp.choices[0].message.content or ""

    import json  # local import to avoid polluting module scope

    try:
        # Handle accidental code fences.
        if "```" in content:
            content = content.split("```")[1].replace("json", "").strip()
        data = json.loads(content)
        rewrites = [s.strip() for s in data if isinstance(s, str) and s.strip()]
    except Exception as exc:
        log.warning("Failed to parse rewrite response (fallback to original): %s", exc)
        rewrites = []

    # Always include the original query as a fallback and to preserve intent.
    if query not in rewrites:
        rewrites.insert(0, query)

    rewrites = rewrites[:4]
    log.info("  Rewrites generated: %d  %s", len(rewrites), rewrites)
    return rewrites


def generate_hyde_document(query: str) -> str:
    """
    Generate a hypothetical reference-style document for HyDE retrieval.
    """
    cfg = get_settings()
    if not cfg.enable_hyde:
        return ""

    log.info("HyDE document generation: model=%s", cfg.hyde_model)
    client = _get_client()
    system_prompt = (
        "You write a neutral, reference-style paragraph that might appear in a "
        "technical document answering the user's question. Do NOT mention that "
        "you are hypothetical; just write the content itself."
    )
    user_prompt = f"Question:\n{query}"

    resp = client.chat.completions.create(
        model=cfg.hyde_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=cfg.hyde_max_tokens,
        temperature=0.3,
    )
    result = (resp.choices[0].message.content or "").strip()
    log.debug("  HyDE document (%d chars): %s…", len(result), result[:120])
    return result


__all__ = ["rewrite_queries", "generate_hyde_document"]

"""
Context compressor.

Responsibilities:
- Take retrieved Document[].
- Call an LLM to produce a shorter, focused context.
- Do NOT answer the question or add reasoning.
"""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from openai import OpenAI

from config.settings import get_settings
from utils.logger import get_logger

log = get_logger(__name__)


def _default_compression_model() -> str:
    cfg = get_settings()
    # Prefer dedicated compression model; fall back to answer/default model.
    return cfg.compression_model or cfg.answer_model or cfg.openai_default_model


def compress_context(docs: List[Document], query: str) -> str:
    """
    Compress retrieved documents into a single context string.

    The model is instructed to:
    - Only summarize and filter for relevance.
    - Not answer the question.
    """
    if not docs:
        log.warning("compress_context called with 0 docs — returning empty")
        return ""

    log.info("━━━ COMPRESSION  docs=%d  model=%s", len(docs), _default_compression_model())

    snippets = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        loc = []
        if "doc_id" in meta:
            loc.append(f"doc_id={meta['doc_id']}")
        if "page" in meta:
            loc.append(f"page={meta['page']}")
        prefix = f"[{i}] ({', '.join(loc)})" if loc else f"[{i}]"
        snippets.append(f"{prefix} {d.page_content}")

    joined = "\n\n".join(snippets)

    system_prompt = (
        "You are a context compressor for a Retrieval-Augmented Generation system.\n"
        "Your ONLY job is to extract and rewrite the parts of the retrieved text\n"
        "that are relevant to the user's question.\n"
        "- Do NOT answer the question.\n"
        "- Do NOT explain your reasoning.\n"
        "- Keep citations or doc/page references if helpful.\n"
    )

    user_prompt = f"Question:\n{query}\n\nRetrieved context:\n{joined}"

    cfg = get_settings()
    client = OpenAI(
        api_key=cfg.openai_api_key,
        timeout=cfg.openai_timeout,
    )
    resp = client.chat.completions.create(
        model=_default_compression_model(),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=cfg.compression_max_tokens,
    )
    body = resp.choices[0].message.content or ""
    log.info("  Compressed: %d chars → %d chars", sum(len(s) for s in snippets), len(body))
    log.debug("  Compressed context preview: %s…", body[:200])
    # Tag the compressed context explicitly to reduce prompt-injection risk.
    return f"<<BEGIN COMPRESSED CONTEXT>>\n{body}\n<<END COMPRESSED CONTEXT>>"


__all__ = ["compress_context"]


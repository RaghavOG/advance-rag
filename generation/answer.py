"""
Answer generation module (no LangGraph, no routing).

Responsibilities:
- Accept a compressed context string and a user query.
- Produce a grounded answer that:
  - Clearly cites the context (doc_id/page if present).
  - Is honest about missing information.
  - Avoids fabricating facts beyond the context.
"""
from __future__ import annotations

from openai import OpenAI

from config.settings import get_settings
from utils.logger import get_logger

log = get_logger(__name__)


def _default_answer_model() -> str:
    cfg = get_settings()
    # Prefer dedicated answer model; fall back to OPENAI_DEFAULT_MODEL.
    return cfg.answer_model or cfg.openai_default_model


def generate_answer(compressed_context: str, query: str) -> str:
    """
    Use an LLM to answer the query based ONLY on the compressed context.
    """
    if not compressed_context.strip():
        log.warning("generate_answer called with empty context — returning early")
        return "I do not have any relevant context to answer this question."

    log.info("━━━ GENERATION  model=%s  context_len=%d", _default_answer_model(), len(compressed_context))

    # NOTE: We currently allow explicit reasoning text for debuggability.
    # This can be replaced with a more concise synthesis format later.
    system_prompt = (
        "You are a retrieval-augmented assistant.\n"
        "You must answer using ONLY the provided compressed context.\n"
        "- Clearly separate your reply into two sections:\n"
        "  1) Retrieved facts / evidence (quote or paraphrase from context, with doc/page IDs if present).\n"
        "  2) Reasoning / synthesis (how you combined those facts).\n"
        "- If the context is insufficient to answer the question, say so explicitly and do NOT guess.\n"
        "- Do not introduce external facts that are not supported by the context.\n"
    )

    user_prompt = f"Compressed context:\n{compressed_context}\n\nQuestion: {query}"

    cfg = get_settings()
    client = OpenAI(
        api_key=cfg.openai_api_key,
        timeout=cfg.answer_timeout,
    )
    resp = client.chat.completions.create(
        model=_default_answer_model(),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=cfg.answer_temperature,
        max_tokens=cfg.max_output_tokens,
    )
    answer = resp.choices[0].message.content or ""
    log.info("  Answer generated: %d chars", len(answer))
    log.debug("  Answer preview: %s…", answer[:200])

    # Minimal structural guard to encourage the model to follow the format.
    if "Retrieved facts" not in answer:
        log.warning("  Answer missing 'Retrieved facts' section — prepending guard message")
        answer = (
            "The retrieved context may have been insufficient to produce a fully "
            "grounded answer in the requested format.\n\n" + answer
        )

    return answer


__all__ = ["generate_answer"]

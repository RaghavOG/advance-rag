"""
Adapts the LangGraph RAGState output to the structured API response models.

This is the only file that knows about both the graph internals and the API schema.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from backend.models import Citation, QueryResponse, SubAnswer


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_citations(answer_text: str, docs_raw: List[Dict[str, Any]]) -> List[Citation]:
    """Extract citation metadata from retrieved docs."""
    citations: List[Citation] = []
    for d in docs_raw:
        meta = d.get("metadata", {})
        snippet = d.get("page_content", "")[:200].strip()
        citations.append(Citation(
            doc_id=meta.get("doc_id"),
            source=meta.get("source"),
            page=meta.get("page"),
            chunk_id=str(meta.get("chunk_id")) if meta.get("chunk_id") is not None else None,
            snippet=snippet or None,
        ))
    return citations


def _split_answer_reasoning(answer: str) -> Tuple[str, Optional[str]]:
    """
    Attempt to split the LLM answer into facts section and reasoning section.
    The generation prompt asks for two labelled sections.
    """
    patterns = [
        r"(?i)(retrieved facts.*?)\n\n(reasoning.*)",
        r"(?i)(retrieved facts.*?)\n(reasoning.*)",
    ]
    for pat in patterns:
        m = re.search(pat, answer, re.DOTALL)
        if m:
            return m.group(1).strip(), m.group(2).strip()
    return answer.strip(), None


def _confidence_from_docs(docs_raw: List[Dict[str, Any]]) -> Optional[float]:
    """Derive a simple confidence signal from min retrieval distance."""
    if not docs_raw:
        return None
    scores = [d.get("score", None) for d in docs_raw if d.get("score") is not None]
    if not scores:
        return None
    # Distance-based: lower is better. Map to [0, 1] confidence (inverted).
    best = min(scores)
    # Clamp to a reasonable range; Chroma cosine distance is in [0, 2].
    confidence = max(0.0, min(1.0, 1.0 - best / 2.0))
    return round(confidence, 2)


# ── Main adapter ──────────────────────────────────────────────────────────────

def graph_state_to_response(
    state: Dict[str, Any],
    conversation_id: str,
) -> QueryResponse:
    """
    Convert a finished RAGState dict into a structured QueryResponse.
    """
    error = state.get("error_message") or ""
    docs_raw = state.get("final_retrieved_docs") or []

    # ── Clarification needed ──────────────────────────────────────────────────
    if error.startswith("CLARIFICATION_NEEDED:"):
        cq = error.removeprefix("CLARIFICATION_NEEDED:").strip()
        current_query = state.get("current_query", "")
        return QueryResponse(
            conversation_id=conversation_id,
            status="clarification_required",
            sub_answers=[SubAnswer(
                query=current_query,
                status="clarification_required",
                clarification_question=cq,
            )],
        )

    # ── Hard failure ──────────────────────────────────────────────────────────
    if error in ("retrieval_failure", "llm_timeout_failure"):
        final = state.get("final_answer") or "I couldn't process this request."
        current_query = state.get("current_query", state.get("normalized_prompt", ""))
        return QueryResponse(
            conversation_id=conversation_id,
            status="failure",
            sub_answers=[SubAnswer(
                query=current_query,
                status="failed",
                answer=final,
                confidence=0.0,
            )],
            error_message=error,
        )

    # ── Normal / multi-query response ─────────────────────────────────────────
    raw_sub = state.get("sub_answers") or []
    sub_answers: List[SubAnswer] = []
    any_failed = False
    any_clarification = False

    for item in raw_sub:
        q = item.get("question", "")
        a = item.get("answer", "")

        if not a or a.startswith("I couldn't find"):
            any_failed = True
            sub_answers.append(SubAnswer(
                query=q,
                status="failed",
                answer=a or "No relevant information found.",
                confidence=0.0,
            ))
            continue

        facts, reasoning = _split_answer_reasoning(a)
        citations = _parse_citations(a, docs_raw)
        confidence = _confidence_from_docs(docs_raw)

        sub_answers.append(SubAnswer(
            query=q,
            status="answered",
            answer=facts,
            reasoning=reasoning,
            citations=citations,
            confidence=confidence,
        ))

    # Single-question path — sub_answers may be empty if graph merged into final_answer.
    if not sub_answers:
        final = state.get("final_answer") or state.get("answer_text") or ""
        if not final:
            return QueryResponse(
                conversation_id=conversation_id,
                status="failure",
                error_message="No answer produced.",
            )
        facts, reasoning = _split_answer_reasoning(final)
        sub_answers.append(SubAnswer(
            query=state.get("normalized_prompt", ""),
            status="answered",
            answer=facts,
            reasoning=reasoning,
            citations=_parse_citations(final, docs_raw),
            confidence=_confidence_from_docs(docs_raw),
        ))

    overall = "failure" if any_failed and not any(s.status == "answered" for s in sub_answers) \
        else "partial" if any_failed \
        else "answered"

    return QueryResponse(
        conversation_id=conversation_id,
        status=overall,
        sub_answers=sub_answers,
    )

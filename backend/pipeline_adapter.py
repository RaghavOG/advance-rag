"""
Adapts the LangGraph RAGState output to the structured API response models.

This is the only file that knows about both the graph internals and the API schema.

Key fixes vs. previous version
-------------------------------
- _confidence_from_docs: scores are NOW normalised confidences in [0,1], higher=better.
  The old logic (1 - best/2) was for raw Chroma distances and is no longer correct.
  Now we simply take max(scores).

- Timings are propagated from state["timings"] into QueryResponse.timings.

- Clarification detection checks the new state field "query_status" in addition to
  the legacy error_message prefix, making the check more robust.

- Citations now include confidence, raw_score, and backend from doc metadata.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from backend.models import Citation, QueryResponse, SubAnswer

# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_citations(docs_raw: List[Dict[str, Any]]) -> List[Citation]:
    """Build Citation objects from final_retrieved_docs entries."""
    citations: List[Citation] = []
    for d in docs_raw:
        meta = d.get("metadata", {})
        snippet = d.get("page_content", "")[:200].strip() or None
        page_raw = meta.get("page")
        try:
            page = int(page_raw) if page_raw is not None else None
        except (TypeError, ValueError):
            page = None
        citations.append(
            Citation(
                doc_id=meta.get("doc_id"),
                source=meta.get("source"),
                page=page,
                chunk_id=str(meta.get("chunk_id")) if meta.get("chunk_id") is not None else None,
                snippet=snippet,
                confidence=round(float(d.get("score", 0.0)), 4),
                raw_score=round(float(d.get("raw_score", d.get("score", 0.0))), 4),
                backend=meta.get("backend"),
            )
        )
    return citations


def _split_answer_reasoning(answer: str) -> Tuple[str, Optional[str]]:
    """
    Attempt to split the LLM answer into a facts section and a reasoning section.
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


def _best_confidence(docs_raw: List[Dict[str, Any]]) -> Optional[float]:
    """
    Return the best (highest) normalised confidence across the retrieved docs.

    Scores stored in final_retrieved_docs are already normalised to [0,1]
    (higher = more relevant) by score_normalizer_node.  We just take the max.
    """
    if not docs_raw:
        return None
    scores = [
        float(d.get("score", 0.0))
        for d in docs_raw
        if d.get("score") is not None
    ]
    return round(max(scores), 2) if scores else None


# ── Main adapter ──────────────────────────────────────────────────────────────

def graph_state_to_response(
    state: Dict[str, Any],
    conversation_id: str,
) -> QueryResponse:
    """
    Convert a finished RAGState dict into a structured QueryResponse.

    Handles all terminal states:
      1. Safety blocked
      2. Clarification needed
      3. Hard failure (retrieval / LLM timeout)
      4. Normal / multi-query answered (including partial)
    """
    error = state.get("error_message") or ""
    docs_raw = state.get("final_retrieved_docs") or []
    timings: Optional[Dict[str, float]] = state.get("timings") or None

    # ── 1. Safety block ───────────────────────────────────────────────────────
    if error.startswith("safety_blocked:"):
        reason = error.removeprefix("safety_blocked:").strip()
        return QueryResponse(
            conversation_id=conversation_id,
            status="failure",
            sub_answers=[
                SubAnswer(
                    query=state.get("normalized_prompt", ""),
                    status="failed",
                    answer=f"Your request was blocked by the content policy. {reason}",
                    confidence=0.0,
                )
            ],
            error_message=error,
            timings=timings,
        )

    # ── 2. Clarification needed ───────────────────────────────────────────────
    is_clarification = (
        error.startswith("CLARIFICATION_NEEDED:")
        or state.get("query_status") == "clarification_needed"
    )
    if is_clarification:
        cq = (
            error.removeprefix("CLARIFICATION_NEEDED:").strip()
            if error.startswith("CLARIFICATION_NEEDED:")
            else state.get("clarification_question") or "Could you please clarify?"
        )
        current_query = state.get("current_query", "")
        return QueryResponse(
            conversation_id=conversation_id,
            status="clarification_required",
            sub_answers=[
                SubAnswer(
                    query=current_query,
                    status="clarification_required",
                    clarification_question=cq,
                )
            ],
            timings=timings,
        )

    # ── 3. Hard failure ───────────────────────────────────────────────────────
    if error in ("retrieval_failure", "llm_timeout_failure"):
        final = state.get("final_answer") or "I couldn't process this request."
        current_query = state.get("current_query", state.get("normalized_prompt", ""))
        return QueryResponse(
            conversation_id=conversation_id,
            status="failure",
            sub_answers=[
                SubAnswer(
                    query=current_query,
                    status="failed",
                    answer=final,
                    confidence=0.0,
                )
            ],
            error_message=error,
            timings=timings,
        )

    # ── 4. Normal / multi-query ───────────────────────────────────────────────
    raw_sub = state.get("sub_answers") or []
    sub_answers: List[SubAnswer] = []
    any_failed = False

    for item in raw_sub:
        q = item.get("question", "")
        a = item.get("answer", "")

        if not a or a.startswith("I couldn't find"):
            any_failed = True
            sub_answers.append(
                SubAnswer(
                    query=q,
                    status="failed",
                    answer=a or "No relevant information found.",
                    confidence=0.0,
                    citations=_parse_citations(docs_raw),
                )
            )
            continue

        facts, reasoning = _split_answer_reasoning(a)
        sub_answers.append(
            SubAnswer(
                query=q,
                status="answered",
                answer=facts,
                reasoning=reasoning,
                citations=_parse_citations(docs_raw),
                confidence=_best_confidence(docs_raw),
            )
        )

    # Single-question path — sub_answers may be empty if graph merged into final_answer.
    if not sub_answers:
        final = state.get("final_answer") or state.get("answer_text") or ""
        if not final:
            return QueryResponse(
                conversation_id=conversation_id,
                status="failure",
                error_message="No answer produced.",
                timings=timings,
            )
        facts, reasoning = _split_answer_reasoning(final)
        sub_answers.append(
            SubAnswer(
                query=state.get("normalized_prompt", ""),
                status="answered",
                answer=facts,
                reasoning=reasoning,
                citations=_parse_citations(docs_raw),
                confidence=_best_confidence(docs_raw),
            )
        )

    overall: str
    if any_failed and not any(s.status == "answered" for s in sub_answers):
        overall = "failure"
    elif any_failed:
        overall = "partial"
    else:
        overall = "answered"

    return QueryResponse(
        conversation_id=conversation_id,
        status=overall,  # type: ignore[arg-type]
        sub_answers=sub_answers,
        timings=timings,
    )

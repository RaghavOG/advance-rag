"""
Simple, rule-based query decomposition and wrapper around the linear pipeline.

Rules (v1, no LLM needed):
- Split on '?'.
- Split on obvious numbered list lines (e.g. '1) ..', '2. ..').
- Split on 'and also', 'also', 'additionally' (lightweight).
- Trim whitespace and drop empty strings.
- If result length > 1 -> treat as multi-query.

Hard limit (from config):
- MAX_SUB_QUERIES=3
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

from config.settings import get_settings


def split_queries(prompt: str) -> List[str]:
    """
    Heuristic, non-LLM query splitter.
    """
    if not prompt.strip():
        return []

    # First, capture numbered-list style questions line by line.
    numbered: List[str] = []
    for line in prompt.splitlines():
        m = re.match(r"^\s*\d+[\.\)\-]\s*(.+)$", line)
        if m:
            q = m.group(1).strip()
            if q:
                numbered.append(q)

    # Normalize whitespace for the rest of the splitting.
    normalized = re.sub(r"\s+", " ", prompt).strip()

    # Split on question marks.
    parts = [p.strip() for p in re.split(r"\?+", normalized) if p.strip()]

    # Further split on simple conjunctions.
    sub_queries: List[str] = []
    for part in parts:
        # Split on 'and also', 'also', 'additionally' (case-insensitive).
        chunks = re.split(r"\b(?:and also|also|additionally)\b", part, flags=re.IGNORECASE)
        for chunk in chunks:
            chunk = chunk.strip(" ,;")
            if chunk:
                sub_queries.append(chunk)

    # Combine with numbered questions (if any) and deduplicate while preserving order.
    all_candidates = numbered + sub_queries
    seen: set[str] = set()
    result: List[str] = []
    for q in all_candidates:
        if q and q not in seen:
            seen.add(q)
            result.append(q)

    return result


def answer_single(pdf_path: str | Path, question: str) -> str:
    """
    Run the existing linear pipeline for a single question.
    Lazy import to avoid circular import: graph.graph → graph.nodes → query.decompose → pipeline.run → graph.graph.
    """
    from pipeline.run import answer_question  # noqa: PLC0415
    return answer_question(pdf_path, question)


def format_multi_answer(sub_queries: List[str], answers: List[str]) -> str:
    """
    Merge multiple Q/A pairs into a clear, structured response.
    """
    lines: List[str] = []
    for i, (q, a) in enumerate(zip(sub_queries, answers), start=1):
        lines.append(f"Question {i}: {q}")
        lines.append("Answer:")
        lines.append(a)
        lines.append("")  # blank line between entries
    return "\n".join(lines).rstrip()


def answer_user_prompt(pdf_path: str | Path, prompt: str) -> str:
    """
    Thin wrapper around the linear pipeline that supports simple multi-question prompts.
    """
    cfg = get_settings()
    sub_queries = split_queries(prompt)

    if not sub_queries:
        return "I couldn't find a concrete question in your prompt."

    if len(sub_queries) == 1:
        return answer_single(pdf_path, sub_queries[0])

    if len(sub_queries) > cfg.max_sub_queries:
        return f"Please ask fewer questions at a time (max {cfg.max_sub_queries})."

    answers: List[str] = []
    for q in sub_queries:
        answers.append(answer_single(pdf_path, q))

    return format_multi_answer(sub_queries, answers)


__all__ = ["split_queries", "answer_user_prompt"]

"""
Pipeline runner.

answer_question() now delegates to the LangGraph.
The linear fallback path is preserved inside the graph itself,
so the external API is unchanged.

Usage
-----
    from pipeline.run import answer_question
    answer = answer_question("sample_docs/paper.pdf", "What is the main contribution?")

Multi-question prompts (e.g. "What is X? And also how does Y work?")
are handled transparently by the graph's detect_multi_query node.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ingestion.ingest import ingest_pdf
from graph.graph import rag_graph
from utils.logger import get_logger, log_stage

log = get_logger(__name__)


def answer_question(pdf_path: str | Path, question: str) -> str:
    """
    Ingest a PDF (idempotency risk noted in ingestion/ingest.py),
    then run the full LangGraph pipeline for the given question.

    Returns the final answer string or an error message if the
    pipeline could not produce a grounded response.
    """
    pdf = Path(pdf_path)
    log.info("═══════════════════════════════════════")
    log.info("PIPELINE START  file=%s  question=%r", pdf.name, question[:80])
    log.info("═══════════════════════════════════════")

    with log_stage(log, "ingest_pdf", file=pdf.name):
        ingest_pdf(pdf)

    initial_state: Dict[str, Any] = {"raw_prompt": question}
    with log_stage(log, "rag_graph.invoke"):
        result: Dict[str, Any] = rag_graph.invoke(initial_state)

    log.info("═══════════════════════════════════════")
    log.info("PIPELINE DONE  status=%s", result.get("error_message") or "ok")

    # Surface structured error messages to the caller when present.
    if result.get("error_message") == "retrieval_failure":
        return result.get("final_answer", "No relevant information found.")
    if result.get("error_message") == "llm_timeout_failure":
        return result.get("final_answer", "The system is temporarily unavailable.")

    # Normal path: return the merged (or single) final answer.
    return result.get("final_answer") or result.get("answer_text") or "No answer generated."


__all__ = ["answer_question"]


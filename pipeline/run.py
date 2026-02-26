"""
Linear RAG pipeline runner.

Provides a single entry point:
    answer_question(pdf_path, question)
"""
from __future__ import annotations

from pathlib import Path

from ingestion.ingest import ingest_pdf
from retrieval.retriever import retrieve_text
from compression.compressor import compress_context
from generation.answer import generate_answer


def answer_question(pdf_path: str | Path, question: str) -> str:
    """
    Ingest one PDF, retrieve, compress, and answer.
    """
    pdf = Path(pdf_path)
    ingest_pdf(pdf)
    docs = retrieve_text(question)
    compressed = compress_context(docs, question)
    return generate_answer(compressed, question)


__all__ = ["answer_question"]


"""
Minimal ingestion pipeline (text-only, PDFs).

Goal:
    "I ingest 1 PDF → ask a question → get a grounded answer."

This module only covers the ingestion half: load → clean → chunk → write to
the configured vector store using a single, centralized embedder.
"""
from __future__ import annotations

from pathlib import Path
from typing import List
from uuid import uuid4

from config.settings import get_settings
from ingestion.loaders import load_pdf_pages
from ingestion.chunking import chunk_pages
from vectorstores.factory import create_vectorstore


def ingest_pdf(path: Path) -> int:
    """
    Ingest a single PDF into the text collection of the configured vector store.

    Returns the number of chunks written.
    """
    cfg = get_settings()
    # NOTE: doc_id is currently a fresh UUID per ingest, which means re-running
    # ingestion on the same file will create duplicate chunks. For a production
    # system you may want to derive doc_id deterministically (e.g. hash of path
    # + modified time) and/or check the vector store for existing entries and
    # delete/re-ingest. That idempotency layer is intentionally left minimal here.
    doc_id = str(uuid4())

    pages = load_pdf_pages(path, doc_id=doc_id)
    chunks = chunk_pages(pages)

    if not chunks:
        return 0

    vs = create_vectorstore(collection_name=cfg.chroma_collection_text)
    texts: List[str] = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    # All supported backends (Chroma, FAISS, Pinecone) expose add_texts.
    # NOTE: minimal idempotency guard would check whether any existing chunks
    # with this doc_id are already present in the vector store and either skip
    # or delete+re-ingest. That cross-backend metadata query is non-trivial,
    # so for now we only document the risk here to avoid silent surprises.
    vs.add_texts(texts=texts, metadatas=metadatas)

    return len(chunks)


__all__ = ["ingest_pdf"]


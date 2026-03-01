"""
Ingestion pipeline — supports PDF, TXT, and Markdown.

Stages:  load → clean → chunk → embed → write to vector store
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List

from config.settings import get_settings
from ingestion.chunking import chunk_pages
from ingestion.loaders import load_document
from utils.logger import get_logger, log_stage
from vectorstores.factory import create_vectorstore

log = get_logger(__name__)

_SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}


def _deterministic_doc_id(path: Path) -> str:
    """
    Derive a stable doc_id from the file path + last-modified time.
    Re-running ingest on an unchanged file produces the same id — useful for
    future idempotency checks. Changing the file changes the id.
    NOTE: Chroma does not yet expose a delete-by-metadata API here, so
    duplicate prevention is still a TODO (see TODO.md §2.6).
    """
    mtime = path.stat().st_mtime if path.exists() else 0
    raw = f"{path.resolve()}::{mtime}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def ingest_document(path: Path) -> int:
    """
    Ingest a single document (PDF, TXT, MD) into the text vector store.
    Returns the number of chunks written.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix not in _SUPPORTED_SUFFIXES:
        log.warning("Skipping unsupported file: %s (suffix=%s)", path.name, suffix)
        return 0

    log.info("━━━ INGEST  %s  (%s) ━━━", path.name, suffix.upper().lstrip("."))

    doc_id = _deterministic_doc_id(path)
    log.debug("  doc_id=%s  (sha256 of path+mtime)", doc_id)

    with log_stage(log, "load_document", file=path.name):
        pages = load_document(path, doc_id=doc_id)

    if not pages:
        log.warning("  No content extracted from %s — skipping", path.name)
        return 0

    with log_stage(log, "chunk_pages", pages=len(pages)):
        chunks = chunk_pages(pages)

    if not chunks:
        log.warning("  Chunking produced 0 chunks from %s — skipping", path.name)
        return 0

    cfg = get_settings()
    with log_stage(log, "vectorstore_add", chunks=len(chunks), collection=cfg.chroma_collection_text):
        vs = create_vectorstore(collection_name=cfg.chroma_collection_text)
        texts: List[str] = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        vs.add_texts(texts=texts, metadatas=metadatas)

    log.info("  ✅ Ingested %d chunks from '%s'  doc_id=%s", len(chunks), path.name, doc_id)
    return len(chunks)


def ingest_pdf(path: Path) -> int:
    """Backward-compatible alias for ingest_document()."""
    return ingest_document(path)


__all__ = ["ingest_document", "ingest_pdf"]

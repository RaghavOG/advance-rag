"""
Ingestion pipeline — supports PDF, TXT, and Markdown.

Stages:  load → clean → chunk → embed → write to vector store

Idempotency
-----------
Each document is identified by a SHA-256 content hash (first 16 hex chars).
Before ingesting, the manifest file (`data/ingest_manifest.json`) is checked.
If a matching hash exists the document is skipped — re-running ingest on an
unchanged file is a safe no-op.

The manifest is a simple JSON dict: { doc_id: chunk_count }.
It survives server restarts and works across all three vector store backends.

To force re-ingestion of a specific file (e.g. after it was deleted from the
vector store) pass `force_reingest=True`.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List

from config.settings import get_settings
from ingestion.chunking import chunk_pages
from ingestion.loaders import load_document
from utils.logger import get_logger, log_stage
from vectorstores.factory import create_vectorstore

log = get_logger(__name__)

_SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}

# Manifest lives under data/ at the project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST_PATH = _PROJECT_ROOT / "data" / "ingest_manifest.json"


# ── Manifest helpers ──────────────────────────────────────────────────────────

def _load_manifest() -> dict[str, object]:
    if _MANIFEST_PATH.exists():
        try:
            return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Could not read ingest manifest (%s) — treating as empty", exc)
    return {}


def _save_manifest(manifest: dict[str, object]) -> None:
    _MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        _MANIFEST_PATH.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
    except Exception as exc:
        log.warning("Could not save ingest manifest: %s", exc)


# ── Doc-ID derivation ─────────────────────────────────────────────────────────

def _content_hash(path: Path) -> str:
    """
    Derive a stable doc_id from the file's raw content bytes.

    Using content rather than path+mtime means:
    - Moving a file doesn't change its id (correct — same content).
    - Modifying a file changes its id (correct — stale chunks should be replaced).
    - The id is reproducible across machines (useful for distributed ingest).

    Returns the first 16 hex chars of the SHA-256 digest.
    """
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()[:16]


# ── Public API ────────────────────────────────────────────────────────────────

def ingest_document(path: Path, *, force_reingest: bool = False) -> int:
    """
    Ingest a single document (PDF, TXT, MD) into the text vector store.

    Parameters
    ----------
    path           : filesystem path to the document
    force_reingest : skip the idempotency check and always re-ingest

    Returns
    -------
    Number of chunks written (0 if skipped or unsupported).
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix not in _SUPPORTED_SUFFIXES:
        log.warning("Skipping unsupported file: %s (suffix=%s)", path.name, suffix)
        return 0

    log.info("━━━ INGEST  %s  (%s) ━━━", path.name, suffix.upper().lstrip("."))

    # ── Idempotency check ────────────────────────────────────────────────────
    doc_id = _content_hash(path)
    log.debug("  doc_id=%s  (sha256 of content)", doc_id)

    manifest = _load_manifest()

    if not force_reingest and doc_id in manifest:
        existing_count = manifest[doc_id]
        log.info(
            "  ⏭  Already ingested '%s' (doc_id=%s, %d chunks) — skipping."
            " Pass force_reingest=True to override.",
            path.name,
            doc_id,
            existing_count,
        )
        return existing_count

    if force_reingest and doc_id in manifest:
        log.info("  force_reingest=True — re-ingesting '%s' (doc_id=%s)", path.name, doc_id)

    # ── Load → chunk → embed → write ─────────────────────────────────────────
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

    # ── Update manifest ───────────────────────────────────────────────────────
    # Backwards-compatible manifest entry; store richer metadata for new writes.
    manifest[doc_id] = {
        "chunks": len(chunks),
        "filename": path.name,
        "path": str(path),
    }
    _save_manifest(manifest)

    log.info(
        "  ✅ Ingested %d chunks from '%s'  doc_id=%s",
        len(chunks),
        path.name,
        doc_id,
    )
    return len(chunks)


def ingest_pdf(path: Path, *, force_reingest: bool = False) -> int:
    """Backward-compatible alias for ingest_document()."""
    return ingest_document(path, force_reingest=force_reingest)


__all__ = ["ingest_document", "ingest_pdf"]

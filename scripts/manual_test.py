"""
Manual validation script — ingest → retrieve → compress (no answering).

Usage:
    python scripts/manual_test.py sample_docs/kafka_architecture.txt "How does Kafka guarantee ordering?"
    python scripts/manual_test.py sample_docs/rag_systems.md "What is HyDE?"
    LOG_LEVEL=DEBUG python scripts/manual_test.py sample_docs/rag_systems.md "What is chunking?"

Supports: .pdf, .txt, .md
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("LOG_LEVEL", "INFO")

from ingestion.ingest import ingest_document
from retrieval.retriever import retrieve_text
from compression.compressor import compress_context
from utils.logger import get_logger

log = get_logger("manual_test")


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python scripts/manual_test.py <doc_path> \"Your question\"")
        print("       Supports: .pdf .txt .md .markdown")
        raise SystemExit(1)

    doc_path = Path(sys.argv[1])
    question = " ".join(sys.argv[2:])

    if not doc_path.exists():
        log.error("File not found: %s", doc_path)
        raise SystemExit(1)

    log.info("═══════ MANUAL TEST ═══════")
    log.info("File    : %s", doc_path)
    log.info("Question: %s", question)
    log.info("")

    # ── Stage 1: Ingest ────────────────────────────────────────────────────
    num_chunks = ingest_document(doc_path)
    print(f"\n✓ Ingested {num_chunks} chunks from '{doc_path.name}'")

    # ── Stage 2: Retrieve ──────────────────────────────────────────────────
    print(f"\n── Retrieving for: {question!r}")
    docs = retrieve_text(question)

    if not docs:
        print("⚠  No documents retrieved. Check your vector store and embedding setup.")
        return

    print(f"\n── Retrieved {len(docs)} chunk(s):\n")
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        source = Path(meta.get("source", "?")).name
        hyde_tag = " [HyDE]" if meta.get("hyde") else ""
        rewrite_tag = f" [rewrite={meta.get('rewrite_id', 0)}]"
        print(f"  Chunk #{i}{hyde_tag}{rewrite_tag}")
        print(f"    source  : {source}")
        print(f"    page    : {meta.get('page')}")
        print(f"    chunk_id: {meta.get('chunk_id')}")
        print(f"    text    : {d.page_content[:300]!r}")
        print()

    # ── Stage 3: Compress ──────────────────────────────────────────────────
    print("── Compressed context:\n")
    compressed = compress_context(docs, question)
    print(compressed)
    print()


if __name__ == "__main__":
    main()


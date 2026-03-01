"""
Ingest all documents from the sample_docs/ folder into the vector store.

Usage:
    python scripts/ingest_docs.py
    python scripts/ingest_docs.py --docs-dir path/to/custom/sample_docs
    python scripts/ingest_docs.py --file sample_docs/kafka_architecture.txt

Supports: .pdf, .txt, .md, .markdown
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Allow root-level imports regardless of CWD.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingestion.ingest import ingest_document
from utils.logger import get_logger

log = get_logger("ingest_docs")

SUPPORTED = {".pdf", ".txt", ".md", ".markdown"}


def ingest_directory(docs_dir: Path) -> None:
    files = sorted(
        f for f in docs_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED
    )

    if not files:
        log.warning("No supported documents found in %s", docs_dir)
        return

    log.info("Found %d document(s) to ingest in '%s'", len(files), docs_dir)
    total_chunks = 0
    failed = 0
    t_start = time.perf_counter()

    for i, path in enumerate(files, start=1):
        log.info("─── [%d/%d] %s", i, len(files), path.name)
        try:
            n = ingest_document(path)
            total_chunks += n
            log.info("    → %d chunks written", n)
        except Exception as exc:
            log.error("    ✗ Failed to ingest %s: %s", path.name, exc)
            failed += 1

    elapsed = time.perf_counter() - t_start
    log.info("")
    log.info("═══════════════════════════════════════")
    log.info("Ingestion complete")
    log.info("  Files processed : %d", len(files) - failed)
    log.info("  Files failed    : %d", failed)
    log.info("  Total chunks    : %d", total_chunks)
    log.info("  Elapsed         : %.1fs", elapsed)
    log.info("═══════════════════════════════════════")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG vector store")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--docs-dir", type=Path, default=ROOT / "sample_docs",
                       help="Directory of documents to ingest (default: sample_docs/)")
    group.add_argument("--file", type=Path, help="Ingest a single file")
    args = parser.parse_args()

    os.environ.setdefault("LOG_LEVEL", "INFO")

    if args.file:
        log.info("Ingesting single file: %s", args.file)
        n = ingest_document(args.file)
        log.info("Done — %d chunk(s) written", n)
    else:
        ingest_directory(args.docs_dir)


if __name__ == "__main__":
    main()

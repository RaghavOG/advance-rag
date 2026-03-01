"""
End-to-end demo of the RAG pipeline.

Ingests all sample_docs/ documents, then runs a set of demo questions through the
full LangGraph pipeline and prints the structured results.

Usage:
    python scripts/run_demo.py
    python scripts/run_demo.py --skip-ingest   # if already ingested
    python scripts/run_demo.py --question "How does Kafka guarantee ordering?"
    LOG_LEVEL=DEBUG python scripts/run_demo.py  # verbose output
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingestion.ingest import ingest_document
from graph.graph import rag_graph
from utils.logger import get_logger

log = get_logger("run_demo")

DOCS_DIR = ROOT / "sample_docs"

DEMO_QUESTIONS = [
    # Single factual question
    "What is the role of partitions in Apache Kafka?",
    # Multi-query prompt (two independent questions)
    "How does Kafka guarantee ordering? And also what is consumer group?",
    # Question about RAG
    "What is HyDE and how does it improve retrieval?",
    # Cross-document question
    "What are the failure modes in RAG systems and how should they be handled?",
    # Python practices question
    "What Python practices should I follow for configuration management?",
]


def ingest_all() -> None:
    files = sorted(
        f for f in DOCS_DIR.rglob("*")
        if f.is_file() and f.suffix.lower() in {".pdf", ".txt", ".md", ".markdown"}
    )
    log.info("Ingesting %d document(s) from %s …", len(files), DOCS_DIR)
    for path in files:
        try:
            n = ingest_document(path)
            log.info("  ✓ %s  →  %d chunks", path.name, n)
        except Exception as exc:
            log.error("  ✗ %s  →  %s", path.name, exc)


def run_question(question: str) -> None:
    log.info("")
    log.info("┌──────────────────────────────────────────────────────────")
    log.info("│ QUESTION: %s", question)
    log.info("└──────────────────────────────────────────────────────────")

    t0 = time.perf_counter()
    state = rag_graph.invoke({"raw_prompt": question})
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    # ── Print structured result ────────────────────────────────────────────
    sub_answers = state.get("sub_answers") or []
    final = state.get("final_answer") or state.get("answer_text") or ""
    error = state.get("error_message") or ""

    print()
    print("=" * 70)
    print(f"  Q: {question}")
    print(f"  elapsed: {elapsed_ms}ms")
    print("=" * 70)

    if error and not sub_answers:
        print(f"\n  ⚠  {error}\n")
        print(f"  {final}")
        print()
        return

    if sub_answers:
        for i, sa in enumerate(sub_answers, 1):
            q = sa.get("question", "")
            a = sa.get("answer", "")
            print(f"\n  Sub-question {i}: {q}")
            print("  " + "─" * 60)
            for line in textwrap.wrap(a, width=66):
                print(f"  {line}")
    elif final:
        for line in textwrap.wrap(final, width=66):
            print(f"  {line}")

    print()

    # ── Retrieved doc summary ──────────────────────────────────────────────
    docs = state.get("final_retrieved_docs") or []
    if docs:
        print(f"  Retrieved {len(docs)} chunk(s):")
        for d in docs[:5]:
            meta = d.get("metadata", {})
            score = d.get("score", "?")
            print(f"    • {meta.get('source', '?').split('/')[-1]} "
                  f"p.{meta.get('page')} chunk={meta.get('chunk_id')} "
                  f"score={round(score, 4) if isinstance(score, float) else score}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG pipeline demo")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Skip document ingestion (use if already done)")
    parser.add_argument("--question", type=str, default=None,
                        help="Run a single custom question instead of the demo set")
    args = parser.parse_args()

    os.environ.setdefault("LOG_LEVEL", "INFO")

    if not args.skip_ingest:
        ingest_all()

    questions = [args.question] if args.question else DEMO_QUESTIONS
    for q in questions:
        run_question(q)

    log.info("Demo complete.")


if __name__ == "__main__":
    main()

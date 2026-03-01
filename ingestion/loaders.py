"""
Document loaders for the ingestion pipeline.

Supported formats:
  - PDF   (.pdf)  — pypdf page-by-page extraction
  - Text  (.txt)  — split on double newlines (paragraph boundaries)
  - Markdown (.md, .markdown) — same as text, strips markdown syntax

All loaders return a list of PageText objects with consistent metadata.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

from pypdf import PdfReader

from ingestion.cleaning import clean_text
from utils.logger import get_logger

log = get_logger(__name__)

_SUPPORTED = {".pdf", ".txt", ".md", ".markdown"}


@dataclass
class PageText:
    doc_id: str
    page: int        # 1-based; for text files, 1 paragraph = 1 "page"
    source: str
    text: str


# ── PDF ───────────────────────────────────────────────────────────────────────

def load_pdf_pages(path: Path, *, doc_id: str) -> List[PageText]:
    """Load a PDF into per-page PageText objects."""
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    log.info("Loading PDF: %s", path.name)
    reader = PdfReader(str(path))
    total = len(reader.pages)
    log.debug("  PDF has %d page(s)", total)

    pages: List[PageText] = []
    empty = 0
    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        cleaned = clean_text(raw)
        if not cleaned:
            empty += 1
            log.debug("  Page %d is empty — skipped", i + 1)
            continue
        pages.append(PageText(doc_id=doc_id, page=i + 1, source=str(path), text=cleaned))

    log.info("  Loaded %d/%d pages from '%s'  (skipped %d empty)", len(pages), total, path.name, empty)
    return pages


# ── Plain text / Markdown ─────────────────────────────────────────────────────

def _strip_markdown(text: str) -> str:
    """Very light markdown stripping (headings, bold/italic, code fences)."""
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)   # fenced code blocks
    text = re.sub(r"`[^`]+`", "", text)                       # inline code
    text = re.sub(r"#{1,6}\s*", "", text)                     # headings
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)     # bold/italic
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)               # images
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)    # links → label
    return text


def load_text_file(path: Path, *, doc_id: str) -> List[PageText]:
    """
    Load a .txt or .md file.
    Splits on paragraph boundaries (blank lines) and treats each paragraph
    as a "page" so chunk metadata stays meaningful.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    log.info("Loading %s file: %s", suffix.upper().lstrip("."), path.name)

    raw = path.read_text(encoding="utf-8", errors="replace")
    if suffix in {".md", ".markdown"}:
        raw = _strip_markdown(raw)

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", raw) if p.strip()]
    log.debug("  Split into %d paragraph(s)", len(paragraphs))

    pages: List[PageText] = []
    for i, para in enumerate(paragraphs):
        cleaned = clean_text(para)
        if cleaned:
            pages.append(PageText(doc_id=doc_id, page=i + 1, source=str(path), text=cleaned))

    log.info("  Loaded %d paragraph(s) from '%s'", len(pages), path.name)
    return pages


# ── Generic dispatcher ────────────────────────────────────────────────────────

def load_document(path: Path, *, doc_id: str) -> List[PageText]:
    """
    Dispatch to the correct loader based on file extension.
    Raises ValueError for unsupported formats.
    """
    suffix = path.suffix.lower()
    if suffix not in _SUPPORTED:
        raise ValueError(
            f"Unsupported format '{suffix}' for {path.name}. "
            f"Supported: {', '.join(sorted(_SUPPORTED))}"
        )
    if suffix == ".pdf":
        return load_pdf_pages(path, doc_id=doc_id)
    return load_text_file(path, doc_id=doc_id)


__all__ = ["PageText", "load_pdf_pages", "load_text_file", "load_document"]

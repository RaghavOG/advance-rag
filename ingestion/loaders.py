"""
PDF loaders for the minimal ingestion pipeline (text-only).

Uses pypdf for simple, robust text extraction.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from pypdf import PdfReader

from ingestion.cleaning import clean_text


@dataclass
class PageText:
    doc_id: str
    page: int
    source: str
    text: str


def load_pdf_pages(path: Path, *, doc_id: str) -> List[PageText]:
    """
    Load a single PDF into per-page PageText objects.
    """
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    reader = PdfReader(str(path))
    pages: List[PageText] = []
    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        cleaned = clean_text(raw)
        pages.append(
            PageText(
                doc_id=doc_id,
                page=i + 1,
                source=str(path),
                text=cleaned,
            )
        )
    return pages


__all__ = ["PageText", "load_pdf_pages"]


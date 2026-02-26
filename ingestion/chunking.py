"""
Recursive chunking for PDF page text.

We use LangChain's RecursiveCharacterTextSplitter to keep chunks within a
reasonable size while preserving as much structure as possible.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import get_settings
from ingestion.loaders import PageText


def chunk_pages(pages: List[PageText]) -> List[Dict[str, Any]]:
    """
    Chunk a list of PageText objects into smaller text segments.

    Returns a list of dicts with keys:
      - text: chunk text
      - metadata: {doc_id, page, source, chunk_id}
    """
    cfg = get_settings()
    # Use the ingestion-related chunk sizes if defined, else fall back.
    chunk_size = getattr(cfg, "max_chunk_char_length", 2000)
    chunk_overlap = getattr(cfg, "min_chunk_char_length", 200)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[Dict[str, Any]] = []
    chunk_id = 0

    for page in pages:
        if not page.text:
            continue
        for piece in splitter.split_text(page.text):
            meta = {
                "doc_id": page.doc_id,
                "page": page.page,
                "source": page.source,
                "chunk_id": chunk_id,
            }
            chunks.append({"text": piece, "metadata": meta})
            chunk_id += 1

    return chunks


__all__ = ["chunk_pages"]


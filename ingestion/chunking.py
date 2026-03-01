"""
Recursive chunking for loaded page/paragraph text.

We use LangChain's RecursiveCharacterTextSplitter to keep chunks within a
reasonable size while preserving as much structure as possible.
"""
from __future__ import annotations

from typing import Any, Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import get_settings
from ingestion.loaders import PageText
from utils.logger import get_logger

log = get_logger(__name__)


def chunk_pages(pages: List[PageText]) -> List[Dict[str, Any]]:
    """
    Chunk a list of PageText objects into smaller text segments.

    Returns a list of dicts with keys:
      - text: chunk text
      - metadata: {doc_id, page, source, chunk_id}
    """
    cfg = get_settings()
    chunk_size = getattr(cfg, "max_chunk_char_length", 2000)
    chunk_overlap = getattr(cfg, "min_chunk_char_length", 200)

    log.debug(
        "Chunking %d page(s)  chunk_size=%d  overlap=%d",
        len(pages), chunk_size, chunk_overlap,
    )

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
        pieces = splitter.split_text(page.text)
        log.debug("  Page/para %d → %d chunk(s)", page.page, len(pieces))
        for piece in pieces:
            meta = {
                "doc_id": page.doc_id,
                "page": page.page,
                "source": page.source,
                "chunk_id": chunk_id,
            }
            chunks.append({"text": piece, "metadata": meta})
            chunk_id += 1

    log.info("Chunking complete: %d total chunk(s) produced", len(chunks))
    return chunks


__all__ = ["chunk_pages"]

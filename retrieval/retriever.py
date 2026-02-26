"""
Retrieval layer for the minimal text-first pipeline.

Responsibilities:
- Accept a user query.
- Call the configured vector store.
- Apply top-k and optional metadata filters.
- Return LangChain Document objects.
"""
from __future__ import annotations

from typing import List, Optional, Dict, Any

from langchain_core.documents import Document

from config.settings import get_settings
from vectorstores.factory import create_vectorstore


def retrieve_text(
    query: str,
    *,
    k: Optional[int] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """
    Retrieve top-k text chunks for a query.

    For now this is text-only and targets the text collection from Settings.
    """
    cfg = get_settings()
    vs = create_vectorstore(collection_name=cfg.chroma_collection_text)

    top_k = k or cfg.top_k_text
    # All supported backends expose similarity_search with filter support.
    docs = vs.similarity_search(
        query,
        k=top_k,
        filter=metadata_filter,
    )
    return docs


__all__ = ["retrieve_text"]

# Responsibilities:
# Accept query
# Call vectorstore
# Return Document[]
# Apply:
    # top-k
    # metadata filters


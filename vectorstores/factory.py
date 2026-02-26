"""
Vector store factory.

Responsibilities:
- Read VECTOR_STORE from configuration.
- Initialize the appropriate backend (Chroma, FAISS, Pinecone).
- Return a common LangChain VectorStore interface.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import Chroma, FAISS
from langchain_pinecone import PineconeVectorStore

from config.settings import get_settings, VectorStoreType
from embeddings.factory import get_embedder


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def create_vectorstore(
    *,
    collection_name: Optional[str] = None,
) -> VectorStore:
    """
    Create a VectorStore instance for the configured backend.

    Conventions:
    - For Chroma: `collection_name` is required and maps to a Chroma collection.
    - For FAISS:  each logical "collection" is stored under
                  `FAISS_INDEX_PATH/<collection_name>/` and MUST NOT be shared
                  across different logical collections.
    - For Pinecone: `collection_name` is ignored; the configured index name is used.
    """
    cfg = get_settings()
    embeddings = get_embedder()

    if cfg.vector_store == VectorStoreType.CHROMA:
        if not collection_name:
            raise ValueError(
                "collection_name is required when using Chroma as the vector store."
            )
        persist_dir = cfg.chroma_persist_directory
        _ensure_dir(persist_dir)
        return Chroma(
            collection_name=collection_name,
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
        )

    if cfg.vector_store == VectorStoreType.FAISS:
        if collection_name is None:
            raise ValueError("FAISS requires an explicit collection_name")

        base = cfg.faiss_index_path
        _ensure_dir(base)
        index_path = base / f"{collection_name}"

        if index_path.exists():
            # Load an existing FAISS index from disk.
            return FAISS.load_local(
                str(index_path),
                embeddings,
                allow_dangerous_deserialization=True,
            )

        # Create a new, empty FAISS index (dimension inferred from embeddings).
        vs = FAISS.from_texts(texts=[], embedding=embeddings)
        vs.save_local(str(index_path))
        return vs

    if cfg.vector_store == VectorStoreType.PINECONE:
        # Pinecone credentials and index configuration are validated in Settings.
        if cfg.pinecone_index_name is None:
            raise ValueError("PINECONE_INDEX_NAME must be set for Pinecone backend.")
        return PineconeVectorStore(
            index_name=cfg.pinecone_index_name,
            embedding=embeddings,
            namespace=cfg.pinecone_namespace,
        )

    raise ValueError(f"Unsupported VECTOR_STORE backend: {cfg.vector_store}")


__all__ = ["create_vectorstore"]


"""
Vector store factory.

Responsibilities:
- Read VECTOR_STORE from configuration.
- Initialize the appropriate backend (Chroma, FAISS, Pinecone).
- Return a common LangChain VectorStore interface.

Score semantics by backend
--------------------------
  chroma  : similarity_search_with_score returns L2/cosine DISTANCE → lower is better.
            Normalize to confidence via  confidence = 1 - raw_score  (clamped to [0,1]).
  pinecone: similarity_search_with_score returns cosine SIMILARITY   → higher is better.
            Score is already a confidence in [0,1]; use as-is.
  faiss   : By default uses L2 distance → lower is better.
            When FAISS_USE_INNER_PRODUCT=true, returns inner-product similarity → higher better.
            Normalize the L2 case via  confidence = 1 / (1 + raw_score)  (always [0,1]).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.vectorstores import VectorStore
from langchain_pinecone import PineconeVectorStore

from config.settings import VectorStoreType, get_settings
from embeddings.factory import get_embedder
from utils.logger import get_logger

_log = get_logger(__name__)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_score(raw_score: float) -> Tuple[float, str]:
    """
    Convert a raw score from the current backend to a confidence in [0, 1].

    Returns (confidence, description) where description briefly explains the
    transformation so callers can log it clearly.
    """
    cfg = get_settings()

    if cfg.vector_store == VectorStoreType.PINECONE:
        # Pinecone cosine similarity: already in [0, 1], higher = better.
        confidence = float(max(0.0, min(1.0, raw_score)))
        return confidence, "pinecone_similarity(as-is)"

    if cfg.vector_store == VectorStoreType.FAISS:
        if cfg.faiss_use_inner_product:
            # Inner-product similarity, typically in [-1, 1] for normalised vectors.
            confidence = float(max(0.0, min(1.0, raw_score)))
            return confidence, "faiss_ip(as-is)"
        # L2 distance: confidence = 1 / (1 + distance) so large distances → ~0.
        confidence = 1.0 / (1.0 + float(raw_score))
        return confidence, f"faiss_l2→conf(1/(1+{raw_score:.4f}))"

    # Chroma (default): cosine / L2 distance, lower is better.
    # Distance is typically in [0, 2]; clamp to [0, 1] before inverting.
    clamped = max(0.0, min(1.0, float(raw_score)))
    confidence = 1.0 - clamped
    return confidence, f"chroma_dist→conf(1-{raw_score:.4f})"


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

    _log.info(
        "create_vectorstore: backend=%s  collection=%r  (env VECTOR_STORE=%s)",
        cfg.vector_store.value,
        collection_name,
        cfg.vector_store.value,
    )

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


__all__ = ["create_vectorstore", "normalize_score"]

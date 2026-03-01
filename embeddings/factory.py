"""
Embedding factory.

Responsibilities:
- Read Settings.
- Instantiate exactly one embedder for the configured provider.
- Centralize embedding creation for HyDE, re-embedding, model swaps, caching, etc.
"""
from functools import lru_cache

from config.settings import VectorStoreType, get_settings
from embeddings.base import BaseEmbedder
from embeddings.openai import OpenAIEmbedder
from embeddings.sentence_transformers import SentenceTransformersEmbedder


@lru_cache(maxsize=1)
def get_embedder() -> BaseEmbedder:
    """
    Return a singleton embedder instance based on configuration.

    Do NOT instantiate embeddings anywhere else; always go through this factory.
    """
    cfg = get_settings()
    provider = cfg.embedding_provider

    if provider == "openai":
        embedder: BaseEmbedder = OpenAIEmbedder()
    elif provider == "sentence-transformers":
        embedder = SentenceTransformersEmbedder()
    else:
        raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {provider!r}")

    # Fail-fast check: ensure embedding dimension matches Pinecone index dimension.
    if cfg.vector_store == VectorStoreType.PINECONE and cfg.pinecone_dimension is not None:
        probe_vec = embedder.embed_query("dimension check")
        dim = len(probe_vec)
        if dim != cfg.pinecone_dimension:
            raise ValueError(
                f"Embedding dimension {dim} does not match PINECONE_DIMENSION={cfg.pinecone_dimension}. "
                "Update either EMBEDDING_MODEL_NAME or PINECONE_DIMENSION so they are consistent."
            )

    return embedder


__all__ = ["get_embedder", "BaseEmbedder"]

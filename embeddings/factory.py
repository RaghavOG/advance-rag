"""
Embedding factory.

Responsibilities:
- Read Settings.
- Instantiate exactly one embedder for the configured provider.
- Centralize embedding creation for HyDE, re-embedding, model swaps, caching, etc.
"""
from functools import lru_cache

from config.settings import get_settings
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
        return OpenAIEmbedder()
    if provider == "sentence-transformers":
        return SentenceTransformersEmbedder()

    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {provider!r}")


__all__ = ["get_embedder", "BaseEmbedder"]


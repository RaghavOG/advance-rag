"""
Base embedding interface for this project.

We subclass LangChain's Embeddings so that all concrete embedders share a
common type and can be swapped without changing callers.
"""
from langchain_core.embeddings import Embeddings


class BaseEmbedder(Embeddings):
    """
    Thin wrapper around LangChain's Embeddings.

    Concrete implementations (OpenAI, sentence-transformers, etc.) should
    subclass this so we can type against a single base.
    """

    # No extra methods for now; LangChain's abstract methods define the surface.
    pass


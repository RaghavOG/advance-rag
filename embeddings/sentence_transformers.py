"""
Sentence-transformers based embedder implementation.
"""
from langchain_community.embeddings import SentenceTransformerEmbeddings

from config.settings import get_settings
from embeddings.base import BaseEmbedder


class SentenceTransformersEmbedder(SentenceTransformerEmbeddings, BaseEmbedder):
    def __init__(self) -> None:
        cfg = get_settings()
        super().__init__(model_name=cfg.embedding_model_name)


__all__ = ["SentenceTransformersEmbedder"]


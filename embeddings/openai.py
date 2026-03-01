"""
OpenAI-based embedder implementation.

Uses the configured EMBEDDING_MODEL_NAME and OPENAI_API_KEY from Settings.
"""
from langchain_openai import OpenAIEmbeddings

from config.settings import get_settings
from embeddings.base import BaseEmbedder


class OpenAIEmbedder(OpenAIEmbeddings, BaseEmbedder):
    def __init__(self) -> None:
        cfg = get_settings()
        super().__init__(
            model=cfg.embedding_model_name,
            openai_api_key=cfg.openai_api_key,
        )


__all__ = ["OpenAIEmbedder"]

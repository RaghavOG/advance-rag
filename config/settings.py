"""
Application and vector store configuration using pydantic-settings.

Goals:
- Load values from .env automatically.
- Validate and normalize critical settings.
- Fail fast on misconfiguration instead of silently proceeding.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class VectorStoreType(str, Enum):
    CHROMA = "chroma"
    FAISS = "faiss"
    PINECONE = "pinecone"


class Settings(BaseSettings):
    """
    Central settings model.

    Only a focused subset of all possible knobs is modeled here; you can extend
    this class as your system grows.
    """

    # Tell pydantic-settings to load from .env and ignore extra env vars.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Core app metadata ---
    app_name: str = Field("advanced-rag", alias="APP_NAME")
    app_env: Literal["development", "staging", "production"] = Field(
        "development", alias="APP_ENV"
    )

    # --- Embeddings / LLM (minimal subset, extend as needed) ---
    embedding_provider: str = Field("openai", alias="EMBEDDING_PROVIDER")
    embedding_model_name: str = Field(
        "all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL_NAME",
    )

    # OpenAI is the most common provider; we validate its key if selected.
    openai_api_key: Optional[str] = Field(
        default=None,
        alias="OPENAI_API_KEY",
    )

    # --- Vector store selection ---
    vector_store: VectorStoreType = Field(
        default=VectorStoreType.CHROMA,
        alias="VECTOR_STORE",
    )

    # Chroma (local)
    chroma_persist_directory: Path = Field(
        default=Path("./data/chroma"),
        alias="CHROMA_PERSIST_DIRECTORY",
    )
    chroma_collection_text: str = Field(
        default="text_index",
        alias="CHROMA_COLLECTION_TEXT",
    )
    chroma_collection_image: str = Field(
        default="image_index",
        alias="CHROMA_COLLECTION_IMAGE",
    )
    chroma_collection_audio: str = Field(
        default="audio_index",
        alias="CHROMA_COLLECTION_AUDIO",
    )

    # FAISS (local)
    faiss_index_path: Path = Field(
        default=Path("./data/faiss"),
        alias="FAISS_INDEX_PATH",
    )
    faiss_use_inner_product: bool = Field(
        default=False,
        alias="FAISS_USE_INNER_PRODUCT",
    )

    # Pinecone (managed, optional)
    pinecone_api_key: Optional[str] = Field(
        default=None,
        alias="PINECONE_API_KEY",
    )
    pinecone_environment: Optional[str] = Field(
        default=None,
        alias="PINECONE_ENVIRONMENT",
    )
    pinecone_index_name: Optional[str] = Field(
        default=None,
        alias="PINECONE_INDEX_NAME",
    )
    pinecone_dimension: Optional[int] = Field(
        default=None,
        alias="PINECONE_DIMENSION",
    )
    pinecone_metric: str = Field(
        default="cosine",
        alias="PINECONE_METRIC",
    )
    pinecone_namespace: str = Field(
        default="default",
        alias="PINECONE_NAMESPACE",
    )
    pinecone_timeout: int = Field(
        default=30,
        alias="PINECONE_TIMEOUT",
    )

    # --- Retrieval ---
    top_k_text: int = Field(5, alias="TOP_K_TEXT")
    top_k_image: int = Field(3, alias="TOP_K_IMAGE")
    top_k_audio: int = Field(3, alias="TOP_K_AUDIO")

    # --- OpenAI client defaults ---
    openai_timeout: int = Field(30, alias="OPENAI_TIMEOUT")
    openai_max_retries: int = Field(3, alias="OPENAI_MAX_RETRIES")
    openai_default_model: str = Field("gpt-4.1-mini", alias="OPENAI_DEFAULT_MODEL")

    # --- Compression / context ---
    compression_model: str = Field("gpt-4.1-mini", alias="COMPRESSION_MODEL")
    compression_max_tokens: int = Field(500, alias="COMPRESSION_MAX_TOKENS")
    max_context_tokens: int = Field(4000, alias="MAX_CONTEXT_TOKENS")

    # --- Answer generation ---
    answer_model: str = Field("gpt-4.1-mini", alias="ANSWER_MODEL")
    answer_temperature: float = Field(0.2, alias="ANSWER_TEMPERATURE")
    answer_timeout: int = Field(40, alias="ANSWER_TIMEOUT")

    # --- Token / safety limits ---
    max_input_tokens: int = Field(800, alias="MAX_INPUT_TOKENS")
    max_output_tokens: int = Field(800, alias="MAX_OUTPUT_TOKENS")

    # --- Query decomposition ---
    max_sub_queries: int = Field(3, alias="MAX_SUB_QUERIES")

    # --- HyDE / query rewriting ---
    enable_hyde: bool = Field(True, alias="ENABLE_HYDE")
    hyde_model: str = Field("gpt-4.1-mini", alias="HYDE_MODEL")
    hyde_max_tokens: int = Field(300, alias="HYDE_MAX_TOKENS")
    hyde_timeout: int = Field(20, alias="HYDE_TIMEOUT")

    # --- Ambiguity / clarification ---
    ambiguity_model: str = Field("gpt-4.1-mini", alias="AMBIGUITY_MODEL")
    ambiguity_timeout: int = Field(15, alias="AMBIGUITY_TIMEOUT")

    # --- Retrieval confidence ---
    retrieval_confidence_threshold: float = Field(0.2, alias="RETRIEVAL_CONFIDENCE_THRESHOLD")

    # --- Graph / pipeline ---
    graph_max_retries: int = Field(2, alias="GRAPH_MAX_RETRIES")
    enable_clarification: bool = Field(True, alias="ENABLE_QUERY_CLASSIFICATION")

    # -------- Validators (fail fast on bad config) --------

    @field_validator("embedding_provider")
    @classmethod
    def _embedding_provider_supported(cls, v: str) -> str:
        allowed = {"openai", "sentence-transformers"}
        if v not in allowed:
            raise ValueError(f"EMBEDDING_PROVIDER must be one of {allowed}, got {v!r}")
        return v

    @field_validator("openai_api_key")
    @classmethod
    def _require_openai_key_if_needed(
        cls,
        v: Optional[str],
        info,
    ) -> Optional[str]:
        # If using OpenAI embeddings and no key is set (or placeholder), fail fast.
        embedding_provider = info.data.get("embedding_provider")
        if embedding_provider == "openai":
            if not v or v.strip() == "" or v.startswith("your_"):
                raise ValueError(
                    "OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai "
                    "and must not be the placeholder value."
                )
        return v

    @field_validator("chroma_persist_directory")
    @classmethod
    def _validate_chroma_dir(cls, v: Path) -> Path:
        if str(v).strip() == "":
            raise ValueError("CHROMA_PERSIST_DIRECTORY cannot be empty")
        return v

    @field_validator("faiss_index_path")
    @classmethod
    def _validate_faiss_path(cls, v: Path) -> Path:
        if str(v).strip() == "":
            raise ValueError("FAISS_INDEX_PATH cannot be empty")
        return v

    @field_validator("pinecone_api_key", mode="after")
    @classmethod
    def _validate_pinecone_if_selected(
        cls,
        v: Optional[str],
        info,
    ) -> Optional[str]:
        vector_store = info.data.get("vector_store")
        if vector_store == VectorStoreType.PINECONE:
            # Fail fast if Pinecone is selected but credentials are missing/placeholder.
            missing = not v or v.startswith("your_")
            env = info.data.get("pinecone_environment")
            index_name = info.data.get("pinecone_index_name")
            dim = info.data.get("pinecone_dimension")
            if missing or not env or not index_name or not dim:
                raise ValueError(
                    "Pinecone selected (VECTOR_STORE=pinecone) but PINECONE_API_KEY, "
                    "PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME, or PINECONE_DIMENSION "
                    "is missing or invalid."
                )
        return v


# Singleton-style accessor so config is evaluated once and reused.
settings = Settings()


def get_settings() -> Settings:
    return settings


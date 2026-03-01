"""
Application configuration via pydantic-settings.

Loads from .env, validates every critical field, and fails fast on
misconfiguration so you never get silent wrong behavior at runtime.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env from project root (parent of config/) so it works regardless of CWD.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"

# Load .env into os.environ before Settings() runs. This guarantees pydantic-settings
# sees all variables even when env_file is not picked up correctly.
if _ENV_FILE.exists():
    load_dotenv(_ENV_FILE, override=False)


class VectorStoreType(str, Enum):
    CHROMA = "chroma"
    FAISS = "faiss"
    PINECONE = "pinecone"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE) if _ENV_FILE.exists() else None,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Core ─────────────────────────────────────────────────────────────────
    app_name: str = Field("advance-rag", alias="APP_NAME")
    app_env: Literal["development", "staging", "production"] = Field(
        "development", alias="APP_ENV"
    )

    # ── OpenAI ───────────────────────────────────────────────────────────────
    openai_api_key: Optional[str] = Field(None, alias="OPENAI_API_KEY")
    openai_default_model: str = Field("gpt-4.1-mini", alias="OPENAI_DEFAULT_MODEL")
    openai_timeout: int = Field(30, alias="OPENAI_TIMEOUT")
    openai_max_retries: int = Field(3, alias="OPENAI_MAX_RETRIES")

    # ── Embeddings ───────────────────────────────────────────────────────────
    embedding_provider: str = Field("openai", alias="EMBEDDING_PROVIDER")
    embedding_model_name: str = Field("text-embedding-3-large", alias="EMBEDDING_MODEL_NAME")
    embedding_batch_size: int = Field(64, alias="EMBEDDING_BATCH_SIZE")

    # ── Vector store ─────────────────────────────────────────────────────────
    vector_store: VectorStoreType = Field(VectorStoreType.CHROMA, alias="VECTOR_STORE")

    # Chroma
    chroma_persist_directory: Path = Field(Path("./data/chroma"), alias="CHROMA_PERSIST_DIRECTORY")
    chroma_collection_text: str = Field("text_index", alias="CHROMA_COLLECTION_TEXT")
    chroma_collection_image: str = Field("image_index", alias="CHROMA_COLLECTION_IMAGE")
    chroma_collection_audio: str = Field("audio_index", alias="CHROMA_COLLECTION_AUDIO")

    # FAISS
    faiss_index_path: Path = Field(Path("./data/faiss"), alias="FAISS_INDEX_PATH")
    faiss_use_inner_product: bool = Field(False, alias="FAISS_USE_INNER_PRODUCT")

    # Pinecone — all defined together so model_validator sees every field
    pinecone_api_key: Optional[str] = Field(None, alias="PINECONE_API_KEY")
    pinecone_environment: Optional[str] = Field(None, alias="PINECONE_ENVIRONMENT")
    pinecone_index_name: Optional[str] = Field(None, alias="PINECONE_INDEX_NAME")
    pinecone_dimension: Optional[int] = Field(None, alias="PINECONE_DIMENSION")
    pinecone_metric: str = Field("cosine", alias="PINECONE_METRIC")
    pinecone_namespace: str = Field("default", alias="PINECONE_NAMESPACE")
    pinecone_timeout: int = Field(30, alias="PINECONE_TIMEOUT")

    # ── Retrieval ────────────────────────────────────────────────────────────
    top_k_text: int = Field(5, alias="TOP_K_TEXT")
    top_k_image: int = Field(3, alias="TOP_K_IMAGE")
    top_k_audio: int = Field(3, alias="TOP_K_AUDIO")
    retrieval_confidence_threshold: float = Field(0.2, alias="RETRIEVAL_CONFIDENCE_THRESHOLD")

    # ── Compression ──────────────────────────────────────────────────────────
    compression_model: str = Field("gpt-4.1-mini", alias="COMPRESSION_MODEL")
    compression_max_tokens: int = Field(500, alias="COMPRESSION_MAX_TOKENS")
    max_context_tokens: int = Field(4000, alias="MAX_CONTEXT_TOKENS")

    # ── Answer generation ────────────────────────────────────────────────────
    answer_model: str = Field("gpt-4.1-mini", alias="ANSWER_MODEL")
    answer_temperature: float = Field(0.2, alias="ANSWER_TEMPERATURE")
    answer_timeout: int = Field(40, alias="ANSWER_TIMEOUT")

    # ── Limits ───────────────────────────────────────────────────────────────
    max_input_tokens: int = Field(800, alias="MAX_INPUT_TOKENS")
    max_output_tokens: int = Field(800, alias="MAX_OUTPUT_TOKENS")

    # ── Ingestion / chunking ─────────────────────────────────────────────────
    min_chunk_char_length: int = Field(200, alias="MIN_CHUNK_CHAR_LENGTH")
    max_chunk_char_length: int = Field(2000, alias="MAX_CHUNK_CHAR_LENGTH")
    ingestion_batch_size: int = Field(10, alias="INGESTION_BATCH_SIZE")

    # ── Query decomposition ──────────────────────────────────────────────────
    max_sub_queries: int = Field(3, alias="MAX_SUB_QUERIES")

    # ── HyDE / query rewriting ───────────────────────────────────────────────
    enable_hyde: bool = Field(True, alias="ENABLE_HYDE")
    hyde_model: str = Field("gpt-4.1-mini", alias="HYDE_MODEL")
    hyde_max_tokens: int = Field(300, alias="HYDE_MAX_TOKENS")
    hyde_timeout: int = Field(20, alias="HYDE_TIMEOUT")

    # ── Ambiguity / clarification ────────────────────────────────────────────
    ambiguity_model: str = Field("gpt-4.1-mini", alias="AMBIGUITY_MODEL")
    ambiguity_timeout: int = Field(15, alias="AMBIGUITY_TIMEOUT")

    # ── Graph / pipeline ─────────────────────────────────────────────────────
    graph_max_retries: int = Field(2, alias="GRAPH_MAX_RETRIES")
    enable_clarification: bool = Field(True, alias="ENABLE_QUERY_CLASSIFICATION")

    # ── MongoDB ──────────────────────────────────────────────────────────────
    mongodb_uri: Optional[str] = Field(None, alias="MONGODB_URI")
    mongodb_db_name: str = Field("advance-rag", alias="MONGODB_DB_NAME")

    # ── LangSmith tracing ────────────────────────────────────────────────────
    langsmith_tracing: bool = Field(False, alias="LANGSMITH_TRACING")
    langsmith_endpoint: str = Field("https://api.smith.langchain.com", alias="LANGSMITH_ENDPOINT")
    langsmith_api_key: Optional[str] = Field(None, alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field("advance-rag", alias="LANGSMITH_PROJECT")

    # ── Validators ───────────────────────────────────────────────────────────

    @field_validator("embedding_provider")
    @classmethod
    def _embedding_provider_supported(cls, v: str) -> str:
        allowed = {"openai", "sentence-transformers"}
        if v not in allowed:
            raise ValueError(
                f"EMBEDDING_PROVIDER must be one of {allowed}, got {v!r}"
            )
        return v

    @field_validator("openai_api_key")
    @classmethod
    def _require_openai_key(cls, v: Optional[str], info) -> Optional[str]:
        if info.data.get("embedding_provider") == "openai":
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

    @model_validator(mode="after")
    def _validate_pinecone(self) -> "Settings":
        """
        Validate Pinecone credentials AFTER all fields are parsed.
        Only runs when VECTOR_STORE=pinecone. .env is loaded from project root (see _ENV_FILE).
        """
        if self.vector_store != VectorStoreType.PINECONE:
            return self

        def _blank(v: Optional[str]) -> bool:
            return v is None or (isinstance(v, str) and not v.strip())

        problems: list[str] = []
        if _blank(self.pinecone_api_key) or (self.pinecone_api_key or "").startswith("your_"):
            problems.append("PINECONE_API_KEY is missing or is a placeholder")
        if _blank(self.pinecone_environment):
            problems.append("PINECONE_ENVIRONMENT is missing (e.g. us-east-1)")
        if _blank(self.pinecone_index_name):
            problems.append("PINECONE_INDEX_NAME is missing")
        if self.pinecone_dimension is None or self.pinecone_dimension <= 0:
            problems.append(
                "PINECONE_DIMENSION is missing or invalid (positive integer, e.g. 3072)"
            )

        if problems:
            raise ValueError(
                "VECTOR_STORE=pinecone but Pinecone settings are incomplete. "
                "Ensure .env is in the project root:\n"
                + "\n".join(f"  • {p}" for p in problems)
                + "\n(Loading .env from: " + str(_ENV_FILE) + ")"
            )
        return self

    @model_validator(mode="after")
    def _validate_langsmith(self) -> "Settings":
        if self.langsmith_tracing:
            if not self.langsmith_api_key or self.langsmith_api_key.startswith("your_"):
                raise ValueError(
                    "LANGSMITH_TRACING=true but LANGSMITH_API_KEY is missing or a placeholder"
                )
        return self


# Singleton — evaluated once at import time.
settings = Settings()


def get_settings() -> Settings:
    return settings

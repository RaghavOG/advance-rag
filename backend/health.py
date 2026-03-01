"""
System health checks for the RAG backend.

Each check is independent — a failure in one does not abort others.
Results are collected into a structured HealthReport and returned from
the GET / and GET /health routes.

Checks performed:
  1. config          — pydantic-settings loads without error
  2. openai_key      — API key present and accepted by OpenAI
  3. embedding       — embedder produces a vector for a probe string
  4. vector_store    — vector store connection + collection exists
  5. docs_folder     — sample_docs/ directory exists and contains ingestable files
  6. chroma_dir      — Chroma persist dir exists and is writable
  7. mongodb         — MongoDB connection (if MONGODB_URI is set)
  8. langgraph       — rag_graph compiles without error
  9. packages        — critical Python packages are importable
"""
from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Pydantic models ───────────────────────────────────────────────────────────

CheckStatus = Literal["ok", "warn", "fail", "skip"]


class CheckResult(BaseModel):
    name: str
    status: CheckStatus
    message: str
    detail: Optional[str] = None
    elapsed_ms: Optional[int] = None


class HealthReport(BaseModel):
    overall: Literal["healthy", "degraded", "unhealthy"]
    service: str = "multimodal-rag"
    version: str = "1.0.0"
    timestamp: str
    checks: List[CheckResult]
    summary: Dict[str, int]          # {"ok": 5, "warn": 1, "fail": 0, "skip": 1}


# ── Individual checks ─────────────────────────────────────────────────────────

def _timed(fn) -> CheckResult:
    """Run fn() and capture elapsed ms. fn must return CheckResult."""
    t0 = time.perf_counter()
    result = fn()
    result.elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return result


def check_config() -> CheckResult:
    try:
        from config.settings import get_settings
        cfg = get_settings()
        return CheckResult(
            name="config",
            status="ok",
            message="Settings loaded and validated",
            detail=(
                f"env={cfg.app_env}  "
                f"vector_store={cfg.vector_store.value}  "
                f"embedding_provider={cfg.embedding_provider}  "
                f"embedding_model={cfg.embedding_model_name}  "
                f"answer_model={cfg.answer_model}"
            ),
        )
    except Exception as exc:
        return CheckResult(
            name="config",
            status="fail",
            message="Settings failed to load",
            detail=str(exc),
        )


def check_openai_key() -> CheckResult:
    try:
        from config.settings import get_settings
        cfg = get_settings()
        key = cfg.openai_api_key

        if not key:
            return CheckResult(
                name="openai_key",
                status="fail",
                message="OPENAI_API_KEY is not set",
                detail="Set OPENAI_API_KEY in your .env file",
            )

        if key.startswith("your_") or key == "sk-...":
            return CheckResult(
                name="openai_key",
                status="fail",
                message="OPENAI_API_KEY is still a placeholder",
                detail=f"Found: {key[:12]}…",
            )

        # Lightweight probe: list models (no token cost).
        from openai import OpenAI, AuthenticationError, APIConnectionError, RateLimitError
        client = OpenAI(api_key=key, timeout=8)
        client.models.list()

        masked = f"{key[:8]}…{key[-4:]}" if len(key) > 12 else "***"
        return CheckResult(
            name="openai_key",
            status="ok",
            message="OpenAI API key is valid and reachable",
            detail=f"key={masked}  timeout={cfg.openai_timeout}s",
        )

    except Exception as exc:
        cls = type(exc).__name__
        if "Authentication" in cls or "Incorrect API key" in str(exc):
            msg, status = "API key rejected by OpenAI", "fail"
        elif "Connection" in cls or "connect" in str(exc).lower():
            msg, status = "Cannot reach OpenAI API (network issue?)", "warn"
        elif "RateLimit" in cls:
            msg, status = "Rate limited — key is valid but quota exceeded", "warn"
        else:
            msg, status = f"OpenAI probe failed: {cls}", "warn"
        return CheckResult(name="openai_key", status=status, message=msg, detail=str(exc)[:200])


def check_embedding() -> CheckResult:
    try:
        from embeddings.factory import get_embedder
        embedder = get_embedder()
        vec = embedder.embed_query("health check probe")
        dim = len(vec)
        if dim == 0:
            return CheckResult(
                name="embedding",
                status="fail",
                message="Embedder returned a zero-length vector",
            )
        return CheckResult(
            name="embedding",
            status="ok",
            message=f"Embedding model operational",
            detail=f"dimension={dim}  model={getattr(embedder, 'model', 'unknown')}",
        )
    except Exception as exc:
        return CheckResult(
            name="embedding",
            status="fail",
            message="Embedding model failed",
            detail=str(exc)[:300],
        )


def check_vector_store() -> CheckResult:
    try:
        from config.settings import get_settings
        from vectorstores.factory import create_vectorstore
        cfg = get_settings()
        vs = create_vectorstore(collection_name=cfg.chroma_collection_text)

        # Attempt a zero-vector similarity search — valid even with an empty store.
        try:
            from embeddings.factory import get_embedder
            embedder = get_embedder()
            vec = embedder.embed_query("ping")
            vs.similarity_search_by_vector(vec, k=1)
            search_ok = True
        except Exception:
            search_ok = False

        return CheckResult(
            name="vector_store",
            status="ok" if search_ok else "warn",
            message="Vector store connected" + ("" if search_ok else " (search probe failed)"),
            detail=(
                f"backend={cfg.vector_store.value}  "
                f"collection={cfg.chroma_collection_text}  "
                f"search_ok={search_ok}"
            ),
        )
    except Exception as exc:
        detail = str(exc)[:300]
        msg = "Vector store connection failed"
        if "404" in detail or "Not Found" in detail:
            from config.settings import get_settings
            cfg = get_settings()
            idx = getattr(cfg, "pinecone_index_name", None) or "?"
            dim = getattr(cfg, "pinecone_dimension", None) or "?"
            msg = "Pinecone index not found (404)"
            detail = (
                f"The index '{idx}' does not exist in your Pinecone project. "
                f"Create it in Pinecone Console: dimension={dim}, metric=cosine, "
                f"region matching PINECONE_ENVIRONMENT. Then re-run the health check."
            )
        return CheckResult(
            name="vector_store",
            status="fail",
            message=msg,
            detail=detail,
        )


def check_docs_folder() -> CheckResult:
    docs_dir = ROOT / "sample_docs"
    if not docs_dir.exists():
        return CheckResult(
            name="docs_folder",
            status="warn",
            message="sample_docs/ folder does not exist",
            detail=f"Expected at {docs_dir}. Run: mkdir sample_docs",
        )

    supported = {".pdf", ".txt", ".md", ".markdown"}
    files = [f for f in docs_dir.rglob("*") if f.is_file() and f.suffix.lower() in supported]
    if not files:
        return CheckResult(
            name="docs_folder",
            status="warn",
            message="sample_docs/ exists but contains no ingestable files",
            detail=f"Supported formats: {', '.join(sorted(supported))}",
        )

    by_type: Dict[str, int] = {}
    total_bytes = 0
    for f in files:
        ext = f.suffix.lower()
        by_type[ext] = by_type.get(ext, 0) + 1
        total_bytes += f.stat().st_size

    breakdown = "  ".join(f"{ext}={n}" for ext, n in sorted(by_type.items()))
    return CheckResult(
        name="docs_folder",
        status="ok",
        message=f"{len(files)} ingestable file(s) found in sample_docs/",
        detail=f"{breakdown}  total_size={total_bytes // 1024}KB",
    )


def check_chroma_dir() -> CheckResult:
    try:
        from config.settings import get_settings
        cfg = get_settings()
        from config.settings import VectorStoreType
        if cfg.vector_store != VectorStoreType.CHROMA:
            return CheckResult(
                name="chroma_dir",
                status="skip",
                message=f"Skipped — vector_store={cfg.vector_store.value}",
            )

        d = cfg.chroma_persist_directory
        exists = d.exists()
        writable = False
        if exists:
            probe = d / ".write_probe"
            try:
                probe.touch()
                probe.unlink()
                writable = True
            except Exception:
                pass

        if not exists:
            d.mkdir(parents=True, exist_ok=True)
            return CheckResult(
                name="chroma_dir",
                status="ok",
                message="Chroma persist directory created",
                detail=str(d),
            )
        if not writable:
            return CheckResult(
                name="chroma_dir",
                status="fail",
                message="Chroma persist directory is not writable",
                detail=str(d),
            )
        return CheckResult(
            name="chroma_dir",
            status="ok",
            message="Chroma persist directory exists and is writable",
            detail=str(d),
        )
    except Exception as exc:
        return CheckResult(name="chroma_dir", status="fail", message=str(exc))


def check_langgraph() -> CheckResult:
    try:
        from graph.graph import build_rag_graph
        g = build_rag_graph()
        # Verify the graph has a compiled invoke method.
        if not callable(getattr(g, "invoke", None)):
            return CheckResult(
                name="langgraph",
                status="fail",
                message="Graph compiled but .invoke() is not callable",
            )
        return CheckResult(
            name="langgraph",
            status="ok",
            message="LangGraph compiled successfully",
            detail=f"nodes={len(g.nodes) if hasattr(g, 'nodes') else 'unknown'}",
        )
    except Exception as exc:
        return CheckResult(
            name="langgraph",
            status="fail",
            message="LangGraph failed to compile",
            detail=str(exc)[:300],
        )


def check_mongodb() -> CheckResult:
    """Check MongoDB connectivity and conversation persistence."""
    try:
        from config.settings import get_settings
        cfg = get_settings()

        if not cfg.mongodb_uri:
            return CheckResult(
                name="mongodb",
                status="warn",
                message="MONGODB_URI not set — using in-memory conversation store (not persistent)",
                detail="Set MONGODB_URI in .env to enable persistence across restarts",
            )

        from database.client import get_db
        db = get_db()

        if db is None:
            return CheckResult(
                name="mongodb",
                status="fail",
                message="MongoDB URI is set but connection failed",
                detail="Check MONGODB_URI credentials and network access",
            )

        # Lightweight ping — no data written.
        db.client.admin.command("ping")
        conv_count = db["conversations"].count_documents({})

        return CheckResult(
            name="mongodb",
            status="ok",
            message="MongoDB connected and responsive",
            detail=(
                f"db={db.name}  "
                f"conversations={conv_count}  "
                f"uri={cfg.mongodb_uri[:40]}…"
            ),
        )

    except Exception as exc:
        return CheckResult(
            name="mongodb",
            status="fail",
            message="MongoDB check failed",
            detail=str(exc)[:300],
        )


def check_packages() -> CheckResult:
    required = [
        ("langchain_core", "langchain-core"),
        ("langchain_openai", "langchain-openai"),
        ("langgraph", "langgraph"),
        ("openai", "openai"),
        ("chromadb", "chromadb"),
        ("pypdf", "pypdf"),
        ("pydantic_settings", "pydantic-settings"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("pymongo", "pymongo[srv]"),
    ]
    optional = [
        ("faiss", "faiss-cpu"),
        ("sentence_transformers", "sentence-transformers"),
        ("pytesseract", "pytesseract"),
        ("faster_whisper", "faster-whisper"),
        ("langsmith", "langsmith"),
    ]

    missing_required: list[str] = []
    missing_optional: list[str] = []

    for mod, pkg in required:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing_required.append(pkg)

    for mod, pkg in optional:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing_optional.append(pkg)

    if missing_required:
        return CheckResult(
            name="packages",
            status="fail",
            message=f"{len(missing_required)} required package(s) missing",
            detail=f"missing_required={missing_required}  missing_optional={missing_optional}",
        )
    if missing_optional:
        return CheckResult(
            name="packages",
            status="warn",
            message=f"All required packages present; {len(missing_optional)} optional package(s) missing",
            detail=f"missing_optional={missing_optional}",
        )
    return CheckResult(
        name="packages",
        status="ok",
        message="All required and optional packages are installed",
    )


# ── Aggregator ────────────────────────────────────────────────────────────────

_CHECKS = [
    check_config,
    check_openai_key,
    check_embedding,
    check_vector_store,
    check_docs_folder,
    check_chroma_dir,
    check_mongodb,
    check_langgraph,
    check_packages,
]


def run_all_checks() -> HealthReport:
    import datetime

    results: List[CheckResult] = []
    for fn in _CHECKS:
        try:
            results.append(_timed(fn))
        except Exception as exc:
            results.append(CheckResult(
                name=fn.__name__.removeprefix("check_"),
                status="fail",
                message=f"Check crashed: {exc}",
            ))

    summary: Dict[str, int] = {"ok": 0, "warn": 0, "fail": 0, "skip": 0}
    for r in results:
        summary[r.status] = summary.get(r.status, 0) + 1

    if summary["fail"] > 0:
        overall = "unhealthy"
    elif summary["warn"] > 0:
        overall = "degraded"
    else:
        overall = "healthy"

    return HealthReport(
        overall=overall,
        timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        checks=results,
        summary=summary,
    )

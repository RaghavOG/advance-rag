"""
Admin / dashboard routes for ingestion and index inspection.

These endpoints are intended for a local-only admin dashboard. They expose:

- GET  /api/admin/ingestion/docs
    → List all ingested documents from the manifest with basic stats.
- DELETE /api/admin/ingestion/docs/{doc_id}
    → Delete all chunks for a document from the vector store and manifest.
- GET  /api/admin/ingestion/docs/{doc_id}/chunks
    → Return chunk-level metadata/snippets for a given document.
"""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from config.settings import VectorStoreType, get_settings
from database import repository as db
from ingestion.ingest import (  # type: ignore[attr-defined]
    _MANIFEST_PATH,
    _load_manifest,
    _save_manifest,
)
from utils.logger import get_logger
from vectorstores.factory import create_vectorstore

router = APIRouter(tags=["admin-ingestion"])
log = get_logger(__name__)


def _normalise_manifest_entry(doc_id: str, value: Any) -> Dict[str, Any]:
    """
    Backwards-compatible manifest normalisation.

    Older manifests stored just an int (chunk_count). Newer ones store an object:
      { "chunks": int, "filename": str, "path": str | null }
    """
    if isinstance(value, int):
        return {"doc_id": doc_id, "chunks": value, "filename": None, "path": None}
    if isinstance(value, dict):
        return {
            "doc_id": doc_id,
            "chunks": int(value.get("chunks", 0)),
            "filename": value.get("filename"),
            "path": value.get("path"),
        }
    return {"doc_id": doc_id, "chunks": 0, "filename": None, "path": None}


@router.get("/admin/ingestion/docs")
async def list_ingested_docs() -> Dict[str, Any]:
    """
    List all ingested documents and aggregate index stats.

    Stats:
      - total_documents
      - total_chunks
      - index_size (alias for total_chunks)
      - embedding_model
      - vector_store_backend
    """
    cfg = get_settings()
    raw_manifest = _load_manifest()

    docs: List[Dict[str, Any]] = []
    total_chunks = 0
    for doc_id, value in raw_manifest.items():
        entry = _normalise_manifest_entry(doc_id, value)
        total_chunks += int(entry["chunks"])
        docs.append(entry)

    return {
        "documents": docs,
        "stats": {
            "total_documents": len(docs),
            "total_chunks": total_chunks,
            "index_size": total_chunks,
            "embedding_model": cfg.embedding_model_name,
            "vector_store_backend": cfg.vector_store.value,
        },
    }


@router.delete("/admin/ingestion/docs/{doc_id}")
async def delete_ingested_doc(doc_id: str) -> Dict[str, Any]:
    """
    Delete a single document's chunks from the vector store and manifest.

    Notes:
      - For Chroma, deletion uses a metadata filter on doc_id.
      - For FAISS, delete-by-filter is not implemented yet — you must rebuild
        the index manually (delete the FAISS directory and re-ingest).
    """
    cfg = get_settings()
    manifest = _load_manifest()

    if doc_id not in manifest:
        raise HTTPException(status_code=404, detail=f"doc_id {doc_id} not found in manifest")

    vs = create_vectorstore(collection_name=cfg.chroma_collection_text)
    try:
        if cfg.vector_store == VectorStoreType.CHROMA:
            # Chroma uses 'where' filters.
            vs.delete(where={"doc_id": doc_id})  # type: ignore[arg-type]
        elif cfg.vector_store == VectorStoreType.PINECONE:
            # PineconeVectorStore expects a 'filter' argument.
            vs.delete(filter={"doc_id": doc_id})  # type: ignore[arg-type]
        else:
            raise HTTPException(
                status_code=400,
                detail="Per-document delete is not implemented for this backend. "
                "Clear the index directory and re-ingest.",
            )
    except HTTPException:
        raise
    except Exception as exc:
        log.warning("Vector store delete for doc_id=%s failed: %s", doc_id, exc)
        raise HTTPException(status_code=500, detail=f"Vector store delete failed: {exc}") from exc

    # Update manifest
    removed_chunks = manifest.pop(doc_id, 0)
    _save_manifest(manifest)

    return {"doc_id": doc_id, "removed_chunks": int(removed_chunks)}


@router.get("/admin/ingestion/docs/{doc_id}/chunks")
async def list_doc_chunks(doc_id: str) -> Dict[str, Any]:
    """
    Return chunk-level metadata for a single document.

    Implementation detail:
      - For Chroma, we rely on the underlying .get(where={doc_id}) API via the
        LangChain wrapper's ._collection attribute.
      - For other backends, this endpoint returns an empty list for now.
    """
    cfg = get_settings()
    if not _MANIFEST_PATH.exists():
        raise HTTPException(status_code=404, detail="No manifest found")

    manifest = _load_manifest()
    if doc_id not in manifest:
        raise HTTPException(status_code=404, detail=f"doc_id {doc_id} not found in manifest")

    if cfg.vector_store != VectorStoreType.CHROMA:
        # Listing chunks is backend-specific; only Chroma is implemented.
        return {"doc_id": doc_id, "chunks": [], "backend": cfg.vector_store.value}

    vs = create_vectorstore(collection_name=cfg.chroma_collection_text)
    try:
        # type: ignore[attr-defined]
        raw = vs._collection.get(where={"doc_id": doc_id}, include=["metadatas", "documents"])
    except Exception as exc:
        log.warning("Chunk listing for doc_id=%s failed: %s", doc_id, exc)
        raise HTTPException(status_code=500, detail=f"Chunk listing failed: {exc}") from exc

    metadatas = raw.get("metadatas") or []
    documents = raw.get("documents") or []

    chunks: List[Dict[str, Any]] = []
    for meta, text in zip(metadatas, documents):
        chunks.append(
            {
                "doc_id": meta.get("doc_id"),
                "source": meta.get("source"),
                "page": meta.get("page"),
                "chunk_id": meta.get("chunk_id"),
                "snippet": (text or "")[:200],
            }
        )

    return {"doc_id": doc_id, "chunks": chunks, "backend": cfg.vector_store.value}


@router.post("/admin/ingestion/clear_all")
async def clear_all() -> Dict[str, Any]:
    """
    Danger: Clear all embeddings and conversation history.

    - Clears the vector store index (for Chroma/Pinecone).
    - Clears the ingestion manifest.
    - Clears MongoDB conversations and in-memory cache.
    """
    cfg = get_settings()
    backend = cfg.vector_store

    # Clear vector store
    vs = create_vectorstore(collection_name=cfg.chroma_collection_text)
    try:
        if backend == VectorStoreType.CHROMA:
            # Delete everything by using an empty where filter.
            vs.delete(where={})  # type: ignore[arg-type]
        elif backend == VectorStoreType.PINECONE:
            vs.delete(delete_all=True)  # type: ignore[arg-type]
        else:
            raise HTTPException(
                status_code=400,
                detail="Global clear_all is not implemented for this backend. "
                "Delete the index directory manually.",
            )
    except HTTPException:
        raise
    except Exception as exc:
        log.warning("Vector store clear_all failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Vector store clear_all failed: {exc}") from exc

    # Clear manifest
    _save_manifest({})

    # Clear conversation store (memory + MongoDB)
    # Reset in-memory map
    from backend import store as store_module  # local import to avoid cycles

    if hasattr(store_module, "_mem"):
        store_module._mem.clear()  # type: ignore[attr-defined]

    # Drop MongoDB collection if available
    if db.is_available():
        col = db._collection()  # type: ignore[attr-defined]
        if col is not None:
            col.delete_many({})

    return {
        "status": "ok",
        "vector_store_backend": backend.value,
        "manifest_cleared": True,
        "conversations_cleared": True,
    }

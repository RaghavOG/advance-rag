"""
Simple upload endpoint for local dashboard.

Accepts a single file, saves it under sample_docs/, and runs the ingestion
pipeline. Intended for use from the frontend dashboard with progress UI.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, File, HTTPException, UploadFile

from ingestion.ingest import ingest_document
from utils.logger import get_logger

router = APIRouter(tags=["upload"])
log = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_UPLOAD_DIR = _PROJECT_ROOT / "sample_docs"


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> Dict[str, Any]:
  """
  Upload a document and ingest it into the text vector store.

  Returns:
    {
      "filename": str,
      "path": str,
      "chunks_ingested": int
    }
  """
  suffix = (file.filename or "").lower()
  if not any(suffix.endswith(ext) for ext in (".pdf", ".txt", ".md", ".markdown")):
      raise HTTPException(status_code=400, detail="Only PDF, TXT, and Markdown files are supported")

  _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
  dest = _UPLOAD_DIR / file.filename

  try:
      content = await file.read()
      dest.write_bytes(content)
  except Exception as exc:
      log.warning("Failed to save uploaded file %s: %s", file.filename, exc)
      raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}") from exc

  try:
      chunks = ingest_document(dest)
  except Exception as exc:
      log.warning("Ingestion failed for %s: %s", dest, exc)
      raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

  return {
      "filename": file.filename,
      "path": str(dest),
      "chunks_ingested": chunks,
  }

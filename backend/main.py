"""
FastAPI entry point for the Multimodal RAG backend.

Run with:
    uvicorn backend.main:app --reload --port 8000

Routes:
    GET /                        → Full HTML health dashboard
    GET /health                  → JSON or HTML health (Accept-header driven)
    GET /health/json             → Raw JSON health report
    POST /api/query              → RAG query
    POST /api/clarify            → Clarification follow-up
    GET  /api/conversation/{id}  → Single conversation
    GET  /api/conversations      → List recent conversations
"""
from __future__ import annotations

import os

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from utils.logger import get_logger

log = get_logger(__name__)


def _configure_langsmith() -> None:
    """
    Set LangSmith environment variables from Settings before any LangChain import.
    Must run early — LangChain reads these vars at import time.
    """
    try:
        from config.settings import get_settings
        cfg = get_settings()
        if cfg.langsmith_tracing and cfg.langsmith_api_key:
            os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
            os.environ.setdefault("LANGCHAIN_ENDPOINT", cfg.langsmith_endpoint)
            os.environ.setdefault("LANGCHAIN_API_KEY", cfg.langsmith_api_key)
            os.environ.setdefault("LANGCHAIN_PROJECT", cfg.langsmith_project)
            log.info("LangSmith tracing enabled: project=%s", cfg.langsmith_project)
        else:
            log.info("LangSmith tracing disabled (LANGSMITH_TRACING=false or key missing)")
    except Exception as exc:
        log.warning("LangSmith config skipped: %s", exc)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # ── startup ──────────────────────────────────────────────────────────────
    _configure_langsmith()

    # Warm MongoDB connection so first request isn't slow.
    try:
        from database.client import get_db
        db = get_db()
        if db is not None:
            log.info("MongoDB ready at startup")
        else:
            log.info("Running without MongoDB (in-memory conversation store)")
    except Exception as exc:
        log.warning("MongoDB startup probe failed: %s", exc)

    # Ensure LangGraph compiles (catches circular import / config errors early).
    try:
        from graph.graph import build_rag_graph
        build_rag_graph()
        log.info("LangGraph compiled successfully")
    except Exception as exc:
        log.warning("LangGraph warm-up failed (will retry on first request): %s", exc)

    log.info("All services started successfully. API ready.")
    print("\n  ✓ All services started successfully. API ready at http://127.0.0.1:8000\n")

    yield

    # ── shutdown ─────────────────────────────────────────────────────────────
    try:
        from database.client import close_db
        close_db()
    except Exception:
        pass


from backend.routes.query import router as query_router

app = FastAPI(
    title="Multimodal RAG API",
    description="Production-grade RAG pipeline with LangGraph orchestration",
    version="1.0.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router, prefix="/api")


# ── Health check helpers ──────────────────────────────────────────────────────

def _status_color(status: str) -> str:
    return {"ok": "#10b981", "warn": "#f59e0b", "fail": "#f43f5e", "skip": "#64748b"}.get(status, "#94a3b8")


def _status_icon(status: str) -> str:
    return {"ok": "✅", "warn": "⚠️", "fail": "❌", "skip": "⏭️"}.get(status, "•")


def _overall_color(overall: str) -> str:
    return {"healthy": "#10b981", "degraded": "#f59e0b", "unhealthy": "#f43f5e"}.get(overall, "#94a3b8")


def _build_html(report) -> str:
    checks_html = ""
    for c in report.checks:
        color = _status_color(c.status)
        icon = _status_icon(c.status)
        detail_row = f'<div class="detail">{c.detail}</div>' if c.detail else ""
        elapsed = f'<span class="elapsed">{c.elapsed_ms}ms</span>' if c.elapsed_ms is not None else ""
        checks_html += f"""
        <div class="check">
          <div class="check-header">
            <span class="check-icon">{icon}</span>
            <span class="check-name">{c.name}</span>
            <span class="check-badge" style="background:{color}22;color:{color};border-color:{color}55">
              {c.status.upper()}
            </span>
            {elapsed}
          </div>
          <div class="check-msg">{c.message}</div>
          {detail_row}
        </div>"""

    ok_count    = report.summary.get("ok",   0)
    warn_count  = report.summary.get("warn",  0)
    fail_count  = report.summary.get("fail",  0)
    skip_count  = report.summary.get("skip",  0)
    overall_col = _overall_color(report.overall)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>RAG Health · {report.overall.upper()}</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{background:#0f1117;color:#cbd5e1;font-family:ui-sans-serif,system-ui,sans-serif;min-height:100vh;padding:2rem 1rem}}
    .page{{max-width:780px;margin:0 auto}}

    /* Header */
    .header{{display:flex;align-items:center;justify-content:space-between;margin-bottom:2rem;flex-wrap:wrap;gap:1rem}}
    .title{{font-size:1.5rem;font-weight:700;color:#f1f5f9}}
    .title span{{color:#818cf8}}
    .overall-badge{{
      display:inline-flex;align-items:center;gap:.5rem;
      border-radius:9999px;border:1px solid {overall_col}55;
      background:{overall_col}15;color:{overall_col};
      padding:.35rem 1rem;font-size:.85rem;font-weight:600
    }}
    .dot{{width:8px;height:8px;border-radius:50%;background:{overall_col};animation:pulse 2s infinite}}
    @keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.4}}}}

    /* Summary bar */
    .summary{{display:grid;grid-template-columns:repeat(4,1fr);gap:.75rem;margin-bottom:1.75rem}}
    .stat{{background:#161b27;border:1px solid #252d3d;border-radius:.75rem;padding:1rem;text-align:center}}
    .stat-val{{font-size:1.75rem;font-weight:700;line-height:1}}
    .stat-label{{font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;color:#64748b;margin-top:.3rem}}

    /* Checks */
    .checks{{display:flex;flex-direction:column;gap:.65rem}}
    .check{{background:#161b27;border:1px solid #252d3d;border-radius:.75rem;padding:1rem 1.2rem}}
    .check-header{{display:flex;align-items:center;gap:.65rem;flex-wrap:wrap}}
    .check-icon{{font-size:1rem;line-height:1}}
    .check-name{{font-size:.85rem;font-weight:600;color:#e2e8f0;font-family:ui-monospace,monospace;min-width:140px}}
    .check-badge{{
      font-size:.65rem;font-weight:700;padding:.2rem .6rem;
      border-radius:9999px;border:1px solid;letter-spacing:.05em
    }}
    .elapsed{{margin-left:auto;font-size:.65rem;color:#475569;font-family:monospace}}
    .check-msg{{font-size:.8rem;color:#94a3b8;margin-top:.5rem;margin-left:1.65rem}}
    .detail{{
      font-size:.7rem;color:#475569;margin-top:.35rem;margin-left:1.65rem;
      font-family:ui-monospace,monospace;
      background:#0f1117;border:1px solid #1e2535;border-radius:.4rem;
      padding:.4rem .7rem;word-break:break-word;line-height:1.6
    }}

    /* Footer */
    .footer{{margin-top:2rem;text-align:center;font-size:.7rem;color:#334155}}
    .footer a{{color:#6366f1;text-decoration:none}}
    .footer a:hover{{text-decoration:underline}}

    /* Refresh hint */
    .refresh{{display:inline-flex;align-items:center;gap:.4rem;font-size:.7rem;color:#475569;cursor:pointer;
      background:#1e2535;border:1px solid #252d3d;border-radius:.5rem;padding:.3rem .7rem;
      text-decoration:none;transition:background .15s}}
    .refresh:hover{{background:#252d3d}}
  </style>
</head>
<body>
  <div class="page">
    <div class="header">
      <div>
        <div class="title">Multimodal <span>RAG</span> — System Health</div>
        <div style="font-size:.75rem;color:#475569;margin-top:.3rem">{report.timestamp}</div>
      </div>
      <div style="display:flex;gap:.75rem;align-items:center;flex-wrap:wrap">
        <a class="refresh" href="/">↻ Refresh</a>
        <a class="refresh" href="/health/json" target="_blank">{{ }} JSON</a>
        <div class="overall-badge">
          <span class="dot"></span>
          {report.overall.upper()}
        </div>
      </div>
    </div>

    <div class="summary">
      <div class="stat">
        <div class="stat-val" style="color:#10b981">{ok_count}</div>
        <div class="stat-label">Passing</div>
      </div>
      <div class="stat">
        <div class="stat-val" style="color:#f59e0b">{warn_count}</div>
        <div class="stat-label">Warnings</div>
      </div>
      <div class="stat">
        <div class="stat-val" style="color:#f43f5e">{fail_count}</div>
        <div class="stat-label">Failing</div>
      </div>
      <div class="stat">
        <div class="stat-val" style="color:#64748b">{skip_count}</div>
        <div class="stat-label">Skipped</div>
      </div>
    </div>

    <div class="checks">
      {checks_html}
    </div>

    <div class="footer">
      <p>
        <a href="/docs">API Docs (Swagger)</a> ·
        <a href="/redoc">ReDoc</a> ·
        v{report.version}
      </p>
    </div>
  </div>
</body>
</html>"""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root(request: Request):
    """HTML health dashboard."""
    from backend.health import run_all_checks
    report = run_all_checks()
    return HTMLResponse(content=_build_html(report))


@app.get("/health", include_in_schema=True, summary="Health check (JSON or HTML)")
async def health(request: Request):
    """
    Returns JSON when Accept: application/json, otherwise redirects to the HTML dashboard.
    """
    from backend.health import run_all_checks
    from fastapi.responses import RedirectResponse
    accept = request.headers.get("accept", "")
    if "application/json" in accept:
        report = run_all_checks()
        status_code = 200 if report.overall == "healthy" else 503 if report.overall == "unhealthy" else 207
        return JSONResponse(content=report.model_dump(), status_code=status_code)
    return RedirectResponse(url="/")


@app.get("/health/json", summary="Health check (JSON only)")
async def health_json():
    """Always returns the health report as JSON."""
    from backend.health import run_all_checks
    report = run_all_checks()
    status_code = 200 if report.overall == "healthy" else 503 if report.overall == "unhealthy" else 207
    return JSONResponse(content=report.model_dump(), status_code=status_code)


# ── Conversation history ───────────────────────────────────────────────────────

@app.get("/api/conversations", summary="List recent conversations")
async def list_conversations(limit: int = 50):
    """
    Returns the most recent conversations from MongoDB (or in-memory store).
    Useful for conversation history in the frontend.
    """
    from backend import store
    entries = store.list_all(limit=limit)
    return {
        "conversations": [e.model_dump() for e in entries],
        "store_backend": store.backend_status(),
        "count": len(entries),
    }

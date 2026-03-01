"""
Centralized structured logger for the entire RAG pipeline.

Usage anywhere in the project:
    from utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Loaded %d chunks", n)

Log levels:
    DEBUG   — per-chunk/per-token details, raw scores, intermediate state
    INFO    — stage entry/exit, counts, latency signals
    WARNING — recoverable anomalies (empty results, fallback used)
    ERROR   — unrecoverable failures caught before re-raise

Environment variable:
    LOG_LEVEL=DEBUG | INFO | WARNING | ERROR  (default: INFO)
"""
from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Generator

# ── Formatting ────────────────────────────────────────────────────────────────

_FMT = (
    "%(asctime)s  %(levelname)-8s  "
    "%(name)-35s  %(message)s"
)
_DATE_FMT = "%H:%M:%S"

_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


class _ColorFormatter(logging.Formatter):
    """ANSI color codes for terminal readability."""

    _COLORS = {
        "DEBUG":    "\033[36m",   # cyan
        "INFO":     "\033[32m",   # green
        "WARNING":  "\033[33m",   # yellow
        "ERROR":    "\033[31m",   # red
        "CRITICAL": "\033[35m",   # magenta
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self._COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname:<8}{self._RESET}"
        return super().format(record)


def _build_handler() -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)
    # Use color formatter when attached to a real terminal.
    if sys.stdout.isatty():
        handler.setFormatter(_ColorFormatter(_FMT, datefmt=_DATE_FMT))
    else:
        handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))
    return handler


# Root handler installed once.
_root_handler = _build_handler()
_configured_loggers: set[str] = set()


def get_logger(name: str) -> logging.Logger:
    """
    Return a module-scoped logger.
    Calling this multiple times with the same name is safe (same logger returned).
    """
    logger = logging.getLogger(name)
    if name not in _configured_loggers:
        logger.setLevel(getattr(logging, _LEVEL, logging.INFO))
        if not logger.handlers:
            logger.addHandler(_root_handler)
        logger.propagate = False
        _configured_loggers.add(name)
    return logger


# ── Timing helper ─────────────────────────────────────────────────────────────

@contextmanager
def log_stage(logger: logging.Logger, stage: str, **kw) -> Generator[None, None, None]:
    """
    Context manager that logs stage entry/exit with elapsed ms.

    Example:
        with log_stage(log, "compression", query=q[:40]):
            ...
    """
    kw_str = "  ".join(f"{k}={v!r}" for k, v in kw.items())
    logger.info("→ START  %-30s  %s", stage, kw_str)
    t0 = time.perf_counter()
    try:
        yield
    except Exception as exc:
        elapsed = int((time.perf_counter() - t0) * 1000)
        logger.error("✗ FAIL   %-30s  elapsed=%dms  error=%s", stage, elapsed, exc)
        raise
    else:
        elapsed = int((time.perf_counter() - t0) * 1000)
        logger.info("✓ DONE   %-30s  elapsed=%dms", stage, elapsed)

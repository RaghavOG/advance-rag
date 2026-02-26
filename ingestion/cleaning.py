"""
Minimal cleaning utilities for text extracted from PDFs.

Keep this simple: the goal is to normalize whitespace and strip obvious noise
without getting clever. Retrieval bugs often hide in over-processed text.
"""
import re


WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if not text:
        return ""
    # Normalize all whitespace to single spaces and strip ends.
    normalized = WHITESPACE_RE.sub(" ", text)
    return normalized.strip()


__all__ = ["clean_text"]


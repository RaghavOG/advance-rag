"""
Safety filter — fast, rule-based (no LLM calls).

Checks every user prompt for:
  1. Excessive length (configurable via SAFETY_MAX_INPUT_CHARS)
  2. Prompt injection patterns — attempts to override system instructions
  3. Disallowed structural markers (XML-style instruction tags, etc.)

Returns (is_safe: bool, reason: str).
  is_safe=True  → prompt is clean; pipeline proceeds normally.
  is_safe=False → prompt is blocked; reason is surfaced to the user.

Design principles
-----------------
- No LLM calls — this runs before any expensive steps.
- False positives are preferred over false negatives for the injection checks.
- All patterns are case-insensitive and whitespace-tolerant.
- The list is intentionally conservative; extend via _INJECTION_PATTERNS below.
"""
from __future__ import annotations

import re
from typing import Tuple

# ---------------------------------------------------------------------------
# Injection pattern library
# Each entry is a compiled regex.  A match means the prompt is blocked.
# ---------------------------------------------------------------------------
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.I),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)", re.I),
    re.compile(r"forget\s+(everything|all|your|the)\s+(previous|prior|above|instructions?)", re.I),
    re.compile(r"new\s+(system\s+)?prompt\s*:", re.I),
    re.compile(r"you\s+are\s+now\s+(a\s+)?different", re.I),
    re.compile(r"<\s*/?\s*instructions?\s*>", re.I),
    re.compile(r"<\s*/?\s*system\s*>", re.I),
    re.compile(r"\[\s*INST\s*\]", re.I),
    re.compile(r"###\s*(System|Instruction|Override)", re.I),
    re.compile(r"reveal\s+(your\s+)?(system\s+)?prompt", re.I),
    re.compile(r"print\s+(your\s+)?(system\s+)?prompt", re.I),
    re.compile(r"output\s+(your\s+)?(system\s+)?instructions?", re.I),
    # DAN / jailbreak patterns
    re.compile(r"\bDAN\b"),
    re.compile(r"developer\s+mode", re.I),
    re.compile(r"jailbreak", re.I),
]

# Repeated character flood (e.g., 200+ consecutive same chars — likely adversarial)
_FLOOD_PATTERN = re.compile(r"(.)\1{199,}")


def check_safety(text: str, max_chars: int = 4000) -> Tuple[bool, str]:
    """
    Validate a user prompt.

    Parameters
    ----------
    text      : the normalized user prompt
    max_chars : maximum allowed character count (from SAFETY_MAX_INPUT_CHARS)

    Returns
    -------
    (True, "")                 → prompt is safe
    (False, "<reason string>") → prompt is blocked
    """
    # 1. Length check
    if len(text) > max_chars:
        return (
            False,
            f"Input too long ({len(text)} chars). Maximum allowed: {max_chars} chars.",
        )

    # 2. Character flood
    if _FLOOD_PATTERN.search(text):
        return False, "Input contains abnormal repeated characters."

    # 3. Injection patterns
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return False, "Potential prompt-injection attempt detected. Please rephrase your question."

    return True, ""


__all__ = ["check_safety"]

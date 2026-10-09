"""Prompt normalization — Step 1 of the layered fallback flow.

Produces a canonical form of the prompt so that trivially different phrasings
("Can you explain X?" vs "explain x") hash to the same exact-match key and
embed closer together in vector space.
"""

from __future__ import annotations

import hashlib
import re

# Ordered longest-first so that e.g. "i would like" is removed before "i would".
FILLER_PHRASES: tuple[str, ...] = (
    "i would like to know",
    "i would like",
    "tell me about",
    "i want to",
    "i need to",
    "could you",
    "can you",
    "help me",
    "explain",
    "what is",
    "please",
)

_WHITESPACE_RE = re.compile(r"\s+")
_DISALLOWED_RE = re.compile(r"[^\w\s?.,]")


def normalize_prompt(prompt: str) -> str:
    """Lowercase, collapse whitespace, strip filler phrases and stray symbols."""
    p = prompt.lower().strip()
    p = _WHITESPACE_RE.sub(" ", p)
    for filler in FILLER_PHRASES:
        p = p.replace(filler, " ")
    p = _DISALLOWED_RE.sub("", p)
    p = _WHITESPACE_RE.sub(" ", p)
    return p.strip()


def exact_match_key(normalized_prompt: str, tenant_id: str, context_hash: str = "") -> str:
    """Deterministic L1 cache key: hash(normalized prompt + tenant + context)."""
    digest = hashlib.sha256(
        f"{tenant_id}\x00{normalized_prompt}\x00{context_hash}".encode()
    ).hexdigest()
    return f"tenant:{tenant_id}:exact:{digest}"


def context_hash(context_turns: list[str] | None) -> str:
    if not context_turns:
        return ""
    return hashlib.sha256("\x00".join(context_turns).encode()).hexdigest()[:16]

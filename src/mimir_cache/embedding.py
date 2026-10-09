"""Embedding pipeline — context-aware cache key embeddings (Contribution 3).

Two embedder implementations behind one protocol:

* ``SentenceTransformerEmbedder`` — production path, all-MiniLM-L6-v2 (384-dim).
* ``HashEmbedder`` — deterministic, dependency-free fallback used in tests/CI and
  when ``sentence-transformers`` is not installed. Similar token sets produce
  similar vectors (bag-of-token-hashes), which is sufficient to exercise every
  code path of the cache without downloading a model.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from .config import Settings
from .normalizer import normalize_prompt


class Embedder(Protocol):
    dim: int

    def encode(self, text: str) -> np.ndarray:
        """Return an L2-normalized vector for ``text``."""
        ...


class HashEmbedder:
    """Deterministic bag-of-token-hashes embedder (no external model)."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = text.split()
        if not tokens:
            return vec
        for token in tokens:
            digest = hashlib.sha256(token.encode()).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec


class SentenceTransformerEmbedder:
    """all-MiniLM-L6-v2 sentence embeddings (lazy-loaded)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, text: str) -> np.ndarray:
        return np.asarray(
            self._model.encode(text, normalize_embeddings=True), dtype=np.float32
        )


def build_embedder(settings: Settings) -> Embedder:
    if settings.use_hash_embedder:
        return HashEmbedder(settings.embedding_dim)
    try:
        return SentenceTransformerEmbedder(settings.embedding_model)
    except ImportError:
        return HashEmbedder(settings.embedding_dim)


def embed_cache_key(
    embedder: Embedder,
    prompt: str,
    context_turns: list[str] | None,
    settings: Settings,
) -> np.ndarray:
    """Weighted combination of the current prompt and recent conversation turns.

    With the context-aware feature flag off (default until the ablation study),
    this is a plain prompt embedding — matching the single-prompt baseline.
    """
    normalized = normalize_prompt(prompt)
    prompt_vec = embedder.encode(normalized)

    if not context_turns or not settings.context_aware_enabled:
        return prompt_vec

    context_summary = " | ".join(context_turns[-settings.context_window_turns :])
    context_vec = embedder.encode(context_summary)
    combined = settings.prompt_weight * prompt_vec + settings.context_weight * context_vec
    norm = float(np.linalg.norm(combined))
    return combined / norm if norm > 0 else combined

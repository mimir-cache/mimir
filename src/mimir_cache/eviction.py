"""Learning-based eviction with semantic importance scoring — Contribution 2.

Replaces LRU/TTL with a weighted score over recency, frequency, regeneration
cost, semantic uniqueness, and predicted future access probability. Entries with
the LOWEST score are evicted first. Weights are configurable so the ablation
study can compare learned weights against LRU (alpha=1, rest=0) and LFU
(beta=1, rest=0) baselines on the same code path.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from .store import CacheEntry

TOKEN_COST_NORMALIZER = 2_000.0  # tokens at which regeneration cost saturates to 1.0
RECENCY_HALF_LIFE_SECONDS = 6 * 3600.0


@dataclass(frozen=True)
class EvictionWeights:
    alpha: float = 0.25  # recency
    beta: float = 0.25  # access frequency
    gamma: float = 0.20  # token cost to regenerate
    delta: float = 0.15  # semantic uniqueness
    epsilon: float = 0.15  # predicted future access probability

    @classmethod
    def lru_baseline(cls) -> EvictionWeights:
        return cls(alpha=1.0, beta=0.0, gamma=0.0, delta=0.0, epsilon=0.0)

    @classmethod
    def lfu_baseline(cls) -> EvictionWeights:
        return cls(alpha=0.0, beta=1.0, gamma=0.0, delta=0.0, epsilon=0.0)


def recency_score(last_accessed_at: float, now: float | None = None) -> float:
    """Exponential time decay in (0, 1]; 1.0 == accessed just now."""
    now = now if now is not None else time.time()
    elapsed = max(now - last_accessed_at, 0.0)
    return math.exp(-elapsed * math.log(2) / RECENCY_HALF_LIFE_SECONDS)


def compute_eviction_score(entry: CacheEntry, weights: EvictionWeights) -> float:
    recency = recency_score(entry.last_accessed_at)
    frequency = min(entry.access_count / (entry.age_seconds / 3600.0 + 1.0), 1.0)
    token_cost = min(entry.token_count / TOKEN_COST_NORMALIZER, 1.0)
    uniqueness = entry.semantic_uniqueness
    future_access = entry.predicted_future_access_prob

    return (
        weights.alpha * recency
        + weights.beta * frequency
        + weights.gamma * token_cost
        + weights.delta * uniqueness
        + weights.epsilon * future_access
    )


def select_eviction_victims(
    entries: list[CacheEntry], evict_count: int, weights: EvictionWeights
) -> list[CacheEntry]:
    """Return the ``evict_count`` lowest-scoring entries (expired ones first)."""
    if evict_count <= 0:
        return []
    expired = [e for e in entries if e.is_expired]
    if len(expired) >= evict_count:
        return expired[:evict_count]

    live = [e for e in entries if not e.is_expired]
    scored = sorted(live, key=lambda e: compute_eviction_score(e, weights))
    remaining = evict_count - len(expired)
    return expired + scored[:remaining]

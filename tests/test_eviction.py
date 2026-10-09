import time

import numpy as np

from mimir_cache.eviction import (
    EvictionWeights,
    compute_eviction_score,
    recency_score,
    select_eviction_victims,
)
from mimir_cache.store import CacheEntry


def make_entry(key: str, **overrides) -> CacheEntry:
    defaults = dict(
        cache_key=key,
        tenant_id="t1",
        normalized_prompt=f"prompt {key}",
        response_text="response",
        embedding=np.zeros(4, dtype=np.float32),
        token_count=100,
    )
    defaults.update(overrides)
    return CacheEntry(**defaults)


def test_recency_score_decays_over_time():
    now = time.time()
    assert recency_score(now, now) == 1.0
    assert recency_score(now - 6 * 3600, now) < 0.51
    assert recency_score(now - 48 * 3600, now) < 0.01


def test_higher_token_cost_scores_higher():
    weights = EvictionWeights()
    cheap = make_entry("cheap", token_count=10)
    expensive = make_entry("expensive", token_count=1900)

    assert compute_eviction_score(expensive, weights) > compute_eviction_score(cheap, weights)


def test_lowest_scoring_entries_are_evicted_first():
    weights = EvictionWeights()
    now = time.time()
    stale = make_entry("stale", token_count=10)
    stale.last_accessed_at = now - 7 * 24 * 3600
    stale.semantic_uniqueness = 0.0
    stale.predicted_future_access_prob = 0.0
    hot = make_entry("hot", token_count=1900)
    hot.access_count = 50
    hot.semantic_uniqueness = 0.9
    hot.predicted_future_access_prob = 0.9

    victims = select_eviction_victims([hot, stale], evict_count=1, weights=weights)

    assert [v.cache_key for v in victims] == ["stale"]


def test_expired_entries_are_evicted_before_scoring():
    weights = EvictionWeights()
    expired = make_entry("expired", ttl_expires_at=time.time() - 10)
    fresh = make_entry("fresh")
    fresh.last_accessed_at = time.time() - 30 * 24 * 3600  # ancient but not expired

    victims = select_eviction_victims([fresh, expired], evict_count=1, weights=weights)

    assert [v.cache_key for v in victims] == ["expired"]


def test_lru_baseline_orders_by_recency_only():
    weights = EvictionWeights.lru_baseline()
    now = time.time()
    old = make_entry("old", token_count=2000)
    old.last_accessed_at = now - 24 * 3600
    recent = make_entry("recent", token_count=1)
    recent.last_accessed_at = now

    victims = select_eviction_victims([recent, old], evict_count=1, weights=weights)

    assert [v.cache_key for v in victims] == ["old"]

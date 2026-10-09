"""Similarity decision logic — Steps 3 and 4 of the layered fallback flow."""

from __future__ import annotations

from enum import StrEnum


class SimilarityVerdict(StrEnum):
    COLLISION_SUSPECT = "collision_suspect"  # score >= ceiling — suspiciously perfect
    SEMANTIC_HIT = "semantic_hit"  # serve cached response
    LOW_CONFIDENCE = "low_confidence"  # ambiguous zone — call LLM, don't serve cache
    CACHE_MISS = "cache_miss"  # call LLM


def evaluate_similarity(
    score: float,
    *,
    semantic_hit_threshold: float,
    ambiguous_zone_low: float,
    collision_ceiling: float,
) -> SimilarityVerdict:
    if score >= collision_ceiling:
        return SimilarityVerdict.COLLISION_SUSPECT
    if score >= semantic_hit_threshold:
        return SimilarityVerdict.SEMANTIC_HIT
    if score >= ambiguous_zone_low:
        return SimilarityVerdict.LOW_CONFIDENCE
    return SimilarityVerdict.CACHE_MISS

import pytest

from mimir_cache.decision import SimilarityVerdict, evaluate_similarity

THRESHOLDS = {
    "semantic_hit_threshold": 0.92,
    "ambiguous_zone_low": 0.80,
    "collision_ceiling": 0.999,
}


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (1.0, SimilarityVerdict.COLLISION_SUSPECT),
        (0.9995, SimilarityVerdict.COLLISION_SUSPECT),
        (0.998, SimilarityVerdict.SEMANTIC_HIT),
        (0.92, SimilarityVerdict.SEMANTIC_HIT),
        (0.9199, SimilarityVerdict.LOW_CONFIDENCE),
        (0.80, SimilarityVerdict.LOW_CONFIDENCE),
        (0.7999, SimilarityVerdict.CACHE_MISS),
        (0.0, SimilarityVerdict.CACHE_MISS),
        (-0.3, SimilarityVerdict.CACHE_MISS),
    ],
)
def test_should_map_score_to_verdict(score, expected):
    assert evaluate_similarity(score, **THRESHOLDS) is expected

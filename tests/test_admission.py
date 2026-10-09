import numpy as np

from mimir_cache.admission import (
    AdmissionFeatures,
    HeuristicAdmissionPolicy,
    extract_features,
)
from mimir_cache.embedding import HashEmbedder
from mimir_cache.vector_index import TenantVectorIndex


def make_features(**overrides) -> AdmissionFeatures:
    defaults = dict(
        prompt_word_count=8,
        cluster_density=0.0,
        query_frequency=0.0,
        hour_of_day=0.5,
        tenant_hit_rate=0.0,
        response_token_count=100,
    )
    defaults.update(overrides)
    return AdmissionFeatures(**defaults)


def test_dense_cluster_increases_reuse_probability():
    policy = HeuristicAdmissionPolicy()
    sparse = policy.predict_reuse_probability(make_features(cluster_density=0.0))
    dense = policy.predict_reuse_probability(make_features(cluster_density=0.9))

    assert dense > sparse


def test_very_long_prompts_are_penalized():
    policy = HeuristicAdmissionPolicy()
    short = policy.predict_reuse_probability(make_features(prompt_word_count=8))
    long = policy.predict_reuse_probability(make_features(prompt_word_count=150))

    assert long < short


def test_probability_stays_in_unit_interval():
    policy = HeuristicAdmissionPolicy()
    high = policy.predict_reuse_probability(
        make_features(cluster_density=1.0, query_frequency=100.0, tenant_hit_rate=1.0)
    )
    low = policy.predict_reuse_probability(make_features(prompt_word_count=500))

    assert 0.0 <= low <= high <= 1.0


def test_should_admit_respects_threshold():
    policy = HeuristicAdmissionPolicy()
    features = make_features(cluster_density=0.9)
    prob = policy.predict_reuse_probability(features)

    assert policy.should_admit(features, threshold=prob - 0.01)
    assert not policy.should_admit(features, threshold=prob + 0.01)


def test_extract_features_uses_neighbour_density():
    embedder = HashEmbedder(dim=128)
    index = TenantVectorIndex("t1", dim=128)
    vec = embedder.encode("how do i reset my password")
    index.add(embedder.encode("how do i reset my password please"), "k1")

    features = extract_features(
        prompt="how do i reset my password",
        embedding=vec,
        index=index,
        tenant_hit_rate=0.5,
        query_frequency=2.0,
        hour_of_day=14,
        response_token_count=120,
    )

    assert features.cluster_density > 0.0
    assert features.prompt_word_count == 6
    assert 0.0 <= features.hour_of_day <= 1.0
    features_vector = features.to_vector()
    assert features_vector.shape == (6,)
    assert features_vector.dtype == np.float32

import numpy as np

from mimir_cache.embedding import HashEmbedder
from mimir_cache.vector_index import TenantIndexRegistry, TenantVectorIndex


def unit(vec):
    vec = np.asarray(vec, dtype=np.float32)
    return vec / np.linalg.norm(vec)


def test_should_return_exact_vector_with_similarity_one():
    index = TenantVectorIndex("t1", dim=4)
    v = unit([1, 0, 0, 0])
    index.add(v, "key1")

    results = index.search(v, k=1)

    assert results[0].cache_key == "key1"
    assert results[0].similarity == np.float32(1.0)


def test_should_rank_closest_vector_first():
    index = TenantVectorIndex("t1", dim=4)
    index.add(unit([1, 0, 0, 0]), "aligned")
    index.add(unit([0, 1, 0, 0]), "orthogonal")

    results = index.search(unit([0.9, 0.1, 0, 0]), k=2)

    assert results[0].cache_key == "aligned"
    assert results[0].similarity > results[1].similarity


def test_removed_keys_are_excluded_from_results():
    index = TenantVectorIndex("t1", dim=4)
    v = unit([1, 0, 0, 0])
    index.add(v, "key1")
    index.remove("key1")

    assert index.search(v, k=1) == []
    assert index.size == 0


def test_empty_index_returns_no_results():
    index = TenantVectorIndex("t1", dim=4)
    assert index.search(unit([1, 0, 0, 0]), k=5) == []


def test_registry_isolates_tenants():
    registry = TenantIndexRegistry(dim=8)
    embedder = HashEmbedder(dim=8)
    vec = embedder.encode("shared secret prompt")

    registry.for_tenant("tenant_a").add(vec, "a_key")

    # Same vector searched in tenant_b's index must find nothing.
    assert registry.for_tenant("tenant_b").search(vec, k=5) == []
    assert registry.for_tenant("tenant_a").search(vec, k=5)[0].cache_key == "a_key"


def test_hash_embedder_is_deterministic_and_normalized():
    embedder = HashEmbedder(dim=64)
    a = embedder.encode("hello world")
    b = embedder.encode("hello world")

    assert np.allclose(a, b)
    assert abs(float(np.linalg.norm(a)) - 1.0) < 1e-5


def test_hash_embedder_similar_token_overlap_scores_higher():
    embedder = HashEmbedder(dim=256)
    base = embedder.encode("semantic cache for llm applications")
    similar = embedder.encode("semantic cache for llm apps")
    unrelated = embedder.encode("banana smoothie recipe ingredients")

    assert float(base @ similar) > float(base @ unrelated)

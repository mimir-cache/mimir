"""Per-tenant vector index — L2 semantic lookup (Contribution 4: tenant isolation).

Each tenant owns a physically separate index object; a lookup can never touch
another tenant's vectors, making cross-tenant leakage architecturally impossible
rather than policy-blocked.

Uses FAISS (``IndexFlatIP`` — inner product on L2-normalized vectors == cosine)
when installed, otherwise a NumPy brute-force index with identical semantics.
Demo scope is <= 10k entries per tenant, where brute force is fine
(see Risk 3 note in system_architecture.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

import numpy as np

try:
    import faiss  # type: ignore

    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


@dataclass
class SearchResult:
    cache_key: str
    similarity: float


class _NumpyFlatIP:
    """Minimal drop-in for ``faiss.IndexFlatIP`` on normalized vectors."""

    def __init__(self, dim: int):
        self.dim = dim
        self._vectors: list[np.ndarray] = []

    @property
    def ntotal(self) -> int:
        return len(self._vectors)

    def add(self, vecs: np.ndarray) -> None:
        for row in np.atleast_2d(vecs):
            self._vectors.append(np.asarray(row, dtype=np.float32))

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if not self._vectors:
            return np.array([[-1.0] * k]), np.array([[-1] * k])
        matrix = np.stack(self._vectors)
        scores = matrix @ np.atleast_2d(query)[0]
        k = min(k, len(scores))
        top = np.argsort(scores)[::-1][:k]
        return scores[top].reshape(1, -1), top.reshape(1, -1)


class TenantVectorIndex:
    """Semantic index scoped to exactly one tenant."""

    def __init__(self, tenant_id: str, dim: int = 384):
        self.tenant_id = tenant_id
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim) if _HAS_FAISS else _NumpyFlatIP(dim)
        self._id_to_cache_key: dict[int, str] = {}
        self._removed: set[str] = set()
        self._lock = Lock()

    @property
    def size(self) -> int:
        return len(self._id_to_cache_key) - len(self._removed)

    def add(self, embedding: np.ndarray, cache_key: str) -> None:
        with self._lock:
            idx = int(self._index.ntotal)
            self._index.add(np.asarray(embedding, dtype=np.float32).reshape(1, -1))
            self._id_to_cache_key[idx] = cache_key
            self._removed.discard(cache_key)

    def remove(self, cache_key: str) -> None:
        """Tombstone removal — flat indexes do not support in-place deletes."""
        with self._lock:
            self._removed.add(cache_key)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[SearchResult]:
        with self._lock:
            if self._index.ntotal == 0:
                return []
            scores, indices = self._index.search(
                np.asarray(query_embedding, dtype=np.float32).reshape(1, -1), k
            )
            results: list[SearchResult] = []
            for score, idx in zip(scores[0], indices[0], strict=False):
                if idx < 0:
                    continue
                cache_key = self._id_to_cache_key.get(int(idx))
                if cache_key is None or cache_key in self._removed:
                    continue
                results.append(SearchResult(cache_key=cache_key, similarity=float(score)))
            return results


@dataclass
class TenantIndexRegistry:
    """Creates and hands out one isolated index per tenant."""

    dim: int = 384
    _indexes: dict[str, TenantVectorIndex] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def for_tenant(self, tenant_id: str) -> TenantVectorIndex:
        with self._lock:
            if tenant_id not in self._indexes:
                self._indexes[tenant_id] = TenantVectorIndex(tenant_id, self.dim)
            return self._indexes[tenant_id]

    def drop_tenant(self, tenant_id: str) -> None:
        with self._lock:
            self._indexes.pop(tenant_id, None)

"""Cache entry store — response payloads and access statistics.

Entries live in a per-tenant dict keyed by cache key. In the deployed system the
hot copy sits in Redis and the cold archive in PostgreSQL; for the demo scope a
process-local store with the same interface keeps the engine testable. The exact
match (L1) lookup and the vector index (L2) both resolve to keys in this store.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock

import numpy as np


@dataclass
class CacheEntry:
    cache_key: str
    tenant_id: str
    normalized_prompt: str
    response_text: str
    embedding: np.ndarray
    token_count: int = 0
    model_used: str = ""
    context_hash: str = ""
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    semantic_uniqueness: float = 0.5
    admission_score: float = 0.5
    predicted_future_access_prob: float = 0.5
    ttl_expires_at: float | None = None

    @property
    def age_seconds(self) -> float:
        return max(time.time() - self.created_at, 1.0)

    @property
    def is_expired(self) -> bool:
        return self.ttl_expires_at is not None and time.time() >= self.ttl_expires_at


class TenantEntryStore:
    """Thread-safe entry store scoped to one tenant."""

    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self._entries: dict[str, CacheEntry] = {}
        self._lock = Lock()

    def get(self, cache_key: str) -> CacheEntry | None:
        with self._lock:
            entry = self._entries.get(cache_key)
            if entry is None:
                return None
            if entry.is_expired:
                del self._entries[cache_key]
                return None
            return entry

    def touch(self, cache_key: str) -> None:
        with self._lock:
            entry = self._entries.get(cache_key)
            if entry is not None:
                entry.last_accessed_at = time.time()
                entry.access_count += 1

    def put(self, entry: CacheEntry) -> None:
        with self._lock:
            self._entries[entry.cache_key] = entry

    def delete(self, cache_key: str) -> None:
        with self._lock:
            self._entries.pop(cache_key, None)

    def all_entries(self) -> list[CacheEntry]:
        with self._lock:
            return list(self._entries.values())

    def flush(self) -> int:
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            return count

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


class EntryStoreRegistry:
    def __init__(self) -> None:
        self._stores: dict[str, TenantEntryStore] = {}
        self._lock = Lock()

    def for_tenant(self, tenant_id: str) -> TenantEntryStore:
        with self._lock:
            if tenant_id not in self._stores:
                self._stores[tenant_id] = TenantEntryStore(tenant_id)
            return self._stores[tenant_id]

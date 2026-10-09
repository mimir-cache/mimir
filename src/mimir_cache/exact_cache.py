"""L1 exact-match key lookup.

Maps an exact-match key (hash of normalized prompt + tenant + context) to the
cache key of the stored entry. Backed by Redis when ``MIMIR_REDIS_URL`` is set,
otherwise by an in-process LRU dict (the L0 failure fallback from Step 6 of the
layered flow doubles as the dev/test backend).
"""

from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from typing import Protocol


class ExactCache(Protocol):
    def get(self, key: str) -> str | None: ...

    def set(self, key: str, cache_key: str, ttl_seconds: int) -> None: ...

    def delete(self, key: str) -> None: ...

    def flush_namespace(self, prefix: str) -> int: ...


class InMemoryExactCache:
    def __init__(self, max_entries: int = 100_000):
        self._data: OrderedDict[str, str] = OrderedDict()
        self._max_entries = max_entries
        self._lock = Lock()

    def get(self, key: str) -> str | None:
        with self._lock:
            value = self._data.get(key)
            if value is not None:
                self._data.move_to_end(key)
            return value

    def set(self, key: str, cache_key: str, ttl_seconds: int) -> None:
        with self._lock:
            self._data[key] = cache_key
            self._data.move_to_end(key)
            while len(self._data) > self._max_entries:
                self._data.popitem(last=False)

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def flush_namespace(self, prefix: str) -> int:
        with self._lock:
            doomed = [k for k in self._data if k.startswith(prefix)]
            for k in doomed:
                del self._data[k]
            return len(doomed)


class RedisExactCache:
    def __init__(self, redis_url: str):
        import redis

        self._client = redis.Redis.from_url(redis_url, decode_responses=True)

    def get(self, key: str) -> str | None:
        return self._client.get(key)

    def set(self, key: str, cache_key: str, ttl_seconds: int) -> None:
        self._client.set(key, cache_key, ex=ttl_seconds)

    def delete(self, key: str) -> None:
        self._client.delete(key)

    def flush_namespace(self, prefix: str) -> int:
        count = 0
        for key in self._client.scan_iter(match=f"{prefix}*"):
            self._client.delete(key)
            count += 1
        return count


def build_exact_cache(redis_url: str | None) -> ExactCache:
    if redis_url:
        try:
            cache = RedisExactCache(redis_url)
            cache.get("mimir:healthcheck")
            return cache
        except Exception:
            # Step 6 failure fallback: Redis unavailable → in-memory L0
            return InMemoryExactCache()
    return InMemoryExactCache()

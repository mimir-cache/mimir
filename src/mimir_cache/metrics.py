"""Per-tenant metrics — feeds the CMS analytics dashboard.

Counters follow the dashboard formulas in section 8 of system_architecture.md.
Cost model uses simple per-token pricing so token savings translate directly
into the cost-reduction metric shown on the dashboard.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock

USD_PER_MILLION_TOKENS = 0.80  # Claude Haiku input pricing — cost model constant


@dataclass
class CacheEvent:
    tenant_id: str
    event_type: str  # exact_hit | semantic_hit | cache_miss | low_confidence |
    #                  collision_suspect | admission_skip | eviction
    latency_ms: float
    similarity_score: float | None = None
    tokens_used: int = 0
    tokens_saved: int = 0
    session_id: str = ""
    created_at: float = field(default_factory=time.time)


@dataclass
class TenantMetrics:
    total_requests: int = 0
    exact_hits: int = 0
    semantic_hits: int = 0
    cache_misses: int = 0
    low_confidence: int = 0
    collision_alerts: int = 0
    admission_skips: int = 0
    evictions: int = 0
    tokens_used: int = 0
    tokens_saved: int = 0
    latency_sum_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.exact_hits + self.semantic_hits) / self.total_requests

    @property
    def avg_latency_ms(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.latency_sum_ms / self.total_requests

    @property
    def cost_saved_usd(self) -> float:
        return self.tokens_saved * USD_PER_MILLION_TOKENS / 1_000_000

    def snapshot(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "exact_hits": self.exact_hits,
            "semantic_hits": self.semantic_hits,
            "cache_misses": self.cache_misses,
            "low_confidence": self.low_confidence,
            "collision_alerts": self.collision_alerts,
            "admission_skips": self.admission_skips,
            "evictions": self.evictions,
            "hit_rate": round(self.hit_rate, 4),
            "tokens_used": self.tokens_used,
            "tokens_saved": self.tokens_saved,
            "cost_saved_usd": round(self.cost_saved_usd, 6),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
        }


_COUNTER_BY_EVENT = {
    "exact_hit": "exact_hits",
    "semantic_hit": "semantic_hits",
    "cache_miss": "cache_misses",
    "low_confidence": "low_confidence",
    "collision_suspect": "collision_alerts",
    "admission_skip": "admission_skips",
    "eviction": "evictions",
}

# Events that represent an incoming request (evictions/admission skips are side effects).
_REQUEST_EVENTS = {
    "exact_hit",
    "semantic_hit",
    "cache_miss",
    "low_confidence",
    "collision_suspect",
}


class MetricsRegistry:
    def __init__(self, event_buffer_size: int = 2_000):
        self._metrics: dict[str, TenantMetrics] = {}
        self._events: deque[CacheEvent] = deque(maxlen=event_buffer_size)
        self._lock = Lock()

    def record(self, event: CacheEvent) -> None:
        with self._lock:
            metrics = self._metrics.setdefault(event.tenant_id, TenantMetrics())
            counter = _COUNTER_BY_EVENT.get(event.event_type)
            if counter is not None:
                setattr(metrics, counter, getattr(metrics, counter) + 1)
            if event.event_type in _REQUEST_EVENTS:
                metrics.total_requests += 1
                metrics.latency_sum_ms += event.latency_ms
            metrics.tokens_used += event.tokens_used
            metrics.tokens_saved += event.tokens_saved
            self._events.append(event)

    def for_tenant(self, tenant_id: str) -> TenantMetrics:
        with self._lock:
            return self._metrics.setdefault(tenant_id, TenantMetrics())

    def snapshot(self, tenant_id: str) -> dict:
        return self.for_tenant(tenant_id).snapshot()

    def platform_snapshot(self) -> dict[str, dict]:
        with self._lock:
            return {tid: m.snapshot() for tid, m in self._metrics.items()}

    def recent_events(self, tenant_id: str | None = None, limit: int = 100) -> list[CacheEvent]:
        with self._lock:
            events = list(self._events)
        if tenant_id is not None:
            events = [e for e in events if e.tenant_id == tenant_id]
        return events[-limit:][::-1]

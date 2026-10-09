"""Collision attack defense and probe detection — Contribution 5.

Two runtime defenses from section 12 of system_architecture.md:

* Similarity ceiling — a match that is *too* perfect (score >= ceiling) is more
  likely an adversarial near-duplicate than an organic rephrasing; the cache is
  bypassed and the request served directly by the LLM.
* Probe detection — many near-identical queries from one tenant inside a short
  window indicates cache probing; flagged for rate limiting.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class SecurityAlert:
    tenant_id: str
    alert_type: str  # collision_suspect | probe_pattern
    details: dict
    created_at: float = field(default_factory=time.time)


class ProbeDetector:
    """Sliding-window counter of near-identical queries per tenant."""

    def __init__(self, window_seconds: int = 60, threshold: int = 10):
        self.window_seconds = window_seconds
        self.threshold = threshold
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def record_and_check(self, tenant_id: str, normalized_prompt: str) -> bool:
        """Record a query; return True when probing behaviour is detected."""
        key = f"{tenant_id}:{normalized_prompt}"
        now = time.time()
        with self._lock:
            window = self._events[key]
            window.append(now)
            while window and window[0] < now - self.window_seconds:
                window.popleft()
            return len(window) >= self.threshold


class SecurityMonitor:
    """Aggregates alerts for the dashboard feed."""

    def __init__(self, probe_window_seconds: int = 60, probe_threshold: int = 10):
        self.probe_detector = ProbeDetector(probe_window_seconds, probe_threshold)
        self._alerts: deque[SecurityAlert] = deque(maxlen=500)
        self._lock = Lock()

    def record_collision_suspect(
        self, tenant_id: str, prompt: str, matched_prompt: str, score: float
    ) -> SecurityAlert:
        alert = SecurityAlert(
            tenant_id=tenant_id,
            alert_type="collision_suspect",
            details={"query": prompt, "matched": matched_prompt, "score": round(score, 6)},
        )
        with self._lock:
            self._alerts.append(alert)
        return alert

    def check_probe(self, tenant_id: str, normalized_prompt: str) -> SecurityAlert | None:
        if not self.probe_detector.record_and_check(tenant_id, normalized_prompt):
            return None
        alert = SecurityAlert(
            tenant_id=tenant_id,
            alert_type="probe_pattern",
            details={
                "normalized_prompt": normalized_prompt,
                "window_seconds": self.probe_detector.window_seconds,
                "threshold": self.probe_detector.threshold,
            },
        )
        with self._lock:
            self._alerts.append(alert)
        return alert

    def recent_alerts(self, tenant_id: str | None = None, limit: int = 50) -> list[SecurityAlert]:
        with self._lock:
            alerts = list(self._alerts)
        if tenant_id is not None:
            alerts = [a for a in alerts if a.tenant_id == tenant_id]
        return alerts[-limit:][::-1]

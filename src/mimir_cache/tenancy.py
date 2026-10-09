"""Multi-tenant registry and API-key authentication — Contribution 4.

Every request authenticates with a tenant API key; all cache operations are then
scoped to that tenant's namespace. Keys are stored hashed (SHA-256), never in
plaintext. The three demo tenants from section 7 (Risk 6) are seeded on startup
with keys taken from environment variables or generated and logged once.
"""

from __future__ import annotations

import hashlib
import os
import secrets
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class TenantConfig:
    tenant_id: str
    name: str
    similarity_threshold: float = 0.92
    ambiguous_zone_low: float = 0.80
    collision_ceiling: float = 0.999
    ttl_seconds: int = 86_400
    max_cache_entries: int = 10_000
    admission_threshold: float = 0.50
    context_window_turns: int = 3


def hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


@dataclass
class TenantRegistry:
    _tenants: dict[str, TenantConfig] = field(default_factory=dict)
    _key_hash_to_tenant: dict[str, str] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def register(self, config: TenantConfig, api_key: str) -> None:
        with self._lock:
            self._tenants[config.tenant_id] = config
            self._key_hash_to_tenant[hash_api_key(api_key)] = config.tenant_id

    def authenticate(self, api_key: str) -> TenantConfig | None:
        with self._lock:
            tenant_id = self._key_hash_to_tenant.get(hash_api_key(api_key))
            if tenant_id is None:
                return None
            return self._tenants.get(tenant_id)

    def get(self, tenant_id: str) -> TenantConfig | None:
        with self._lock:
            return self._tenants.get(tenant_id)

    def all_tenants(self) -> list[TenantConfig]:
        with self._lock:
            return list(self._tenants.values())

    def update(self, tenant_id: str, **overrides) -> TenantConfig | None:
        with self._lock:
            config = self._tenants.get(tenant_id)
            if config is None:
                return None
            for key, value in overrides.items():
                if value is not None and hasattr(config, key):
                    setattr(config, key, value)
            return config


DEMO_TENANTS = (
    ("tenant_a", "Customer Service QA"),
    ("tenant_b", "Coding Assistant"),
    ("tenant_c", "General Knowledge QA"),
)


def seed_demo_tenants(registry: TenantRegistry) -> dict[str, str]:
    """Register the three demo tenants; returns tenant_id -> api_key.

    Keys come from MIMIR_API_KEY_TENANT_A/B/C when set (stable across restarts),
    otherwise they are generated per process.
    """
    issued: dict[str, str] = {}
    for tenant_id, name in DEMO_TENANTS:
        env_var = f"MIMIR_API_KEY_{tenant_id.upper()}"
        api_key = os.environ.get(env_var) or f"mimir_{tenant_id}_{secrets.token_hex(16)}"
        registry.register(TenantConfig(tenant_id=tenant_id, name=name), api_key)
        issued[tenant_id] = api_key
    return issued

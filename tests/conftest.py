import pytest
from fastapi.testclient import TestClient

from mimir_cache.config import Settings
from mimir_cache.main import create_app

TEST_KEYS = {
    "tenant_a": "test_key_tenant_a",
    "tenant_b": "test_key_tenant_b",
    "tenant_c": "test_key_tenant_c",
}
ADMIN_KEY = "test_admin_key"


@pytest.fixture
def settings() -> Settings:
    return Settings(
        use_hash_embedder=True,
        redis_url=None,
        anthropic_api_key=None,
        api_keys_seed=True,
        probe_threshold=5,
        probe_window_seconds=60,
    )


@pytest.fixture
def client(settings, monkeypatch) -> TestClient:
    for tenant_id, key in TEST_KEYS.items():
        monkeypatch.setenv(f"MIMIR_API_KEY_{tenant_id.upper()}", key)
    monkeypatch.setenv("MIMIR_ADMIN_API_KEY", ADMIN_KEY)
    app = create_app(settings)
    return TestClient(app)


def auth(tenant_id: str = "tenant_a") -> dict:
    return {"Authorization": f"Bearer {TEST_KEYS[tenant_id]}"}


def admin_auth() -> dict:
    return {"Authorization": f"Bearer {ADMIN_KEY}"}

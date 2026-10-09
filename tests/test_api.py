"""End-to-end API tests exercising the full layered flow via HTTP."""

from tests.conftest import admin_auth, auth


def query(client, prompt: str, tenant: str = "tenant_a", **extra):
    payload = {"prompt": prompt, "session_id": "s1", **extra}
    response = client.post("/v1/cache/query", json=payload, headers=auth(tenant))
    assert response.status_code == 200, response.text
    return response.json()


def test_healthz(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_rejects_missing_and_invalid_api_keys(client):
    assert client.post("/v1/cache/query", json={"prompt": "x"}).status_code == 401
    assert (
        client.post(
            "/v1/cache/query",
            json={"prompt": "x"},
            headers={"Authorization": "Bearer wrong_key"},
        ).status_code
        == 403
    )


def test_first_query_is_a_miss_then_exact_hit(client):
    first = query(client, "What is semantic caching?")
    assert first["cache_status"] == "cache_miss"
    assert first["admitted_to_cache"] is True

    second = query(client, "What is semantic caching?")
    assert second["cache_status"] == "exact_hit"
    assert second["tokens_saved"] > 0
    assert second["response"] == first["response"]


def test_normalized_variants_share_the_exact_match_key(client):
    query(client, "Tell me about vector databases")
    variant = query(client, "  tell me about VECTOR databases  ")
    assert variant["cache_status"] == "exact_hit"


def test_tenants_are_isolated(client):
    query(client, "What is my account balance?", tenant="tenant_a")

    other = query(client, "What is my account balance?", tenant="tenant_b")
    assert other["cache_status"] == "cache_miss"


def test_stats_reflect_traffic(client):
    query(client, "stats probe prompt one")
    query(client, "stats probe prompt one")

    response = client.get("/v1/cache/stats", headers=auth())
    assert response.status_code == 200
    stats = response.json()
    assert stats["tenant_id"] == "tenant_a"
    assert stats["metrics"]["total_requests"] >= 2
    assert stats["metrics"]["exact_hits"] >= 1
    assert stats["entry_count"] >= 1


def test_entries_listing(client):
    query(client, "list me please")
    response = client.get("/v1/cache/entries", headers=auth())
    assert response.status_code == 200
    entries = response.json()
    assert len(entries) >= 1
    assert {"cache_key", "normalized_prompt", "access_count"} <= entries[0].keys()


def test_invalidate_flush_all(client):
    query(client, "flush target prompt")
    response = client.request(
        "DELETE",
        "/v1/cache/invalidate",
        json={"flush_all": True},
        headers=auth(),
    )
    assert response.status_code == 200
    assert response.json()["invalidated"] >= 1

    after = query(client, "flush target prompt")
    assert after["cache_status"] == "cache_miss"


def test_invalidate_by_pattern(client):
    query(client, "pattern alpha prompt")
    query(client, "unrelated beta prompt")

    response = client.request(
        "DELETE",
        "/v1/cache/invalidate",
        json={"prompt_pattern": "pattern alpha"},
        headers=auth(),
    )
    assert response.json()["invalidated"] == 1

    assert query(client, "pattern alpha prompt")["cache_status"] == "cache_miss"
    assert query(client, "unrelated beta prompt")["cache_status"] == "exact_hit"


def test_admin_endpoints_require_admin_key(client):
    assert client.get("/v1/admin/tenants", headers=auth()).status_code == 403

    response = client.get("/v1/admin/tenants", headers=admin_auth())
    assert response.status_code == 200
    tenant_ids = {t["tenant_id"] for t in response.json()}
    assert {"tenant_a", "tenant_b", "tenant_c"} <= tenant_ids


def test_admin_can_update_tenant_config(client):
    response = client.patch(
        "/v1/admin/tenants/tenant_a",
        json={"similarity_threshold": 0.95, "ttl_seconds": 3600},
        headers=admin_auth(),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["similarity_threshold"] == 0.95
    assert body["ttl_seconds"] == 3600


def test_probe_detection_raises_alert(client):
    for _ in range(6):  # probe_threshold=5 in test settings
        query(client, "identical probing prompt", tenant="tenant_c")

    response = client.get("/v1/cache/alerts", headers=auth("tenant_c"))
    assert response.status_code == 200
    alerts = response.json()
    assert any(a["alert_type"] == "probe_pattern" for a in alerts)


def test_websocket_metrics_feed(client):
    query(client, "warm up metrics")
    from tests.conftest import TEST_KEYS

    with client.websocket_connect(
        f"/ws/metrics/tenant_a?api_key={TEST_KEYS['tenant_a']}"
    ) as ws:
        message = ws.receive_json()
        assert message["tenant_id"] == "tenant_a"
        assert message["metrics"]["total_requests"] >= 1


def test_websocket_rejects_bad_key(client):
    import pytest
    from starlette.websockets import WebSocketDisconnect

    with (
        pytest.raises(WebSocketDisconnect),
        client.websocket_connect("/ws/metrics/tenant_a?api_key=bad") as ws,
    ):
        ws.receive_json()

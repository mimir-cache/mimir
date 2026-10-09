"""HTTP + WebSocket API — section 11 of system_architecture.md."""

from __future__ import annotations

import asyncio
import os

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .engine import CacheEngine
from .schemas import (
    CacheEntrySummary,
    CacheQueryRequest,
    CacheQueryResponse,
    InvalidateRequest,
    InvalidateResponse,
    SecurityAlertOut,
    TenantStats,
    TenantUpdateRequest,
)
from .tenancy import TenantConfig, TenantRegistry

_bearer = HTTPBearer(auto_error=False)


def get_engine(request: Request) -> CacheEngine:
    return request.app.state.engine


def get_registry(request: Request) -> TenantRegistry:
    return request.app.state.tenants


async def get_current_tenant(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> TenantConfig:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing API key")
    tenant = get_registry(request).authenticate(credentials.credentials)
    if tenant is None:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return tenant


async def require_admin(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    admin_key = os.environ.get("MIMIR_ADMIN_API_KEY")
    if not admin_key:
        raise HTTPException(
            status_code=403, detail="Admin API disabled (MIMIR_ADMIN_API_KEY unset)"
        )
    if credentials is None or credentials.credentials != admin_key:
        raise HTTPException(status_code=403, detail="Invalid admin key")


cache_router = APIRouter(prefix="/v1/cache", tags=["cache"])
admin_router = APIRouter(prefix="/v1/admin", tags=["admin"], dependencies=[Depends(require_admin)])
ws_router = APIRouter()


@cache_router.post("/query", response_model=CacheQueryResponse)
async def query_cache(
    body: CacheQueryRequest,
    tenant: TenantConfig = Depends(get_current_tenant),
    engine: CacheEngine = Depends(get_engine),
) -> CacheQueryResponse:
    return await engine.query(body, tenant)


@cache_router.delete("/invalidate", response_model=InvalidateResponse)
async def invalidate(
    body: InvalidateRequest,
    tenant: TenantConfig = Depends(get_current_tenant),
    engine: CacheEngine = Depends(get_engine),
) -> InvalidateResponse:
    count = engine.invalidate(tenant, body.prompt_pattern, body.flush_all)
    return InvalidateResponse(invalidated=count)


@cache_router.get("/entries", response_model=list[CacheEntrySummary])
async def list_entries(
    tenant: TenantConfig = Depends(get_current_tenant),
    engine: CacheEngine = Depends(get_engine),
    limit: int = 100,
) -> list[CacheEntrySummary]:
    entries = engine.stores.for_tenant(tenant.tenant_id).all_entries()
    entries.sort(key=lambda e: e.last_accessed_at, reverse=True)
    return [
        CacheEntrySummary(
            cache_key=e.cache_key,
            normalized_prompt=e.normalized_prompt,
            token_count=e.token_count,
            access_count=e.access_count,
            created_at=e.created_at,
            last_accessed_at=e.last_accessed_at,
            admission_score=e.admission_score,
        )
        for e in entries[:limit]
    ]


@cache_router.get("/stats", response_model=TenantStats)
async def tenant_stats(
    tenant: TenantConfig = Depends(get_current_tenant),
    engine: CacheEngine = Depends(get_engine),
) -> TenantStats:
    return TenantStats(
        tenant_id=tenant.tenant_id,
        entry_count=len(engine.stores.for_tenant(tenant.tenant_id)),
        metrics=engine.metrics.snapshot(tenant.tenant_id),
    )


@cache_router.get("/alerts", response_model=list[SecurityAlertOut])
async def tenant_alerts(
    tenant: TenantConfig = Depends(get_current_tenant),
    engine: CacheEngine = Depends(get_engine),
    limit: int = 50,
) -> list[SecurityAlertOut]:
    return [
        SecurityAlertOut(
            tenant_id=a.tenant_id, alert_type=a.alert_type,
            details=a.details, created_at=a.created_at,
        )
        for a in engine.security.recent_alerts(tenant.tenant_id, limit)
    ]


@admin_router.get("/tenants")
async def list_tenants(request: Request) -> list[dict]:
    return [vars(t) for t in get_registry(request).all_tenants()]


@admin_router.patch("/tenants/{tenant_id}")
async def update_tenant(tenant_id: str, body: TenantUpdateRequest, request: Request) -> dict:
    updated = get_registry(request).update(tenant_id, **body.model_dump(exclude_none=True))
    if updated is None:
        raise HTTPException(status_code=404, detail="Tenant not found")
    return vars(updated)


@admin_router.get("/metrics")
async def platform_metrics(request: Request) -> dict:
    engine: CacheEngine = request.app.state.engine
    return engine.metrics.platform_snapshot()


@admin_router.get("/alerts", response_model=list[SecurityAlertOut])
async def all_alerts(request: Request, limit: int = 100) -> list[SecurityAlertOut]:
    engine: CacheEngine = request.app.state.engine
    return [
        SecurityAlertOut(
            tenant_id=a.tenant_id, alert_type=a.alert_type,
            details=a.details, created_at=a.created_at,
        )
        for a in engine.security.recent_alerts(None, limit)
    ]


@ws_router.websocket("/ws/metrics/{tenant_id}")
async def metrics_feed(websocket: WebSocket, tenant_id: str) -> None:
    """Push tenant metrics every N seconds. Authenticate via ?api_key= query param."""
    engine: CacheEngine = websocket.app.state.engine
    registry: TenantRegistry = websocket.app.state.tenants
    api_key = websocket.query_params.get("api_key", "")
    tenant = registry.authenticate(api_key)
    if tenant is None or tenant.tenant_id != tenant_id:
        await websocket.close(code=4403)
        return
    await websocket.accept()
    interval = websocket.app.state.settings.metrics_push_interval_seconds
    try:
        while True:
            await websocket.send_json({
                "tenant_id": tenant_id,
                "metrics": engine.metrics.snapshot(tenant_id),
                "entry_count": len(engine.stores.for_tenant(tenant_id)),
            })
            await asyncio.sleep(interval)
    except WebSocketDisconnect:
        return

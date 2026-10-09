"""Application factory and uvicorn entry point."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from . import __version__
from .admission import build_admission_policy
from .api import admin_router, cache_router, ws_router
from .config import Settings, get_settings
from .embedding import build_embedder
from .engine import CacheEngine
from .exact_cache import build_exact_cache
from .llm import build_llm_client
from .metrics import MetricsRegistry
from .security import SecurityMonitor
from .store import EntryStoreRegistry
from .tenancy import TenantRegistry, seed_demo_tenants
from .vector_index import TenantIndexRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mimir")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    embedder = build_embedder(settings)
    engine = CacheEngine(
        settings=settings,
        embedder=embedder,
        exact_cache=build_exact_cache(settings.redis_url),
        indexes=TenantIndexRegistry(dim=embedder.dim),
        stores=EntryStoreRegistry(),
        metrics=MetricsRegistry(),
        security=SecurityMonitor(settings.probe_window_seconds, settings.probe_threshold),
        llm=build_llm_client(
            settings.anthropic_api_key,
            settings.llm_model,
            settings.llm_max_tokens,
            settings.openrouter_api_key,
            settings.openrouter_model,
            settings.ollama_model,
            settings.ollama_base_url,
        ),
        admission=build_admission_policy(),
    )

    tenants = TenantRegistry()
    if settings.api_keys_seed:
        issued = seed_demo_tenants(tenants)
        for tenant_id, key in issued.items():
            logger.info("Seeded tenant %s with API key %s", tenant_id, key)

    app = FastAPI(
        title="MIMIR — Adaptive Predictive Semantic Cache",
        version=__version__,
        description=(
            "Middleware between AI web applications and LLM APIs. "
            "Serves semantically cached responses, applies ML-based admission and "
            "learning-based eviction, and enforces per-tenant isolation."
        ),
    )
    app.state.settings = settings
    app.state.engine = engine
    app.state.tenants = tenants

    app.include_router(cache_router)
    app.include_router(admin_router)
    app.include_router(ws_router)

    @app.get("/healthz", tags=["health"])
    async def healthz() -> dict:
        return {"status": "ok", "version": __version__}

    return app


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run("mimir_cache.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    run()

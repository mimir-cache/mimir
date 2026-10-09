"""Application settings.

All thresholds mirror section 3 of ``system_architecture.md`` (Similarity Threshold
Configuration). Every value can be overridden per tenant (see ``tenancy.TenantConfig``)
or globally via environment variables prefixed with ``MIMIR_``.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MIMIR_", env_file=".env", extra="ignore")

    # --- Similarity decision thresholds ---
    semantic_hit_threshold: float = 0.92
    ambiguous_zone_low: float = 0.80
    collision_ceiling: float = 0.999

    # --- Context-aware cache keys (Contribution 3) ---
    context_aware_enabled: bool = False  # feature flag (Risk 4) — enable for ablation study
    context_window_turns: int = 3
    prompt_weight: float = 0.70
    context_weight: float = 0.30

    # --- Admission / eviction (Contributions 1 & 2) ---
    admission_threshold: float = 0.50
    max_cache_entries_per_tenant: int = 10_000
    eviction_batch_fraction: float = 0.10  # evict 10% of entries when the cap is reached
    ttl_seconds: int = 86_400

    # --- Eviction score weights (learned; defaults are uniform priors) ---
    eviction_alpha: float = 0.25  # recency
    eviction_beta: float = 0.25  # access frequency
    eviction_gamma: float = 0.20  # token cost to regenerate
    eviction_delta: float = 0.15  # semantic uniqueness
    eviction_epsilon: float = 0.15  # predicted future access probability

    # --- Security (Contribution 5) ---
    probe_window_seconds: int = 60
    probe_threshold: int = 10  # near-identical queries within window → probe alert

    # --- Embeddings ---
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    use_hash_embedder: bool = False  # force deterministic fallback embedder (tests/CI)

    # --- Infrastructure ---
    redis_url: str | None = None  # e.g. redis://localhost:6379/0 — in-memory L0 when unset
    # Real LLM calls are opt-in only: both must be set, otherwise the echo stub is used.
    anthropic_api_key: str | None = None
    llm_model: str = ""
    llm_max_tokens: int = 1024
    # Alternative real-LLM backend via OpenRouter (OpenAI-compatible API). Also opt-in only.
    openrouter_api_key: str | None = None
    openrouter_model: str = ""
    # Alternative real-LLM backend via a local Ollama server. No key needed, opt-in via model name.
    ollama_model: str = ""
    ollama_base_url: str = "http://localhost:11434"

    # --- API ---
    api_keys_seed: bool = True  # seed the 3 demo tenants (tenant_a/b/c) on startup
    metrics_push_interval_seconds: float = 5.0


@lru_cache
def get_settings() -> Settings:
    return Settings()

"""API request/response models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CacheQueryRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=32_000)
    session_id: str = ""
    context_turns: list[str] = Field(default_factory=list, max_length=20)
    model: str = "claude-haiku"
    metadata: dict = Field(default_factory=dict)


class CacheQueryResponse(BaseModel):
    response: str
    cache_status: str  # exact_hit | semantic_hit | cache_miss | low_confidence | collision_suspect
    similarity_score: float | None = None
    latency_ms: float
    tokens_used: int = 0
    tokens_saved: int = 0
    admitted_to_cache: bool | None = None
    cache_key: str | None = None


class InvalidateRequest(BaseModel):
    prompt_pattern: str | None = None  # substring match on normalized prompt
    flush_all: bool = False


class InvalidateResponse(BaseModel):
    invalidated: int


class CacheEntrySummary(BaseModel):
    cache_key: str
    normalized_prompt: str
    token_count: int
    access_count: int
    created_at: float
    last_accessed_at: float
    admission_score: float


class TenantStats(BaseModel):
    tenant_id: str
    entry_count: int
    metrics: dict


class TenantUpdateRequest(BaseModel):
    similarity_threshold: float | None = Field(default=None, ge=0.5, le=1.0)
    ttl_seconds: int | None = Field(default=None, ge=60)
    max_cache_entries: int | None = Field(default=None, ge=10)
    admission_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    collision_ceiling: float | None = Field(default=None, ge=0.9, le=1.0)
    context_window_turns: int | None = Field(default=None, ge=0, le=10)


class SecurityAlertOut(BaseModel):
    tenant_id: str
    alert_type: str
    details: dict
    created_at: float

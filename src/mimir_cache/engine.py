"""Cache engine — orchestrates the layered fallback flow (section 3).

Per request:
  1. normalize prompt
  2. L1 exact key lookup
  3. L2 semantic vector search in the tenant's isolated index
  4. collision check / confidence check on the best match
  5. on miss: LLM call, then ML admission decision before storing
  6. eviction sweep when the tenant's entry cap is reached
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

import numpy as np

from .admission import HeuristicAdmissionPolicy, MLAdmissionPolicy, extract_features
from .config import Settings
from .decision import SimilarityVerdict, evaluate_similarity
from .embedding import Embedder, embed_cache_key
from .eviction import EvictionWeights, select_eviction_victims
from .exact_cache import ExactCache
from .llm import LLMClient
from .metrics import CacheEvent, MetricsRegistry
from .normalizer import context_hash, exact_match_key, normalize_prompt
from .schemas import CacheQueryRequest, CacheQueryResponse
from .security import SecurityMonitor
from .store import CacheEntry, EntryStoreRegistry
from .tenancy import TenantConfig
from .vector_index import TenantIndexRegistry


@dataclass
class CacheEngine:
    settings: Settings
    embedder: Embedder
    exact_cache: ExactCache
    indexes: TenantIndexRegistry
    stores: EntryStoreRegistry
    metrics: MetricsRegistry
    security: SecurityMonitor
    llm: LLMClient
    admission: HeuristicAdmissionPolicy | MLAdmissionPolicy

    async def query(self, request: CacheQueryRequest, tenant: TenantConfig) -> CacheQueryResponse:
        started = time.perf_counter()
        normalized = normalize_prompt(request.prompt)
        ctx_hash = context_hash(request.context_turns or None)

        # Probe detection runs on every request, hit or miss.
        self.security.check_probe(tenant.tenant_id, normalized)

        store = self.stores.for_tenant(tenant.tenant_id)
        index = self.indexes.for_tenant(tenant.tenant_id)

        # --- Step 2: L1 exact match ---
        l1_key = exact_match_key(normalized, tenant.tenant_id, ctx_hash)
        cached_key = self.exact_cache.get(l1_key)
        if cached_key is not None:
            entry = store.get(cached_key)
            if entry is not None:
                store.touch(cached_key)
                return self._respond(
                    request, tenant, "exact_hit", entry.response_text,
                    similarity=1.0, tokens_saved=entry.token_count,
                    cache_key=cached_key, started=started,
                )
            self.exact_cache.delete(l1_key)  # dangling key — entry evicted/expired

        # --- Step 3: L2 semantic similarity search (tenant-scoped) ---
        embedding = embed_cache_key(
            self.embedder, request.prompt, request.context_turns, self.settings
        )
        best_entry: CacheEntry | None = None
        best_score = -1.0
        for result in index.search(embedding, k=5):
            entry = store.get(result.cache_key)
            if entry is None:
                index.remove(result.cache_key)
                continue
            if result.similarity > best_score:
                best_score = result.similarity
                best_entry = entry

        if best_entry is not None:
            verdict = evaluate_similarity(
                best_score,
                semantic_hit_threshold=tenant.similarity_threshold,
                ambiguous_zone_low=tenant.ambiguous_zone_low,
                collision_ceiling=tenant.collision_ceiling,
            )
            is_collision = (
                verdict is SimilarityVerdict.COLLISION_SUSPECT
                and normalized != best_entry.normalized_prompt
            )
            if is_collision:
                # Suspiciously perfect near-duplicate that is not the same prompt:
                # bypass the cache and answer from the LLM directly.
                self.security.record_collision_suspect(
                    tenant.tenant_id, request.prompt, best_entry.normalized_prompt, best_score
                )
                llm_response = await self.llm.complete(request.prompt, request.context_turns)
                return self._respond(
                    request, tenant, "collision_suspect", llm_response.text,
                    similarity=best_score, tokens_used=llm_response.total_tokens,
                    started=started,
                )
            if verdict in (SimilarityVerdict.SEMANTIC_HIT, SimilarityVerdict.COLLISION_SUSPECT):
                store.touch(best_entry.cache_key)
                return self._respond(
                    request, tenant, "semantic_hit", best_entry.response_text,
                    similarity=best_score, tokens_saved=best_entry.token_count,
                    cache_key=best_entry.cache_key, started=started,
                )
            if verdict is SimilarityVerdict.LOW_CONFIDENCE:
                # Step 4: ambiguous zone — answer from the LLM, never the cache.
                llm_response = await self.llm.complete(request.prompt, request.context_turns)
                admitted, cache_key = await self._admit(
                    request, tenant, normalized, ctx_hash, embedding, llm_response.text,
                    llm_response.total_tokens, llm_response.model, best_score,
                )
                return self._respond(
                    request, tenant, "low_confidence", llm_response.text,
                    similarity=best_score, tokens_used=llm_response.total_tokens,
                    admitted=admitted, cache_key=cache_key, started=started,
                )

        # --- Step 5: cache miss → LLM call + admission decision ---
        llm_response = await self.llm.complete(request.prompt, request.context_turns)
        admitted, cache_key = await self._admit(
            request, tenant, normalized, ctx_hash, embedding, llm_response.text,
            llm_response.total_tokens, llm_response.model, best_score,
        )
        return self._respond(
            request, tenant, "cache_miss", llm_response.text,
            similarity=best_score if best_score > 0 else None,
            tokens_used=llm_response.total_tokens,
            admitted=admitted, cache_key=cache_key, started=started,
        )

    async def _admit(
        self,
        request: CacheQueryRequest,
        tenant: TenantConfig,
        normalized: str,
        ctx_hash: str,
        embedding: np.ndarray,
        response_text: str,
        token_count: int,
        model: str,
        best_score: float,
    ) -> tuple[bool, str | None]:
        store = self.stores.for_tenant(tenant.tenant_id)
        index = self.indexes.for_tenant(tenant.tenant_id)
        tenant_metrics = self.metrics.for_tenant(tenant.tenant_id)

        features = extract_features(
            prompt=normalized,
            embedding=embedding,
            index=index,
            tenant_hit_rate=tenant_metrics.hit_rate,
            query_frequency=tenant_metrics.total_requests / (time.time() % 3600 / 3600 + 1),
            hour_of_day=time.localtime().tm_hour,
            response_token_count=token_count,
        )
        reuse_prob = self.admission.predict_reuse_probability(features)
        if reuse_prob < tenant.admission_threshold:
            self.metrics.record(CacheEvent(
                tenant_id=tenant.tenant_id, event_type="admission_skip", latency_ms=0.0,
            ))
            return False, None

        self._evict_if_full(tenant)

        cache_key = f"tenant:{tenant.tenant_id}:entry:{uuid.uuid4().hex}"
        # Semantic uniqueness: distance to the closest existing entry.
        uniqueness = 1.0 - max(best_score, 0.0)
        entry = CacheEntry(
            cache_key=cache_key,
            tenant_id=tenant.tenant_id,
            normalized_prompt=normalized,
            response_text=response_text,
            embedding=embedding,
            token_count=token_count,
            model_used=model,
            context_hash=ctx_hash,
            semantic_uniqueness=uniqueness,
            admission_score=reuse_prob,
            predicted_future_access_prob=reuse_prob,
            ttl_expires_at=time.time() + tenant.ttl_seconds,
        )
        store.put(entry)
        index.add(embedding, cache_key)
        self.exact_cache.set(
            exact_match_key(normalized, tenant.tenant_id, ctx_hash), cache_key, tenant.ttl_seconds
        )
        return True, cache_key

    def _evict_if_full(self, tenant: TenantConfig) -> None:
        store = self.stores.for_tenant(tenant.tenant_id)
        if len(store) < tenant.max_cache_entries:
            return
        index = self.indexes.for_tenant(tenant.tenant_id)
        weights = EvictionWeights(
            alpha=self.settings.eviction_alpha,
            beta=self.settings.eviction_beta,
            gamma=self.settings.eviction_gamma,
            delta=self.settings.eviction_delta,
            epsilon=self.settings.eviction_epsilon,
        )
        evict_count = max(int(tenant.max_cache_entries * self.settings.eviction_batch_fraction), 1)
        victims = select_eviction_victims(store.all_entries(), evict_count, weights)
        for victim in victims:
            store.delete(victim.cache_key)
            index.remove(victim.cache_key)
            self.exact_cache.delete(
                exact_match_key(victim.normalized_prompt, tenant.tenant_id, victim.context_hash)
            )
            self.metrics.record(CacheEvent(
                tenant_id=tenant.tenant_id, event_type="eviction", latency_ms=0.0,
            ))

    def invalidate(self, tenant: TenantConfig, prompt_pattern: str | None, flush_all: bool) -> int:
        store = self.stores.for_tenant(tenant.tenant_id)
        index = self.indexes.for_tenant(tenant.tenant_id)
        if flush_all:
            count = store.flush()
            self.indexes.drop_tenant(tenant.tenant_id)
            self.exact_cache.flush_namespace(f"tenant:{tenant.tenant_id}:")
            return count
        if not prompt_pattern:
            return 0
        pattern = normalize_prompt(prompt_pattern)
        count = 0
        for entry in store.all_entries():
            if pattern in entry.normalized_prompt:
                store.delete(entry.cache_key)
                index.remove(entry.cache_key)
                self.exact_cache.delete(
                    exact_match_key(entry.normalized_prompt, tenant.tenant_id, entry.context_hash)
                )
                count += 1
        return count

    def _respond(
        self,
        request: CacheQueryRequest,
        tenant: TenantConfig,
        status: str,
        response_text: str,
        *,
        similarity: float | None = None,
        tokens_used: int = 0,
        tokens_saved: int = 0,
        admitted: bool | None = None,
        cache_key: str | None = None,
        started: float,
    ) -> CacheQueryResponse:
        latency_ms = (time.perf_counter() - started) * 1000
        self.metrics.record(CacheEvent(
            tenant_id=tenant.tenant_id,
            event_type=status,
            latency_ms=latency_ms,
            similarity_score=similarity,
            tokens_used=tokens_used,
            tokens_saved=tokens_saved,
            session_id=request.session_id,
        ))
        return CacheQueryResponse(
            response=response_text,
            cache_status=status,
            similarity_score=round(similarity, 6) if similarity is not None else None,
            latency_ms=round(latency_ms, 3),
            tokens_used=tokens_used,
            tokens_saved=tokens_saved,
            admitted_to_cache=admitted,
            cache_key=cache_key,
        )

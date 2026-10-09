"""Predictive cache admission — Contribution 1.

Decides, per LLM response, whether the entry is worth caching. Ships with two
policies behind one interface:

* ``HeuristicAdmissionPolicy`` — the cold-start default (Risk 1 mitigation):
  interpretable feature scoring used until enough labelled trace data exists.
* ``MLAdmissionPolicy`` — GradientBoosting classifier trained offline on trace
  replay labels ("accessed again within 7 days"), the InstCache methodology.

Feature extraction is shared so the heuristic phase produces exactly the
feature vectors later used as ML training rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .vector_index import TenantVectorIndex


@dataclass
class AdmissionFeatures:
    prompt_word_count: int
    cluster_density: float  # mean similarity of nearest cached neighbours
    query_frequency: float  # accesses of nearest cluster / hour
    hour_of_day: float  # 0-1 normalized
    tenant_hit_rate: float  # tenant's historical hit rate
    response_token_count: int

    def to_vector(self) -> np.ndarray:
        return np.array(
            [
                self.prompt_word_count,
                self.cluster_density,
                self.query_frequency,
                self.hour_of_day,
                self.tenant_hit_rate,
                self.response_token_count,
            ],
            dtype=np.float32,
        )


def extract_features(
    *,
    prompt: str,
    embedding: np.ndarray,
    index: TenantVectorIndex,
    tenant_hit_rate: float,
    query_frequency: float,
    hour_of_day: int,
    response_token_count: int,
) -> AdmissionFeatures:
    neighbours = index.search(embedding, k=5)
    cluster_density = (
        float(np.mean([n.similarity for n in neighbours])) if neighbours else 0.0
    )
    return AdmissionFeatures(
        prompt_word_count=len(prompt.split()),
        cluster_density=cluster_density,
        query_frequency=query_frequency,
        hour_of_day=hour_of_day / 23.0,
        tenant_hit_rate=tenant_hit_rate,
        response_token_count=response_token_count,
    )


class HeuristicAdmissionPolicy:
    """Interpretable admission scoring for the cold-start phase.

    Rationale per feature (section 6 of system_architecture.md):
    dense clusters and frequent query patterns predict reuse; very long,
    highly specific prompts rarely repeat.
    """

    def predict_reuse_probability(self, features: AdmissionFeatures) -> float:
        score = 0.5
        score += 0.30 * min(features.cluster_density, 1.0)
        score += 0.15 * min(features.query_frequency / 10.0, 1.0)
        score += 0.10 * min(features.tenant_hit_rate, 1.0)
        # Long prompts (> 50 words) are increasingly unlikely to repeat.
        specificity_penalty = min(max(features.prompt_word_count - 50, 0) / 100.0, 0.4)
        score -= specificity_penalty
        return float(np.clip(score, 0.0, 1.0))

    def should_admit(self, features: AdmissionFeatures, threshold: float = 0.5) -> bool:
        return self.predict_reuse_probability(features) >= threshold


class MLAdmissionPolicy:
    """GradientBoosting classifier over the shared feature vector."""

    def __init__(self) -> None:
        from sklearn.ensemble import GradientBoostingClassifier

        self._model = GradientBoostingClassifier(n_estimators=100)
        self._is_trained = False
        self._fallback = HeuristicAdmissionPolicy()

    def train(self, feature_rows: np.ndarray, labels: np.ndarray) -> None:
        self._model.fit(feature_rows, labels)
        self._is_trained = True

    def predict_reuse_probability(self, features: AdmissionFeatures) -> float:
        if not self._is_trained:
            return self._fallback.predict_reuse_probability(features)
        proba = self._model.predict_proba(features.to_vector().reshape(1, -1))
        return float(proba[0][1])

    def should_admit(self, features: AdmissionFeatures, threshold: float = 0.5) -> bool:
        return self.predict_reuse_probability(features) >= threshold


def build_admission_policy() -> HeuristicAdmissionPolicy | MLAdmissionPolicy:
    try:
        return MLAdmissionPolicy()
    except ImportError:
        return HeuristicAdmissionPolicy()

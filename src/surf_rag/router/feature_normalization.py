"""Train-only z-score and identity normalization for router features."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from surf_rag.router.query_features import V1_FEATURE_NAMES

_BINARY_FEATURES: frozenset[str] = frozenset(
    {
        "multi_entity_indicator",
        "bridge_composition_indicator",
        "comparison_indicator",
        "temporal_indicator",
        "numeric_count_indicator",
    }
)


@dataclass
class FeatureNormalizerV1:
    """Fitted on train split only: z-score for continuous, copy for binary."""

    means: Dict[str, float] = field(default_factory=dict)
    stds: Dict[str, float] = field(default_factory=dict)
    version: str = "1"

    def to_json(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "means": dict(self.means),
            "stds": dict(self.stds),
            "z_score_features": [
                n for n in V1_FEATURE_NAMES if n not in _BINARY_FEATURES
            ],
            "identity_features": sorted(_BINARY_FEATURES),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "FeatureNormalizerV1":
        return cls(
            means={k: float(v) for k, v in (data.get("means") or {}).items()},
            stds={k: float(v) for k, v in (data.get("stds") or {}).items()},
            version=str(data.get("version", "1")),
        )


def fit_normalizer_v1(
    feature_rows: Sequence[Mapping[str, float]],
) -> FeatureNormalizerV1:
    """Compute mean and std per continuous feature over ``feature_rows``."""
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    n = max(len(feature_rows), 1)
    for name in V1_FEATURE_NAMES:
        if name in _BINARY_FEATURES:
            continue
        vals = [float(r.get(name, 0.0)) for r in feature_rows]
        m = sum(vals) / max(len(vals), 1)
        if len(vals) <= 1:
            var = 0.0
        else:
            var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
        sd = var**0.5
        if sd < 1e-8:
            sd = 1.0
        means[name] = m
        stds[name] = sd
    return FeatureNormalizerV1(means=means, stds=stds)


def transform_row(
    features: Mapping[str, float],
    normalizer: FeatureNormalizerV1,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name in V1_FEATURE_NAMES:
        x = float(features.get(name, 0.0))
        if name in _BINARY_FEATURES:
            out[name] = x
            continue
        m = float(normalizer.means.get(name, 0.0))
        s = float(normalizer.stds.get(name, 1.0))
        if s < 1e-12:
            s = 1.0
        out[name] = (x - m) / s
    return out


def prefix_raw_norm(
    raw: Mapping[str, float], norm: Mapping[str, float]
) -> Dict[str, float]:
    row: Dict[str, float] = {}
    for k, v in raw.items():
        row[f"feature_raw__{k}"] = float(v)
    for k, v in norm.items():
        row[f"feature_norm__{k}"] = float(v)
    return row

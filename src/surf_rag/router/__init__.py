"""Router package: feature extraction, soft labels, MLP, training (see submodules)."""

from __future__ import annotations

from typing import Any

from surf_rag.router.feature_normalization import (
    FeatureNormalizerV1,
    fit_normalizer_v1,
    transform_row,
)
from surf_rag.router.query_features import (
    FEATURE_SET_VERSION,
    V1_FEATURE_NAMES,
    QueryFeatureContext,
    extract_features_v1,
    feature_vector_ordered,
)
from surf_rag.router.soft_labels import (
    beta_scaled_softmax,
    entropy,
    expected_weight,
    kl_divergence,
    materialize_soft_labels,
    soft_label_from_scores,
)
from surf_rag.router.splits import assign_splits_stratified

__all__ = [
    "beta_scaled_softmax",
    "entropy",
    "expected_weight",
    "kl_divergence",
    "materialize_soft_labels",
    "soft_label_from_scores",
    "FEATURE_SET_VERSION",
    "V1_FEATURE_NAMES",
    "QueryFeatureContext",
    "extract_features_v1",
    "feature_vector_ordered",
    "FeatureNormalizerV1",
    "fit_normalizer_v1",
    "transform_row",
    "assign_splits_stratified",
    "build_router_dataframe",
]


def __getattr__(name: str) -> Any:
    """Lazy import for ``build_router_dataframe`` to avoid import cycles with ``evaluation``."""
    if name == "build_router_dataframe":
        from surf_rag.router.dataset import build_router_dataframe

        return build_router_dataframe
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*__all__, *list(globals().keys())})

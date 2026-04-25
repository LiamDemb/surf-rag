from surf_rag.router.soft_labels import (
    beta_scaled_softmax,
    entropy,
    expected_weight,
    kl_divergence,
    materialize_soft_labels,
    soft_label_from_scores,
)
from surf_rag.router.query_features import (
    FEATURE_SET_VERSION,
    V1_FEATURE_NAMES,
    QueryFeatureContext,
    extract_features_v1,
    feature_vector_ordered,
)
from surf_rag.router.feature_normalization import (
    FeatureNormalizerV1,
    fit_normalizer_v1,
    transform_row,
)
from surf_rag.router.splits import assign_splits_stratified
from surf_rag.router.dataset import build_router_dataframe

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

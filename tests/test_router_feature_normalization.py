"""Tests for router feature normalization."""

from __future__ import annotations

from surf_rag.router.feature_normalization import fit_normalizer_v1, transform_row
from surf_rag.router.query_features import V1_FEATURE_NAMES


def test_binary_unchanged() -> None:
    base = {n: 0.0 for n in V1_FEATURE_NAMES}
    base["multi_entity_indicator"] = 1.0
    base["content_token_len"] = 0.0
    train = [dict(base) for _ in range(2)]
    fit = fit_normalizer_v1(train)
    base["multi_entity_indicator"] = 1.0
    t = transform_row(base, fit)
    assert t["multi_entity_indicator"] == 1.0


def test_json_round_trip() -> None:
    from surf_rag.router.feature_normalization import FeatureNormalizerV1

    f = fit_normalizer_v1([{n: float(i) for n in V1_FEATURE_NAMES} for i in range(3)])
    j = f.to_json()
    g = FeatureNormalizerV1.from_json(j)
    assert g.means == f.means

from __future__ import annotations

import pytest

from surf_rag.router.architectures.polyreg_v1 import expanded_monomial_count
from surf_rag.router.architectures.registry import get_architecture
from surf_rag.router.query_features import V1_FEATURE_NAMES


def test_mlp_v1_rejects_unknown_kwargs() -> None:
    arch = get_architecture("mlp-v1")
    with pytest.raises(ValueError, match="unknown architecture kwargs"):
        arch.validate_kwargs({"bogus": 1})


def test_mlp_v1_validates_dropout_range() -> None:
    arch = get_architecture("mlp-v1")
    with pytest.raises(ValueError, match="dropout"):
        arch.validate_kwargs({"dropout": 1.2})


def test_logreg_v1_rejects_unknown_kwargs() -> None:
    arch = get_architecture("logreg-v1")
    with pytest.raises(ValueError, match="does not accept architecture kwargs"):
        arch.validate_kwargs({"hidden_dim": 8})


def test_logreg_v1_accepts_excluded_features_only() -> None:
    arch = get_architecture("logreg-v1")
    assert arch.validate_kwargs({"excluded_features": ["content_token_len"]}) == {
        "excluded_features": ("content_token_len",)
    }


def test_mlp_v1_normalizes_defaults() -> None:
    arch = get_architecture("mlp-v1")
    out = arch.validate_kwargs({})
    assert out == {
        "embed_proj_dim": 16,
        "feat_proj_dim": 16,
        "hidden_dim": 32,
        "dropout": 0.1,
        "excluded_features": (),
    }


def test_tower_v01_rejects_unknown_kwargs() -> None:
    arch = get_architecture("tower_v01")
    with pytest.raises(ValueError, match="unknown architecture kwargs"):
        arch.validate_kwargs({"extra": 1})


def test_tower_v01_validates_embed_dims_length() -> None:
    arch = get_architecture("tower_v01")
    with pytest.raises(ValueError, match="embed_dims"):
        arch.validate_kwargs({"embed_dims": [128, 64]})


def test_tower_v01_defaults() -> None:
    arch = get_architecture("tower_v01")
    out = arch.validate_kwargs({})
    assert out["feat_hidden"] == 32
    assert out["dropout"] == 0.0
    assert out["embed_dims"] == (128, 64, 32)


def test_polyreg_v1_defaults() -> None:
    arch = get_architecture("polyreg-v1")
    assert arch.validate_kwargs({}) == {
        "degree": 2,
        "max_expanded_features": 150_000,
        "excluded_features": (),
    }


def test_polyreg_v1_degree_and_linear_width() -> None:
    arch = get_architecture("polyreg-v1")
    cfg = arch.build_model_config(4, 14, "query-features", {"degree": 2})
    assert cfg.degree == 2
    model = arch.build_model(cfg)
    n_phi = expanded_monomial_count(14, 2)
    assert model.head.in_features == n_phi
    assert n_phi == 119


def test_polyreg_v1_rejects_degree_too_large() -> None:
    arch = get_architecture("polyreg-v1")
    with pytest.raises(ValueError, match="degree must be"):
        arch.validate_kwargs({"degree": 0})


def test_polyreg_v1_rejects_huge_expansion() -> None:
    arch = get_architecture("polyreg-v1")
    with pytest.raises(ValueError, match="max_expanded"):
        arch.build_model_config(384, 14, "both", {"degree": 3})


def test_polyreg_v1_rejects_unknown_kw() -> None:
    arch = get_architecture("polyreg-v1")
    with pytest.raises(ValueError, match="unknown architecture kwargs"):
        arch.validate_kwargs({"bogus": 1})


def test_polyreg_v1_excluded_raises_when_none_left_for_query_features() -> None:
    arch = get_architecture("polyreg-v1")
    with pytest.raises(ValueError, match="query-features mode requires"):
        arch.build_model_config(
            4,
            14,
            "query-features",
            {"degree": 1, "excluded_features": list(V1_FEATURE_NAMES)},
        )


def test_polyreg_v1_excluded_reduces_polynomial_width() -> None:
    arch = get_architecture("polyreg-v1")
    cfg = arch.build_model_config(
        4,
        14,
        "query-features",
        {"degree": 2, "excluded_features": ["content_token_len"]},
    )
    assert cfg.excluded_features == ("content_token_len",)
    model = arch.build_model(cfg)
    n_kept = 13
    n_phi = expanded_monomial_count(n_kept, 2)
    assert model.head.in_features == n_phi

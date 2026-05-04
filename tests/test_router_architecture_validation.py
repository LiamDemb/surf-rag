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
    cfg = arch.build_model_config(4, 14, "query-features", "regression", {"degree": 2})
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
        arch.build_model_config(384, 14, "both", "regression", {"degree": 3})


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
            "regression",
            {"degree": 1, "excluded_features": list(V1_FEATURE_NAMES)},
        )


def test_mlp_v2_defaults() -> None:
    arch = get_architecture("mlp-v2")
    out = arch.validate_kwargs({})
    assert out["hidden_dim_1"] == 32
    assert out["hidden_dim_2"] == 8
    assert out["dropout_1"] == 0.2
    assert out["dropout_2"] == 0.1
    assert out["activation"] == "gelu"
    assert out["excluded_features"] == ()


def test_mlp_v2_rejects_unknown_kwargs() -> None:
    arch = get_architecture("mlp-v2")
    with pytest.raises(ValueError, match="unknown architecture kwargs"):
        arch.validate_kwargs({"bogus": 1})


def test_mlp_v2_rejects_excluded_features() -> None:
    arch = get_architecture("mlp-v2")
    with pytest.raises(ValueError, match="does not support excluded_features"):
        arch.validate_kwargs({"excluded_features": ["content_token_len"]})


def test_mlp_v2_validates_dropout() -> None:
    arch = get_architecture("mlp-v2")
    with pytest.raises(ValueError, match="dropout_1"):
        arch.validate_kwargs({"dropout_1": 1.0})
    with pytest.raises(ValueError, match="dropout_2"):
        arch.validate_kwargs({"dropout_2": -0.1})


def test_mlp_v2_rejects_non_embedding_input_mode() -> None:
    arch = get_architecture("mlp-v2")
    with pytest.raises(ValueError, match="mlp-v2 only supports input_mode=embedding"):
        arch.build_model_config(256, 14, "both", "regression", {})
    with pytest.raises(ValueError, match="mlp-v2 only supports input_mode=embedding"):
        arch.build_model_config(256, 14, "query-features", "regression", {})


def test_mlp_v2_builds_and_forward() -> None:
    arch = get_architecture("mlp-v2")
    cfg = arch.build_model_config(
        8,
        14,
        "embedding",
        "regression",
        {"hidden_dim_1": 4, "hidden_dim_2": 2, "dropout_1": 0.2, "dropout_2": 0.1},
    )
    m = arch.build_model(cfg)
    import torch

    xe = torch.randn(3, 8)
    xf = torch.zeros(3, 14)
    w = m.predict_weight(xe, xf)
    assert w.shape == (3,)


def test_mlp_v2_classification_head_width() -> None:
    arch = get_architecture("mlp-v2")
    cfg = arch.build_model_config(4, 0, "embedding", "classification", {})
    m = arch.build_model(cfg)
    import torch

    xe = torch.randn(2, 4)
    logits = m.predict_class_logits(xe, torch.zeros(2, 0))
    assert logits.shape == (2, 2)


def test_polyreg_v1_excluded_reduces_polynomial_width() -> None:
    arch = get_architecture("polyreg-v1")
    cfg = arch.build_model_config(
        4,
        14,
        "query-features",
        "regression",
        {"degree": 2, "excluded_features": ["content_token_len"]},
    )
    assert cfg.excluded_features == ("content_token_len",)
    model = arch.build_model(cfg)
    n_kept = 13
    n_phi = expanded_monomial_count(n_kept, 2)
    assert model.head.in_features == n_phi

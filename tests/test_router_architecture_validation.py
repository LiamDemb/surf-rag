from __future__ import annotations

import pytest

from surf_rag.router.architectures.registry import get_architecture


def test_mlp_v1_rejects_unknown_kwargs() -> None:
    arch = get_architecture("mlp-v1")
    with pytest.raises(ValueError, match="unknown architecture kwargs"):
        arch.validate_kwargs({"bogus": 1})


def test_mlp_v1_validates_dropout_range() -> None:
    arch = get_architecture("mlp-v1")
    with pytest.raises(ValueError, match="dropout"):
        arch.validate_kwargs({"dropout": 1.2})


def test_logreg_v1_requires_empty_kwargs() -> None:
    arch = get_architecture("logreg-v1")
    with pytest.raises(ValueError, match="does not accept architecture kwargs"):
        arch.validate_kwargs({"hidden_dim": 8})


def test_mlp_v1_normalizes_defaults() -> None:
    arch = get_architecture("mlp-v1")
    out = arch.validate_kwargs({})
    assert out == {
        "embed_proj_dim": 16,
        "feat_proj_dim": 16,
        "hidden_dim": 32,
        "dropout": 0.1,
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

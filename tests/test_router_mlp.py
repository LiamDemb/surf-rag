"""RouterMLP shape and expected-weight tests."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch

from surf_rag.router.model import (
    RouterMLP,
    RouterMLPConfig,
    expected_dense_weight,
    ROUTER_INPUT_MODE_BOTH,
    ROUTER_INPUT_MODE_EMBEDDING,
    ROUTER_INPUT_MODE_QUERY_FEATURES,
)


def test_forward_shapes_both() -> None:
    cfg = RouterMLPConfig(
        embedding_dim=8,
        feature_dim=4,
        input_mode=ROUTER_INPUT_MODE_BOTH,
        embed_proj_dim=3,
        feat_proj_dim=2,
        hidden_dim=5,
        num_bins=11,
        dropout=0.0,
    )
    m = RouterMLP(cfg)
    b = 3
    e = torch.randn(b, 8)
    f = torch.randn(b, 4)
    logits = m(e, f)
    assert logits.shape == (b, 11)
    d = m.predict_distribution(e, f)
    assert d.shape == (b, 11)
    assert torch.allclose(d.sum(dim=-1), torch.ones(b), atol=1e-5)


def test_expected_weight_uniform() -> None:
    grid = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
    p = torch.tensor([[1.0 / 3, 1.0 / 3, 1.0 / 3]], dtype=torch.float32)
    ev = expected_dense_weight(p, grid)
    assert ev.shape == (1,)
    assert abs(float(ev[0].item() - 0.5) < 0.01)


def test_forward_shapes_query_features_only() -> None:
    cfg = RouterMLPConfig(
        embedding_dim=8,
        feature_dim=4,
        input_mode=ROUTER_INPUT_MODE_QUERY_FEATURES,
        embed_proj_dim=3,
        feat_proj_dim=2,
        hidden_dim=5,
        num_bins=11,
        dropout=0.0,
    )
    m = RouterMLP(cfg)
    b = 2
    e = torch.zeros(b, 8)
    f = torch.randn(b, 4)
    logits = m(e, f)
    assert logits.shape == (b, 11)
    d = m.predict_distribution(e, f)
    assert d.shape == (b, 11)


def test_forward_shapes_embedding_only() -> None:
    cfg = RouterMLPConfig(
        embedding_dim=8,
        feature_dim=4,
        input_mode=ROUTER_INPUT_MODE_EMBEDDING,
        embed_proj_dim=3,
        feat_proj_dim=2,
        hidden_dim=5,
        num_bins=11,
        dropout=0.0,
    )
    m = RouterMLP(cfg)
    b = 2
    e = torch.randn(b, 8)
    f = torch.zeros(b, 4)
    logits = m(e, f)
    assert logits.shape == (b, 11)


def test_config_json_input_mode_roundtrip() -> None:
    cfg = RouterMLPConfig(
        embedding_dim=4,
        feature_dim=2,
        input_mode=ROUTER_INPUT_MODE_QUERY_FEATURES,
        embed_proj_dim=2,
        feat_proj_dim=2,
        hidden_dim=3,
        num_bins=11,
        dropout=0.0,
    )
    d = cfg.to_json()
    assert d["input_mode"] == "query-features"
    cfg2 = RouterMLPConfig.from_json(d)
    assert cfg2.input_mode == ROUTER_INPUT_MODE_QUERY_FEATURES

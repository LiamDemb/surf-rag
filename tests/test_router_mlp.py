"""RouterMLP shape and expected-weight tests."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch

from surf_rag.router.model import (
    RouterMLP,
    RouterMLPConfig,
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
        dropout=0.0,
    )
    m = RouterMLP(cfg)
    b = 3
    e = torch.randn(b, 8)
    f = torch.randn(b, 4)
    logits = m(e, f)
    assert logits.shape == (b, 1)
    w = m.predict_weight(e, f)
    assert w.shape == (b,)
    assert torch.all((w >= 0.0) & (w <= 1.0))


def test_forward_shapes_query_features_only() -> None:
    cfg = RouterMLPConfig(
        embedding_dim=8,
        feature_dim=4,
        input_mode=ROUTER_INPUT_MODE_QUERY_FEATURES,
        embed_proj_dim=3,
        feat_proj_dim=2,
        hidden_dim=5,
        dropout=0.0,
    )
    m = RouterMLP(cfg)
    b = 2
    e = torch.zeros(b, 8)
    f = torch.randn(b, 4)
    logits = m(e, f)
    assert logits.shape == (b, 1)
    w = m.predict_weight(e, f)
    assert w.shape == (b,)


def test_forward_shapes_embedding_only() -> None:
    cfg = RouterMLPConfig(
        embedding_dim=8,
        feature_dim=4,
        input_mode=ROUTER_INPUT_MODE_EMBEDDING,
        embed_proj_dim=3,
        feat_proj_dim=2,
        hidden_dim=5,
        dropout=0.0,
    )
    m = RouterMLP(cfg)
    b = 2
    e = torch.randn(b, 8)
    f = torch.zeros(b, 4)
    logits = m(e, f)
    assert logits.shape == (b, 1)


def test_config_json_input_mode_roundtrip() -> None:
    cfg = RouterMLPConfig(
        embedding_dim=4,
        feature_dim=2,
        input_mode=ROUTER_INPUT_MODE_QUERY_FEATURES,
        embed_proj_dim=2,
        feat_proj_dim=2,
        hidden_dim=3,
        dropout=0.0,
    )
    d = cfg.to_json()
    assert d["input_mode"] == "query-features"
    cfg2 = RouterMLPConfig.from_json(d)
    assert cfg2.input_mode == ROUTER_INPUT_MODE_QUERY_FEATURES

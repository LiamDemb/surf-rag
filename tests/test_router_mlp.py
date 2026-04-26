"""RouterMLP shape and expected-weight tests."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch

from surf_rag.router.model import RouterMLP, RouterMLPConfig, expected_dense_weight


def test_forward_shapes() -> None:
    cfg = RouterMLPConfig(
        embedding_dim=8,
        feature_dim=4,
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

"""Lightweight integration smoke for scalar router inference + policy."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from surf_rag.router.inference import LoadedRouter, predict_batch
from surf_rag.router.model import RouterMLP, RouterMLPConfig
from surf_rag.router.policies import RoutingPolicyName, decide_routing


def test_scalar_predict_and_policy_smoke() -> None:
    cfg = RouterMLPConfig(
        embedding_dim=4,
        feature_dim=2,
        embed_proj_dim=2,
        feat_proj_dim=2,
        hidden_dim=4,
        dropout=0.0,
    )
    model = RouterMLP(cfg)
    router = LoadedRouter(
        model=model,
        config=cfg,
        weight_grid=np.asarray([float(i) / 10.0 for i in range(11)], dtype=np.float32),
        device="cpu",
        manifest={},
    )
    qe = np.ones((1, 4), dtype=np.float32)
    qf = np.zeros((1, 2), dtype=np.float32)
    pred_w = predict_batch(router, qe, qf)
    assert pred_w.shape == (1,)
    assert 0.0 <= float(pred_w[0]) <= 1.0

    soft = decide_routing(
        RoutingPolicyName.LEARNED_SOFT, predicted_weight=float(pred_w[0])
    )
    hard = decide_routing(
        RoutingPolicyName.LEARNED_HARD, predicted_weight=float(pred_w[0])
    )
    assert soft.run_dense and soft.run_graph
    assert hard.hard_branch in ("dense", "graph")

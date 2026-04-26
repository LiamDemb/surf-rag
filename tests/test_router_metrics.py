"""Router metric aggregation."""

from __future__ import annotations

import numpy as np

from surf_rag.router.router_metrics import aggregate_router_metrics


def test_perfect_match_metrics() -> None:
    grid = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    p = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    logp = np.log(p + 1e-12)
    m = aggregate_router_metrics([p], [logp], grid)
    assert m["kl_mean"] < 1e-3
    assert m["argmax_bin_accuracy"] == 1.0
    assert m["expected_weight_mae"] < 1e-5

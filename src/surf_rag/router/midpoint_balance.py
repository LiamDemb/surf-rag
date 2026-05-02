"""Train-only plateau midpoint bucketing and balanced undersampling.

Oracle curves often have tied maxima across neighboring weight bins. We summarize
each row by the midpoint in *weight space* of the contiguous plateau around the
first ``argmax`` (same tie-break as ``oracle_label_from_curve``), using an
additive score tolerance ``epsilon`` on the oracle metric (e.g. NDCG).

Five equal-width buckets on ``[0, 1]`` are ``[0, 0.2)``, …, ``[0.8, 1.0]``
(last interval closed on the right). Undersampling picks ``min_g n_g`` rows
per bucket uniformly at random so each bucket contributes equally to training.
Parquet and dev/test splits are not modified; only the in-memory train frame
used for the training loop is subset.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

RULE_ID = "contiguous_argmax_plateau"
N_MIDPOINT_GROUPS = 5
GROUP_WIDTH = 1.0 / N_MIDPOINT_GROUPS


def plateau_midpoint(
    curve: Sequence[float], grid: Sequence[float], epsilon: float
) -> float:
    """Midpoint weight of the contiguous plateau around the first maximum."""
    scores = np.asarray(list(curve), dtype=np.float64)
    g = np.asarray(list(grid), dtype=np.float64)
    if scores.size == 0 or g.size == 0 or scores.shape != g.shape:
        raise ValueError("oracle_curve and weight_grid must be same-length non-empty")
    max_s = float(scores.max())
    thr = max_s - float(epsilon)
    i0 = int(np.argmax(scores))
    lo = i0
    while lo > 0 and float(scores[lo - 1]) >= thr:
        lo -= 1
    hi = i0
    while hi + 1 < scores.size and float(scores[hi + 1]) >= thr:
        hi += 1
    mid = 0.5 * (float(g[lo]) + float(g[hi]))
    return float(max(0.0, min(1.0, mid)))


def midpoint_to_group(mid: float, *, n_groups: int = N_MIDPOINT_GROUPS) -> int:
    """Map midpoint in ``[0, 1]`` to ``0 .. n_groups-1`` (equal-width bins)."""
    if n_groups < 1:
        raise ValueError("n_groups must be positive")
    width = 1.0 / float(n_groups)
    m = max(0.0, min(1.0, float(mid)))
    if m >= 1.0:
        return n_groups - 1
    if m <= 0.0:
        return 0
    return min(n_groups - 1, int(m / width))


def _row_curve_grid(row: Mapping[str, Any]) -> Tuple[List[float], List[float]]:
    curve = row.get("oracle_curve")
    grid = row.get("weight_grid")
    if curve is None or grid is None:
        return [], []
    if hasattr(curve, "tolist"):
        curve = curve.tolist()
    if hasattr(grid, "tolist"):
        grid = grid.tolist()
    c = [float(x) for x in list(curve)]
    w = [float(x) for x in list(grid)]
    return c, w


def build_train_midpoint_balance_indices(
    df_train_eligible: pd.DataFrame,
    *,
    epsilon: float,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return sorted row indices (into ``df_train_eligible``) to keep and a stats dict."""
    n = len(df_train_eligible)
    if n == 0:
        raise ValueError("df_train_eligible is empty")

    by_group: Dict[int, List[int]] = defaultdict(list)
    groups: List[int] = []
    for pos, (_, row) in enumerate(df_train_eligible.iterrows()):
        c, w = _row_curve_grid(row)
        mid = plateau_midpoint(c, w, epsilon)
        g = midpoint_to_group(mid)
        groups.append(g)
        by_group[g].append(pos)

    counts_before = [int(len(by_group[i])) for i in range(N_MIDPOINT_GROUPS)]
    if any(c == 0 for c in counts_before):
        missing = [i for i, c in enumerate(counts_before) if c == 0]
        raise ValueError(
            "midpoint_balance_masking requires every midpoint group to have at least "
            f"one train row; empty groups (0-based): {missing}. "
            "Disable masking or add data covering all five [0,1] width-0.2 buckets."
        )

    target = min(counts_before)
    rng = np.random.default_rng(int(seed))
    kept: List[int] = []
    for g in range(N_MIDPOINT_GROUPS):
        pool = list(by_group[g])
        rng.shuffle(pool)
        kept.extend(pool[:target])
    kept_arr = np.sort(np.asarray(kept, dtype=np.int64))

    counts_after = [0] * N_MIDPOINT_GROUPS
    for pos in kept_arr.tolist():
        counts_after[int(groups[pos])] += 1

    dropped_per_group = [
        counts_before[i] - counts_after[i] for i in range(N_MIDPOINT_GROUPS)
    ]
    edges = [
        [0.0, 0.2],
        [0.2, 0.4],
        [0.4, 0.6],
        [0.6, 0.8],
        [0.8, 1.0],
    ]
    stats: Dict[str, Any] = {
        "enabled": True,
        "rule": RULE_ID,
        "epsilon": float(epsilon),
        "seed": int(seed),
        "n_groups": N_MIDPOINT_GROUPS,
        "group_edges": edges,
        "counts_before_per_group": counts_before,
        "counts_after_per_group": counts_after,
        "dropped_per_group": dropped_per_group,
        "target_per_group": int(target),
        "train_rows_before": int(n),
        "train_rows_after": int(len(kept_arr)),
    }
    return kept_arr, stats

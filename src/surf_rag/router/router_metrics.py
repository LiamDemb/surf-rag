"""Router metrics for scalar weight predictions and regret."""

from __future__ import annotations

from typing import Dict

import numpy as np

from surf_rag.evaluation.oracle_argmax_intervals import (
    DEFAULT_ARGMAX_INTERVAL_ATOL,
    DEFAULT_ARGMAX_INTERVAL_RTOL,
    dense_weight_argmax_intervals,
    distance_weight_to_argmax_intervals,
)


def _as_float32(x: object) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _interpolate_curve(curve: np.ndarray, w_hat: float) -> float:
    if len(curve) < 2:
        return float(curve[0]) if len(curve) == 1 else 0.0
    w = float(np.clip(w_hat, 0.0, 1.0))
    scaled = w * float(len(curve) - 1)
    lo = int(np.floor(scaled))
    lo = int(np.clip(lo, 0, len(curve) - 2))
    hi = lo + 1
    alpha = scaled - float(lo)
    return float((1.0 - alpha) * curve[lo] + alpha * curve[hi])


def aggregate_router_metrics(
    oracle_curves: np.ndarray,
    predicted_weights: np.ndarray,
    valid_mask: np.ndarray,
    weight_grid: np.ndarray,
) -> Dict[str, float]:
    """Aggregate scalar/regret router metrics.

    Weight-error metrics use **distance to dense-weight argmax intervals** (oracle
    curve plateaus), not a single ``np.argmax`` bin.
    """
    if len(oracle_curves) == 0 or len(predicted_weights) == 0:
        return {}
    curves = _as_float32(oracle_curves)
    preds = _as_float32(predicted_weights).reshape(-1)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    n = min(len(curves), len(preds), len(valid))
    if n == 0:
        return {}
    curves = curves[:n]
    preds = preds[:n]
    valid = valid[:n]
    wg = _as_float32(weight_grid).reshape(-1)

    regrets = []
    dists = []
    oracle_best_scores = []
    qbucket: dict[str, list[float]] = {"q1": [], "q2": [], "q3": [], "q4": []}
    stds = np.std(curves, axis=1)
    if len(stds):
        q1, q2, q3 = np.quantile(stds, [0.25, 0.5, 0.75])
    else:
        q1 = q2 = q3 = 0.0

    for i in range(n):
        if not valid[i]:
            continue
        curve = curves[i].reshape(-1)
        if len(curve) != len(wg):
            continue
        c_star = float(np.max(curve))
        c_hat = _interpolate_curve(curve, float(preds[i]))
        reg = c_star - c_hat
        regrets.append(reg)
        oracle_best_scores.append(c_star)

        intervals = dense_weight_argmax_intervals(
            curve,
            wg,
            rtol=DEFAULT_ARGMAX_INTERVAL_RTOL,
            atol=DEFAULT_ARGMAX_INTERVAL_ATOL,
        )
        d = distance_weight_to_argmax_intervals(float(preds[i]), intervals)
        if np.isfinite(d):
            dists.append(float(d))

        s = float(stds[i])
        if s <= q1:
            qbucket["q1"].append(reg)
        elif s <= q2:
            qbucket["q2"].append(reg)
        elif s <= q3:
            qbucket["q3"].append(reg)
        else:
            qbucket["q4"].append(reg)

    regrets_arr = np.asarray(regrets, dtype=np.float32)
    best_score_arr = np.asarray(oracle_best_scores, dtype=np.float32)
    dist_arr = np.asarray(dists, dtype=np.float64)

    return {
        "mean_regret": float(np.mean(regrets_arr)) if len(regrets_arr) else 0.0,
        "normalized_regret": (
            float(np.mean(regrets_arr)) / max(1e-12, float(np.mean(best_score_arr)))
            if len(best_score_arr)
            else 0.0
        ),
        "argmax_interval_distance_mae": (
            float(np.mean(dist_arr)) if len(dist_arr) else 0.0
        ),
        "argmax_interval_distance_rmse": (
            float(np.sqrt(np.mean(dist_arr**2))) if len(dist_arr) else 0.0
        ),
        "stratified_regret_q1": float(np.mean(qbucket["q1"])) if qbucket["q1"] else 0.0,
        "stratified_regret_q2": float(np.mean(qbucket["q2"])) if qbucket["q2"] else 0.0,
        "stratified_regret_q3": float(np.mean(qbucket["q3"])) if qbucket["q3"] else 0.0,
        "stratified_regret_q4": float(np.mean(qbucket["q4"])) if qbucket["q4"] else 0.0,
        "num_rows": float(len(regrets_arr)),
    }

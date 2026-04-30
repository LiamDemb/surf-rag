"""Router metrics for scalar weight predictions and regret."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


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


def _hard_side_from_weight(w_hat: float, threshold: float = 0.5) -> str:
    if float(w_hat) >= threshold:
        return "dense"
    return "graph"


def aggregate_router_metrics(
    oracle_curves: np.ndarray,
    predicted_weights: np.ndarray,
    valid_mask: np.ndarray,
    weight_grid: np.ndarray,
) -> Dict[str, float]:
    """Aggregate scalar/regret router metrics."""
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
    maes = []
    sqes = []
    hard_match = []
    oracle_best_weights = []
    oracle_best_scores = []
    pred_valid_weights = []
    cls_true = []
    cls_pred = []
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

        best_idx = int(np.argmax(curve))
        oracle_w = float(wg[best_idx])
        oracle_best_weights.append(oracle_w)
        pred_valid_weights.append(float(preds[i]))
        err = float(preds[i] - oracle_w)
        maes.append(abs(err))
        sqes.append(err * err)
        hard_match.append(
            1.0
            if _hard_side_from_weight(float(preds[i]))
            == _hard_side_from_weight(oracle_w)
            else 0.0
        )

        if oracle_w >= 0.6:
            cls_true.append(1)
            cls_pred.append(1 if float(preds[i]) >= 0.5 else 0)
        elif oracle_w <= 0.4:
            cls_true.append(0)
            cls_pred.append(1 if float(preds[i]) >= 0.5 else 0)

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
    best_arr = np.asarray(oracle_best_weights, dtype=np.float32)
    best_score_arr = np.asarray(oracle_best_scores, dtype=np.float32)
    cls_true_arr = np.asarray(cls_true, dtype=np.int64)
    cls_pred_arr = np.asarray(cls_pred, dtype=np.int64)

    precision = recall = f1 = accuracy = 0.0
    tn = fp = fn = tp = 0.0
    if len(cls_true_arr) > 0:
        tp = float(np.sum((cls_true_arr == 1) & (cls_pred_arr == 1)))
        tn = float(np.sum((cls_true_arr == 0) & (cls_pred_arr == 0)))
        fp = float(np.sum((cls_true_arr == 0) & (cls_pred_arr == 1)))
        fn = float(np.sum((cls_true_arr == 1) & (cls_pred_arr == 0)))
        precision = tp / max(1.0, tp + fp)
        recall = tp / max(1.0, tp + fn)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        accuracy = (tp + tn) / max(1.0, tp + tn + fp + fn)

    pearson = 0.0
    r2 = 0.0
    if len(best_arr) > 1:
        p = np.asarray(pred_valid_weights, dtype=np.float32)
        pearson = float(np.corrcoef(p, best_arr)[0, 1])
        ss_res = float(np.sum((p - best_arr) ** 2))
        ss_tot = float(np.sum((best_arr - float(np.mean(best_arr))) ** 2))
        r2 = 1.0 - (ss_res / max(1e-12, ss_tot))

    return {
        "mean_regret": float(np.mean(regrets_arr)) if len(regrets_arr) else 0.0,
        "normalized_regret": (
            float(np.mean(regrets_arr)) / max(1e-12, float(np.mean(best_score_arr)))
            if len(best_score_arr)
            else 0.0
        ),
        "expected_weight_mae": float(np.mean(maes)) if maes else 0.0,
        "expected_weight_rmse": float(np.sqrt(np.mean(sqes))) if sqes else 0.0,
        "pearson_r": pearson,
        "r2": r2,
        "hard_preference_accuracy": float(np.mean(hard_match)) if hard_match else 0.0,
        "classification_accuracy": accuracy,
        "classification_precision_dense": precision,
        "classification_recall_dense": recall,
        "classification_f1_dense": f1,
        "classification_confusion_tn": tn,
        "classification_confusion_fp": fp,
        "classification_confusion_fn": fn,
        "classification_confusion_tp": tp,
        "classification_num_rows": float(len(cls_true_arr)),
        "stratified_regret_q1": float(np.mean(qbucket["q1"])) if qbucket["q1"] else 0.0,
        "stratified_regret_q2": float(np.mean(qbucket["q2"])) if qbucket["q2"] else 0.0,
        "stratified_regret_q3": float(np.mean(qbucket["q3"])) if qbucket["q3"] else 0.0,
        "stratified_regret_q4": float(np.mean(qbucket["q4"])) if qbucket["q4"] else 0.0,
        "num_rows": float(len(regrets_arr)),
    }

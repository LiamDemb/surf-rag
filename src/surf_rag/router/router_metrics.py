"""Router-only metrics vs oracle soft labels (distributions on the weight grid)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

import numpy as np
import torch


def _as_float32(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def kl_p_q(p: np.ndarray, log_q: np.ndarray) -> float:
    """KL(p || q) with p a distribution, log_q = log q (same length)."""
    p = _as_float32(p).reshape(-1)
    log_q = _as_float32(log_q).reshape(-1)
    if len(p) != len(log_q):
        raise ValueError("p and log_q length mismatch")
    q = np.exp(log_q)
    total = 0.0
    for pi, qi, lq in zip(p, q, log_q):
        if pi <= 0.0:
            continue
        if qi <= 0.0:
            return float("inf")
        total += float(pi * (math.log(float(pi)) - float(lq)))
    return total


def cross_entropy_soft(p: np.ndarray, log_q: np.ndarray) -> float:
    """-sum p log q."""
    p = _as_float32(p).reshape(-1)
    log_q = _as_float32(log_q).reshape(-1)
    return float(-np.sum(p * log_q))


def entropy_np(p: np.ndarray) -> float:
    p = _as_float32(p).reshape(-1)
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def expected_value(p: np.ndarray, grid: np.ndarray) -> float:
    p = _as_float32(p).reshape(-1)
    g = _as_float32(grid).reshape(-1)
    return float(np.dot(p, g))


def argmax_bin(p: np.ndarray) -> int:
    p = _as_float32(p).reshape(-1)
    return int(np.argmax(p))


def hard_side(p: np.ndarray, grid: np.ndarray, threshold: float = 0.5) -> str:
    """Return 'dense' if E[w] > threshold, 'graph' if <, 'tie' if ==."""
    ev = expected_value(p, grid)
    if ev > threshold:
        return "dense"
    if ev < threshold:
        return "graph"
    return "tie"


def mass_balance(p: np.ndarray, grid: np.ndarray, threshold: float = 0.5) -> float:
    """Sum of p_i where w_i > threshold minus sum where w_i < threshold."""
    p = _as_float32(p).reshape(-1)
    g = _as_float32(grid).reshape(-1)
    above = float(np.sum(p[g > threshold]))
    below = float(np.sum(p[g < threshold]))
    return above - below


def kl_divergence_torch(log_q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Batch mean of KL(p || softmax(log_q)); p is target distribution."""
    # log_q: (N, C) log-probabilities; p: (N, C)
    return (p * (p.clamp(min=1e-12).log() - log_q)).sum(dim=-1).mean()


def aggregate_router_metrics(
    target_dists: List[np.ndarray],
    pred_log_probs: List[np.ndarray],
    weight_grid: np.ndarray,
) -> Dict[str, float]:
    """Aggregate metrics over a list of per-row distributions and log_softmax predictions."""
    if not target_dists or not pred_log_probs:
        return {}
    kls: List[float] = []
    ces: List[float] = []
    maes: List[float] = []
    rmses: List[float] = []
    hard_match: List[float] = []
    argmax_match: List[float] = []
    within_one: List[float] = []
    target_ent: List[float] = []
    pred_ent: List[float] = []

    n_bins = len(weight_grid)

    for p_t, log_p in zip(target_dists, pred_log_probs):
        p = _as_float32(p_t).reshape(-1)
        lp = _as_float32(log_p).reshape(-1)
        if len(p) != n_bins or len(lp) != n_bins:
            continue
        pred = np.exp(lp)
        kls.append(kl_p_q(p, lp))
        ces.append(cross_entropy_soft(p, lp))
        ev_t = expected_value(p, weight_grid)
        ev_p = expected_value(pred, weight_grid)
        maes.append(abs(ev_t - ev_p))
        rmses.append((ev_t - ev_p) ** 2)
        ht = hard_side(p, weight_grid)
        hp = hard_side(pred, weight_grid)
        if ht == "tie" and hp == "tie":
            hard_match.append(1.0)
        elif ht != "tie" and hp != "tie" and ht == hp:
            hard_match.append(1.0)
        else:
            hard_match.append(0.0)
        ia = argmax_bin(p)
        ip = argmax_bin(pred)
        argmax_match.append(1.0 if ia == ip else 0.0)
        within_one.append(1.0 if abs(ia - ip) <= 1 else 0.0)
        target_ent.append(entropy_np(p))
        pred_ent.append(entropy_np(pred))

    return {
        "kl_mean": float(np.mean(kls)) if kls else 0.0,
        "cross_entropy_mean": float(np.mean(ces)) if ces else 0.0,
        "expected_weight_mae": float(np.mean(maes)) if maes else 0.0,
        "expected_weight_rmse": float(np.sqrt(np.mean(rmses))) if rmses else 0.0,
        "hard_preference_accuracy": float(np.mean(hard_match)) if hard_match else 0.0,
        "argmax_bin_accuracy": float(np.mean(argmax_match)) if argmax_match else 0.0,
        "within_one_bin_accuracy": float(np.mean(within_one)) if within_one else 0.0,
        "target_entropy_mean": float(np.mean(target_ent)) if target_ent else 0.0,
        "predicted_entropy_mean": float(np.mean(pred_ent)) if pred_ent else 0.0,
        "num_rows": float(len(kls)),
    }

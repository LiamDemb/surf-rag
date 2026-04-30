"""Regret/interpolation utilities for scalar router training and evaluation."""

from __future__ import annotations

from typing import Tuple

import torch


def interpolate_curve_torch(
    w_hat: torch.Tensor, oracle_curves: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Interpolate oracle curves at predicted dense weights.

    Returns:
        c_interp: interpolated score at each prediction, shape [B]
        c_star: best oracle score per row, shape [B]
    """
    if w_hat.dim() != 1:
        raise ValueError(f"w_hat must be 1-D [B], got shape {tuple(w_hat.shape)}")
    if oracle_curves.dim() != 2:
        raise ValueError(
            f"oracle_curves must be 2-D [B, C], got shape {tuple(oracle_curves.shape)}"
        )
    if oracle_curves.shape[0] != w_hat.shape[0]:
        raise ValueError("Batch size mismatch between w_hat and oracle_curves")
    if oracle_curves.shape[1] < 2:
        raise ValueError("oracle_curves must have at least 2 bins")

    n_bins = int(oracle_curves.shape[1])
    w = w_hat.clamp(0.0, 1.0)
    scaled = w * float(n_bins - 1)
    idx_lo = scaled.long().clamp(0, n_bins - 2)
    idx_hi = (idx_lo + 1).clamp(0, n_bins - 1)
    alpha = scaled - idx_lo.float()

    c_lo = oracle_curves.gather(1, idx_lo.unsqueeze(1)).squeeze(1)
    c_hi = oracle_curves.gather(1, idx_hi.unsqueeze(1)).squeeze(1)
    c_interp = (1.0 - alpha) * c_lo + alpha * c_hi
    c_star = oracle_curves.max(dim=-1).values
    return c_interp, c_star


def regret_loss(w_hat: torch.Tensor, oracle_curves: torch.Tensor) -> torch.Tensor:
    """Mean regret over a batch."""
    c_interp, c_star = interpolate_curve_torch(w_hat, oracle_curves)
    regret = c_star - c_interp
    return regret.mean()

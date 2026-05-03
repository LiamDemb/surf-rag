"""Training loss registry for scalar router models."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Tuple

import torch

from surf_rag.router.regret import interpolate_curve_torch, regret_loss

logger = logging.getLogger(__name__)

RouterLossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def hinge_squared_regret_loss(
    w_hat: torch.Tensor,
    oracle_curves: torch.Tensor,
    *,
    epsilon: float = 0.0,
) -> torch.Tensor:
    """Mean squared excess regret above ``epsilon`` (zero when regret <= epsilon)."""
    c_interp, c_star = interpolate_curve_torch(w_hat, oracle_curves)
    regret = c_star - c_interp
    excess = torch.relu(regret - float(epsilon))
    return (excess * excess).mean()


def boundary_magnet_loss(
    w_hat: torch.Tensor,
    oracle_curves: torch.Tensor,
    *,
    regret_threshold: float = 0.05,
    magnet_alpha: float = 0.02,
) -> torch.Tensor:
    """Pull weights toward 0/1 while strongly correcting bad oracle scores.

    For each sample, let ``b`` be the closer endpoint in ``{0, 1}`` to ``w`` (tie: ``0``).
    Define boundary offset ``ε = w - b``. When oracle regret ``c* - c_interp`` exceeds
    ``regret_threshold``, add ``ε²`` (sharp correction away from the interior). Always add
    ``magnet_alpha * min(w, 1-w)`` so gradients nudge predictions toward the nearest extreme
    without flattening the middle when regret is low.
    """
    w = w_hat.clamp(0.0, 1.0)
    c_interp, c_star = interpolate_curve_torch(w, oracle_curves)
    regret = c_star - c_interp
    # Closer boundary in [0, 1]; at w==0.5 prefer 0 so ε = w.
    b = torch.where(w <= 1.0 - w, torch.zeros_like(w), torch.ones_like(w))
    boundary_eps = w - b
    far = (regret > float(regret_threshold)).to(w.dtype)
    hinge_sq = far * (boundary_eps * boundary_eps)
    dist_to_extreme = torch.minimum(w, 1.0 - w)
    magnet = float(magnet_alpha) * dist_to_extreme
    return hinge_sq.mean() + magnet.mean()


def _make_regret(_kwargs: Dict[str, Any]) -> RouterLossFn:
    def fn(w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return regret_loss(w, y)

    return fn


def _make_hinge_squared(kw: Dict[str, Any]) -> RouterLossFn:
    epsilon = float(kw.get("epsilon", 0.0))

    def fn(w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return hinge_squared_regret_loss(w, y, epsilon=epsilon)

    return fn


def _make_boundary_magnet(kw: Dict[str, Any]) -> RouterLossFn:
    regret_threshold = float(kw.get("regret_threshold", 0.05))
    magnet_alpha = float(kw.get("magnet_alpha", 0.02))

    def fn(w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return boundary_magnet_loss(
            w, y, regret_threshold=regret_threshold, magnet_alpha=magnet_alpha
        )

    return fn


_LOSS_FACTORIES: Dict[str, Callable[[Dict[str, Any]], RouterLossFn]] = {
    "regret": _make_regret,
    "hinge_squared_regret": _make_hinge_squared,
    "boundary_magnet": _make_boundary_magnet,
}


def resolve_router_training_loss(
    loss_requested: str | None,
    loss_kwargs: Dict[str, Any] | None,
) -> Tuple[RouterLossFn, str, bool]:
    """Return (loss_fn, loss_effective_id, loss_fallback).

    Unknown ``loss_requested`` values fall back to ``regret``; ``loss_fallback`` is True.
    """
    raw = (loss_requested or "regret").strip()
    key = raw.lower().replace("-", "_")
    kwargs = dict(loss_kwargs or {})
    if key not in _LOSS_FACTORIES:
        logger.warning(
            "Unknown router training loss %r; falling back to regret. Known: %s",
            raw,
            ", ".join(sorted(_LOSS_FACTORIES)),
        )
        key = "regret"
        fallback = True
    else:
        fallback = False
    fn = _LOSS_FACTORIES[key](kwargs)
    return fn, key, fallback

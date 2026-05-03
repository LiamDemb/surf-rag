from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch

from surf_rag.router.losses import (
    boundary_magnet_loss,
    hinge_squared_regret_loss,
    resolve_router_training_loss,
)
from surf_rag.router.regret import regret_loss


def test_hinge_squared_zero_when_regret_within_epsilon() -> None:
    curves = torch.tensor([[0.5] * 11], dtype=torch.float32)
    w = torch.tensor([0.3], dtype=torch.float32)
    loss = hinge_squared_regret_loss(w, curves, epsilon=0.01)
    assert float(loss.item()) == pytest.approx(0.0, abs=1e-7)


def test_resolve_unknown_loss_falls_back_to_regret() -> None:
    fn, effective, fallback = resolve_router_training_loss("not_a_registered_loss", {})
    assert effective == "regret"
    assert fallback is True
    w = torch.tensor([0.2], dtype=torch.float32)
    y = torch.tensor([[0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    assert float(fn(w, y).item()) == float(regret_loss(w, y).item())


def test_boundary_magnet_flat_curve_only_magnet_term() -> None:
    curves = torch.tensor([[0.5] * 11], dtype=torch.float32)
    w = torch.tensor([0.3], dtype=torch.float32)
    loss = boundary_magnet_loss(w, curves, regret_threshold=0.05, magnet_alpha=0.1)
    assert float(loss.item()) == pytest.approx(0.1 * 0.3, abs=1e-6)


def test_boundary_magnet_high_regret_adds_squared_boundary_term() -> None:
    # w=0.5, closer boundary 0 -> boundary_eps=0.5 -> sq=0.25 when regret > threshold
    curves = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    w = torch.tensor([0.5], dtype=torch.float32)
    loss = boundary_magnet_loss(w, curves, regret_threshold=0.05, magnet_alpha=0.0)
    assert float(loss.item()) == pytest.approx(0.25, abs=1e-5)


def test_resolve_boundary_magnet_factory() -> None:
    fn, effective, fallback = resolve_router_training_loss(
        "boundary_magnet", {"regret_threshold": 0.05, "magnet_alpha": 0.0}
    )
    assert effective == "boundary_magnet"
    assert fallback is False
    curves = torch.tensor([[0.5] * 11], dtype=torch.float32)
    w = torch.tensor([0.4], dtype=torch.float32)
    assert float(fn(w, curves).item()) == pytest.approx(0.0, abs=1e-6)


def test_resolve_hinge_squared_regret_factory() -> None:
    fn, effective, fallback = resolve_router_training_loss(
        "hinge_squared_regret", {"epsilon": 0.1}
    )
    assert effective == "hinge_squared_regret"
    assert fallback is False
    curves = torch.tensor([[0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    w = torch.tensor([0.0], dtype=torch.float32)
    out = float(fn(w, curves).item())
    reg = float(regret_loss(w, curves).item())
    assert reg > 0.1
    assert out == pytest.approx((reg - 0.1) ** 2, rel=1e-5, abs=1e-6)

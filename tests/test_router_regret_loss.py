from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch

from surf_rag.router.regret import interpolate_curve_torch, regret_loss


def test_flat_curve_has_zero_regret_any_weight() -> None:
    curves = torch.tensor([[0.3] * 11], dtype=torch.float32)
    for w in (0.0, 0.17, 0.5, 0.83, 1.0):
        loss = regret_loss(torch.tensor([w], dtype=torch.float32), curves)
        assert float(loss.item()) == pytest.approx(0.0, abs=1e-7)


def test_single_peak_curve_prefers_peak_region() -> None:
    curve = torch.tensor([[0.0, 0.1, 0.2, 0.4, 0.7, 1.0, 0.8, 0.6, 0.3, 0.1, 0.0]])
    low = float(regret_loss(torch.tensor([0.0]), curve).item())
    peak = float(regret_loss(torch.tensor([0.5]), curve).item())
    high = float(regret_loss(torch.tensor([1.0]), curve).item())
    assert peak < low
    assert peak < high


def test_monotone_curve_gradient_direction() -> None:
    curve = torch.tensor([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
    w = torch.tensor([0.2], dtype=torch.float32, requires_grad=True)
    loss = regret_loss(w, curve)
    loss.backward()
    # Increasing curve: gradient should push prediction upward (negative dL/dw).
    assert w.grad is not None
    assert float(w.grad.item()) < 0.0


def test_boundary_interpolation_no_oob() -> None:
    curves = torch.tensor([[float(i) / 10.0 for i in range(11)]], dtype=torch.float32)
    for w in (0.0, 1.0, -0.5, 1.5):
        c_interp, c_star = interpolate_curve_torch(
            torch.tensor([w], dtype=torch.float32), curves
        )
        assert c_interp.shape == (1,)
        assert c_star.shape == (1,)


def test_known_linear_interpolation_value() -> None:
    # At w=0.25 between bins 0.2 and 0.3 for this linear curve.
    curves = torch.tensor([[float(i) / 10.0 for i in range(11)]], dtype=torch.float32)
    c_interp, _ = interpolate_curve_torch(
        torch.tensor([0.25], dtype=torch.float32), curves
    )
    assert float(c_interp.item()) == pytest.approx(0.25, abs=1e-6)

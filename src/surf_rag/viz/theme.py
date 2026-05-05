"""Central matplotlib style for SuRF-RAG figures."""

from __future__ import annotations

from typing import Any, Final, Mapping

import matplotlib as mpl

PALETTE: Final[dict[str, str]] = {
    "primary": "#2C7FB8",
    "identity_line": "#636363",
    "grid": "#B0B0B0",
    "text": "#1A1A1A",
    "face": "#FFFFFF",
    "light-blue": "#C6C4FC",
    "dark-blue": "#0F008A",
}


def apply_theme(
    *,
    name: str = "default",
    dpi: int | None = None,
    overrides: Mapping[str, Any] | None = None,
    backend: str | None = None,
) -> None:
    """Apply consistent rcParams. Safe to call multiple times.

    Parameters
    ----------
    name:
        Reserved for future named presets; only ``default`` is defined today.
    dpi:
        If set, updates ``figure.dpi`` and ``savefig.dpi``.
    overrides:
        Extra rcParam keys merged last (e.g. from YAML).
    backend:
        Optional backend to set before updating rcParams (e.g. "pgf").
    """
    if backend is not None:
        mpl.use(backend)

    _ = name  # single preset for now
    base: dict[str, Any] = {
        "figure.facecolor": PALETTE["face"],
        "axes.facecolor": PALETTE["face"],
        "axes.edgecolor": PALETTE["text"],
        "axes.labelcolor": PALETTE["text"],
        "axes.titlecolor": PALETTE["text"],
        "text.color": PALETTE["text"],
        "xtick.color": PALETTE["text"],
        "ytick.color": PALETTE["text"],
        "axes.grid": True,
        "grid.color": PALETTE["grid"],
        "grid.alpha": 0.35,
        "grid.linewidth": 0.6,
        "axes.linewidth": 0.9,
        "lines.linewidth": 1.2,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "font.family": "serif",
        "font.serif": ["Libertinus"],
    }
    if dpi is not None:
        base["figure.dpi"] = float(dpi)
        base["savefig.dpi"] = float(dpi)
    mpl.rcParams.update(base)
    if overrides:
        mpl.rcParams.update(dict(overrides))

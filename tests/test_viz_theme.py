from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl

from surf_rag.viz.theme import PALETTE, apply_theme


def test_apply_theme_sets_savefig_dpi() -> None:
    apply_theme(dpi=123)
    assert mpl.rcParams["savefig.dpi"] == 123.0
    assert mpl.rcParams["figure.dpi"] == 123.0


def test_apply_theme_idempotent_on_subset() -> None:
    apply_theme(dpi=100)
    a = {k: mpl.rcParams[k] for k in ("savefig.dpi", "figure.dpi", "font.size")}
    apply_theme(dpi=100)
    b = {k: mpl.rcParams[k] for k in ("savefig.dpi", "figure.dpi", "font.size")}
    assert a == b


def test_palette_keys_complete() -> None:
    for key in ("primary", "identity_line", "grid", "text", "face"):
        assert key in PALETTE
        assert isinstance(PALETTE[key], str)

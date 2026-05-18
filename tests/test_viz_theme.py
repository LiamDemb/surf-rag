from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl

from surf_rag.config.schema import FiguresThemeSection
from surf_rag.viz.theme import (
    BAR_ALPHA,
    DATASET_SOURCE_COLORS,
    PALETTE,
    POLICY_COLORS,
    apply_figures_theme,
    apply_theme,
    bar_style,
    policy_color,
)


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


def test_dataset_source_colors() -> None:
    assert DATASET_SOURCE_COLORS["nq"] == PALETTE["secondary"]
    assert DATASET_SOURCE_COLORS["2wiki"] == PALETTE["primary"]
    assert DATASET_SOURCE_COLORS["nq"] != DATASET_SOURCE_COLORS["2wiki"]


def test_policy_color_cycles() -> None:
    assert policy_color(0) == POLICY_COLORS[0]
    assert policy_color(len(POLICY_COLORS)) == POLICY_COLORS[0]


def test_bar_style_defaults() -> None:
    style = bar_style()
    assert style["alpha"] == BAR_ALPHA
    assert style["color"] == PALETTE["light-blue"]


def test_apply_figures_theme_returns_format() -> None:
    fmt = apply_figures_theme(FiguresThemeSection(dpi=120), image_format="pdf")
    assert fmt == "pdf"
    assert mpl.rcParams["savefig.dpi"] == 120.0

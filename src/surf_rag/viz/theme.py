"""Central matplotlib style for SuRF-RAG figures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, Mapping

if TYPE_CHECKING:
    from surf_rag.config.schema import FiguresThemeSection

import matplotlib as mpl
import numpy as np

# Wong colorblind-friendly accents (https://www.nature.com/articles/nmeth.1618)
PALETTE: Final[dict[str, str]] = {
    "primary": "#1F6EF5",  # light-blue
    "secondary": "#FDCC40",  # light-green
    "red": "#AD0909",  # regret / loss heatmaps
    "identity_line": "#636363",
    "grid": "#B0B0B0",
    "text": "#1A1A1A",
    "face": "#FFFFFF",
    "light-blue": "#CED2FE",
    "dark-blue": "#0072B2",
    "light-green": "#90BE91",
}

# Opacity steps for single-hue heatmaps (blend accent onto ``face``).
HEATMAP_SHADE_ALPHAS: Final[tuple[float, ...]] = (
    0.0,
    0.7,
    1.0,
)

# Cell annotation flips to face colour above this normalized heat level.
HEATMAP_ANNOTATION_CONTRAST_THRESHOLD: Final[float] = 0.52

# Dataset colours: high contrast (orange vs blue), not two similar blues.
DATASET_SOURCE_COLORS: Final[dict[str, str]] = {
    "nq": PALETTE["primary"],
    "2wiki": PALETTE["secondary"],
    "all": "#333333",
}

DATASET_SOURCE_MARKERS: Final[dict[str, str]] = {
    "nq": "o",
    "2wiki": "s",
    "all": "D",
}

DATASET_SOURCE_LABELS: Final[dict[str, str]] = {
    "nq": "NQ",
    "2wiki": "2Wiki",
    "all": "All",
}

# Default opacity for bar chart fills.
BAR_ALPHA: Final[float] = 0.85

# Training / dev learning-curve lines (solid primary vs secondary).
LEARNING_CURVE_LINE_ALPHA: Final[float] = 0.85

POLICY_COLORS: Final[tuple[str, ...]] = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#F0E442",
    "#56B4E9",
    "#E69F00",
    "#000000",
)


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
        "legend.framealpha": 0.9,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
    if dpi is not None:
        base["figure.dpi"] = float(dpi)
        base["savefig.dpi"] = float(dpi)
    mpl.rcParams.update(base)
    if overrides:
        mpl.rcParams.update(dict(overrides))


def policy_color(index: int) -> str:
    """Cycle palette color for grouped policy bar charts."""
    return POLICY_COLORS[index % len(POLICY_COLORS)]


def blend_on_face(
    foreground_hex: str,
    alpha: float,
    *,
    background_hex: str | None = None,
) -> tuple[float, float, float]:
    """Blend ``foreground_hex`` over ``background_hex`` (default ``face``) by ``alpha``."""
    from matplotlib.colors import to_rgb

    bg = to_rgb(background_hex or PALETTE["face"])
    fg = to_rgb(foreground_hex)
    a = float(np.clip(alpha, 0.0, 1.0))
    return tuple((1.0 - a) * bg[i] + a * fg[i] for i in range(3))


def sequential_cmap_from_palette(
    palette_key: str,
    *,
    n_steps: int = 256,
):
    """Sequential colormap: white/face at low values, palette accent at high (opacity ramp)."""
    from matplotlib.colors import LinearSegmentedColormap

    accent = PALETTE.get(palette_key)
    if accent is None:
        raise KeyError(f"Unknown palette key: {palette_key!r}")
    colors = [blend_on_face(accent, a) for a in np.linspace(0.0, 1.0, num=n_steps)]
    return LinearSegmentedColormap.from_list(
        f"surf_{palette_key}_sequential", colors, N=n_steps
    )


def bar_style(
    *,
    color: str | None = None,
) -> dict[str, Any]:
    """Shared kwargs for ``Axes.bar`` (semi-transparent fill, no outline)."""
    return {
        "color": color or PALETTE["light-blue"],
        "alpha": BAR_ALPHA,
        "edgecolor": "none",
        "linewidth": 0,
    }


def style_bar_axes(ax: Any) -> None:
    """Axes styling shared by bar charts: horizontal grid only, open top/right."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)


def apply_figures_theme(
    theme: FiguresThemeSection,
    *,
    image_format: str = "png",
) -> str:
    """Apply a ``FiguresThemeSection`` and return normalized image format."""
    overrides = dict(theme.overrides or {})
    if theme.font_size is not None:
        overrides["font.size"] = float(theme.font_size)
    apply_theme(
        name=theme.name,
        dpi=theme.dpi,
        overrides=overrides or None,
        backend=theme.backend,
    )
    fmt = str(image_format or "png").strip().lower()
    if fmt not in ("png", "pdf"):
        raise ValueError(f"image_format must be png or pdf, got {fmt!r}")
    return fmt

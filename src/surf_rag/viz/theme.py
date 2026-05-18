"""Central matplotlib style for SuRF-RAG figures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, Mapping

if TYPE_CHECKING:
    from surf_rag.config.schema import FiguresThemeSection

import matplotlib as mpl

# Wong colorblind-friendly accents (https://www.nature.com/articles/nmeth.1618)
PALETTE: Final[dict[str, str]] = {
    "primary": "#0072B2",
    "secondary": "#D55E00",
    "identity_line": "#636363",
    "grid": "#B0B0B0",
    "text": "#1A1A1A",
    "face": "#FFFFFF",
    "light-blue": "#CED2FE",
    "dark-blue": "#0072B2",
}

# Dataset colours: high contrast (orange vs blue), not two similar blues.
DATASET_SOURCE_COLORS: Final[dict[str, str]] = {
    "nq": "#D55E00",
    "2wiki": "#0072B2",
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

# Default opacity for bar charts so grid lines remain visible through fills.
BAR_ALPHA: Final[float] = 0.85

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


def bar_style(
    *,
    color: str | None = None,
    edgecolor: str | None = None,
) -> dict[str, Any]:
    """Shared kwargs for ``Axes.bar`` (semi-transparent fill, themed edge)."""
    return {
        "color": color or PALETTE["light-blue"],
        "alpha": BAR_ALPHA,
        "edgecolor": edgecolor or PALETTE["text"],
        "linewidth": 0.4,
    }


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

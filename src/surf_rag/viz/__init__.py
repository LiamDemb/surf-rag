"""Publication-style figures for SuRF-RAG experiments."""

from surf_rag.viz.context import FigureRunContext
from surf_rag.viz.registry import list_figure_kinds
from surf_rag.viz.runner import render_figures_from_config
from surf_rag.viz.theme import PALETTE, apply_theme

__all__ = [
    "FigureRunContext",
    "PALETTE",
    "apply_theme",
    "list_figure_kinds",
    "render_figures_from_config",
]

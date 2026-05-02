"""Orchestrate rendering all figures declared in pipeline config."""

from __future__ import annotations

from collections.abc import Sequence

from surf_rag.config.schema import PipelineConfig
from surf_rag.viz.context import FigureRunContext
from surf_rag.viz.specs import figure_spec_from_mapping
from surf_rag.viz.theme import apply_theme
from surf_rag.viz.types import FigureOutput


def render_figures_from_config(
    cfg: PipelineConfig,
    *,
    output_dir_override: str | None = None,
    only_kinds: frozenset[str] | None = None,
    force: bool = False,
) -> list[FigureOutput]:
    """Render each entry in ``cfg.figures.plots`` when ``figures.enabled`` is true."""
    if not cfg.figures.enabled:
        return []
    import surf_rag.viz.renderers  # noqa: F401 — register built-in renderers

    t = cfg.figures.theme
    overrides: dict[str, object] = dict(t.overrides or {})
    if t.font_size is not None:
        overrides["font.size"] = float(t.font_size)
    apply_theme(name=t.name, dpi=t.dpi, overrides=overrides or None)
    ctx = FigureRunContext.from_pipeline(
        cfg,
        output_dir_override=output_dir_override,
        force=force,
    )
    from surf_rag.viz.registry import render as registry_render

    outputs: list[FigureOutput] = []
    plots: Sequence[dict[str, object]] = cfg.figures.plots
    for plot in plots:
        kind = plot.get("kind")
        if only_kinds is not None and str(kind) not in only_kinds:
            continue
        spec = figure_spec_from_mapping(plot)
        outputs.append(registry_render(spec, ctx))
    return outputs

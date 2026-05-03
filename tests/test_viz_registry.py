from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import surf_rag.viz.renderers  # noqa: F401 — registration side effect
from surf_rag.config.schema import (
    PathsSection,
    PipelineConfig,
    RouterSection,
    RouterTrainSection,
)
from surf_rag.viz.context import FigureRunContext
from surf_rag.viz.registry import list_figure_kinds, render


def test_list_figure_kinds_includes_built_ins() -> None:
    kinds = list_figure_kinds()
    assert "router_pred_vs_oracle" in kinds
    assert "router_pred_vs_oracle_intervals" in kinds
    assert "benchmark_oracle_ndcg_heatmap" in kinds
    assert "oracle_argmax_weight_histogram" in kinds


def test_render_unknown_kind_raises(tmp_path: Path) -> None:
    class _FakeSpec:
        kind = "no-such-kind"

    cfg = replace(
        PipelineConfig(),
        paths=replace(
            PathsSection(),
            data_base=str(tmp_path),
            router_id="r",
            router_base=str(tmp_path / "router"),
            figures_base=str(tmp_path / "fig"),
        ),
        router=replace(
            RouterSection(),
            train=replace(RouterTrainSection(), input_mode="both"),
        ),
    )
    ctx = FigureRunContext.from_pipeline(cfg, force=True)
    with pytest.raises(ValueError, match="No renderer registered"):
        render(_FakeSpec(), ctx)  # type: ignore[arg-type]

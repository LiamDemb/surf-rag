from __future__ import annotations

from pathlib import Path

import pytest

import surf_rag.viz.renderers  # noqa: F401 — registration side effect
from surf_rag.evaluation.router_model_artifacts import RouterModelPaths
from surf_rag.viz.context import FigureRunContext
from surf_rag.viz.registry import list_figure_kinds, render


def test_list_figure_kinds_includes_built_ins() -> None:
    kinds = list_figure_kinds()
    assert "router_pred_vs_oracle" in kinds
    assert "router_pred_vs_oracle_intervals" in kinds


def test_render_unknown_kind_raises() -> None:
    class _FakeSpec:
        kind = "no-such-kind"

    ctx = FigureRunContext(
        router_id="r",
        router_architecture_id=None,
        input_mode="both",
        router_base=Path("."),
        model_paths=RouterModelPaths(run_root=Path(".")),
        output_dir=Path("."),
        experiment_id=None,
        image_format="png",
        force=True,
    )
    with pytest.raises(ValueError, match="No renderer registered"):
        render(_FakeSpec(), ctx)  # type: ignore[arg-type]

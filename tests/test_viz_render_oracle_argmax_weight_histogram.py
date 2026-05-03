from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

from surf_rag.config.schema import (
    PathsSection,
    PipelineConfig,
    RouterSection,
    RouterTrainSection,
)
from surf_rag.viz.context import FigureRunContext
from surf_rag.viz.renderers.oracle_argmax_weight_histogram import (
    render_oracle_argmax_weight_histogram,
)
from surf_rag.viz.specs import OracleArgmaxWeightHistogramSpec
from surf_rag.viz.theme import apply_theme


def _ctx_and_parquet(tmp_path: Path) -> tuple[FigureRunContext, Path]:
    rb = tmp_path / "router"
    ds = rb / "rid" / "dataset"
    ds.mkdir(parents=True)
    p = ds / "router_dataset.parquet"
    w = [0.0, 0.5, 1.0]
    pd.DataFrame(
        [
            {
                "question_id": "unique_mid",
                "split": "test",
                "weight_grid": w,
                "oracle_curve": [0.1, 0.9, 0.2],
            },
            {
                "question_id": "tie_ends",
                "split": "test",
                "weight_grid": w,
                "oracle_curve": [1.0, 0.3, 1.0],
            },
        ]
    ).to_parquet(p, index=False)
    cfg = replace(
        PipelineConfig(),
        paths=replace(
            PathsSection(),
            data_base=str(tmp_path),
            router_base=str(rb),
            router_id="rid",
            benchmark_name="bname",
            benchmark_id="bid",
            figures_base=str(tmp_path / "fig"),
        ),
        router=replace(
            RouterSection(),
            train=replace(RouterTrainSection(), input_mode="both"),
        ),
    )
    ctx = FigureRunContext.from_pipeline(
        cfg, resolved_output_dir=tmp_path / "out", force=True
    )
    return ctx, p


def test_oracle_argmax_histogram_stacked_counts(tmp_path: Path) -> None:
    apply_theme(dpi=100)
    ctx, parq = _ctx_and_parquet(tmp_path)
    assert ctx.router_dataset_parquet == parq
    spec = OracleArgmaxWeightHistogramSpec(
        kind="oracle_argmax_weight_histogram", split="test"
    )
    out = render_oracle_argmax_weight_histogram(spec, ctx)
    assert out.path_image.is_file()
    meta = json.loads(out.path_meta.read_text(encoding="utf-8"))
    assert meta["figure_kind"] == "oracle_argmax_weight_histogram"
    assert meta["n_queries"] == 2
    assert meta["n_queries_multi_argmax"] == 1
    assert meta["total_contributions"] == pytest.approx(3.0)
    assert meta["hist_binning"] == "weight_grid"
    assert meta["hist_bins"] == 0
    assert meta["n_weight_grid_points"] == 3


def test_oracle_argmax_histogram_equal_width_bins_meta(tmp_path: Path) -> None:
    apply_theme(dpi=100)
    ctx, _parq = _ctx_and_parquet(tmp_path)
    spec = OracleArgmaxWeightHistogramSpec(
        kind="oracle_argmax_weight_histogram",
        split="test",
        hist_bins=10,
    )
    out = render_oracle_argmax_weight_histogram(spec, ctx)
    meta = json.loads(out.path_meta.read_text(encoding="utf-8"))
    assert meta["hist_binning"] == "equal_width_0_1"
    assert meta["hist_bins"] == 10


def test_hist_bins_too_large_raises() -> None:
    with pytest.raises(ValueError, match="hist_bins"):
        OracleArgmaxWeightHistogramSpec(
            kind="oracle_argmax_weight_histogram",
            split="all",
            hist_bins=501,
        )

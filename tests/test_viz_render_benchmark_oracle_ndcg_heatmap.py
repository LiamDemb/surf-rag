from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from surf_rag.config.schema import (
    PathsSection,
    PipelineConfig,
    RouterSection,
    RouterTrainSection,
)
from surf_rag.viz.context import FigureRunContext
from surf_rag.viz.renderers.benchmark_oracle_ndcg_heatmap import (
    _curve_all_ones,
    _oracle_mid_band_strictly_beats_tails,
    _oracle_peak_has_interior_dense_weight,
    _score_weighted_centroid_dense_weight,
    render_benchmark_oracle_ndcg_heatmap,
)
from surf_rag.viz.specs import BenchmarkOracleHeatmapSpec
from surf_rag.viz.theme import apply_theme


def _write_parquet_and_context(tmp_path: Path) -> tuple[FigureRunContext, Path]:
    rb = tmp_path / "router"
    ds = rb / "rid" / "dataset"
    ds.mkdir(parents=True)
    p = ds / "router_dataset.parquet"
    w = [0.0, 0.5, 1.0]
    df = pd.DataFrame(
        [
            {
                "question_id": "b",
                "split": "test",
                "weight_grid": w,
                "oracle_curve": [0.0, 0.9, 0.0],
            },
            {
                "question_id": "a",
                "split": "test",
                "weight_grid": w,
                "oracle_curve": [0.0, 0.4, 0.0],
            },
        ]
    )
    df.to_parquet(p, index=False)
    out_dir = tmp_path / "out"
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
    ctx = FigureRunContext.from_pipeline(cfg, resolved_output_dir=out_dir, force=True)
    return ctx, p


def test_render_benchmark_heatmap_writes_files(tmp_path: Path) -> None:
    apply_theme(dpi=100)
    ctx, parq = _write_parquet_and_context(tmp_path)
    assert ctx.router_dataset_parquet == parq
    spec = BenchmarkOracleHeatmapSpec(
        kind="benchmark_oracle_ndcg_heatmap", split="test"
    )
    out = render_benchmark_oracle_ndcg_heatmap(spec, ctx)
    assert out.path_image.is_file()
    meta = json.loads(out.path_meta.read_text(encoding="utf-8"))
    assert meta["n_queries"] == 2
    assert meta["figure_kind"] == "benchmark_oracle_ndcg_heatmap"
    assert meta["query_sort"] == "score_weighted_centroid_dense_weight"
    assert meta["color_norm"] == "log"
    assert meta["color_power_gamma"] == 2.0
    assert meta["interior_peak_only"] is False
    assert meta["exclude_all_one_queries"] is False
    assert meta["n_queries_excluded_all_one"] == 0
    assert meta["n_queries_excluded_endpoint_peak_only"] == 0
    assert meta["mid_dense_band_strict_best"] is False
    assert meta["mid_dense_band_strict_best_w_lo"] == 0.2
    assert meta["mid_dense_band_strict_best_w_hi"] == 0.8
    assert meta["n_queries_excluded_mid_band_not_strict_best"] == 0
    assert meta["show_plot_subtitle"] is True
    assert meta["vmin_log_used"] is not None


def test_oracle_mid_band_strictly_beats_tails() -> None:
    w = np.array([0.0, 0.1, 0.5, 0.9, 1.0], dtype=np.float64)
    assert _oracle_mid_band_strictly_beats_tails([0.5, 0.4, 0.9, 0.3, 0.2], w)
    assert not _oracle_mid_band_strictly_beats_tails([0.9, 0.4, 0.5, 0.3, 0.2], w)
    w_no_mid = np.array([0.0, 0.1, 0.15], dtype=np.float64)
    assert not _oracle_mid_band_strictly_beats_tails([0.1, 0.2, 0.3], w_no_mid)


def test_curve_all_ones() -> None:
    rt, at = 1e-5, 1e-8
    assert _curve_all_ones([1.0, 1.0, 1.0], rtol=rt, atol=at)
    assert not _curve_all_ones([1.0, 0.9, 1.0], rtol=rt, atol=at)
    assert not _curve_all_ones([], rtol=rt, atol=at)


def test_oracle_peak_has_interior_dense_weight() -> None:
    w = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    rt, at = 1e-5, 1e-8
    assert not _oracle_peak_has_interior_dense_weight(
        [1.0, 0.0, 0.0], w, rtol=rt, atol=at
    )
    assert not _oracle_peak_has_interior_dense_weight(
        [0.0, 0.0, 1.0], w, rtol=rt, atol=at
    )
    assert _oracle_peak_has_interior_dense_weight([0.0, 1.0, 0.0], w, rtol=rt, atol=at)
    assert _oracle_peak_has_interior_dense_weight([0.4, 0.5, 0.4], w, rtol=rt, atol=at)


def test_score_weighted_centroid_extremes() -> None:
    w = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    assert _score_weighted_centroid_dense_weight([1.0, 0.0, 0.0], w) == pytest.approx(
        0.0
    )
    assert _score_weighted_centroid_dense_weight([0.0, 0.0, 1.0], w) == pytest.approx(
        1.0
    )
    assert _score_weighted_centroid_dense_weight([0.0, 1.0, 0.0], w) == pytest.approx(
        0.5
    )


def test_all_queries_all_zero_raises(tmp_path: Path) -> None:
    apply_theme(dpi=100)
    rb = tmp_path / "router"
    ds = rb / "rid" / "dataset"
    ds.mkdir(parents=True)
    p = ds / "router_dataset.parquet"
    w = [0.0, 1.0]
    pd.DataFrame(
        [
            {
                "question_id": "x",
                "split": "train",
                "weight_grid": w,
                "oracle_curve": [0.0, 0.0],
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
            figures_base=str(tmp_path / "fig"),
        ),
        router=replace(
            RouterSection(),
            train=replace(RouterTrainSection(), input_mode="both"),
        ),
    )
    ctx = FigureRunContext.from_pipeline(
        cfg, resolved_output_dir=tmp_path / "o", force=True
    )
    spec = BenchmarkOracleHeatmapSpec(
        kind="benchmark_oracle_ndcg_heatmap", split="train"
    )
    with pytest.raises(ValueError, match="No queries left"):
        render_benchmark_oracle_ndcg_heatmap(spec, ctx)


def test_colormap_blues_in_meta(tmp_path: Path) -> None:
    apply_theme(dpi=100)
    ctx, parq = _write_parquet_and_context(tmp_path)
    spec = BenchmarkOracleHeatmapSpec(
        kind="benchmark_oracle_ndcg_heatmap",
        split="test",
        colormap="Blues",
    )
    out = render_benchmark_oracle_ndcg_heatmap(spec, ctx)
    meta = json.loads(out.path_meta.read_text(encoding="utf-8"))
    assert meta["colormap"] == "Blues"


def test_show_plot_subtitle_false_in_meta(tmp_path: Path) -> None:
    apply_theme(dpi=100)
    ctx, _parq = _write_parquet_and_context(tmp_path)
    spec = BenchmarkOracleHeatmapSpec(
        kind="benchmark_oracle_ndcg_heatmap",
        split="test",
        show_plot_subtitle=False,
    )
    out = render_benchmark_oracle_ndcg_heatmap(spec, ctx)
    meta = json.loads(out.path_meta.read_text(encoding="utf-8"))
    assert meta["show_plot_subtitle"] is False


def test_color_norm_linear_meta(tmp_path: Path) -> None:
    apply_theme(dpi=100)
    ctx, _parq = _write_parquet_and_context(tmp_path)
    spec = BenchmarkOracleHeatmapSpec(
        kind="benchmark_oracle_ndcg_heatmap",
        split="test",
        color_norm="linear",
    )
    out = render_benchmark_oracle_ndcg_heatmap(spec, ctx)
    meta = json.loads(out.path_meta.read_text(encoding="utf-8"))
    assert meta["color_norm"] == "linear"
    assert meta["vmin_log_used"] is None


def test_color_norm_power_meta(tmp_path: Path) -> None:
    apply_theme(dpi=100)
    ctx, _parq = _write_parquet_and_context(tmp_path)
    spec = BenchmarkOracleHeatmapSpec(
        kind="benchmark_oracle_ndcg_heatmap",
        split="test",
        color_norm="power",
        color_power_gamma=2.5,
    )
    out = render_benchmark_oracle_ndcg_heatmap(spec, ctx)
    meta = json.loads(out.path_meta.read_text(encoding="utf-8"))
    assert meta["color_norm"] == "power"
    assert meta["color_power_gamma"] == 2.5
    assert meta["vmin_log_used"] is None


def test_unknown_colormap_raises(tmp_path: Path) -> None:
    apply_theme(dpi=100)
    ctx, _parq = _write_parquet_and_context(tmp_path)
    spec = BenchmarkOracleHeatmapSpec(
        kind="benchmark_oracle_ndcg_heatmap",
        split="test",
        colormap="not-a-real-cmap-xyz",
    )
    with pytest.raises(ValueError, match="Unknown colormap"):
        render_benchmark_oracle_ndcg_heatmap(spec, ctx)


def test_excludes_only_fully_zero_queries(tmp_path: Path) -> None:
    """One column all zeros dropped; mixed zero/non-zero curves stay with full columns."""
    apply_theme(dpi=100)
    rb = tmp_path / "router"
    ds = rb / "rid" / "dataset"
    ds.mkdir(parents=True)
    p = ds / "router_dataset.parquet"
    w = [0.0, 0.5, 1.0]
    pd.DataFrame(
        [
            {
                "question_id": "all_zero",
                "split": "test",
                "weight_grid": w,
                "oracle_curve": [0.0, 0.0, 0.0],
            },
            {
                "question_id": "mixed",
                "split": "test",
                "weight_grid": w,
                "oracle_curve": [0.0, 0.8, 0.0],
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
            figures_base=str(tmp_path / "fig"),
        ),
        router=replace(
            RouterSection(),
            train=replace(RouterTrainSection(), input_mode="both"),
        ),
    )
    ctx = FigureRunContext.from_pipeline(
        cfg, resolved_output_dir=tmp_path / "out2", force=True
    )
    spec = BenchmarkOracleHeatmapSpec(
        kind="benchmark_oracle_ndcg_heatmap", split="test"
    )
    out = render_benchmark_oracle_ndcg_heatmap(spec, ctx)
    meta = json.loads(out.path_meta.read_text(encoding="utf-8"))
    assert meta["n_queries"] == 1
    assert meta["n_queries_excluded_all_zero"] == 1


def test_mid_dense_band_strict_best_filters_tails(tmp_path: Path) -> None:
    apply_theme(dpi=100)
    rb = tmp_path / "router"
    ds = rb / "rid" / "dataset"
    ds.mkdir(parents=True)
    p = ds / "router_dataset.parquet"
    w = [0.0, 0.1, 0.5, 0.9, 1.0]
    pd.DataFrame(
        [
            {
                "question_id": "tail_best",
                "split": "test",
                "weight_grid": w,
                "oracle_curve": [0.9, 0.4, 0.5, 0.3, 0.2],
            },
            {
                "question_id": "mid_best",
                "split": "test",
                "weight_grid": w,
                "oracle_curve": [0.5, 0.4, 0.9, 0.3, 0.2],
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
            figures_base=str(tmp_path / "fig"),
        ),
        router=replace(
            RouterSection(),
            train=replace(RouterTrainSection(), input_mode="both"),
        ),
    )
    ctx = FigureRunContext.from_pipeline(
        cfg, resolved_output_dir=tmp_path / "out_mdb", force=True
    )
    spec = BenchmarkOracleHeatmapSpec(
        kind="benchmark_oracle_ndcg_heatmap",
        split="test",
        mid_dense_band_strict_best=True,
    )
    out = render_benchmark_oracle_ndcg_heatmap(spec, ctx)
    meta = json.loads(out.path_meta.read_text(encoding="utf-8"))
    assert meta["n_queries"] == 1
    assert meta["n_queries_excluded_mid_band_not_strict_best"] == 1
    assert meta["mid_dense_band_strict_best"] is True


def test_mid_dense_band_strict_best_all_filtered_raises(tmp_path: Path) -> None:
    apply_theme(dpi=100)
    rb = tmp_path / "router"
    ds = rb / "rid" / "dataset"
    ds.mkdir(parents=True)
    p = ds / "router_dataset.parquet"
    w = [0.0, 0.1, 0.5, 0.9, 1.0]
    pd.DataFrame(
        [
            {
                "question_id": "x",
                "split": "test",
                "weight_grid": w,
                "oracle_curve": [0.9, 0.4, 0.5, 0.3, 0.2],
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
            figures_base=str(tmp_path / "fig"),
        ),
        router=replace(
            RouterSection(),
            train=replace(RouterTrainSection(), input_mode="both"),
        ),
    )
    ctx = FigureRunContext.from_pipeline(
        cfg, resolved_output_dir=tmp_path / "out_mdb2", force=True
    )
    spec = BenchmarkOracleHeatmapSpec(
        kind="benchmark_oracle_ndcg_heatmap",
        split="test",
        mid_dense_band_strict_best=True,
    )
    with pytest.raises(ValueError, match="mid_dense_band_strict_best"):
        render_benchmark_oracle_ndcg_heatmap(spec, ctx)


def test_excludes_uniform_all_one_queries_when_enabled(tmp_path: Path) -> None:
    """Flat oracle curve at 1.0 everywhere is dropped when exclude_all_one_queries."""
    apply_theme(dpi=100)
    rb = tmp_path / "router"
    ds = rb / "rid" / "dataset"
    ds.mkdir(parents=True)
    p = ds / "router_dataset.parquet"
    w = [0.0, 0.5, 1.0]
    pd.DataFrame(
        [
            {
                "question_id": "all_ones",
                "split": "test",
                "weight_grid": w,
                "oracle_curve": [1.0, 1.0, 1.0],
            },
            {
                "question_id": "mixed",
                "split": "test",
                "weight_grid": w,
                "oracle_curve": [0.2, 0.9, 0.3],
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
            figures_base=str(tmp_path / "fig"),
        ),
        router=replace(
            RouterSection(),
            train=replace(RouterTrainSection(), input_mode="both"),
        ),
    )
    ctx = FigureRunContext.from_pipeline(
        cfg, resolved_output_dir=tmp_path / "out_ao", force=True
    )
    spec = BenchmarkOracleHeatmapSpec(
        kind="benchmark_oracle_ndcg_heatmap",
        split="test",
        exclude_all_one_queries=True,
    )
    out = render_benchmark_oracle_ndcg_heatmap(spec, ctx)
    meta = json.loads(out.path_meta.read_text(encoding="utf-8"))
    assert meta["n_queries"] == 1
    assert meta["n_queries_excluded_all_one"] == 1
    assert meta["exclude_all_one_queries"] is True


def test_interior_peak_only_keeps_queries_with_interior_max(tmp_path: Path) -> None:
    apply_theme(dpi=100)
    rb = tmp_path / "router"
    ds = rb / "rid" / "dataset"
    ds.mkdir(parents=True)
    p = ds / "router_dataset.parquet"
    w = [0.0, 0.5, 1.0]
    pd.DataFrame(
        [
            {
                "question_id": "endpoint",
                "split": "test",
                "weight_grid": w,
                "oracle_curve": [1.0, 0.0, 0.0],
            },
            {
                "question_id": "interior",
                "split": "test",
                "weight_grid": w,
                "oracle_curve": [0.0, 1.0, 0.0],
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
            figures_base=str(tmp_path / "fig"),
        ),
        router=replace(
            RouterSection(),
            train=replace(RouterTrainSection(), input_mode="both"),
        ),
    )
    ctx = FigureRunContext.from_pipeline(
        cfg, resolved_output_dir=tmp_path / "out_ip", force=True
    )
    spec = BenchmarkOracleHeatmapSpec(
        kind="benchmark_oracle_ndcg_heatmap",
        split="test",
        interior_peak_only=True,
    )
    out = render_benchmark_oracle_ndcg_heatmap(spec, ctx)
    meta = json.loads(out.path_meta.read_text(encoding="utf-8"))
    assert meta["n_queries"] == 1
    assert meta["n_queries_excluded_endpoint_peak_only"] == 1
    assert meta["interior_peak_only"] is True


def test_interior_peak_only_all_filtered_raises(tmp_path: Path) -> None:
    apply_theme(dpi=100)
    rb = tmp_path / "router"
    ds = rb / "rid" / "dataset"
    ds.mkdir(parents=True)
    p = ds / "router_dataset.parquet"
    w = [0.0, 0.5, 1.0]
    pd.DataFrame(
        [
            {
                "question_id": "a",
                "split": "test",
                "weight_grid": w,
                "oracle_curve": [1.0, 0.0, 0.0],
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
            figures_base=str(tmp_path / "fig"),
        ),
        router=replace(
            RouterSection(),
            train=replace(RouterTrainSection(), input_mode="both"),
        ),
    )
    ctx = FigureRunContext.from_pipeline(
        cfg, resolved_output_dir=tmp_path / "out_ip2", force=True
    )
    spec = BenchmarkOracleHeatmapSpec(
        kind="benchmark_oracle_ndcg_heatmap",
        split="test",
        interior_peak_only=True,
    )
    with pytest.raises(ValueError, match="excluded_endpoint_peak_only"):
        render_benchmark_oracle_ndcg_heatmap(spec, ctx)

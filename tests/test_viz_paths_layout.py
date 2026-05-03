from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from surf_rag.config.loader import resolve_paths
from surf_rag.config.schema import (
    PathsSection,
    PipelineConfig,
    RouterSection,
    RouterTrainSection,
)
from surf_rag.viz.paths_layout import (
    canonical_benchmark_figure_dir,
    canonical_router_figure_dir,
    resolve_figure_output_dir,
)
from surf_rag.viz.specs import (
    BenchmarkOracleHeatmapSpec,
    OracleArgmaxWeightHistogramSpec,
    RouterPredVsOracleSpec,
)


def test_canonical_router_figure_dir_default_figures_base(tmp_path: Path) -> None:
    cfg = replace(
        PipelineConfig(),
        paths=replace(
            PathsSection(),
            data_base=str(tmp_path),
            figures_base=str(tmp_path / "figures"),
            benchmark_name="surf-bench",
            benchmark_id="main",
            router_id="rid",
            router_architecture_id="tower-v01",
        ),
    )
    rp = resolve_paths(cfg)
    out = canonical_router_figure_dir(
        rp,
        router_id="rid",
        router_architecture_id="tower-v01",
        input_mode="both",
    )
    assert rp.figures_base == tmp_path / "figures"
    assert out == (
        tmp_path
        / "figures"
        / "router"
        / "rid"
        / "tower-v01"
        / "both"
        / "benchmark"
        / "surf-bench__main"
    )


def test_canonical_router_figure_dir_legacy_arch_segment(tmp_path: Path) -> None:
    cfg = replace(
        PipelineConfig(),
        paths=replace(
            PathsSection(),
            data_base=str(tmp_path),
            benchmark_name="b",
            benchmark_id="id",
            router_id="r1",
            router_architecture_id=None,
        ),
    )
    rp = resolve_paths(cfg)
    out = canonical_router_figure_dir(
        rp,
        router_id="r1",
        router_architecture_id=None,
        input_mode="embedding",
    )
    assert "legacy-model" in str(out)
    assert out.parts[-2] == "benchmark"
    assert out.name == "b__id"


def test_resolve_paths_custom_figures_base(tmp_path: Path) -> None:
    alt = tmp_path / "my-figures"
    cfg = replace(
        PipelineConfig(),
        paths=replace(
            PathsSection(),
            data_base=str(tmp_path),
            figures_base=str(alt),
            benchmark_name="b",
            benchmark_id="v",
            router_id="r",
        ),
    )
    rp = resolve_paths(cfg)
    assert rp.figures_base == alt.resolve()


def test_canonical_benchmark_figure_dir(tmp_path: Path) -> None:
    cfg = replace(
        PipelineConfig(),
        paths=replace(
            PathsSection(),
            data_base=str(tmp_path),
            figures_base=str(tmp_path / "froot"),
            benchmark_name="hotpotqa",
            benchmark_id="100",
        ),
    )
    rp = resolve_paths(cfg)
    out = canonical_benchmark_figure_dir(rp)
    assert out == tmp_path / "froot" / "benchmarks" / "hotpotqa" / "100"


def test_resolve_figure_output_dir_router_vs_benchmark(tmp_path: Path) -> None:
    cfg = replace(
        PipelineConfig(),
        paths=replace(
            PathsSection(),
            data_base=str(tmp_path),
            figures_base=str(tmp_path / "f"),
            router_id="rid",
            router_base=str(tmp_path / "router"),
            router_architecture_id="arch",
            benchmark_name="b",
            benchmark_id="id",
        ),
        router=replace(
            RouterSection(),
            train=replace(RouterTrainSection(), input_mode="both"),
        ),
    )
    rp = resolve_paths(cfg)
    router_spec = RouterPredVsOracleSpec(kind="router_pred_vs_oracle", split="test")
    out_r = resolve_figure_output_dir(cfg, rp, router_spec, None)
    assert "router" in out_r.parts
    heat_spec = BenchmarkOracleHeatmapSpec(
        kind="benchmark_oracle_ndcg_heatmap", split="all"
    )
    out_b = resolve_figure_output_dir(cfg, rp, heat_spec, None)
    assert out_b == tmp_path / "f" / "benchmarks" / "b" / "id"
    hist_spec = OracleArgmaxWeightHistogramSpec(
        kind="oracle_argmax_weight_histogram", split="all"
    )
    out_h = resolve_figure_output_dir(cfg, rp, hist_spec, None)
    assert out_h == out_b

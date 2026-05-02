from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from surf_rag.config.loader import resolve_paths
from surf_rag.config.schema import PathsSection, PipelineConfig
from surf_rag.viz.paths_layout import canonical_router_figure_dir


def test_canonical_router_figure_dir_default_figures_base(tmp_path: Path) -> None:
    cfg = replace(
        PipelineConfig(),
        paths=replace(
            PathsSection(),
            data_base=str(tmp_path),
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

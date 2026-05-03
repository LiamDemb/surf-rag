"""Canonical on-disk layout for rendered figures (under ``ResolvedPaths.figures_base``)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from surf_rag.config.loader import ResolvedPaths
from surf_rag.evaluation.artifact_paths import safe_benchmark_bundle_subpath

if TYPE_CHECKING:
    from surf_rag.config.schema import PipelineConfig
    from surf_rag.viz.specs import BaseFigureSpec


def canonical_router_figure_dir(
    rp: ResolvedPaths,
    *,
    router_id: str,
    router_architecture_id: str | None,
    input_mode: str,
) -> Path:
    """Default output directory for router-centric figures.

    Layout::

        {figures_base}/router/{router_id}/{architecture|legacy-model}/{input_mode}/benchmark/{name}__{id}/

    Benchmark scope lives under the router branch so all plots for one router stay
    grouped; ``figures_base`` defaults to ``~/figures`` (see
    :func:`surf_rag.config.loader.resolve_paths`).
    """
    rid = safe_benchmark_bundle_subpath(router_id)
    arch = safe_benchmark_bundle_subpath(
        str(router_architecture_id).strip()
        if router_architecture_id and str(router_architecture_id).strip()
        else "legacy-model"
    )
    mode = safe_benchmark_bundle_subpath(input_mode)
    bench = safe_benchmark_bundle_subpath(f"{rp.benchmark_name}__{rp.benchmark_id}")
    return rp.figures_base / "router" / rid / arch / mode / "benchmark" / bench


def canonical_benchmark_figure_dir(rp: ResolvedPaths) -> Path:
    """Default output for benchmark-scoped figures (no router run path).

    Layout::

        {figures_base}/benchmarks/{benchmark_name}/{benchmark_id}/
    """
    name = safe_benchmark_bundle_subpath(rp.benchmark_name)
    bid = safe_benchmark_bundle_subpath(rp.benchmark_id)
    return rp.figures_base / "benchmarks" / name / bid


def resolve_figure_output_dir(
    cfg: "PipelineConfig",
    rp: ResolvedPaths,
    spec: "BaseFigureSpec",
    output_dir_override: str | Path | None,
) -> Path:
    """Pick figure directory for one plot kind (CLI override > YAML > canonical)."""
    if output_dir_override is not None:
        return Path(str(output_dir_override)).expanduser().resolve()
    fig = cfg.figures
    if fig.output_dir and str(fig.output_dir).strip():
        return Path(str(fig.output_dir).strip()).expanduser().resolve()

    from surf_rag.router.model import parse_router_input_mode
    from surf_rag.viz.specs import (
        BenchmarkOracleHeatmapSpec,
        OracleArgmaxWeightHistogramSpec,
    )

    if isinstance(spec, (BenchmarkOracleHeatmapSpec, OracleArgmaxWeightHistogramSpec)):
        return canonical_benchmark_figure_dir(rp).resolve()

    rt = cfg.router.train
    input_mode = parse_router_input_mode(str(rt.input_mode or "both").strip())
    rid = str(cfg.paths.router_id).strip()
    arch_raw = (
        str(cfg.paths.router_architecture_id).strip()
        if cfg.paths.router_architecture_id
        else ""
    )
    arch_id = arch_raw if arch_raw else None
    return canonical_router_figure_dir(
        rp,
        router_id=rid,
        router_architecture_id=arch_id,
        input_mode=input_mode,
    ).resolve()

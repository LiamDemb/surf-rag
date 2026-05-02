"""Canonical on-disk layout for rendered figures (under ``ResolvedPaths.figures_base``)."""

from __future__ import annotations

from pathlib import Path

from surf_rag.config.loader import ResolvedPaths
from surf_rag.evaluation.artifact_paths import safe_benchmark_bundle_subpath


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
    grouped; ``figures_base`` defaults to ``{{data_base}}/figures`` (see
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

"""Canonical filesystem layout for benchmark bundles, router bundles, and evaluations."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

# Fixed policy folder names for benchmark-local evaluations (future evaluate script).
POLICY_50_50: Final[str] = "50-50"
POLICY_GRAPH_ONLY: Final[str] = "graph-only"
POLICY_DENSE_ONLY: Final[str] = "dense-only"


def default_data_base() -> Path:
    """Root data directory (override with ``DATA_BASE``)."""
    return Path(os.getenv("DATA_BASE", "data"))


def default_benchmark_base() -> Path:
    """Root for benchmark bundles: ``$BENCHMARK_BASE`` or ``$DATA_BASE/benchmarks``."""
    bb = os.getenv("BENCHMARK_BASE")
    if bb and str(bb).strip():
        return Path(bb)
    return default_data_base() / "benchmarks"


def default_router_base() -> Path:
    """Directory containing ``<router_id>/`` router bundles (override ``ROUTER_BASE``)."""
    rb = os.getenv("ROUTER_BASE")
    if rb and str(rb).strip():
        return Path(rb)
    return default_data_base() / "router"


def benchmark_bundle_dir(
    benchmark_base: Path, benchmark_name: str, benchmark_id: str
) -> Path:
    """``benchmark_base`` should be ``default_benchmark_base()`` (not raw ``DATA_BASE``)."""
    return benchmark_base / benchmark_name / benchmark_id


def benchmark_subdir(
    benchmark_base: Path, benchmark_name: str, benchmark_id: str, name: str
) -> Path:
    return benchmark_bundle_dir(benchmark_base, benchmark_name, benchmark_id) / name


def router_bundle_dir(router_base: Path, router_id: str) -> Path:
    return router_base / router_id


def router_oracle_dir(router_base: Path, router_id: str) -> Path:
    return router_bundle_dir(router_base, router_id) / "oracle"


def router_dataset_dir(router_base: Path, router_id: str) -> Path:
    return router_bundle_dir(router_base, router_id) / "dataset"


def router_model_dir(router_base: Path, router_id: str) -> Path:
    return router_bundle_dir(router_base, router_id) / "model"


def evaluations_root(
    benchmark_base: Path, benchmark_name: str, benchmark_id: str
) -> Path:
    return benchmark_subdir(benchmark_base, benchmark_name, benchmark_id, "evaluations")


def evaluation_policy_dir(
    benchmark_base: Path,
    benchmark_name: str,
    benchmark_id: str,
    policy_identifier: str,
) -> Path:
    return (
        evaluations_root(benchmark_base, benchmark_name, benchmark_id)
        / policy_identifier
    )


def trained_router_policy_id(router_id: str) -> str:
    return f"trained-router-{router_id}"


def hard_router_policy_id(router_id: str) -> str:
    return f"hard-router-{router_id}"


def e2e_policy_run_dir(
    benchmark_base: Path,
    benchmark_name: str,
    benchmark_id: str,
    policy: str,
    run_id: str,
) -> Path:
    """One e2e run: ``.../evaluations/<policy>/<run_id>/``."""
    safe = policy.replace("/", "-").replace(" ", "_")
    return evaluation_policy_dir(
        benchmark_base, benchmark_name, benchmark_id, safe
    ) / run_id

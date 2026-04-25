"""Router dataset run directories, manifests, and parquet/JSONL helpers.

Router dataset builds live under a shallow root parallel to oracle runs:

    data/router/<benchmark>/<dataset_id>/
        manifest.json
        router_dataset.parquet
        split_summary.json
        feature_stats.json
        reports/   (optional)

The base directory is overridable via ``ROUTER_DATASET_BASE`` (defaults to
``data/router``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from surf_rag.evaluation.oracle_artifacts import read_jsonl, utc_now_iso


def default_router_dataset_base() -> Path:
    return Path(os.getenv("ROUTER_DATASET_BASE", "data/router"))


def build_router_dataset_root(
    router_base: Path,
    benchmark: str,
    dataset_id: str,
) -> Path:
    return router_base / benchmark / dataset_id


@dataclass(frozen=True)
class RouterDatasetPaths:
    """Standard subpaths inside one router dataset run root."""

    run_root: Path

    @property
    def manifest(self) -> Path:
        return self.run_root / "manifest.json"

    @property
    def router_dataset_parquet(self) -> Path:
        return self.run_root / "router_dataset.parquet"

    @property
    def split_summary(self) -> Path:
        return self.run_root / "split_summary.json"

    @property
    def feature_stats(self) -> Path:
        return self.run_root / "feature_stats.json"

    @property
    def reports_dir(self) -> Path:
        return self.run_root / "reports"

    def ensure_dirs(self) -> None:
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)


def make_router_dataset_paths_for_cli(
    benchmark: str,
    dataset_id: str,
    router_base: Optional[Path] = None,
) -> RouterDatasetPaths:
    base = router_base if router_base is not None else default_router_dataset_base()
    return RouterDatasetPaths(
        run_root=build_router_dataset_root(base, benchmark, dataset_id)
    )


def write_router_dataset_manifest(
    paths: RouterDatasetPaths,
    *,
    dataset_id: str,
    benchmark: str,
    benchmark_path: str,
    oracle_base: str,
    oracle_benchmark: str,
    oracle_split: str,
    oracle_run_id: str,
    oracle_run_root: str,
    labels_selected_path: str,
    selected_beta: float,
    feature_set_version: str,
    embedding_model: str,
    split_seed: int,
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Write ``manifest.json`` for a router dataset build."""
    paths.ensure_dirs()
    data: Dict[str, Any] = {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "dataset_id": dataset_id,
        "benchmark": benchmark,
        "benchmark_path": benchmark_path,
        "oracle": {
            "oracle_base": oracle_base,
            "benchmark": oracle_benchmark,
            "split": oracle_split,
            "oracle_run_id": oracle_run_id,
            "run_root": oracle_run_root,
            "labels_selected": labels_selected_path,
            "selected_beta": float(selected_beta),
        },
        "feature_set_version": feature_set_version,
        "embedding_model": embedding_model,
        "split": {
            "seed": int(split_seed),
            "train_ratio": float(train_ratio),
            "dev_ratio": float(dev_ratio),
            "test_ratio": float(test_ratio),
        },
        "artifacts": {
            "router_dataset": paths.router_dataset_parquet.name,
            "split_summary": paths.split_summary.name,
            "feature_stats": paths.feature_stats.name,
            "reports_dir": paths.reports_dir.name,
        },
    }
    if extra:
        data.update(extra)
    paths.manifest.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_router_dataset_manifest(paths: RouterDatasetPaths) -> Dict[str, Any]:
    return json.loads(paths.manifest.read_text(encoding="utf-8"))


def update_router_dataset_manifest(
    paths: RouterDatasetPaths, updates: Dict[str, Any]
) -> None:
    data = read_router_dataset_manifest(paths)
    data.update(updates)
    data["updated_at"] = utc_now_iso()
    paths.manifest.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_split_summary(
    path: Path, summary: Dict[str, Any], *, run_root: Optional[Path] = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(summary)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    if run_root is not None:
        payload["run_root"] = str(run_root)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_feature_stats(path: Path, stats: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_jsonl_dict(path: Path, key: str) -> Dict[str, Dict[str, Any]]:
    """Read JSONL into ``{row[key]: row}`` for string keys; skips rows missing key."""
    out: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        k = str(row.get(key, "")).strip()
        if k:
            out[k] = row
    return out


def write_parquet(
    path: Path,
    df: pd.DataFrame,
    *,
    engine: str = "pyarrow",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine=engine)

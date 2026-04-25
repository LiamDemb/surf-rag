"""Router dataset run directories, manifests, and parquet/JSONL helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from surf_rag.evaluation.artifact_paths import default_router_base
from surf_rag.evaluation.oracle_artifacts import read_jsonl, utc_now_iso


def build_router_dataset_root(router_base: Path, router_id: str) -> Path:
    return router_base / router_id / "dataset"


@dataclass(frozen=True)
class RouterDatasetPaths:
    """Standard subpaths inside one router's dataset directory."""

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
    def split_question_ids(self) -> Path:
        return self.run_root / "split_question_ids.json"

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
    router_id: str,
    router_base: Optional[Path] = None,
) -> RouterDatasetPaths:
    base = router_base if router_base is not None else default_router_base()
    return RouterDatasetPaths(run_root=build_router_dataset_root(base, router_id))


def build_split_question_ids_dict(
    df: pd.DataFrame,
    *,
    router_id: str,
    source_benchmark_name: str,
    source_benchmark_id: str,
    split_seed: int,
) -> Dict[str, Any]:
    """Shape written to ``split_question_ids.json`` (audit + overlap reporting)."""
    out: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
    for _, row in df.iterrows():
        sp = str(row.get("split", "") or "")
        qid = str(row.get("question_id", "") or "")
        if sp in out and qid:
            out[sp].append(qid)
    counts = {k: len(v) for k, v in out.items()}
    return {
        "router_id": router_id,
        "source_benchmark_name": source_benchmark_name,
        "source_benchmark_id": source_benchmark_id,
        "split_seed": int(split_seed),
        "train": out["train"],
        "dev": out["dev"],
        "test": out["test"],
        "counts": counts,
        "canonical_question_hash_available": False,
    }


def write_split_question_ids(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_router_dataset_manifest(
    paths: RouterDatasetPaths,
    *,
    router_id: str,
    source_benchmark_name: str,
    source_benchmark_id: str,
    benchmark_path: str,
    retrieval_asset_dir: str,
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
    """Write ``manifest.json`` for a router dataset build (schema v2)."""
    paths.ensure_dirs()
    data: Dict[str, Any] = {
        "schema_version": 2,
        "created_at": utc_now_iso(),
        "router_id": router_id,
        "dataset_id": router_id,
        "source_benchmark": {
            "name": source_benchmark_name,
            "id": source_benchmark_id,
            "benchmark_path": benchmark_path,
        },
        "source_corpus": {
            "retrieval_asset_dir": retrieval_asset_dir,
        },
        "oracle": {
            "router_id": router_id,
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
            "split_question_ids": paths.split_question_ids.name,
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

"""Trained router bundle under ``data/router/<router_id>/model/``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from surf_rag.evaluation.artifact_paths import default_router_base
from surf_rag.evaluation.oracle_artifacts import utc_now_iso


def build_router_model_root(router_base: Path, router_id: str) -> Path:
    return router_base / router_id / "model"


@dataclass(frozen=True)
class RouterModelPaths:
    """Standard paths under one router's ``model`` directory."""

    run_root: Path

    @property
    def manifest(self) -> Path:
        return self.run_root / "manifest.json"

    @property
    def checkpoint(self) -> Path:
        return self.run_root / "model.pt"

    @property
    def metrics(self) -> Path:
        return self.run_root / "metrics.json"

    @property
    def training_history(self) -> Path:
        return self.run_root / "training_history.json"

    @property
    def reports_dir(self) -> Path:
        return self.run_root / "reports"

    def predictions(self, split: str) -> Path:
        return self.run_root / f"predictions_{split}.jsonl"

    def ensure_dirs(self) -> None:
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)


def make_router_model_paths_for_cli(
    router_id: str,
    router_base: Optional[Path] = None,
) -> RouterModelPaths:
    base = router_base if router_base is not None else default_router_base()
    return RouterModelPaths(run_root=build_router_model_root(base, router_id))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_router_model_manifest(paths: RouterModelPaths) -> Dict[str, Any]:
    return read_json(paths.manifest)


def update_router_model_manifest(
    paths: RouterModelPaths, updates: Dict[str, Any]
) -> None:
    data = read_router_model_manifest(paths)
    data.update(updates)
    data["updated_at"] = utc_now_iso()
    write_json(paths.manifest, data)


def write_router_model_manifest(
    paths: RouterModelPaths,
    *,
    router_id: str,
    dataset_manifest_path: str,
    model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    feature_set_version: str,
    embedding_model: str,
    weight_grid: List[float],
    source_files: Optional[Dict[str, str]] = None,
) -> None:
    """Write ``manifest.json`` for a trained router (schema v1)."""
    paths.ensure_dirs()
    data: Dict[str, Any] = {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "router_id": router_id,
        "model_id": router_id,
        "source": {
            "router_dataset_manifest": dataset_manifest_path,
        },
        "model": {
            "feature_set_version": feature_set_version,
            "embedding_model": embedding_model,
            "weight_grid": [float(x) for x in weight_grid],
            "architecture": model_config,
        },
        "training": training_config,
        "artifacts": {
            "checkpoint": paths.checkpoint.name,
            "metrics": paths.metrics.name,
            "training_history": paths.training_history.name,
            "reports_dir": paths.reports_dir.name,
        },
    }
    if source_files:
        data["source"]["files"] = source_files
    paths.manifest.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_predictions_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

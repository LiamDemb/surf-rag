"""Trained router bundle paths under ``data/router/<router_id>/models/<arch>/<task>/<mode>/``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from surf_rag.evaluation.artifact_paths import default_router_base
from surf_rag.evaluation.oracle_artifacts import utc_now_iso
from surf_rag.evaluation.run_artifacts import as_resolved_path
from surf_rag.router.model import (
    active_inputs_for_mode,
    parse_router_input_mode,
)

ROUTER_TASK_REGRESSION = "regression"
ROUTER_TASK_CLASSIFICATION = "classification"


def parse_router_task_type(value: str | None) -> str:
    s = str(value or ROUTER_TASK_REGRESSION).strip().lower().replace("_", "-")
    aliases = {
        "regression": ROUTER_TASK_REGRESSION,
        "reg": ROUTER_TASK_REGRESSION,
        "classification": ROUTER_TASK_CLASSIFICATION,
        "class": ROUTER_TASK_CLASSIFICATION,
        "cls": ROUTER_TASK_CLASSIFICATION,
    }
    if s not in aliases:
        raise ValueError(
            "router_task_type must be one of "
            f"{sorted(set(aliases.values()))}, got {value!r}"
        )
    return aliases[s]


def build_router_model_root(
    router_base: Path,
    router_id: str,
    input_mode: str = "both",
    router_architecture_id: str | None = None,
    router_task_type: str = ROUTER_TASK_REGRESSION,
) -> Path:
    mode = parse_router_input_mode(input_mode)
    task = parse_router_task_type(router_task_type)
    if router_architecture_id and str(router_architecture_id).strip():
        rid = str(router_architecture_id).strip()
        return router_base / router_id / "models" / rid / task / mode
    return router_base / router_id / "model" / mode


@dataclass(frozen=True)
class RouterModelPaths:
    """Standard paths under one router's ``model`` directory."""

    run_root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_root", as_resolved_path(self.run_root))

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
    input_mode: str = "both",
    router_architecture_id: str | None = None,
    router_task_type: str = ROUTER_TASK_REGRESSION,
) -> RouterModelPaths:
    base = router_base if router_base is not None else default_router_base()
    return RouterModelPaths(
        run_root=build_router_model_root(
            base,
            router_id,
            input_mode,
            router_architecture_id=router_architecture_id,
            router_task_type=router_task_type,
        )
    )


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
    router_architecture_id: str | None = None,
    input_mode: str = "both",
    dataset_manifest_path: str,
    architecture_name: str = "mlp-v1",
    architecture_kwargs: Optional[Dict[str, Any]] = None,
    model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    feature_set_version: str,
    embedding_model: str,
    weight_grid: List[float],
    source_files: Optional[Dict[str, str]] = None,
    task_type: str = ROUTER_TASK_REGRESSION,
    target_spec: Optional[Dict[str, Any]] = None,
    class_to_weight_map: Optional[Dict[str, float]] = None,
) -> None:
    """Write ``manifest.json`` for a trained router (schema v1)."""
    paths.ensure_dirs()
    mode = parse_router_input_mode(input_mode)
    task = parse_router_task_type(task_type)
    data: Dict[str, Any] = {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "router_id": router_id,
        "router_architecture_id": (
            str(router_architecture_id).strip()
            if router_architecture_id and str(router_architecture_id).strip()
            else None
        ),
        "model_id": (
            f"{router_id}:{str(router_architecture_id).strip()}:{task}:{mode}"
            if router_architecture_id and str(router_architecture_id).strip()
            else f"{router_id}:{task}:{mode}"
        ),
        "task_type": task,
        "source": {
            "router_dataset_manifest": dataset_manifest_path,
        },
        "model": {
            "input_mode": mode,
            "active_inputs": active_inputs_for_mode(mode),
            "feature_set_version": feature_set_version,
            "embedding_model": embedding_model,
            "weight_grid": [float(x) for x in weight_grid],
            "architecture_name": architecture_name,
            "architecture_kwargs": dict(architecture_kwargs or {}),
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
    data["target_spec"] = dict(target_spec or {"name": "oracle_curve"})
    if task == ROUTER_TASK_CLASSIFICATION:
        data["class_to_weight_map"] = dict(
            class_to_weight_map or {"graph": 0.0, "dense": 1.0}
        )
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

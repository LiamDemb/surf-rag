"""Read/write evaluation run manifest.json."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from surf_rag.evaluation.run_artifacts import RunArtifactPaths


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_manifest(
    paths: RunArtifactPaths,
    *,
    run_id: str,
    benchmark: str,
    split: str,
    pipeline_name: str,
    retrieval_asset_dir: str,
    generator_model: str,
    include_graph_provenance: bool,
    completion_window: str,
    artifact_paths: Optional[Dict[str, str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Write manifest relative to run root."""
    paths.run_root.mkdir(parents=True, exist_ok=True)
    rel = artifact_paths or {}
    data: Dict[str, Any] = {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "run_id": run_id,
        "benchmark": benchmark,
        "split": split,
        "pipeline_name": pipeline_name,
        "retrieval_asset_dir": retrieval_asset_dir,
        "generator_model": generator_model,
        "include_graph_provenance": include_graph_provenance,
        "completion_window": completion_window,
        "artifacts": rel,
    }
    if extra:
        data.update(extra)
    paths.manifest.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def update_manifest_artifacts(paths: RunArtifactPaths, artifact_paths: Dict[str, str]) -> None:
    """Merge artifact path entries into existing manifest."""
    if not paths.manifest.is_file():
        return
    data = json.loads(paths.manifest.read_text(encoding="utf-8"))
    art = data.setdefault("artifacts", {})
    art.update(artifact_paths)
    data["updated_at"] = utc_now_iso()
    paths.manifest.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def read_manifest(paths: RunArtifactPaths) -> Dict[str, Any]:
    return json.loads(paths.manifest.read_text(encoding="utf-8"))

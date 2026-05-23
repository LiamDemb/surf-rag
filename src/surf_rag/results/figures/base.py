"""Shared figure IO for results bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from surf_rag.results.bundle import ResultsBundle


def figure_paths(bundle: ResultsBundle, artifact_id: str) -> tuple[Path, Path]:
    fig_dir = bundle.output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ext = bundle.image_format
    return fig_dir / f"{artifact_id}.{ext}", fig_dir / f"{artifact_id}.meta.json"


def write_figure_meta(
    meta_path: Path,
    artifact_id: str,
    extra: dict[str, Any],
) -> None:
    payload = {"artifact_id": artifact_id, **extra}
    meta_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

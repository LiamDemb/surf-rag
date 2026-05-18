"""Write CSV tables and sidecar metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from surf_rag.results.bundle import ResultsBundle


def write_table(
    bundle: ResultsBundle,
    artifact_id: str,
    df: pd.DataFrame,
    meta: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    tables_dir = bundle.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / f"{artifact_id}.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")
    meta_path = tables_dir / f"{artifact_id}.meta.json"
    payload = {"artifact_id": artifact_id, **(meta or {})}
    meta_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return csv_path, meta_path

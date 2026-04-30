"""Exit-code checks for Makefile validate-* targets (paths from resolved config)."""

from __future__ import annotations

import sys
from pathlib import Path

from surf_rag.config.loader import load_pipeline_config, resolve_paths


def _fail(msg: str) -> int:
    print(msg, file=sys.stderr)
    return 1


def validate_oracle(config_path: Path) -> int:
    cfg = load_pipeline_config(config_path)
    r = resolve_paths(cfg)
    if not r.benchmark_path.is_file():
        return _fail(f"Missing benchmark: {r.benchmark_path}")
    if not r.corpus_dir.is_dir():
        return _fail(f"Missing corpus dir: {r.corpus_dir}")
    return 0


def validate_router_dataset(config_path: Path) -> int:
    rc = validate_oracle(config_path)
    if rc != 0:
        return rc
    cfg = load_pipeline_config(config_path)
    r = resolve_paths(cfg)
    labels = r.router_oracle_dir / "router_labels.jsonl"
    if not labels.is_file():
        return _fail(
            f"Missing oracle labels {labels} — run make oracle-labels (CONFIG=...)"
        )
    return 0


def validate_router_train(config_path: Path) -> int:
    rc = validate_router_dataset(config_path)
    if rc != 0:
        return rc
    r = resolve_paths(load_pipeline_config(config_path))
    pq = r.router_dataset_dir / "router_dataset.parquet"
    if not pq.is_file():
        return _fail(f"Missing {pq} — run make router-build-dataset (CONFIG=...)")
    return 0

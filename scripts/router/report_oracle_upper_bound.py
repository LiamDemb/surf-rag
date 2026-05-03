"""Aggregate oracle-upper-bound retrieval metrics on router test split."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from surf_rag.config.env import apply_pipeline_env_from_config, load_app_env
from surf_rag.config.loader import load_pipeline_config, resolve_paths
from surf_rag.evaluation.oracle_artifacts import (
    OracleRunPaths,
    build_oracle_run_root,
    read_jsonl,
)
from surf_rag.evaluation.router_dataset_artifacts import (
    make_router_dataset_paths_for_cli,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Report oracle-upper-bound retrieval metrics for router test split."
    )
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--router-id", default=None)
    p.add_argument("--router-base", type=Path, default=None)
    p.add_argument("--benchmark-path", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _load_test_qids(split_question_ids_path: Path) -> set[str]:
    payload = json.loads(split_question_ids_path.read_text(encoding="utf-8"))
    return {str(qid).strip() for qid in (payload.get("test") or []) if str(qid).strip()}


def _read_oracle_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        qid = str(row.get("question_id", "")).strip()
        if qid:
            rows[qid] = row
    return rows


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    ks = (5, 10, 20)
    out: dict[str, dict[str, float]] = {}
    for k in ks:
        ndcgs: list[float] = []
        hits: list[float] = []
        recalls: list[float] = []
        key = str(k)
        for row in rows:
            scores = list(row.get("scores") or [])
            idx = int(row["best_bin_index"])
            picked = scores[idx]
            ndcgs.append(float((picked.get("diagnostic_ndcg") or {}).get(key, 0.0)))
            hits.append(float((picked.get("diagnostic_hit") or {}).get(key, 0.0)))
            recalls.append(float((picked.get("diagnostic_recall") or {}).get(key, 0.0)))
        out[key] = {"ndcg": _mean(ndcgs), "hit": _mean(hits), "recall": _mean(recalls)}
    return out


def main() -> int:
    load_app_env()
    load_dotenv()
    args = _parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s: %(message)s")
    log = logging.getLogger(__name__)

    cfg = None
    if args.config:
        cfg = load_pipeline_config(args.config.resolve())
        apply_pipeline_env_from_config(cfg)
    rp = resolve_paths(cfg) if cfg else None

    router_id = (
        str(args.router_id).strip()
        if args.router_id
        else (str(rp.router_id).strip() if rp else "")
    )
    if not router_id:
        log.error(
            "Missing router id. Provide --router-id or --config with paths.router_id."
        )
        return 2
    router_base = (
        args.router_base.resolve()
        if args.router_base
        else (rp.router_base if rp else Path("data/router").resolve())
    )
    benchmark_path = (
        args.benchmark_path.resolve()
        if args.benchmark_path
        else (rp.benchmark_path if rp else None)
    )
    if benchmark_path is None or not benchmark_path.is_file():
        log.error("Missing benchmark path. Provide --benchmark-path or --config.")
        return 2

    oracle_paths = OracleRunPaths(
        run_root=build_oracle_run_root(router_base, router_id)
    )
    ds_paths = make_router_dataset_paths_for_cli(router_id, router_base=router_base)
    if not ds_paths.split_question_ids.is_file():
        log.error("Missing split ids: %s", ds_paths.split_question_ids)
        return 1
    if not oracle_paths.oracle_scores.is_file():
        log.error("Missing oracle scores: %s", oracle_paths.oracle_scores)
        return 1

    test_qids = _load_test_qids(ds_paths.split_question_ids)
    if not test_qids:
        log.error("Router test split is empty: %s", ds_paths.split_question_ids)
        return 1
    oracle_by_qid = _read_oracle_rows(oracle_paths.oracle_scores)
    missing = sorted(qid for qid in test_qids if qid not in oracle_by_qid)
    if missing:
        preview = ", ".join(missing[:10])
        log.error(
            "Strict check failed: %d test qids missing from oracle_scores (showing up to 10): %s",
            len(missing),
            preview,
        )
        return 1
    selected_rows = [oracle_by_qid[qid] for qid in sorted(test_qids)]
    retrieval_at_k = _aggregate(selected_rows)

    payload = {
        "router_id": router_id,
        "benchmark_name": (rp.benchmark_name if rp else None),
        "benchmark_id": (rp.benchmark_id if rp else None),
        "benchmark_path": str(benchmark_path),
        "oracle_scores_path": str(oracle_paths.oracle_scores),
        "split_question_ids_path": str(ds_paths.split_question_ids),
        "expected_test_qids": len(test_qids),
        "evaluated_qids": len(selected_rows),
        "missing_qids": [],
        "retrieval_at_k": retrieval_at_k,
    }
    out_path = (
        args.output.resolve()
        if args.output
        else oracle_paths.reports_dir / "oracle_upper_bound_test.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    log.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

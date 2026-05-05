#!/usr/bin/env python3
"""Compute the mean score achieved by a steady fuser that constantly predicts each specific weight."""

import argparse
import json
import logging
from pathlib import Path
from collections import defaultdict
import numpy as np

from surf_rag.config.loader import load_pipeline_config
from surf_rag.evaluation.oracle_artifacts import (
    make_run_paths_for_cli,
    read_oracle_score_rows,
)
from surf_rag.evaluation.router_dataset_artifacts import (
    make_router_dataset_paths_for_cli,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to pipeline config yaml (e.g. configs/router-rg.yaml)",
    )
    args = parser.parse_args()

    cfg = load_pipeline_config(Path(args.config))

    benchmark_base = Path(cfg.paths.benchmark_base)
    benchmark_name = cfg.paths.benchmark_name
    benchmark_id = cfg.paths.benchmark_id
    router_base = Path(cfg.paths.router_base)
    router_id = cfg.paths.router_id

    # Output directory
    metrics_dir = benchmark_base / benchmark_name / benchmark_id / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / "mean_steady_fuser_scores.json"

    # Load splits
    ds_paths = make_router_dataset_paths_for_cli(router_id, router_base=router_base)

    qid_to_split = {}
    if ds_paths.split_question_ids.is_file():
        splits = json.loads(ds_paths.split_question_ids.read_text(encoding="utf-8"))
        for split_name in ["train", "dev", "test"]:
            for qid in splits.get(split_name, []):
                qid_to_split[qid] = split_name
    else:
        logging.warning(
            "Could not find split_question_ids. Split breakdown will not be available."
        )

    # Get dataset source and scores from oracle scores
    oracle_paths = make_run_paths_for_cli(router_id, router_base=router_base)
    if not oracle_paths.oracle_scores.is_file():
        logging.error(f"Oracle scores not found: {oracle_paths.oracle_scores}")
        return

    # Dictionary mapping: group_name -> { weight_str -> list_of_scores }
    split_weight_scores = defaultdict(lambda: defaultdict(list))

    for row in read_oracle_score_rows(oracle_paths):
        qid = row["question_id"]
        split = qid_to_split.get(qid, "unknown")
        source = row.get("dataset_source", "unknown")

        weight_grid = np.array(row["weight_grid"], dtype=float)
        scores = np.array(
            [s["oracle_objective_value"] for s in row["scores"]], dtype=float
        )

        if len(weight_grid) == 0:
            continue

        for w, s in zip(weight_grid, scores):
            w_str = f"{w:.1f}"
            split_weight_scores[split][w_str].append(s)
            split_weight_scores["overall"][w_str].append(s)
            split_weight_scores[source][w_str].append(s)

    results = {}
    for group, weight_scores in split_weight_scores.items():
        group_results = {}
        for w_str in sorted(weight_scores.keys(), key=float):
            scores = weight_scores[w_str]
            if len(scores) > 0:
                group_results[w_str] = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "count": len(scores),
                }
            else:
                group_results[w_str] = {"mean": 0.0, "std": 0.0, "count": 0}
        results[group] = group_results

    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    logging.info(f"Wrote mean steady fuser scores to {out_path}")


if __name__ == "__main__":
    main()

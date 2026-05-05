#!/usr/bin/env python3
"""Compute the mean predictions of the router for a given benchmark."""

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
from surf_rag.evaluation.router_model_artifacts import make_router_model_paths_for_cli

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

    router_arch_id = getattr(cfg.paths, "router_architecture_id", None)
    safe_arch_id = router_arch_id.split("/")[-1] if router_arch_id else "unknown_arch"

    # Output directory
    metrics_dir = router_base / router_id / "models" / safe_arch_id / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_filename = "mean_router_predictions.json"
    out_path = metrics_dir / out_filename

    # Get dataset source from oracle scores
    oracle_paths = make_run_paths_for_cli(router_id, router_base=router_base)
    qid_to_source = {}
    if oracle_paths.oracle_scores.is_file():
        for row in read_oracle_score_rows(oracle_paths):
            qid = row["question_id"]
            qid_to_source[qid] = row.get("dataset_source", "unknown")
    else:
        logging.warning(
            "Oracle scores not found. Dataset source breakdown will not be accurate."
        )

    # Get predictions
    input_mode = getattr(cfg.router.train, "input_mode", "both")
    task_type = getattr(cfg.router.train, "task_type", "regression")

    model_paths = make_router_model_paths_for_cli(
        router_id=router_id,
        router_base=router_base,
        input_mode=input_mode,
        router_architecture_id=router_arch_id,
        router_task_type=task_type,
    )

    split_sums = defaultdict(float)
    split_counts = defaultdict(int)
    split_preds = defaultdict(list)

    for split in ["train", "dev", "test"]:
        pred_path = model_paths.predictions(split)
        if not pred_path.is_file() and router_arch_id and "/" in router_arch_id:
            # Try basename fallback
            fallback_arch = router_arch_id.split("/")[-1]
            fallback_paths = make_router_model_paths_for_cli(
                router_id=router_id,
                router_base=router_base,
                input_mode=input_mode,
                router_architecture_id=fallback_arch,
                router_task_type=task_type,
            )
            pred_path = fallback_paths.predictions(split)

        if not pred_path.is_file():
            logging.warning(f"Prediction file not found for split {split}: {pred_path}")
            continue

        with pred_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                qid = row["question_id"]
                pred = float(row["predicted_weight"])
                source = qid_to_source.get(qid, "unknown")

                split_sums[split] += pred
                split_counts[split] += 1
                split_preds[split].append(pred)

                split_sums["overall"] += pred
                split_counts["overall"] += 1
                split_preds["overall"].append(pred)

                split_sums[source] += pred
                split_counts[source] += 1
                split_preds[source].append(pred)

    results = {}
    for key in split_counts:
        mean_val = split_sums[key] / split_counts[key] if split_counts[key] > 0 else 0.0
        std_val = float(np.std(split_preds[key])) if len(split_preds[key]) > 0 else 0.0
        results[key] = {"mean": mean_val, "std": std_val, "count": split_counts[key]}

    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    logging.info(f"Wrote mean router predictions to {out_path}")


if __name__ == "__main__":
    main()

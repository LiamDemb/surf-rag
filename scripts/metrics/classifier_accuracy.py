#!/usr/bin/env python3
"""Compute accuracy and confusion matrix for a hard routing classifier."""

import argparse
import json
import logging
from pathlib import Path
from collections import defaultdict

from surf_rag.config.loader import load_pipeline_config
from surf_rag.evaluation.oracle_artifacts import (
    make_run_paths_for_cli,
    read_oracle_score_rows,
)
from surf_rag.evaluation.router_model_artifacts import make_router_model_paths_for_cli

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def compute_metrics(predictions):
    if not predictions:
        return {"accuracy": 0.0, "confusion_matrix": {}, "count": 0}

    correct = 0
    # confusion_matrix[target][predicted]
    conf_matrix = defaultdict(lambda: defaultdict(int))

    for p in predictions:
        # Ignore if missing target or predicted class
        if "target_class_id" not in p or "predicted_class_id" not in p:
            continue

        target = p["target_class_id"]
        pred = p["predicted_class_id"]

        # Sometimes predictions for invalid queries might be None, ignore them or treat as wrong
        if pred is None:
            continue

        if target == pred:
            correct += 1
        conf_matrix[target][pred] += 1

    # We only count predictions where `pred` was not None
    total_valid = sum(sum(preds.values()) for preds in conf_matrix.values())
    acc = correct / total_valid if total_valid > 0 else 0.0

    # Format confusion matrix for JSON (convert int keys to string)
    cm_formatted = {}
    for t_class, p_dict in sorted(conf_matrix.items()):
        cm_formatted[str(t_class)] = {
            str(p_class): count for p_class, count in sorted(p_dict.items())
        }

    return {"accuracy": acc, "confusion_matrix": cm_formatted, "count": total_valid}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to pipeline config yaml (e.g. configs/router-cls.yaml)",
    )
    args = parser.parse_args()

    cfg = load_pipeline_config(Path(args.config))

    router_base = Path(cfg.paths.router_base)
    router_id = cfg.paths.router_id

    router_arch_id = getattr(cfg.paths, "router_architecture_id", None)
    safe_arch_id = router_arch_id.split("/")[-1] if router_arch_id else "unknown_arch"

    # Output directory
    metrics_dir = router_base / router_id / "models" / safe_arch_id / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_filename = "classifier_accuracy.json"
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
    # Determine task_type, but default to classification if it's a cls model
    task_type = getattr(cfg.router.train, "task_type", "classification")

    model_paths = make_router_model_paths_for_cli(
        router_id=router_id,
        router_base=router_base,
        input_mode=input_mode,
        router_architecture_id=router_arch_id,
        router_task_type=task_type,
    )

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
                source = qid_to_source.get(qid, "unknown")

                split_preds[split].append(row)
                split_preds["overall"].append(row)
                split_preds[source].append(row)

    results = {}
    for key, preds in split_preds.items():
        results[key] = compute_metrics(preds)

    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    logging.info(f"Wrote classifier accuracy metrics to {out_path}")


if __name__ == "__main__":
    main()

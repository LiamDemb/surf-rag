"""Classifier quality and routing regret table."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from surf_rag.evaluation.router_model_artifacts import (
    ROUTER_TASK_CLASSIFICATION,
    make_router_model_paths_for_cli,
)
from surf_rag.results.bundle import ResultsBundle, get_router_arch
from surf_rag.results.loaders import iter_predictions_jsonl
from surf_rag.results.tables.regressor_metrics import SkippedArtifact
from surf_rag.results.tables.writer import write_table


def build_classifier_metrics(bundle: ResultsBundle) -> tuple[pd.DataFrame, dict]:
    arch = get_router_arch(bundle, "classifier")
    if arch is None or not arch.architecture_id.strip():
        raise SkippedArtifact("results.router.classifier not configured")

    paths = make_router_model_paths_for_cli(
        bundle.resolved.router_id,
        router_base=bundle.resolved.router_base,
        input_mode=arch.input_mode,
        router_architecture_id=arch.architecture_id,
        router_task_type=ROUTER_TASK_CLASSIFICATION,
    )
    if not paths.checkpoint.is_file():
        raise SkippedArtifact(f"Classifier checkpoint missing: {paths.checkpoint}")

    split = bundle.results.split
    pred_path = paths.predictions(split)
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for row in iter_predictions_jsonl(pred_path):
        qid = str(row.get("question_id", "")).strip()
        if qid not in bundle.split_qids:
            continue
        if row.get("predicted_class_id") is None or row.get("target_class_id") is None:
            continue
        pred = int(row["predicted_class_id"])
        target = int(row["target_class_id"])
        if pred not in (0, 1) or target not in (0, 1):
            continue
        src = bundle.qid_to_source.get(qid, "unknown")
        curve = list(row.get("oracle_curve") or [])
        best = float(row.get("target_oracle_best_score", 0.0))
        score = float(curve[-1] if pred == 1 and curve else curve[0] if curve else 0.0)
        regret = max(0.0, best - score)
        rec = {"correct": pred == target, "target": target, "regret": regret}
        buckets[(split, "all")].append(rec)
        if src in ("nq", "2wiki"):
            buckets[(split, src)].append(rec)

    rows: list[dict] = []
    for (sp, src), items in sorted(buckets.items()):
        if not items:
            continue
        n = len(items)
        correct = sum(1 for x in items if x["correct"])
        targets = [x["target"] for x in items]
        majority = max(set(targets), key=targets.count)
        maj_acc = sum(1 for t in targets if t == majority) / n
        rows.append(
            {
                "split": sp,
                "dataset_source": src,
                "accuracy": correct / n,
                "majority_baseline_accuracy": maj_acc,
                "routing_regret_mean": float(np.mean([x["regret"] for x in items])),
                "n": n,
            }
        )
    if not rows:
        raise SkippedArtifact(f"No classifier predictions on split {split!r}")
    df = pd.DataFrame(rows)
    paths_out = write_table(
        bundle,
        "classifier_metrics",
        df,
        {"architecture_id": arch.architecture_id},
    )
    return df, {"csv": str(paths_out[0]), "meta": str(paths_out[1])}

"""Classifier routing confusion with mean oracle-curve regret per cell (results split)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from surf_rag.config.schema import ResultsArtifactSpec
from surf_rag.evaluation.router_model_artifacts import (
    ROUTER_TASK_CLASSIFICATION,
    make_router_model_paths_for_cli,
)
from surf_rag.results.bundle import ResultsBundle, get_router_arch
from surf_rag.results.figures._confusion_matrix import (
    accumulate_mean_regret_matrix,
    new_regret_buckets,
    render_binary_confusion_heatmap,
    routing_regret_for_prediction,
)
from surf_rag.results.figures.base import figure_paths, write_figure_meta
from surf_rag.results.loaders import iter_predictions_jsonl
from surf_rag.results.tables.regressor_metrics import SkippedArtifact


def render_classifier_regret_confusion(
    bundle: ResultsBundle, spec: ResultsArtifactSpec
) -> dict[str, str]:
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

    split = bundle.results.split
    pred_path = paths.predictions(split)
    if not pred_path.is_file():
        raise SkippedArtifact(f"Missing predictions: {pred_path}")

    buckets = new_regret_buckets()
    for row in iter_predictions_jsonl(pred_path):
        qid = str(row.get("question_id", "")).strip()
        if qid not in bundle.split_qids:
            continue
        pred = row.get("predicted_class_id")
        target = row.get("target_class_id")
        if pred is None or target is None:
            continue
        pred_i, tgt_i = int(pred), int(target)
        if pred_i not in (0, 1) or tgt_i not in (0, 1):
            continue
        regret = routing_regret_for_prediction(row, pred_i)
        buckets[(tgt_i, pred_i)].append(regret)

    mean_m, counts = accumulate_mean_regret_matrix(buckets)
    if not counts.any():
        raise SkippedArtifact(f"No classifier predictions on split {split!r}")

    def annotate_cell(i: int, j: int, val: float) -> str:
        n = int(counts[i, j])
        if not np.isfinite(val):
            return f"—\n(n={n})" if n else "—"
        # return f"{val:.4f}\n(n={n})"
        return f"{val:.4f}"

    fig = render_binary_confusion_heatmap(
        mean_m,
        palette_key="red",
        title=f"Mean routing regret by route ({split} split)",
        annotate=annotate_cell,
        vmin=0.0,
        cbar_label="Mean regret",
    )
    img_path, meta_path = figure_paths(bundle, spec.id)
    fig.savefig(img_path, format=bundle.image_format)
    plt.close(fig)
    write_figure_meta(
        meta_path,
        spec.id,
        {
            "split": split,
            "mean_regret": [
                [
                    None if not np.isfinite(mean_m[i, j]) else float(mean_m[i, j])
                    for j in range(2)
                ]
                for i in range(2)
            ],
            "counts": counts.tolist(),
            "palette": "red",
        },
    )
    return {"image": str(img_path), "meta": str(meta_path)}

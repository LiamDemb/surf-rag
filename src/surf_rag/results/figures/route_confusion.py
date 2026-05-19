"""Routing classifier confusion matrix."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from surf_rag.config.schema import ResultsArtifactSpec
from surf_rag.evaluation.router_model_artifacts import (
    ROUTER_TASK_CLASSIFICATION,
    make_router_model_paths_for_cli,
)
from surf_rag.results.bundle import ResultsBundle, get_router_arch
from surf_rag.results.figures.base import figure_paths, write_figure_meta
from surf_rag.results.loaders import iter_predictions_jsonl
from surf_rag.results.tables.regressor_metrics import SkippedArtifact
from surf_rag.viz.theme import PALETTE


def render_route_confusion(
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
    pred_path = paths.predictions(bundle.results.split)
    if not pred_path.is_file():
        raise SkippedArtifact(f"Missing predictions: {pred_path}")

    cm = np.zeros((2, 2), dtype=int)
    for row in iter_predictions_jsonl(pred_path):
        qid = str(row.get("question_id", "")).strip()
        if qid not in bundle.split_qids:
            continue
        pred = row.get("predicted_class_id")
        target = row.get("target_class_id")
        if pred is None or target is None:
            continue
        pred_i, tgt_i = int(pred), int(target)
        if pred_i in (0, 1) and tgt_i in (0, 1):
            cm[tgt_i, pred_i] += 1

    row_sum = cm.sum(axis=1, keepdims=True)
    pct = np.zeros_like(cm, dtype=float)
    np.divide(cm, row_sum, out=pct, where=row_sum > 0)

    img_path, meta_path = figure_paths(bundle, spec.id)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["GraphRAG", "DenseRAG"]
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Routing confusion")
    for i in range(2):
        for j in range(2):
            color = "white" if im.norm(cm[i, j]) > 0.5 else PALETTE["text"]
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({100 * pct[i, j]:.0f}%)",
                ha="center",
                va="center",
                color=color,
                fontsize=10,
            )
    fig.tight_layout()
    fig.savefig(img_path, format=bundle.image_format)
    plt.close(fig)
    write_figure_meta(meta_path, spec.id, {"counts": cm.tolist()})
    return {"image": str(img_path), "meta": str(meta_path)}

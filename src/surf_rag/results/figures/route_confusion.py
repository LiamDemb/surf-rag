"""Routing classifier confusion matrix (counts) on the configured results split."""

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
    accumulate_confusion_counts,
    render_binary_confusion_heatmap,
)
from surf_rag.results.figures.base import figure_paths, write_figure_meta
from surf_rag.results.loaders import iter_predictions_jsonl
from surf_rag.results.tables.regressor_metrics import SkippedArtifact


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
    split = bundle.results.split
    pred_path = paths.predictions(split)
    if not pred_path.is_file():
        raise SkippedArtifact(f"Missing predictions: {pred_path}")

    pairs: list[tuple[int, int]] = []
    for row in iter_predictions_jsonl(pred_path):
        qid = str(row.get("question_id", "")).strip()
        if qid not in bundle.split_qids:
            continue
        pred = row.get("predicted_class_id")
        target = row.get("target_class_id")
        if pred is None or target is None:
            continue
        pairs.append((int(target), int(pred)))

    cm = accumulate_confusion_counts(pairs)
    row_sum = cm.sum(axis=1, keepdims=True)
    pct = np.zeros_like(cm, dtype=float)
    np.divide(cm, row_sum, out=pct, where=row_sum > 0)

    def annotate(i: int, j: int, val: float) -> str:
        count = int(cm[i, j])
        return f"{count}\n({100 * pct[i, j]:.0f}%)"

    fig = render_binary_confusion_heatmap(
        cm.astype(float),
        palette_key="primary",
        title=f"Routing confusion ({split} split)",
        annotate=annotate,
        vmin=0.0,
        vmax=float(cm.max()) if cm.max() > 0 else 1.0,
        cbar_label="Count",
    )
    img_path, meta_path = figure_paths(bundle, spec.id)
    fig.savefig(img_path, format=bundle.image_format)
    plt.close(fig)
    write_figure_meta(
        meta_path,
        spec.id,
        {"split": split, "counts": cm.tolist(), "palette": "primary"},
    )
    return {"image": str(img_path), "meta": str(meta_path)}

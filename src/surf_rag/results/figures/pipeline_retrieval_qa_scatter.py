"""Scatter: mean retrieval NDCG@k vs mean LLM-judge QA accuracy, one point per pipeline."""

from __future__ import annotations

import matplotlib.pyplot as plt

from surf_rag.config.schema import ResultsArtifactSpec
from surf_rag.results.bundle import ResultsBundle, policy_list
from surf_rag.results.figures.base import figure_paths, write_figure_meta
from surf_rag.results.figures.retrieval_answer_gain import _aggregate_for_source
from surf_rag.results.metric_fields import resolve_retrieval_metric_k
from surf_rag.viz.theme import PALETTE, policy_color

_POLICY_LABELS: dict[str, str] = {
    "dense-only": "DenseRAG-only",
    "graph-only": "GraphRAG-only",
    "50-50": "Fixed Fusion",
    "learned-soft": "SuRF-RAG",
    "hard-routing": "Hard-Routing",
    "rrf": "RRF",
    "oracle-upper-bound": "Fusion Oracle",
    "oracle-classification": "Routing Oracle",
}

# Offset annotations in points to reduce overlap (one policy per direction cycle).
_LABEL_OFFSETS: tuple[tuple[float, float], ...] = (
    (10, 6),
    (10, -8),
    (-10, 6),
    (-10, -8),
    (14, 0),
    (-14, 0),
    (0, 10),
    (0, -10),
)


def _retrieval_axis_label(metric: str, k: int) -> str:
    m = metric.strip().lower()
    if m in ("stateful_ndcg", "ndcg"):
        return f"NDCG@{k}"
    if m == "hit":
        return f"Hit@{k}"
    if m == "recall":
        return f"Recall@{k}"
    return f"{metric}@{k}"


def _policy_label(policy: str) -> str:
    return _POLICY_LABELS.get(policy, policy.replace("-", " ").title())


def render_pipeline_retrieval_qa_scatter(
    bundle: ResultsBundle, spec: ResultsArtifactSpec
) -> dict[str, str]:
    metric, k = resolve_retrieval_metric_k(spec, bundle)
    split = bundle.results.split
    policies = policy_list(bundle)

    points: list[dict] = []
    for i, policy in enumerate(policies):
        agg = _aggregate_for_source(bundle, policy, "all", metric=metric, k=k)
        if not agg["retrieval"] and not agg["accuracy"]:
            continue
        points.append(
            {
                "policy": policy,
                "label": _policy_label(policy),
                "ndcg": agg["retrieval"],
                "accuracy": agg["accuracy"],
                "color_idx": i,
            }
        )

    if not points:
        raise ValueError(
            "No pipeline points with retrieval and QA accuracy on the split"
        )

    img_path, meta_path = figure_paths(bundle, spec.id)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    for i, pt in enumerate(points):
        ax.scatter(
            pt["ndcg"],
            pt["accuracy"],
            color=policy_color(pt["color_idx"]),
            s=88,
            edgecolors=PALETTE["text"],
            linewidths=0.4,
            zorder=2,
        )
        ox, oy = _LABEL_OFFSETS[i % len(_LABEL_OFFSETS)]
        ha = "left" if ox >= 0 else "right"
        va = "bottom" if oy >= 0 else "top"
        ax.annotate(
            pt["label"],
            (pt["ndcg"], pt["accuracy"]),
            xytext=(ox, oy),
            textcoords="offset points",
            fontsize=9,
            color=PALETTE["text"],
            ha=ha,
            va=va,
            clip_on=False,
        )

    ax.set_xlabel(_retrieval_axis_label(metric, k))
    ax.set_ylabel("QA accuracy (LLM judge)")
    # ax.set_title(f"Retrieval vs answer quality ({split} split)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(img_path, format=bundle.image_format)
    plt.close(fig)
    write_figure_meta(
        meta_path,
        spec.id,
        {
            "split": split,
            "metric": metric,
            "k": k,
            "policies": [
                {
                    "policy": p["policy"],
                    "ndcg": p["ndcg"],
                    "accuracy": p["accuracy"],
                }
                for p in points
            ],
        },
    )
    return {"image": str(img_path), "meta": str(meta_path)}

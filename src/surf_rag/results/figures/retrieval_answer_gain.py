"""Scatter: retrieval gain vs answer-accuracy gain vs dense-only baseline."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from surf_rag.config.schema import ResultsArtifactSpec
from surf_rag.results.bundle import ResultsBundle, policy_list
from surf_rag.results.figures.base import figure_paths, write_figure_meta
from surf_rag.results.loaders import load_answerability, load_policy_metrics
from surf_rag.results.metric_fields import (
    e2e_retrieval_value,
    resolve_retrieval_metric_k,
)
from surf_rag.viz.theme import (
    DATASET_SOURCE_LABELS,
    DATASET_SOURCE_MARKERS,
    PALETTE,
    policy_color,
)


def render_retrieval_answer_gain(
    bundle: ResultsBundle, spec: ResultsArtifactSpec
) -> dict[str, str]:
    metric, k = resolve_retrieval_metric_k(spec, bundle)
    baseline = "dense-only"
    if baseline not in bundle.policies:
        raise FileNotFoundError(
            f"retrieval_answer_gain requires baseline policy {baseline!r}"
        )
    policies = [p for p in policy_list(bundle) if p != baseline]
    policy_to_idx = {p: i for i, p in enumerate(policies)}

    points: list[dict] = []
    for policy in policies:
        for src in ("nq", "2wiki", "all"):
            cur = _aggregate_for_source(bundle, policy, src, metric=metric, k=k)
            base = _aggregate_for_source(bundle, baseline, src, metric=metric, k=k)
            points.append(
                {
                    "policy": policy,
                    "dataset_source": src,
                    "retrieval_gain": cur["retrieval"] - base["retrieval"],
                    "accuracy_gain": cur["accuracy"] - base["accuracy"],
                    "color_idx": policy_to_idx[policy],
                }
            )

    img_path, meta_path = figure_paths(bundle, spec.id)
    fig, ax = plt.subplots(figsize=(6, 6.5))
    for pt in points:
        src = pt["dataset_source"]
        ax.scatter(
            pt["retrieval_gain"],
            pt["accuracy_gain"],
            color=policy_color(pt["color_idx"]),
            marker=DATASET_SOURCE_MARKERS.get(src, "o"),
            s=72,
            edgecolors=PALETTE["text"],
            linewidths=0.35,
            zorder=2,
        )

    if points:
        x_vals = [p["retrieval_gain"] for p in points]
        y_vals = [p["accuracy_gain"] for p in points]
        lo = min(min(x_vals), min(y_vals)) - 0.02
        hi = max(max(x_vals), max(y_vals)) + 0.02
    else:
        lo, hi = -0.05, 0.05
    ax.plot(
        [lo, hi],
        [lo, hi],
        color=PALETTE["identity_line"],
        linestyle="--",
        linewidth=1.0,
        zorder=0,
    )
    ax.axhline(0, color=PALETTE["grid"], linewidth=0.8, zorder=0)
    ax.axvline(0, color=PALETTE["grid"], linewidth=0.8, zorder=0)
    ax.set_xlabel(f"Δ {metric}@{k} vs {baseline}")
    ax.set_ylabel("Δ answer accuracy vs baseline")
    ax.set_title("Retrieval vs answer gain")

    policy_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=policy_color(i),
            markeredgecolor=PALETTE["text"],
            markeredgewidth=0.35,
            markersize=8,
            linestyle="None",
            label=policy,
        )
        for i, policy in enumerate(policies)
    ]
    dataset_handles = [
        Line2D(
            [0],
            [0],
            marker=DATASET_SOURCE_MARKERS.get(src, "o"),
            color="w",
            markerfacecolor=PALETTE["grid"],
            markeredgecolor=PALETTE["text"],
            markeredgewidth=0.5,
            markersize=8,
            linestyle="None",
            label=DATASET_SOURCE_LABELS.get(src, src),
        )
        for src in ("nq", "2wiki", "all")
    ]
    ncol_policy = min(4, max(1, len(policy_handles)))
    fig.subplots_adjust(bottom=0.26)
    leg_policy = fig.legend(
        handles=policy_handles,
        title="Policy (colour)",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.11),
        ncol=ncol_policy,
        frameon=True,
    )
    fig.legend(
        handles=dataset_handles,
        title="Dataset (marker)",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=True,
    )
    ax.add_artist(leg_policy)

    fig.tight_layout(rect=(0, 0.22, 1, 1))
    fig.savefig(img_path, format=bundle.image_format)
    plt.close(fig)
    write_figure_meta(
        meta_path,
        spec.id,
        {
            "n_points": len(points),
            "metric": metric,
            "baseline": baseline,
            "color_legend": "policy",
            "marker_legend": "dataset_source",
        },
    )
    return {"image": str(img_path), "meta": str(meta_path)}


def _aggregate_for_source(
    bundle: ResultsBundle,
    policy: str,
    src: str,
    *,
    metric: str,
    k: int,
) -> dict[str, float]:
    answerable = load_answerability(bundle.answerability_path)
    per_q = (
        load_policy_metrics(bundle.policies[policy].metrics_path).get("per_question")
        or []
    )
    ret_vals: list[float] = []
    acc_vals: list[float] = []
    for row in per_q:
        qid = str(row.get("question_id", "")).strip()
        if qid not in bundle.split_qids:
            continue
        qsrc = bundle.qid_to_source.get(qid, "")
        if src != "all" and qsrc != src:
            continue
        ret_vals.append(e2e_retrieval_value(row, metric=metric, k=k))
        judge = row.get("qa_llm_judge")
        if isinstance(judge, dict) and "correct" in judge:
            acc_vals.append(1.0 if judge["correct"] else 0.0)
    return {
        "retrieval": float(sum(ret_vals) / len(ret_vals)) if ret_vals else 0.0,
        "accuracy": float(sum(acc_vals) / len(acc_vals)) if acc_vals else 0.0,
    }

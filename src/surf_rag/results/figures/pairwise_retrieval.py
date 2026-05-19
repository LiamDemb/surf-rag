"""Per-query scatter comparing two policies' retrieval scores."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from surf_rag.config.schema import ResultsArtifactSpec
from surf_rag.results.bundle import ResultsBundle
from surf_rag.results.figures.base import figure_paths, write_figure_meta
from surf_rag.results.loaders import load_policy_metrics
from surf_rag.results.metric_fields import (
    e2e_retrieval_value,
    resolve_retrieval_metric_k,
)
from surf_rag.viz.theme import (
    DATASET_SOURCE_COLORS,
    DATASET_SOURCE_LABELS,
    DATASET_SOURCE_MARKERS,
    PALETTE,
)


def render_pairwise_retrieval(
    bundle: ResultsBundle, spec: ResultsArtifactSpec
) -> dict[str, str]:
    x_policy = spec.x_policy
    y_policy = spec.y_policy
    if not x_policy or not y_policy:
        raise ValueError(
            f"Artifact {spec.id!r} requires x_policy and y_policy in config"
        )
    if x_policy not in bundle.policies or y_policy not in bundle.policies:
        raise FileNotFoundError(
            f"Policies {x_policy!r} and/or {y_policy!r} not in results.policies"
        )
    metric, k = resolve_retrieval_metric_k(spec, bundle)

    def scores(policy: str) -> dict[str, float]:
        per_q = (
            load_policy_metrics(bundle.policies[policy].metrics_path).get(
                "per_question"
            )
            or []
        )
        out: dict[str, float] = {}
        for row in per_q:
            qid = str(row.get("question_id", "")).strip()
            if qid in bundle.split_qids:
                out[qid] = e2e_retrieval_value(row, metric=metric, k=k)
        return out

    xs = scores(x_policy)
    ys = scores(y_policy)
    common = sorted(set(xs) & set(ys))

    img_path, meta_path = figure_paths(bundle, spec.id)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    for src in ("nq", "2wiki"):
        qids = [q for q in common if bundle.qid_to_source.get(q, "") == src]
        if not qids:
            continue
        x_vals = [xs[q] for q in qids]
        y_vals = [ys[q] for q in qids]
        ax.scatter(
            x_vals,
            y_vals,
            label=DATASET_SOURCE_LABELS.get(src, src),
            c=DATASET_SOURCE_COLORS.get(src, PALETTE["primary"]),
            marker=DATASET_SOURCE_MARKERS.get(src, "o"),
            s=28,
            alpha=0.85,
            edgecolors=PALETTE["text"],
            linewidths=0.35,
        )

    other_qids = [
        q for q in common if bundle.qid_to_source.get(q, "") not in ("nq", "2wiki")
    ]
    if other_qids:
        ax.scatter(
            [xs[q] for q in other_qids],
            [ys[q] for q in other_qids],
            label="Other",
            c=PALETTE["grid"],
            marker="^",
            s=24,
            alpha=0.85,
            edgecolors=PALETTE["text"],
            linewidths=0.35,
        )

    all_vals = [xs[q] for q in common] + [ys[q] for q in common]
    lo = min(all_vals + [0.0])
    hi = max(all_vals + [1.0])
    pad = 0.02 * (hi - lo) if hi > lo else 0.05
    ax.plot(
        [lo - pad, hi + pad],
        [lo - pad, hi + pad],
        color=PALETTE["identity_line"],
        linestyle="--",
        linewidth=1.0,
        zorder=0,
        label="No change",
    )
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(f"{x_policy} — {metric}@{k}")
    ax.set_ylabel(f"{y_policy} — {metric}@{k}")
    ax.set_title(f"Per-query: {y_policy} vs {x_policy}")
    ax.legend(title="Dataset", loc="lower right", framealpha=0.95)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(img_path, format=bundle.image_format)
    plt.close(fig)
    write_figure_meta(
        meta_path,
        spec.id,
        {
            "x_policy": x_policy,
            "y_policy": y_policy,
            "metric": metric,
            "k": k,
            "n": len(common),
            "color_legend": "dataset_source",
        },
    )
    return {"image": str(img_path), "meta": str(meta_path)}

"""Bar chart: retrieval metric@k by policy (all test questions) with bootstrap 95% CIs."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from surf_rag.config.schema import ResultsArtifactSpec
from surf_rag.evaluation.latency_metrics import bootstrap_mean_ci95
from surf_rag.results.bundle import ResultsBundle, policy_list
from surf_rag.results.figures.base import figure_paths, write_figure_meta
from surf_rag.results.loaders import load_policy_metrics
from surf_rag.results.metric_fields import (
    e2e_retrieval_value,
    resolve_retrieval_metric_k,
)
from surf_rag.results.tables.pipeline_retrieval import DEFAULT_EXCLUDE
from surf_rag.viz.theme import BAR_ALPHA, PALETTE, bar_style, style_bar_axes

_CI_SAMPLES = 10_000
_CI_SEED = 42
_DEFAULT_YLIM_MIN = 0.6
_DEFAULT_YLIM_MAX = 1.0


def _resolve_ylim(spec: ResultsArtifactSpec) -> tuple[float, float]:
    y_min = float(spec.ylim_min if spec.ylim_min is not None else _DEFAULT_YLIM_MIN)
    y_max = float(spec.ylim_max if spec.ylim_max is not None else _DEFAULT_YLIM_MAX)
    if y_min >= y_max:
        raise ValueError(f"ylim_min ({y_min}) must be less than ylim_max ({y_max})")
    return y_min, y_max


def _metric_axis_label(metric: str, k: int) -> str:
    m = metric.strip().lower()
    if m in ("stateful_ndcg", "ndcg"):
        return f"NDCG@{k}" if m == "ndcg" else f"Stateful NDCG@{k}"
    if m == "recall":
        return f"Recall@{k}"
    if m == "hit":
        return f"Hit@{k}"
    return f"{metric}@{k}"


def _per_question_values(
    bundle: ResultsBundle,
    policy: str,
    *,
    metric: str,
    k: int,
) -> list[float]:
    per_q = (
        load_policy_metrics(bundle.policies[policy].metrics_path).get("per_question")
        or []
    )
    vals: list[float] = []
    for row in per_q:
        qid = str(row.get("question_id", "")).strip()
        if qid not in bundle.split_qids:
            continue
        vals.append(e2e_retrieval_value(row, metric=metric, k=k))
    return vals


def _mean_and_ci95(vals: list[float]) -> tuple[float, float, float, int]:
    """Return (mean, ci_low, ci_high, n). Bootstrap percentile CI on the question mean."""
    if not vals:
        return 0.0, 0.0, 0.0, 0
    mean = float(np.mean(vals))
    lo, hi = bootstrap_mean_ci95(vals, samples=_CI_SAMPLES, seed=_CI_SEED)
    return mean, float(lo), float(hi), len(vals)


def render_retrieval_by_policy_ci(
    bundle: ResultsBundle, spec: ResultsArtifactSpec
) -> dict[str, str]:
    metric, k = resolve_retrieval_metric_k(spec, bundle)
    if spec.exclude_policies is not None:
        exclude = set(spec.exclude_policies)
    else:
        exclude = set(DEFAULT_EXCLUDE)
    policies = policy_list(bundle, exclude=list(exclude))

    stats: dict[str, tuple[float, float, float, int]] = {}
    for policy in policies:
        vals = _per_question_values(bundle, policy, metric=metric, k=k)
        stats[policy] = _mean_and_ci95(vals)

    img_path, meta_path = figure_paths(bundle, spec.id)
    fig, ax = plt.subplots(figsize=(max(8, len(policies) * 1.2), 4.5))
    x = np.arange(len(policies))
    width = 0.65
    style = bar_style(color=PALETTE["light-blue"])

    means: list[float] = []
    yerr_lo: list[float] = []
    yerr_hi: list[float] = []
    for policy in policies:
        mean, lo, hi, _n = stats.get(policy, (0.0, 0.0, 0.0, 0))
        means.append(mean)
        yerr_lo.append(max(0.0, mean - lo))
        yerr_hi.append(max(0.0, hi - mean))

    ax.bar(
        x,
        means,
        width=width,
        yerr=np.array([yerr_lo, yerr_hi]),
        capsize=3,
        error_kw={
            "color": PALETTE["text"],
            "linewidth": 1.0,
            "capthick": 1.0,
            "alpha": 1.0,
            "zorder": 4,
        },
        zorder=3,
        **style,
    )

    ax.set_xticks(x, policies, rotation=30, ha="right")
    ax.set_ylabel(_metric_axis_label(metric, k))
    ax.set_title("Retrieval by policy (95% bootstrap CI)")
    y_min, y_max = _resolve_ylim(spec)
    ax.set_ylim(y_min, y_max)
    style_bar_axes(ax)
    fig.tight_layout()
    fig.savefig(img_path, format=bundle.image_format)
    plt.close(fig)
    write_figure_meta(
        meta_path,
        spec.id,
        {
            "metric": metric,
            "k": k,
            "dataset_source": "all",
            "policies": policies,
            "ci_method": "bootstrap_percentile",
            "ci_level": 0.95,
            "ci_samples": _CI_SAMPLES,
            "ci_seed": _CI_SEED,
            "ci_unit": "question",
            "bar_color": PALETTE["light-blue"],
            "bar_alpha": BAR_ALPHA,
            "per_policy_n": {p: stats.get(p, (0, 0, 0, 0))[3] for p in policies},
            "ylim": [y_min, y_max],
        },
    )
    return {"image": str(img_path), "meta": str(meta_path)}

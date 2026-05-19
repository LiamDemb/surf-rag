"""Orchestrate dissertation results tables and figures."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

from surf_rag.config.schema import PipelineConfig, ResultsArtifactSpec
from surf_rag.results.bundle import ResultsBundle, enabled_artifacts
from surf_rag.results.figures.dispersion_curve import render_dispersion_curve
from surf_rag.results.figures.endpoint_pref import render_endpoint_pref
from surf_rag.results.figures.retrieval_by_policy_ci import (
    render_retrieval_by_policy_ci,
)
from surf_rag.results.figures.pairwise_retrieval import render_pairwise_retrieval
from surf_rag.results.figures.retrieval_answer_gain import render_retrieval_answer_gain
from surf_rag.results.figures.route_confusion import render_route_confusion
from surf_rag.results.figures.router_training_learning_curve import (
    render_router_training_learning_curve_results,
)
from surf_rag.results.figures.weight_dists import render_weight_dists
from surf_rag.results.manifest import (
    ArtifactRecord,
    BuildManifest,
    manifest_from_bundle,
)
from surf_rag.results.tables.bench_splits import build_bench_splits
from surf_rag.results.tables.branch_retrieval import build_branch_retrieval
from surf_rag.results.tables.classifier_metrics import build_classifier_metrics
from surf_rag.results.tables.oracle_policy_comparison import (
    build_oracle_policy_comparison,
)
from surf_rag.results.tables.oracle_stats import build_oracle_stats
from surf_rag.results.tables.pipeline_answers import build_pipeline_answers
from surf_rag.results.tables.pairwise_wilcoxon import build_pairwise_wilcoxon
from surf_rag.results.tables.pipeline_retrieval import build_pipeline_retrieval
from surf_rag.results.tables.regressor_metrics import (
    SkippedArtifact,
    build_regressor_metrics,
)
from surf_rag.results.tables.regressor_test_summary import build_regressor_test_summary

log = logging.getLogger(__name__)

FIGURE_RENDERERS: dict[str, Callable] = {
    "endpoint_pref": render_endpoint_pref,
    "dispersion_curve": render_dispersion_curve,
    "weight_dists": render_weight_dists,
    "route_confusion": render_route_confusion,
    "retrieval_by_policy_ci": render_retrieval_by_policy_ci,
    "pairwise_retrieval": render_pairwise_retrieval,
    "retrieval_answer_gain": render_retrieval_answer_gain,
    "router_training_learning_curve": render_router_training_learning_curve_results,
}

TABLE_BUILDERS: dict[str, Callable] = {
    "bench_splits": lambda b, s: build_bench_splits(b),
    "oracle_stats": lambda b, s: build_oracle_stats(b, s),
    "regressor_metrics": lambda b, s: build_regressor_metrics(b),
    "regressor_test_summary": lambda b, s: build_regressor_test_summary(b),
    "classifier_metrics": lambda b, s: build_classifier_metrics(b),
    "branch_retrieval": lambda b, s: build_branch_retrieval(b, s),
    "pipeline_retrieval": lambda b, s: build_pipeline_retrieval(b, s),
    "pipeline_retrieval_full": lambda b, s: build_pipeline_retrieval(b, s),
    "pipeline_answers": lambda b, s: build_pipeline_answers(b),
    "pairwise_wilcoxon": lambda b, s: build_pairwise_wilcoxon(b, s),
    "oracle_policy_comparison": lambda b, s: build_oracle_policy_comparison(b, s),
}

DEFAULT_ARTIFACT_IDS: list[tuple[str, str, str | None]] = [
    ("bench_splits", "table", None),
    ("oracle_stats", "table", None),
    ("endpoint_pref", "figure", "endpoint_pref"),
    ("dispersion_curve", "figure", "dispersion_curve"),
    ("regressor_metrics", "table", None),
    ("regressor_test_summary", "table", None),
    ("classifier_metrics", "table", None),
    ("weight_dists", "figure", "weight_dists"),
    ("route_confusion", "figure", "route_confusion"),
    ("branch_retrieval", "table", None),
    ("pipeline_retrieval", "table", None),
    ("pipeline_answers", "table", None),
    ("pairwise_wilcoxon", "table", None),
    ("oracle_policy_comparison", "table", None),
    ("retrieval_by_policy_ci", "figure", "retrieval_by_policy_ci"),
    ("retrieval_answer_gain", "figure", "retrieval_answer_gain"),
]


def _default_artifacts() -> list[ResultsArtifactSpec]:
    return [
        ResultsArtifactSpec(id=aid, kind=kind, figure=fig)
        for aid, kind, fig in DEFAULT_ARTIFACT_IDS
    ]


def build_results(
    cfg: PipelineConfig,
    *,
    config_path: Path | None = None,
    only_ids: frozenset[str] | None = None,
) -> BuildManifest:
    bundle = ResultsBundle.from_config(cfg, config_path=config_path)
    specs = enabled_artifacts(bundle)
    if not specs:
        specs = _default_artifacts()

    records: list[ArtifactRecord] = []
    had_failure = False

    for spec in specs:
        if only_ids is not None and spec.id not in only_ids:
            continue
        try:
            rec = _build_one(bundle, spec)
            records.append(rec)
            if rec.status == "failed":
                had_failure = True
        except Exception as exc:
            had_failure = True
            log.exception("Artifact %s failed", spec.id)
            records.append(
                ArtifactRecord(
                    id=spec.id,
                    kind=spec.kind,
                    status="failed",
                    error=str(exc),
                )
            )

    manifest = manifest_from_bundle(
        bundle, config_path=config_path, artifact_records=records
    )
    out_path = bundle.output_dir / "manifest.json"
    manifest.write(out_path)
    log.info("Wrote manifest %s", out_path)
    if had_failure:
        raise RuntimeError("One or more artefacts failed; see manifest.json")
    return manifest


def _build_one(bundle: ResultsBundle, spec: ResultsArtifactSpec) -> ArtifactRecord:
    if spec.kind == "table":
        builder = TABLE_BUILDERS.get(spec.id)
        if builder is None:
            raise ValueError(f"Unknown table artifact id: {spec.id!r}")
        try:
            _df, paths = builder(bundle, spec)
            return ArtifactRecord(
                id=spec.id,
                kind="table",
                status="ok",
                table_csv=paths.get("csv"),
            )
        except SkippedArtifact as exc:
            bundle.warnings.append(f"{spec.id}: skipped — {exc.reason}")
            log.warning("Skipped %s: %s", spec.id, exc.reason)
            return ArtifactRecord(
                id=spec.id,
                kind="table",
                status="skipped",
                error=exc.reason,
            )

    if spec.kind == "figure":
        figure_kind = (spec.figure or spec.id).strip()
        renderer = FIGURE_RENDERERS.get(figure_kind)
        if renderer is None:
            raise ValueError(f"Unknown figure kind: {figure_kind!r}")
        try:
            paths = renderer(bundle, spec)
            return ArtifactRecord(
                id=spec.id,
                kind="figure",
                status="ok",
                figure_image=paths.get("image"),
                figure_meta=paths.get("meta"),
            )
        except SkippedArtifact as exc:
            bundle.warnings.append(f"{spec.id}: skipped — {exc.reason}")
            log.warning("Skipped %s: %s", spec.id, exc.reason)
            return ArtifactRecord(
                id=spec.id,
                kind="figure",
                status="skipped",
                error=exc.reason,
            )

    raise ValueError(f"artifact kind must be table or figure, got {spec.kind!r}")

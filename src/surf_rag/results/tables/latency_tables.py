"""Latency and wall-clock tables for results builds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from surf_rag.config.schema import ResultsArtifactSpec
from surf_rag.evaluation.latency_metrics import summarize_latency
from surf_rag.evaluation.router_model_artifacts import (
    ROUTER_TASK_CLASSIFICATION,
    ROUTER_TASK_REGRESSION,
    make_router_model_paths_for_cli,
    read_json,
)
from surf_rag.results.bundle import ResultsBundle, get_router_arch, policy_list
from surf_rag.results.loaders import load_policy_metrics
from surf_rag.results.tables.regressor_metrics import SkippedArtifact
from surf_rag.results.tables.writer import write_table

# Per-question ``latency_ms`` keys (E2E protocol v2) -> stable metric labels for tables.
LATENCY_COMPONENTS: tuple[tuple[str, str], ...] = (
    ("retrieval_reported_total_ms", "retrieval_reported_total"),
    ("retrieval_stage_total_ms", "retrieval_stage_total"),
    ("routing_input_ms", "routing_input"),
    ("router_predict_ms", "router_predict"),
    ("dense_branch_ms", "dense_branch"),
    ("graph_branch_ms", "graph_branch"),
    ("fusion_ms", "fusion"),
)


def _flatten_latency_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Turn :func:`summarize_latency` output into flat CSV-friendly fields."""
    out: dict[str, Any] = {
        "n": int(summary.get("count", 0)),
        "valid_n": int(summary.get("valid_count", 0)),
        "missing_n": int(summary.get("missing_count", 0)),
        "mean_ms": float(summary.get("mean_ms", 0.0)),
        "median_ms": float(summary.get("median_ms", 0.0)),
        "p90_ms": float(summary.get("p90_ms", 0.0)),
        "p95_ms": float(summary.get("p95_ms", 0.0)),
        "std_ms": float(summary.get("std_ms", 0.0)),
        "min_ms": float(summary.get("min_ms", 0.0)),
        "max_ms": float(summary.get("max_ms", 0.0)),
    }
    ci = summary.get("mean_ci95_ms")
    if isinstance(ci, (list, tuple)) and len(ci) == 2:
        out["mean_ci95_lo_ms"] = float(ci[0])
        out["mean_ci95_hi_ms"] = float(ci[1])
    return out


def _expected_split_count(
    split_qids: set[str],
    qid_to_source: dict[str, str],
    dataset_source: str,
) -> int:
    if dataset_source == "all":
        return len(split_qids)
    return sum(
        1 for qid in split_qids if qid_to_source.get(qid, "unknown") == dataset_source
    )


def _collect_latency_values(
    per_question: list[dict[str, Any]],
    *,
    split_qids: set[str],
    dataset_source: str,
    qid_to_source: dict[str, str],
    latency_key: str,
) -> list[float]:
    vals: list[float] = []
    for row in per_question:
        qid = str(row.get("question_id", "") or "").strip()
        if not qid or qid not in split_qids:
            continue
        if dataset_source != "all":
            src = qid_to_source.get(qid, "unknown")
            if src != dataset_source:
                continue
        lat = row.get("latency_ms")
        if not isinstance(lat, dict):
            continue
        raw = lat.get(latency_key)
        try:
            x = float(raw)
        except (TypeError, ValueError):
            continue
        if x == x:  # finite
            vals.append(x)
    return vals


def _router_metrics_path(bundle: ResultsBundle, role: str) -> Path | None:
    arch = get_router_arch(bundle, role)
    if arch is None or not str(arch.architecture_id or "").strip():
        return None
    task = ROUTER_TASK_REGRESSION if role == "regressor" else ROUTER_TASK_CLASSIFICATION
    paths = make_router_model_paths_for_cli(
        bundle.resolved.router_id,
        router_base=bundle.resolved.router_base,
        input_mode=arch.input_mode,
        router_architecture_id=arch.architecture_id,
        router_task_type=task,
    )
    if not paths.metrics.is_file():
        return None
    return paths.metrics


def build_oracle_ops_summary(
    bundle: ResultsBundle, spec: ResultsArtifactSpec
) -> tuple[pd.DataFrame, dict]:
    summary_path = bundle.resolved.router_oracle_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Oracle summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    wall_s = float(summary.get("oracle_sweep_wall_s", 0.0) or 0.0)
    newly = int(summary.get("newly_scored", 0) or 0)
    scored = int(summary.get("oracle_scored", 0) or 0)
    denom = newly if newly > 0 else scored
    per_q = (wall_s / denom) if denom > 0 else ""

    row = {
        "router_id": str(summary.get("router_id", bundle.resolved.router_id)),
        "oracle_sweep_wall_s": round(wall_s, 3),
        "questions_snapshot": int(summary.get("questions_snapshot", 0) or 0),
        "oracle_scored": scored,
        "newly_scored": newly,
        "dense_cached": int(summary.get("dense_cached", 0) or 0),
        "graph_cached": int(summary.get("graph_cached", 0) or 0),
        "sweep_s_per_question": round(per_q, 6) if per_q != "" else "",
        "note": (
            "oracle_sweep_wall_s is the weight-grid sweep only (last prepare call); "
            "retrieval wall time is not recorded in summary.json."
        ),
    }
    df = pd.DataFrame([row])
    artifact_id = spec.id or "oracle_ops_summary"
    paths = write_table(
        bundle,
        artifact_id,
        df,
        {"source": str(summary_path)},
    )
    return df, {"csv": str(paths[0]), "meta": str(paths[1])}


def build_pipeline_ops_timing(
    bundle: ResultsBundle, spec: ResultsArtifactSpec
) -> tuple[pd.DataFrame, dict]:
    rows: list[dict[str, Any]] = []
    summary_path = bundle.resolved.router_oracle_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        wall_s = float(summary.get("oracle_sweep_wall_s", 0.0) or 0.0)
        rows.append(
            {
                "stage": "oracle_weight_sweep",
                "wall_s": round(wall_s, 3),
                "n_items": int(summary.get("newly_scored", 0) or 0)
                or int(summary.get("oracle_scored", 0) or 0),
                "unit": "questions",
                "notes": "In-memory fusion/metric sweep; last oracle-prepare invocation.",
            }
        )

    for role, label in (
        ("regressor", "router_train_regression"),
        ("classifier", "router_train_classification"),
    ):
        mpath = _router_metrics_path(bundle, role)
        if mpath is None:
            continue
        metrics = read_json(mpath)
        tw = metrics.get("training_wall_s")
        if tw is None:
            continue
        arch = get_router_arch(bundle, role)
        rows.append(
            {
                "stage": label,
                "wall_s": round(float(tw), 3),
                "n_items": int(metrics.get("best_epoch", 0) or 0) + 1,
                "unit": "epochs_completed",
                "notes": (
                    f"task={metrics.get('task_type', role)} "
                    f"arch={arch.architecture_id if arch else ''} "
                    f"input_mode={arch.input_mode if arch else ''}"
                ),
            }
        )

    if not rows:
        raise FileNotFoundError(
            "No pipeline ops timing sources found (oracle summary or router metrics)."
        )

    df = pd.DataFrame(rows)
    artifact_id = spec.id or "pipeline_ops_timing"
    paths = write_table(bundle, artifact_id, df, {"split": bundle.results.split})
    return df, {"csv": str(paths[0]), "meta": str(paths[1])}


def build_router_training_timing(
    bundle: ResultsBundle, spec: ResultsArtifactSpec
) -> tuple[pd.DataFrame, dict]:
    rows: list[dict[str, Any]] = []
    for role in ("regressor", "classifier"):
        mpath = _router_metrics_path(bundle, role)
        if mpath is None:
            continue
        metrics = read_json(mpath)
        arch = get_router_arch(bundle, role)
        rows.append(
            {
                "router_role": role,
                "architecture_id": arch.architecture_id if arch else "",
                "input_mode": arch.input_mode if arch else "",
                "task_type": str(metrics.get("task_type", role)),
                "training_wall_s": round(float(metrics.get("training_wall_s", 0.0)), 3),
                "best_epoch": int(metrics.get("best_epoch", 0) or 0),
                "architecture": str(metrics.get("architecture", "")),
                "metrics_path": str(mpath),
            }
        )

    if not rows:
        raise SkippedArtifact(
            "results.router regressor/classifier not configured or metrics.json missing"
        )

    df = pd.DataFrame(rows)
    artifact_id = spec.id or "router_training_timing"
    paths = write_table(bundle, artifact_id, df, {})
    return df, {"csv": str(paths[0]), "meta": str(paths[1])}


def build_e2e_startup_latency(
    bundle: ResultsBundle, spec: ResultsArtifactSpec
) -> tuple[pd.DataFrame, dict]:
    rows: list[dict[str, Any]] = []
    for policy in policy_list(bundle):
        run = bundle.policies[policy]
        manifest_path = run.run_dir / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        e2e = dict(manifest.get("e2e") or {})
        startup = dict(e2e.get("startup_latency_ms") or {})
        components = dict(startup.get("startup_components") or {})
        rows.append(
            {
                "policy": policy,
                "run_id": run.run_id,
                "startup_total_ms": float(startup.get("startup_total_ms", 0.0) or 0.0),
                "dense_init_ms": float(components.get("dense_init_ms", 0.0) or 0.0),
                "graph_init_ms": float(components.get("graph_init_ms", 0.0) or 0.0),
                "router_init_ms": float(components.get("router_init_ms", 0.0) or 0.0),
            }
        )

    if not rows:
        raise FileNotFoundError("No E2E manifests with startup_latency_ms found.")

    df = pd.DataFrame(rows)
    artifact_id = spec.id or "e2e_startup_latency"
    paths = write_table(bundle, artifact_id, df, {"split": bundle.results.split})
    return df, {"csv": str(paths[0]), "meta": str(paths[1])}


def build_pipeline_latency_summary(
    bundle: ResultsBundle, spec: ResultsArtifactSpec
) -> tuple[pd.DataFrame, dict]:
    if spec.exclude_policies is not None:
        policies = policy_list(bundle, exclude=list(spec.exclude_policies))
    else:
        policies = policy_list(bundle)

    rows: list[dict[str, Any]] = []
    for policy in policies:
        metrics = load_policy_metrics(bundle.policies[policy].metrics_path)
        per_q = metrics.get("per_question") or []
        if not isinstance(per_q, list):
            continue
        for latency_key, metric_label in LATENCY_COMPONENTS:
            for src in ("all", "nq", "2wiki"):
                vals = _collect_latency_values(
                    per_q,
                    split_qids=bundle.split_qids,
                    dataset_source=src,
                    qid_to_source=bundle.qid_to_source,
                    latency_key=latency_key,
                )
                expected = _expected_split_count(
                    bundle.split_qids, bundle.qid_to_source, src
                )
                if expected == 0:
                    continue
                summary = summarize_latency(vals, total_count=expected)
                row = {
                    "policy": policy,
                    "run_id": bundle.policies[policy].run_id,
                    "dataset_source": src,
                    "metric": metric_label,
                    **_flatten_latency_summary(summary),
                }
                rows.append(row)

    if not rows:
        raise ValueError("pipeline_latency_summary: no latency values in E2E metrics.")

    df = pd.DataFrame(rows)
    artifact_id = spec.id or "pipeline_latency_summary"
    paths = write_table(
        bundle,
        artifact_id,
        df,
        {
            "split": bundle.results.split,
            "components": [m for _, m in LATENCY_COMPONENTS],
        },
    )
    return df, {"csv": str(paths[0]), "meta": str(paths[1])}

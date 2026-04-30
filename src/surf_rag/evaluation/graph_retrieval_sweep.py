from __future__ import annotations

import csv
import hashlib
import itertools
import json
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Iterator, Mapping, Sequence

import yaml

from surf_rag.config.env import apply_pipeline_env_from_config
from surf_rag.config.loader import (
    PipelineConfig,
    load_pipeline_config,
    resolve_paths,
    validate_e2e_config,
)
from surf_rag.config.resolved import write_resolved_config_yaml
from surf_rag.config.schema import RetrievalSection
from surf_rag.evaluation.artifact_paths import (
    benchmark_bundle_dir,
    safe_benchmark_bundle_subpath,
)
from surf_rag.evaluation.e2e_policies import parse_routing_policy
from surf_rag.evaluation.e2e_runner import evaluate_e2e_run
from surf_rag.evaluation.router_dataset_artifacts import (
    make_router_dataset_paths_for_cli,
)
from surf_rag.evaluation.run_artifacts import RunArtifactPaths
from surf_rag.evaluation.retrieval_jsonl import write_retrieval_line
from surf_rag.reranking.reranker import build_reranker
from surf_rag.retrieval.routed import RoutedFusionPipeline
from surf_rag.router.policies import RoutingPolicyName
from surf_rag.strategies.factory import build_dense_retriever, build_graph_retriever

_VALID_RETRIEVAL_GRID_KEYS: frozenset[str] = frozenset(
    f.name for f in fields(RetrievalSection)
)


def default_sweep_id() -> str:
    return datetime.now(timezone.utc).strftime("sweep-%Y%m%d-%H%M%S")


def default_sweep_dir(
    *,
    benchmark_base: Path,
    benchmark_name: str,
    benchmark_id: str,
    sweep_id: str,
) -> Path:
    """Session directory for compact sweep artifacts."""
    session = safe_benchmark_bundle_subpath(sweep_id)
    return benchmark_bundle_dir(benchmark_base, benchmark_name, benchmark_id) / session


def normalize_sweep_grid(raw: Mapping[str, Any]) -> dict[str, list[Any]]:
    """Validate keys and coerce each axis to a non-empty list."""
    if not raw:
        raise ValueError(
            "Sweep grid is empty. Set ``graph_retrieval_sweep.grid`` in the pipeline YAML "
            "or pass ``--grid`` to a YAML file of RetrievalSection field names → lists."
        )
    if not isinstance(raw, Mapping):
        raise ValueError(f"Sweep grid must be a mapping, got {type(raw)}")
    out: dict[str, list[Any]] = {}
    for k, v in raw.items():
        if not isinstance(k, str) or not k.strip():
            raise ValueError(f"Invalid grid key: {k!r}")
        key = k.strip()
        if key not in _VALID_RETRIEVAL_GRID_KEYS:
            raise ValueError(
                f"Unknown retrieval grid key {key!r}. "
                f"Valid keys: {sorted(_VALID_RETRIEVAL_GRID_KEYS)}"
            )
        seq = v if isinstance(v, list) else [v]
        if not seq:
            raise ValueError(f"Grid axis {key!r} must be a non-empty list")
        out[key] = list(seq)
    if not out:
        raise ValueError("Sweep grid has no axes after filtering.")
    return out


def load_grid_yaml(path: Path) -> dict[str, list[Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        raise ValueError(f"Grid file is empty: {path}")
    if not isinstance(raw, dict):
        raise ValueError(f"Grid YAML must be a mapping at top level, got {type(raw)}")
    return normalize_sweep_grid(raw)


def iter_grid_combos(grid: Mapping[str, Sequence[Any]]) -> Iterator[dict[str, Any]]:
    keys = list(grid)
    for combo in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, combo, strict=True))


def pipeline_with_retrieval_overrides(
    base: PipelineConfig, overrides: Mapping[str, Any]
) -> PipelineConfig:
    unknown = set(overrides) - _VALID_RETRIEVAL_GRID_KEYS
    if unknown:
        raise ValueError(f"Unknown retrieval override keys: {sorted(unknown)}")
    new_retrieval = replace(base.retrieval, **dict(overrides))
    return replace(base, retrieval=new_retrieval)


def get_nested(obj: Any, dotted_path: str) -> Any:
    cur: Any = obj
    for part in dotted_path.split("."):
        if not part:
            raise ValueError(f"Invalid objective path: {dotted_path!r}")
        if isinstance(cur, dict):
            cur = cur[part]
        else:
            cur = getattr(cur, part)
    return cur


def trial_run_id(trial_index: int) -> str:
    return f"trial-{trial_index:04d}"


def combo_key(overrides: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        dict(sorted(overrides.items())), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class TrialRecord:
    trial_index: int
    run_id: str
    combo_key: str
    retrieval_overrides: dict[str, Any]
    objective_path: str
    objective_score: float | None
    status: str
    error: str | None
    started_at: str
    finished_at: str
    elapsed_ms: int
    metrics_path: str | None = None
    run_root: str | None = None
    metric_all_count: int | None = None
    metric_all_ndcg_at_10: float | None = None
    metric_all_hit_at_10: float | None = None
    metric_all_recall_at_10: float | None = None

    @classmethod
    def from_json(cls, row: Mapping[str, Any]) -> "TrialRecord":
        return cls(
            trial_index=int(row["trial_index"]),
            run_id=str(row["run_id"]),
            combo_key=str(row["combo_key"]),
            retrieval_overrides=dict(row.get("retrieval_overrides") or {}),
            objective_path=str(row["objective_path"]),
            objective_score=(
                float(row["objective_score"])
                if row.get("objective_score") is not None
                else None
            ),
            status=str(row.get("status", "error")),
            error=str(row["error"]) if row.get("error") else None,
            started_at=str(row.get("started_at") or ""),
            finished_at=str(row.get("finished_at") or ""),
            elapsed_ms=int(row.get("elapsed_ms") or 0),
            metrics_path=str(row["metrics_path"]) if row.get("metrics_path") else None,
            run_root=str(row["run_root"]) if row.get("run_root") else None,
            metric_all_count=(
                int(row["metric_all_count"])
                if row.get("metric_all_count") is not None
                else None
            ),
            metric_all_ndcg_at_10=(
                float(row["metric_all_ndcg_at_10"])
                if row.get("metric_all_ndcg_at_10") is not None
                else None
            ),
            metric_all_hit_at_10=(
                float(row["metric_all_hit_at_10"])
                if row.get("metric_all_hit_at_10") is not None
                else None
            ),
            metric_all_recall_at_10=(
                float(row["metric_all_recall_at_10"])
                if row.get("metric_all_recall_at_10") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class SweepSummary:
    sweep_id: str
    sweep_dir: str
    base_config_path: str
    grid_source: str
    objective_path: str
    policy: str
    trials: list[TrialRecord]
    best_trial_index: int | None
    completed_trials: int
    skipped_trials: int


def _trials_jsonl_path(sweep_dir: Path) -> Path:
    return sweep_dir / "trials.jsonl"


def _manifest_path(sweep_dir: Path) -> Path:
    return sweep_dir / "manifest.json"


def append_trial_row(trials_jsonl: Path, trial: TrialRecord) -> None:
    trials_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with trials_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(trial), ensure_ascii=False) + "\n")
        f.flush()


def read_trial_rows(trials_jsonl: Path) -> list[TrialRecord]:
    if not trials_jsonl.is_file():
        return []
    rows: list[TrialRecord] = []
    with trials_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(TrialRecord.from_json(json.loads(line)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
    return rows


def completed_combo_keys(trials: Iterable[TrialRecord]) -> set[str]:
    return {t.combo_key for t in trials if t.status == "ok"}


def pick_best_trial(trials: Sequence[TrialRecord]) -> int | None:
    best_i: int | None = None
    best_s = float("-inf")
    for i, t in enumerate(trials):
        s = t.objective_score
        if t.status != "ok" or s is None:
            continue
        if best_i is None or s > best_s:
            best_i, best_s = i, s
        elif s == best_s and t.trial_index < trials[best_i].trial_index:
            best_i, best_s = i, s
    return best_i


def write_best_json(sweep_dir: Path, summary: SweepSummary) -> None:
    best = None
    if summary.best_trial_index is not None and 0 <= summary.best_trial_index < len(
        summary.trials
    ):
        best = summary.trials[summary.best_trial_index]
    payload = {
        "sweep_id": summary.sweep_id,
        "objective_path": summary.objective_path,
        "best_trial_index": summary.best_trial_index,
        "best_retrieval_overrides": best.retrieval_overrides if best else None,
        "best_score": best.objective_score if best else None,
        "best_combo_key": best.combo_key if best else None,
        "best_run_root": best.run_root if best else None,
    }
    (sweep_dir / "best.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_leaderboard_csv(sweep_dir: Path, trials: Sequence[TrialRecord]) -> None:
    all_keys = sorted({k for t in trials for k in t.retrieval_overrides})
    out = sweep_dir / "leaderboard.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "trial_index",
                "combo_key",
                "status",
                "score",
                "elapsed_ms",
                "error",
                *all_keys,
            ]
        )
        for t in sorted(trials, key=lambda x: x.trial_index):
            w.writerow(
                [
                    t.trial_index,
                    t.combo_key,
                    t.status,
                    "" if t.objective_score is None else t.objective_score,
                    t.elapsed_ms,
                    "" if t.error is None else t.error,
                    *[t.retrieval_overrides.get(k, "") for k in all_keys],
                ]
            )


def write_sweep_summary_json(sweep_dir: Path, summary: SweepSummary) -> None:
    (sweep_dir / "sweep_summary.json").write_text(
        json.dumps(asdict(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_manifest(
    *,
    sweep_dir: Path,
    sweep_id: str,
    base_config_path: Path,
    grid_source: str,
    objective_path: str,
    policy: str,
) -> None:
    payload = {
        "schema": "surf-rag/graph-retrieval-sweep/v2",
        "created_at": _now_iso(),
        "sweep_id": sweep_id,
        "base_config_path": str(base_config_path.resolve()),
        "grid_source": grid_source,
        "objective_path": objective_path,
        "policy": policy,
        "artifacts": {
            "trials_jsonl": "trials.jsonl",
            "best_json": "best.json",
            "leaderboard_csv": "leaderboard.csv",
            "sweep_summary_json": "sweep_summary.json",
            "resolved_config_yaml": "resolved_config.yaml",
        },
    }
    _manifest_path(sweep_dir).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _load_benchmark_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def run_retrieval_only_trial(
    *,
    base_config: PipelineConfig,
    retrieval_overrides: Mapping[str, Any],
    objective_path: str,
    split_question_ids_path: Path | None,
    trial_index: int,
    run_id: str,
    run_root: Path,
) -> TrialRecord:
    start_t = perf_counter()
    started_at = _now_iso()
    cfg = pipeline_with_retrieval_overrides(base_config, retrieval_overrides)
    apply_pipeline_env_from_config(cfg)
    validate_e2e_config(cfg)
    e = cfg.e2e
    policy = parse_routing_policy(e.policy)
    if policy != RoutingPolicyName.GRAPH_ONLY:
        raise ValueError(
            "graph_retrieval_sweep requires e2e.policy == graph-only "
            f"(got {policy.value!r})."
        )

    rp = resolve_paths(cfg)
    run_paths = RunArtifactPaths(run_root=run_root.resolve())
    run_paths.retrieval_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_benchmark_rows(rp.benchmark_path.resolve(), e.limit)
    if e.only_question_ids:
        keep = set(str(x).strip() for x in e.only_question_ids if str(x).strip())
        rows = [r for r in rows if str(r.get("question_id", "")).strip() in keep]
    rows = [r for r in rows if str(r.get("question", "")).strip()]

    dense_retriever = build_dense_retriever(str(rp.corpus_dir.resolve()))
    graph_retriever = build_graph_retriever(
        str(rp.corpus_dir.resolve()),
        pipeline_config=cfg,
    )
    pipeline = RoutedFusionPipeline(
        dense_retriever,
        graph_retriever,
        fusion_keep_k=e.fusion_keep_k,
        router=None,
    )
    reranker = build_reranker(e.reranker, cross_encoder_model=e.cross_encoder_model)

    retrieval_path = run_paths.retrieval_results_jsonl()
    with retrieval_path.open("w", encoding="utf-8") as retrieval_fp:
        for sample in rows:
            question = str(sample.get("question", "")).strip()
            qid = str(sample.get("question_id", "")).strip()
            rr = pipeline.run(question, policy)
            rr = reranker.rerank(question, rr, top_k=e.rerank_top_k)
            write_retrieval_line(retrieval_fp, rr, qid)

    report = evaluate_e2e_run(
        run_paths=run_paths,
        benchmark_path=rp.benchmark_path.resolve(),
        split_question_ids_path=split_question_ids_path,
    )
    score = float(get_nested(report, objective_path))
    all10 = (
        ((report.get("overlap") or {}).get("all") or {})
        .get("retrieval_at_k", {})
        .get("10", {})
    )
    metrics_path = run_paths.run_root / "metrics.json"
    metrics_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    elapsed_ms = int((perf_counter() - start_t) * 1000)
    return TrialRecord(
        trial_index=trial_index,
        run_id=run_id,
        combo_key=combo_key(retrieval_overrides),
        retrieval_overrides=dict(retrieval_overrides),
        objective_path=objective_path,
        objective_score=score,
        status="ok",
        error=None,
        started_at=started_at,
        finished_at=_now_iso(),
        elapsed_ms=elapsed_ms,
        metrics_path=str(metrics_path.resolve()),
        run_root=str(run_paths.run_root.resolve()),
        metric_all_count=int(
            ((report.get("overlap") or {}).get("all") or {}).get("count", 0)
        ),
        metric_all_ndcg_at_10=(
            float(all10["ndcg"])
            if isinstance(all10, dict) and "ndcg" in all10
            else None
        ),
        metric_all_hit_at_10=(
            float(all10["hit"]) if isinstance(all10, dict) and "hit" in all10 else None
        ),
        metric_all_recall_at_10=(
            float(all10["recall"])
            if isinstance(all10, dict) and "recall" in all10
            else None
        ),
    )


def build_summary(
    *,
    sweep_id: str,
    sweep_dir: Path,
    base_config_path: Path,
    grid_source: str,
    objective_path: str,
    policy: str,
    trials: Sequence[TrialRecord],
    skipped_trials: int,
) -> SweepSummary:
    t_list = list(trials)
    best_idx = pick_best_trial(t_list)
    return SweepSummary(
        sweep_id=sweep_id,
        sweep_dir=str(sweep_dir.resolve()),
        base_config_path=str(base_config_path.resolve()),
        grid_source=grid_source,
        objective_path=objective_path,
        policy=policy,
        trials=t_list,
        best_trial_index=best_idx,
        completed_trials=sum(1 for t in t_list if t.status == "ok"),
        skipped_trials=skipped_trials,
    )


def run_cartesian_sweep(
    *,
    base_config_path: Path,
    grid_path: Path | None = None,
    sweep_id: str | None = None,
    objective_path: str | None = None,
) -> SweepSummary:
    base = load_pipeline_config(base_config_path.resolve())
    sweep_cfg = base.graph_retrieval_sweep
    if grid_path is not None:
        grid = load_grid_yaml(grid_path.resolve())
        grid_source = str(grid_path.resolve())
    else:
        grid = normalize_sweep_grid(sweep_cfg.grid)
        grid_source = f"{base_config_path.resolve()}#graph_retrieval_sweep.grid"

    resolved_objective = (
        objective_path
        if objective_path is not None
        else (sweep_cfg.objective or "overlap.all.retrieval_at_k.10.ndcg")
    )
    resolved_sweep_id = (
        sweep_id
        or (str(sweep_cfg.sweep_id).strip() if sweep_cfg.sweep_id else None)
        or (str(base.experiment_id).strip() if base.experiment_id else None)
        or default_sweep_id()
    )

    rp0 = resolve_paths(base)
    validate_e2e_config(base)
    pol = parse_routing_policy(base.e2e.policy)
    if pol != RoutingPolicyName.GRAPH_ONLY:
        raise ValueError(
            "This sweep requires e2e.policy: graph-only in the base config."
        )

    split_q: Path | None = None
    if sweep_cfg.use_router_overlap_splits:
        if base.paths.router_id and str(base.paths.router_id).strip():
            dsp = make_router_dataset_paths_for_cli(
                str(base.paths.router_id), router_base=rp0.router_base
            )
            split_q = dsp.split_question_ids

    sweep_dir = default_sweep_dir(
        benchmark_base=rp0.benchmark_base,
        benchmark_name=rp0.benchmark_name,
        benchmark_id=rp0.benchmark_id,
        sweep_id=resolved_sweep_id,
    )
    sweep_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(
        sweep_dir=sweep_dir,
        sweep_id=resolved_sweep_id,
        base_config_path=base_config_path,
        grid_source=grid_source,
        objective_path=resolved_objective,
        policy=RoutingPolicyName.GRAPH_ONLY.value,
    )
    write_resolved_config_yaml(
        sweep_dir / "resolved_config.yaml",
        base,
        rp0,
    )

    trials_jsonl = _trials_jsonl_path(sweep_dir)
    trials = read_trial_rows(trials_jsonl)
    done_keys = completed_combo_keys(trials)
    skipped = 0

    for idx, overrides in enumerate(iter_grid_combos(grid)):
        k = combo_key(overrides)
        if k in done_keys:
            skipped += 1
            continue
        run_id = trial_run_id(idx)
        # No per-trial heavy artifact folders: keep only compact outputs.
        trial_root = sweep_dir / run_id
        try:
            trial = run_retrieval_only_trial(
                base_config=base,
                retrieval_overrides=overrides,
                objective_path=resolved_objective,
                split_question_ids_path=split_q,
                trial_index=idx,
                run_id=run_id,
                run_root=trial_root,
            )
        except (OSError, ValueError, RuntimeError) as ex:
            trial = TrialRecord(
                trial_index=idx,
                run_id=run_id,
                combo_key=k,
                retrieval_overrides=dict(overrides),
                objective_path=resolved_objective,
                objective_score=None,
                status="error",
                error=str(ex),
                started_at=_now_iso(),
                finished_at=_now_iso(),
                elapsed_ms=0,
            )
        append_trial_row(trials_jsonl, trial)
        trials.append(trial)
        if trial.status == "ok":
            done_keys.add(trial.combo_key)
        summary = build_summary(
            sweep_id=resolved_sweep_id,
            sweep_dir=sweep_dir,
            base_config_path=base_config_path,
            grid_source=grid_source,
            objective_path=resolved_objective,
            policy=RoutingPolicyName.GRAPH_ONLY.value,
            trials=trials,
            skipped_trials=skipped,
        )
        write_best_json(sweep_dir, summary)

    summary = build_summary(
        sweep_id=resolved_sweep_id,
        sweep_dir=sweep_dir,
        base_config_path=base_config_path,
        grid_source=grid_source,
        objective_path=resolved_objective,
        policy=RoutingPolicyName.GRAPH_ONLY.value,
        trials=trials,
        skipped_trials=skipped,
    )
    write_best_json(sweep_dir, summary)
    write_leaderboard_csv(sweep_dir, trials)
    write_sweep_summary_json(sweep_dir, summary)
    return summary

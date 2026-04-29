from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import yaml

from surf_rag.config.env import apply_pipeline_env_from_config
from surf_rag.config.loader import (
    PipelineConfig,
    load_pipeline_config,
    resolve_paths,
    validate_e2e_config,
)
from surf_rag.config.schema import RetrievalSection
from surf_rag.evaluation.artifact_paths import (
    benchmark_bundle_dir,
    safe_benchmark_bundle_subpath,
)
from surf_rag.evaluation.e2e_policies import parse_routing_policy
from surf_rag.evaluation.e2e_runner import (
    evaluate_e2e_run,
    e2e_prepare_and_submit,
)
from surf_rag.evaluation.router_dataset_artifacts import (
    make_router_dataset_paths_for_cli,
)
from surf_rag.evaluation.run_artifacts import RunArtifactPaths
from surf_rag.router.policies import RoutingPolicyName

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
    """Session directory for ``sweep_summary.json``, ``best.json``, ``leaderboard.csv``."""
    session = safe_benchmark_bundle_subpath(sweep_id)
    return benchmark_bundle_dir(benchmark_base, benchmark_name, benchmark_id) / session


def normalize_sweep_grid(raw: Mapping[str, Any]) -> dict[str, list[Any]]:
    """Validate keys and coerce each axis to a non-empty list (Cartesian product input)."""
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
        if isinstance(v, list):
            seq = v
        else:
            seq = [v]
        if not seq:
            raise ValueError(f"Grid axis {key!r} must be a non-empty list")
        out[key] = list(seq)
    if not out:
        raise ValueError("Sweep grid has no axes after filtering.")
    return out


def load_grid_yaml(path: Path) -> dict[str, list[Any]]:
    """Load a YAML file of retrieval field names → lists (Cartesian product)."""
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
    """Stable run id for manifests and batch custom_ids (one folder per trial)."""
    return f"trial-{trial_index:04d}"


@dataclass(frozen=True)
class TrialRecord:
    trial_index: int
    run_id: str
    run_root: str
    metrics_path: str | None
    retrieval_overrides: dict[str, Any]
    objective_score: float | None
    prepare_exit_code: int
    error: str | None = None


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


def run_retrieval_only_trial(
    *,
    base_config: PipelineConfig,
    resolved_benchmark_base: Path,
    retrieval_overrides: Mapping[str, Any],
    trial_run_root: Path,
    run_id: str,
    split_question_ids_path: Path | None,
) -> TrialRecord:
    """``dry_run`` prepare + evaluate; no OpenAI generation when ``dry_run`` is True."""
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
    paths = RunArtifactPaths(run_root=trial_run_root.resolve())
    prep = e2e_prepare_and_submit(
        rp.benchmark_path.resolve(),
        benchmark_base=resolved_benchmark_base,
        benchmark_name=rp.benchmark_name,
        benchmark_id=rp.benchmark_id,
        split=e.split,
        run_id=run_id,
        routing_policy=policy,
        retrieval_asset_dir=rp.corpus_dir.resolve(),
        router_id=cfg.paths.router_id,
        router_base=rp.router_base,
        fusion_keep_k=e.fusion_keep_k,
        reranker_kind=e.reranker,
        rerank_top_k=e.rerank_top_k,
        cross_encoder_model=e.cross_encoder_model,
        limit=e.limit,
        only_question_ids=set(e.only_question_ids) if e.only_question_ids else None,
        completion_window=e.completion_window or cfg.generation.completion_window,
        include_graph_provenance=e.include_graph_provenance,
        dry_run=True,
        router_device=e.router_device,
        router_input_mode=e.router_input_mode,
        router_inference_batch_size=e.router_inference_batch_size,
        dev_sync=False,
        pipeline_config_for_artifact=cfg,
        run_paths_override=paths,
    )

    metrics_path = paths.run_root / "metrics.json"
    if prep != 0:
        return TrialRecord(
            trial_index=0,
            run_id=run_id,
            run_root=str(paths.run_root.resolve()),
            metrics_path=None,
            retrieval_overrides=dict(retrieval_overrides),
            objective_score=None,
            prepare_exit_code=int(prep),
            error=f"e2e_prepare_and_submit exited with code {prep}",
        )

    report = evaluate_e2e_run(
        run_paths=paths,
        benchmark_path=rp.benchmark_path.resolve(),
        split_question_ids_path=split_question_ids_path,
    )
    metrics_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return TrialRecord(
        trial_index=0,
        run_id=run_id,
        run_root=str(paths.run_root.resolve()),
        metrics_path=str(metrics_path.resolve()),
        retrieval_overrides=dict(retrieval_overrides),
        objective_score=None,
        prepare_exit_code=0,
        error=None,
    )


def score_trial_metrics(report: Mapping[str, Any], objective_path: str) -> float:
    return float(get_nested(report, objective_path))


def pick_best_trial(
    trials: Sequence[TrialRecord], scores: Sequence[float | None]
) -> int | None:
    """Highest score wins; ties broken by lower ``trial_index`` (earlier combo)."""
    best_i: int | None = None
    best_s = float("-inf")
    for i, (t, s) in enumerate(zip(trials, scores, strict=True)):
        if s is None:
            continue
        if best_i is None:
            best_i, best_s = i, s
        elif s > best_s:
            best_i, best_s = i, s
        elif s == best_s and t.trial_index < trials[best_i].trial_index:
            best_i, best_s = i, s
    return best_i


def write_sweep_artifacts(
    sweep_dir: Path,
    summary: SweepSummary,
    *,
    write_csv: bool = True,
) -> None:
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / "sweep_summary.json").write_text(
        json.dumps(asdict(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    best = None
    if summary.best_trial_index is not None and 0 <= summary.best_trial_index < len(
        summary.trials
    ):
        best = summary.trials[summary.best_trial_index]
    best_payload = {
        "sweep_id": summary.sweep_id,
        "objective_path": summary.objective_path,
        "best_trial_index": summary.best_trial_index,
        "best_retrieval_overrides": (best.retrieval_overrides if best else None),
        "best_run_root": best.run_root if best else None,
        "best_metrics_path": best.metrics_path if best else None,
        "best_score": (
            best.objective_score if best and best.objective_score is not None else None
        ),
    }
    (sweep_dir / "best.json").write_text(
        json.dumps(best_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if write_csv and summary.trials:
        all_keys = sorted({k for t in summary.trials for k in t.retrieval_overrides})
        header = [
            "trial_index",
            "run_id",
            "score",
            "prepare_exit_code",
            "run_root",
            *all_keys,
        ]
        lines = [",".join(header)]
        for t in summary.trials:
            cells = [
                str(t.trial_index),
                t.run_id,
                "" if t.objective_score is None else str(t.objective_score),
                str(t.prepare_exit_code),
                t.run_root,
            ]
            for k in all_keys:
                v = t.retrieval_overrides.get(k, "")
                cells.append("" if v is None else str(v))
            lines.append(",".join(_csv_escape(c) for c in cells))
        (sweep_dir / "leaderboard.csv").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )


def _csv_escape(s: str) -> str:
    if any(x in s for x in (",", '"', "\n")):
        return '"' + s.replace('"', '""') + '"'
    return s


def run_cartesian_sweep(
    *,
    base_config_path: Path,
    grid_path: Path | None = None,
    sweep_id: str | None = None,
    objective_path: str | None = None,
) -> SweepSummary:
    """Load ``base_config_path``; grid from ``--grid`` file if set, else ``graph_retrieval_sweep.grid``."""
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

    trials: list[TrialRecord] = []
    scores: list[float | None] = []

    for idx, overrides in enumerate(iter_grid_combos(grid)):
        run_id = trial_run_id(idx)
        trial_root = sweep_dir / run_id
        raw = run_retrieval_only_trial(
            base_config=base,
            resolved_benchmark_base=rp0.benchmark_base,
            retrieval_overrides=overrides,
            trial_run_root=trial_root,
            run_id=run_id,
            split_question_ids_path=split_q,
        )
        score: float | None = None
        err = raw.error
        if raw.prepare_exit_code == 0 and raw.metrics_path:
            try:
                report = json.loads(Path(raw.metrics_path).read_text(encoding="utf-8"))
                score = score_trial_metrics(report, resolved_objective)
            except (KeyError, TypeError, ValueError) as ex:
                err = f"objective / metrics parse failed: {ex}"
        trials.append(
            TrialRecord(
                trial_index=idx,
                run_id=raw.run_id,
                run_root=raw.run_root,
                metrics_path=raw.metrics_path,
                retrieval_overrides=raw.retrieval_overrides,
                objective_score=score,
                prepare_exit_code=raw.prepare_exit_code,
                error=err,
            )
        )
        scores.append(score)

    best_idx = pick_best_trial(trials, scores)
    summary = SweepSummary(
        sweep_id=resolved_sweep_id,
        sweep_dir=str(sweep_dir.resolve()),
        base_config_path=str(base_config_path.resolve()),
        grid_source=grid_source,
        objective_path=resolved_objective,
        policy=RoutingPolicyName.GRAPH_ONLY.value,
        trials=trials,
        best_trial_index=best_idx,
    )
    write_sweep_artifacts(sweep_dir, summary)
    return summary

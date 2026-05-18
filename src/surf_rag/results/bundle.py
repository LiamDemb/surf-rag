"""Resolve a results build bundle from pipeline config."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from surf_rag.config.loader import ResolvedPaths, load_pipeline_config, resolve_paths
from surf_rag.config.schema import (
    PipelineConfig,
    ResultsArtifactSpec,
    ResultsOracleConfig,
    ResultsPolicyEntry,
    ResultsRouterArch,
    ResultsSection,
)
from surf_rag.evaluation.artifact_paths import e2e_policy_run_dir
from surf_rag.results.loaders import split_question_ids_path


@dataclass(frozen=True)
class PolicyRun:
    policy: str
    run_id: str
    router_role: str | None
    metrics_path: Path
    run_dir: Path
    resolved_config_path: Path | None


@dataclass
class ResultsBundle:
    cfg: PipelineConfig
    resolved: ResolvedPaths
    results: ResultsSection
    output_dir: Path
    split_qids: set[str]
    qid_to_source: dict[str, str]
    policies: dict[str, PolicyRun]
    oracle_scores_path: Path
    split_question_ids_path: Path
    retrieval_graph_path: Path
    answerability_path: Path
    image_format: str
    warnings: list[str] = field(default_factory=list)

    @staticmethod
    def apply_env_policy_overrides(cfg: PipelineConfig) -> PipelineConfig:
        """Merge ``RESULTS_POLICY_<POLICY>`` env vars into ``results.policies``."""
        from dataclasses import replace

        policies = dict(cfg.results.policies)
        for name in list(policies.keys()):
            env_key = f"RESULTS_POLICY_{name.upper().replace('-', '_')}"
            override = os.environ.get(env_key, "").strip()
            if override:
                entry = policies[name]
                policies[name] = ResultsPolicyEntry(
                    run_id=override, router_role=entry.router_role
                )
        if policies == cfg.results.policies:
            return cfg
        return replace(cfg, results=replace(cfg.results, policies=policies))

    @classmethod
    def from_config(
        cls,
        cfg: PipelineConfig,
        *,
        config_path: Path | None = None,
    ) -> ResultsBundle:
        cfg = cls.apply_env_policy_overrides(cfg)
        rp = resolve_paths(cfg)
        rs = cfg.results
        if not rs.bundle_id.strip():
            raise ValueError("results.bundle_id is required")
        if not rs.policies:
            raise ValueError("results.policies must list at least one policy")

        output_dir = Path(rs.output_root).expanduser().resolve() / rs.bundle_id
        split_path = split_question_ids_path(rp.router_dataset_dir)
        if not split_path.is_file():
            raise FileNotFoundError(f"split_question_ids.json not found: {split_path}")

        from surf_rag.results.loaders import load_benchmark_sources, load_split_qids

        split_qids = load_split_qids(split_path, rs.split)
        if not split_qids and rs.split != "all":
            raise ValueError(f"No question ids for split {rs.split!r} in {split_path}")

        if not rp.benchmark_path.is_file():
            raise FileNotFoundError(f"Benchmark not found: {rp.benchmark_path}")

        qid_to_source = load_benchmark_sources(rp.benchmark_path)
        oracle_scores = rp.router_oracle_dir / "oracle_scores.jsonl"
        if not oracle_scores.is_file():
            raise FileNotFoundError(f"Oracle scores not found: {oracle_scores}")

        retrieval_graph = rp.router_oracle_dir / "retrieval_graph.jsonl"
        answerability = rp.bundle / "audit" / "answerability" / "verdicts.jsonl"

        policies: dict[str, PolicyRun] = {}
        for policy, entry in rs.policies.items():
            if not entry.run_id.strip():
                raise ValueError(f"results.policies.{policy}.run_id is empty")
            run_dir = e2e_policy_run_dir(
                rp.benchmark_base,
                rp.benchmark_name,
                rp.benchmark_id,
                policy,
                entry.run_id,
            )
            metrics_path = run_dir / "metrics.json"
            if not metrics_path.is_file():
                raise FileNotFoundError(
                    f"Missing metrics for policy {policy!r} run {entry.run_id!r}: "
                    f"{metrics_path}"
                )
            rc = run_dir / "resolved_config.yaml"
            policies[policy] = PolicyRun(
                policy=policy,
                run_id=entry.run_id,
                router_role=entry.router_role,
                metrics_path=metrics_path,
                run_dir=run_dir,
                resolved_config_path=rc if rc.is_file() else None,
            )

        theme = _resolve_theme(cfg)
        from surf_rag.viz.theme import apply_figures_theme

        image_format = apply_figures_theme(theme, image_format=rs.image_format)

        return cls(
            cfg=cfg,
            resolved=rp,
            results=rs,
            output_dir=output_dir,
            split_qids=split_qids,
            qid_to_source=qid_to_source,
            policies=policies,
            oracle_scores_path=oracle_scores,
            split_question_ids_path=split_path,
            retrieval_graph_path=retrieval_graph,
            answerability_path=answerability,
            image_format=image_format,
        )


def _resolve_theme(cfg: PipelineConfig):
    """Prefer results.theme; fall back to figures.theme for non-default fields."""
    from dataclasses import replace

    rt = cfg.results.theme
    ft = cfg.figures.theme
    merged_overrides = {**(ft.overrides or {}), **(rt.overrides or {})}
    dpi = rt.dpi if rt.dpi != 200 or ft.dpi == 200 else ft.dpi
    if rt.dpi == 200 and ft.dpi != 200:
        dpi = ft.dpi
    backend = rt.backend if rt.backend is not None else ft.backend
    return replace(
        rt,
        dpi=dpi,
        backend=backend,
        overrides=merged_overrides,
    )


def get_router_arch(bundle: ResultsBundle, role: str) -> ResultsRouterArch | None:
    return bundle.results.router.get(role)


def enabled_artifacts(bundle: ResultsBundle) -> list[ResultsArtifactSpec]:
    return [a for a in bundle.results.artifacts if a.enabled and a.id.strip()]


def policy_list(
    bundle: ResultsBundle, *, exclude: list[str] | None = None
) -> list[str]:
    ex = set(exclude or [])
    return [p for p in bundle.policies if p not in ex]

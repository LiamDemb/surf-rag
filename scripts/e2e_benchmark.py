#!/usr/bin/env python3
"""End-to-end benchmark: routed retrieval, batch generation, collect, evaluate."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from surf_rag.config.env import load_app_env, apply_pipeline_env_from_config
from surf_rag.config.loader import load_pipeline_config, validate_e2e_config
from surf_rag.config.merge import (
    merge_e2e_common_args,
    merge_e2e_evaluate_args,
    merge_e2e_prepare_args,
)
from surf_rag.evaluation.artifact_paths import (
    default_benchmark_base,
    default_router_base,
)
from surf_rag.evaluation.e2e_runner import (
    evaluate_e2e_run,
    e2e_prepare_and_submit,
    make_e2e_run_paths,
)
from surf_rag.evaluation.e2e_policies import (
    ORACLE_E2E_POLICIES,
    parse_routing_policy,
)
from surf_rag.evaluation.router_dataset_artifacts import (
    make_router_dataset_paths_for_cli,
)
from surf_rag.generation.batch_orchestrator import (
    collect_batches,
    submit_dry_run_batches,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--benchmark-base", type=Path, default=None)
    p.add_argument("--benchmark-name", default=None)
    p.add_argument("--benchmark-id", default=None)
    p.add_argument("--benchmark-path", type=Path, default=None)
    p.add_argument(
        "--split", default="test", help="Split label for batch custom_id / manifest."
    )
    p.add_argument("--run-id", default=None)
    p.add_argument(
        "--policy",
        default=None,
        help=(
            "Routing policy: learned-soft, learned-soft-log, hard-routing, hybrid, "
            "50-50, 50-50-log, rrf, "
            "dense-only, graph-only, oracle-upper-bound, oracle-classification"
        ),
    )
    p.add_argument("--retrieval-asset-dir", type=Path, default=None)


def _e2e_config(args: argparse.Namespace):
    return getattr(args, "_pipeline_config", None)


def cmd_prepare(args: argparse.Namespace) -> int:
    cfg = _e2e_config(args)
    if (
        not args.benchmark_name
        or not args.benchmark_id
        or not args.benchmark_path
        or not args.run_id
        or not args.policy
        or not args.retrieval_asset_dir
    ):
        log.error(
            "Missing required fields. Use --config path.yaml or pass "
            "--benchmark-name, --benchmark-id, --benchmark-path, --run-id, "
            "--policy, --retrieval-asset-dir."
        )
        return 2
    if cfg:
        try:
            validate_e2e_config(cfg)
        except ValueError as e:
            log.error("%s", e)
            return 2
    try:
        policy = parse_routing_policy(args.policy)
    except ValueError as e:
        log.error("%s", e)
        return 2
    if policy in ORACLE_E2E_POLICIES:
        if not args.router_id or not str(args.router_id).strip():
            log.error(
                "Oracle e2e policies require --router-id (or config paths.router_id)."
            )
            return 2
        if str(args.split).strip().lower() != "test":
            log.error("%s is test-only; use --split test.", policy)
            return 2
    bb = args.benchmark_base or default_benchmark_base()
    return e2e_prepare_and_submit(
        args.benchmark_path.resolve(),
        benchmark_base=bb,
        benchmark_name=args.benchmark_name,
        benchmark_id=args.benchmark_id,
        split=args.split,
        run_id=args.run_id,
        routing_policy=policy,
        retrieval_asset_dir=args.retrieval_asset_dir.resolve(),
        router_id=args.router_id,
        router_architecture_id=getattr(args, "router_architecture_id", None),
        router_base=args.router_base,
        fusion_keep_k=args.fusion_keep_k,
        rrf_k=int(getattr(args, "rrf_k", 60)),
        reranker_kind=args.reranker,
        rerank_top_k=args.rerank_top_k,
        cross_encoder_model=args.cross_encoder_model,
        cross_encoder_device=args.cross_encoder_device,
        limit=args.limit,
        only_question_ids=set(args.only_question_id) if args.only_question_id else None,
        completion_window=args.completion_window,
        include_graph_provenance=args.include_graph_provenance,
        dry_run=args.dry_run,
        router_device=args.router_device,
        router_input_mode=args.router_input_mode,
        router_task_type=args.router_task_type,
        router_confidence_threshold=args.router_confidence_threshold,
        router_fallback_regressor_id=args.router_fallback_regressor_id,
        router_fallback_architecture_id=args.router_fallback_architecture_id,
        router_inference_batch_size=args.router_inference_batch_size,
        latency_warmup_questions=args.latency_warmup_questions,
        dev_sync=args.dev_sync,
        pipeline_config_for_artifact=cfg,
        router_embedding_provider=getattr(args, "router_embedding_provider", None),
        router_embedding_cache_mode=getattr(args, "router_embedding_cache_mode", None),
        router_embedding_cache_id=getattr(args, "router_embedding_cache_id", None),
        router_embedding_cache_path=getattr(args, "router_embedding_cache_path", None),
        router_embedding_cache_writeback=getattr(
            args, "router_embedding_cache_writeback", None
        ),
        router_openai_embedding_dimensions=getattr(
            args, "router_openai_embedding_dimensions", None
        ),
        branch_top_k=int(getattr(args, "branch_top_k", 20)),
        branch_cache_mode=str(getattr(args, "branch_cache_mode", "off")),
        branch_cache_oracle_router_id=getattr(
            args, "branch_cache_oracle_router_id", None
        ),
        branch_cache_dense_jsonl=(
            str(args.branch_cache_dense_jsonl)
            if getattr(args, "branch_cache_dense_jsonl", None)
            else None
        ),
        branch_cache_graph_jsonl=(
            str(args.branch_cache_graph_jsonl)
            if getattr(args, "branch_cache_graph_jsonl", None)
            else None
        ),
        branch_cache_strict_manifest=(
            True
            if getattr(args, "branch_cache_strict_manifest", None) is None
            else bool(args.branch_cache_strict_manifest)
        ),
        branch_cache_sequential_retrieval_total=(
            False
            if getattr(args, "branch_cache_sequential_retrieval_total", None) is None
            else bool(args.branch_cache_sequential_retrieval_total)
        ),
    )


def cmd_collect(args: argparse.Namespace) -> int:
    if (
        not args.benchmark_name
        or not args.benchmark_id
        or not args.run_id
        or not args.policy
    ):
        log.error("Missing required E2E fields (use --config or CLI flags).")
        return 2
    bb = args.benchmark_base or default_benchmark_base()
    try:
        policy = parse_routing_policy(args.policy)
    except ValueError as e:
        log.error("%s", e)
        return 2
    paths = make_e2e_run_paths(
        benchmark_base=bb,
        benchmark_name=args.benchmark_name,
        benchmark_id=args.benchmark_id,
        policy=policy,
        run_id=args.run_id,
    )
    return collect_batches(run_root=paths.run_root)


def cmd_submit_batch(args: argparse.Namespace) -> int:
    if (
        not args.benchmark_name
        or not args.benchmark_id
        or not args.run_id
        or not args.policy
    ):
        log.error("Missing required E2E fields (use --config or CLI flags).")
        return 2
    bb = args.benchmark_base or default_benchmark_base()
    try:
        policy = parse_routing_policy(args.policy)
    except ValueError as e:
        log.error("%s", e)
        return 2
    paths = make_e2e_run_paths(
        benchmark_base=bb,
        benchmark_name=args.benchmark_name,
        benchmark_id=args.benchmark_id,
        policy=policy,
        run_id=args.run_id,
    )
    return submit_dry_run_batches(run_root=paths.run_root)


def cmd_evaluate(args: argparse.Namespace) -> int:
    if (
        not args.benchmark_name
        or not args.benchmark_id
        or not args.benchmark_path
        or not args.run_id
        or not args.policy
    ):
        log.error("Missing required E2E fields (use --config or CLI flags).")
        return 2
    bb = args.benchmark_base or default_benchmark_base()
    try:
        policy = parse_routing_policy(args.policy)
    except ValueError as e:
        log.error("%s", e)
        return 2
    paths = make_e2e_run_paths(
        benchmark_base=bb,
        benchmark_name=args.benchmark_name,
        benchmark_id=args.benchmark_id,
        policy=policy,
        run_id=args.run_id,
    )
    split_path = args.split_question_ids
    if split_path is None and args.router_id:
        dsp = make_router_dataset_paths_for_cli(
            args.router_id, router_base=args.router_base
        )
        split_path = dsp.split_question_ids
    report = evaluate_e2e_run(
        run_paths=paths,
        benchmark_path=args.benchmark_path.resolve(),
        split_question_ids_path=split_path,
        apply_answerability_audit=bool(
            getattr(args, "apply_answerability_audit", False)
        ),
    )
    out = paths.run_root / "metrics.json"
    out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    log.info("Wrote %s", out)
    return 0


def cmd_print_config(args: argparse.Namespace) -> int:
    if (
        not args.benchmark_name
        or not args.benchmark_id
        or not args.run_id
        or not args.policy
    ):
        log.error("Missing required E2E fields (use --config or CLI flags).")
        return 2
    bb = args.benchmark_base or default_benchmark_base()
    try:
        policy = parse_routing_policy(args.policy)
    except ValueError as e:
        log.error("%s", e)
        return 2
    paths = make_e2e_run_paths(
        benchmark_base=bb,
        benchmark_name=args.benchmark_name,
        benchmark_id=args.benchmark_id,
        policy=policy,
        run_id=args.run_id,
    )
    print("run_root", paths.run_root)
    print("retrieval", paths.retrieval_results_jsonl())
    print("batch_state", paths.batch_state_json())
    print("answers", paths.generation_answers_jsonl())
    print("metrics", paths.run_root / "metrics.json")
    return 0


def main() -> int:
    load_app_env()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML pipeline/E2E config (see configs/templates/)",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser(
        "prepare",
        help="Routed retrieval + write batch JSONL (+ submit unless --dry-run)",
    )
    _add_common(p_prep)
    p_prep.add_argument("--router-id", default=None)
    p_prep.add_argument("--router-architecture-id", default=None)
    p_prep.add_argument("--router-base", type=Path, default=None)
    p_prep.add_argument(
        "--fusion-keep-k",
        type=int,
        default=25,
        help=(
            "Generation-context retrieval depth before LLM prompting; pure retrieval "
            "metrics are sourced from pre-truncation retrieval artifacts."
        ),
    )
    p_prep.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF smoothing constant k in 1/(k+rank) for policy rrf (default: 60).",
    )
    p_prep.add_argument(
        "--branch-top-k",
        type=int,
        default=20,
        help="Per-branch top-k for dense/graph retrievers (align with frozen oracle cache).",
    )
    p_prep.add_argument(
        "--branch-cache-mode",
        default="off",
        help="Frozen replay: off | router_oracle | explicit_jsonl.",
    )
    p_prep.add_argument(
        "--branch-cache-oracle-router-id",
        default=None,
        help="Override router id for router_oracle cache directory (default: paths.router_id).",
    )
    p_prep.add_argument(
        "--branch-cache-dense-jsonl",
        type=Path,
        default=None,
        help="Explicit dense retrieval JSONL (mode explicit_jsonl).",
    )
    p_prep.add_argument(
        "--branch-cache-graph-jsonl",
        type=Path,
        default=None,
        help="Explicit graph retrieval JSONL (mode explicit_jsonl).",
    )
    p_prep.add_argument(
        "--branch-cache-strict-manifest",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Strictly validate oracle manifest vs benchmark/corpus (default: true from YAML).",
    )
    p_prep.add_argument(
        "--branch-cache-sequential-retrieval-total",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "When using frozen dual-branch fusion, set fused total_ms to sum of cached "
            "branch totals + fusion (default: false from YAML)."
        ),
    )
    p_prep.add_argument(
        "--reranker",
        default="none",
        help="none | cross_encoder (default model: cross-encoder/ms-marco-MiniLM-L-6-v2)",
    )
    p_prep.add_argument("--rerank-top-k", type=int, default=10)
    p_prep.add_argument("--cross-encoder-model", default=None)
    p_prep.add_argument("--cross-encoder-device", default=None)
    p_prep.add_argument("--limit", type=int, default=None)
    p_prep.add_argument("--only-question-id", action="append", default=[])
    p_prep.add_argument("--completion-window", default="24h")
    p_prep.add_argument("--include-graph-provenance", action="store_true")
    p_prep.add_argument("--dry-run", action="store_true")
    p_prep.add_argument(
        "--dev-sync",
        action="store_true",
        help="Dev-only: call chat completions synchronously and ingest immediately.",
    )
    p_prep.add_argument("--router-device", default="cpu")
    p_prep.add_argument("--router-input-mode", default="both")
    p_prep.add_argument("--router-task-type", default="regression")
    p_prep.add_argument("--router-confidence-threshold", type=float, default=0.7)
    p_prep.add_argument("--router-fallback-regressor-id", default=None)
    p_prep.add_argument("--router-fallback-architecture-id", default=None)
    p_prep.add_argument(
        "--router-inference-batch-size",
        type=int,
        default=32,
        help="Mini-batch size for learned-router query embeddings + features.",
    )
    p_prep.add_argument(
        "--latency-warmup-questions",
        type=int,
        default=0,
        help="Warmup retrieval questions excluded from latency reporting.",
    )
    p_prep.add_argument(
        "--router-embedding-provider",
        default=None,
        help="Override dataset embedding provider for router query tensors (st|openai).",
    )
    p_prep.add_argument(
        "--router-embedding-cache-mode",
        default=None,
        help="off | prefer | required | auto (default from config; auto follows dataset).",
    )
    p_prep.add_argument(
        "--router-embedding-cache-id",
        default=None,
        help="Benchmark embedding cache id (under benchmark bundle query_embeddings/...).",
    )
    p_prep.add_argument(
        "--router-embedding-cache-path",
        default=None,
        help="Explicit cache directory root (overrides default layout).",
    )
    p_prep.add_argument(
        "--router-embedding-cache-writeback",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="When using prefer mode, append cache misses (default: true).",
    )
    p_prep.add_argument(
        "--router-openai-embedding-dimensions",
        type=int,
        default=None,
        help="OpenAI embedding output dimensions override for router query tensors.",
    )
    p_prep.set_defaults(func=cmd_prepare)

    p_sb = sub.add_parser(
        "submit-batch", help="Submit an already generated batch from a dry-run"
    )
    _add_common(p_sb)
    p_sb.set_defaults(func=cmd_submit_batch)

    p_col = sub.add_parser("collect", help="Download batch outputs → answers.jsonl")
    _add_common(p_col)
    p_col.set_defaults(func=cmd_collect)

    p_ev = sub.add_parser(
        "evaluate", help="Compute metrics.json from retrieval + answers"
    )
    _add_common(p_ev)
    p_ev.add_argument(
        "--router-id", default=None, help="Default split_question_ids path"
    )
    p_ev.add_argument("--router-base", type=Path, default=None)
    p_ev.add_argument(
        "--split-question-ids",
        type=Path,
        default=None,
        help="Override router dataset split JSON (train/dev/test ids)",
    )
    audit_grp = p_ev.add_mutually_exclusive_group()
    audit_grp.add_argument(
        "--apply-answerability-audit",
        action="store_true",
        help="Use canonical mask/manifest under benchmark bundle (see audit/answerability/).",
    )
    audit_grp.add_argument(
        "--no-apply-answerability-audit",
        action="store_true",
        help="Disable answerability masking even if set in YAML.",
    )
    p_ev.set_defaults(func=cmd_evaluate, apply_answerability_audit=False)

    p_pc = sub.add_parser("print-config", help="Print resolved paths for one run")
    _add_common(p_pc)
    p_pc.set_defaults(func=cmd_print_config)

    args = ap.parse_args()
    if args.config:
        cfg = load_pipeline_config(args.config.resolve())
        apply_pipeline_env_from_config(cfg)
        args._pipeline_config = cfg
        if args.cmd == "prepare":
            merge_e2e_prepare_args(args, cfg)
        elif args.cmd in ("collect", "submit-batch", "print-config"):
            merge_e2e_common_args(args, cfg)
        elif args.cmd == "evaluate":
            merge_e2e_evaluate_args(args, cfg)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""End-to-end benchmark: routed retrieval, batch generation, collect, evaluate."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from argparse import BooleanOptionalAction
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
from surf_rag.evaluation.router_dataset_artifacts import (
    make_router_dataset_paths_for_cli,
)
from surf_rag.generation.batch_orchestrator import collect_batches

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
        help="Routing policy: learned-soft, learned-hard, 50-50, dense-only, graph-only",
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
    bb = args.benchmark_base or default_benchmark_base()
    return e2e_prepare_and_submit(
        args.benchmark_path.resolve(),
        benchmark_base=bb,
        benchmark_name=args.benchmark_name,
        benchmark_id=args.benchmark_id,
        split=args.split,
        run_id=args.run_id,
        routing_policy=args.policy,
        retrieval_asset_dir=args.retrieval_asset_dir.resolve(),
        router_id=args.router_id,
        router_base=args.router_base,
        fusion_keep_k=args.fusion_keep_k,
        branch_top_k=args.branch_top_k,
        reranker_kind=args.reranker,
        rerank_top_k=args.rerank_top_k,
        cross_encoder_model=args.cross_encoder_model,
        sentence_window_radius=args.sentence_window_radius,
        sentence_window_max_windows=args.sentence_window_max_windows,
        sentence_window_min_windows=args.sentence_window_min_windows,
        sentence_window_max_words=args.sentence_window_max_words,
        sentence_window_max_subwindow_words=args.sentence_window_max_subwindow_words,
        sentence_window_min_top_chunk_coverage=args.sentence_window_min_top_chunk_coverage,
        sentence_window_min_distinct_parent_chunks=args.sentence_window_min_distinct_parent_chunks,
        sentence_window_max_per_chunk=args.sentence_window_max_per_chunk,
        sentence_window_iou_select_threshold=args.sentence_window_iou_select_threshold,
        sentence_window_premerge_iou=args.sentence_window_premerge_iou,
        sentence_window_premerge_max_gap_chars=args.sentence_window_premerge_max_gap_chars,
        sentence_window_ce_relax_margin=args.sentence_window_ce_relax_margin,
        sentence_window_ce_filler_top_ranks=args.sentence_window_ce_filler_top_ranks,
        sentence_window_filler_title_overlap=args.sentence_window_filler_title_overlap,
        sentence_window_filler_novel_parent_max_rank=args.sentence_window_filler_novel_parent_max_rank,
        sentence_window_merge_overlaps=args.sentence_window_merge_overlaps,
        sentence_window_duplicate_filter=args.sentence_window_duplicate_filter,
        sentence_window_include_title=args.sentence_window_include_title,
        limit=args.limit,
        only_question_ids=set(args.only_question_id) if args.only_question_id else None,
        completion_window=args.completion_window,
        include_graph_provenance=args.include_graph_provenance,
        dry_run=args.dry_run,
        router_device=args.router_device,
        router_input_mode=args.router_input_mode,
        router_inference_batch_size=args.router_inference_batch_size,
        pipeline_config_for_artifact=cfg,
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
    from surf_rag.evaluation.e2e_policies import parse_routing_policy

    policy = parse_routing_policy(args.policy)
    paths = make_e2e_run_paths(
        benchmark_base=bb,
        benchmark_name=args.benchmark_name,
        benchmark_id=args.benchmark_id,
        policy=policy,
        run_id=args.run_id,
    )
    return collect_batches(run_root=paths.run_root)


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
    from surf_rag.evaluation.e2e_policies import parse_routing_policy

    policy = parse_routing_policy(args.policy)
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
    from surf_rag.evaluation.e2e_policies import parse_routing_policy

    policy = parse_routing_policy(args.policy)
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
    p_prep.add_argument("--router-base", type=Path, default=None)
    p_prep.add_argument(
        "--branch-top-k",
        type=int,
        default=20,
        help="Per-branch (dense / graph) retrieval depth before fusion trim.",
    )
    p_prep.add_argument("--fusion-keep-k", type=int, default=20)
    p_prep.add_argument(
        "--reranker",
        default="none",
        help="none | cross_encoder | sentence_window (cross-encoder model from config)",
    )
    p_prep.add_argument(
        "--rerank-top-k",
        type=int,
        default=5,
        help="For cross_encoder: number of chunks after rerank. Ignored for sentence_window max windows (use --sentence-window-max-windows).",
    )
    p_prep.add_argument("--cross-encoder-model", default=None)
    p_prep.add_argument("--sentence-window-radius", type=int, default=1)
    p_prep.add_argument(
        "--sentence-window-max-windows",
        type=int,
        default=12,
        help="Upper cap on selected windows (target 8–12 in default configs).",
    )
    p_prep.add_argument(
        "--sentence-window-min-windows",
        type=int,
        default=8,
        help="Try to select at least this many windows when pool and word budget allow.",
    )
    p_prep.add_argument(
        "--sentence-window-max-words",
        type=int,
        default=1280,
        help="Total evidence word budget across selected windows.",
    )
    p_prep.add_argument(
        "--sentence-window-max-subwindow-words",
        type=int,
        default=180,
        help="Hard cap per window before scoring; oversized spans are split.",
    )
    p_prep.add_argument(
        "--sentence-window-iou-select-threshold",
        type=float,
        default=0.35,
    )
    p_prep.add_argument(
        "--sentence-window-premerge-iou",
        type=float,
        default=0.35,
    )
    p_prep.add_argument(
        "--sentence-window-premerge-max-gap-chars",
        type=int,
        default=48,
    )
    p_prep.add_argument(
        "--sentence-window-ce-relax-margin",
        type=float,
        default=3.0,
        help="Filler pass: allow scores within this margin of the per-question max.",
    )
    p_prep.add_argument(
        "--sentence-window-ce-filler-top-ranks",
        type=int,
        default=3,
        help="Filler: always-allow parent chunks ranked below this (0..k-1).",
    )
    p_prep.add_argument(
        "--sentence-window-filler-title-overlap",
        action=BooleanOptionalAction,
        default=True,
    )
    p_prep.add_argument(
        "--sentence-window-filler-novel-parent-max-rank",
        type=int,
        default=10,
        help="Max fused parent rank for weak-score filler from a not-yet-covered chunk.",
    )
    p_prep.add_argument("--sentence-window-min-top-chunk-coverage", type=int, default=3)
    p_prep.add_argument(
        "--sentence-window-min-distinct-parent-chunks",
        type=int,
        default=4,
        help="Prefer evidence from at least this many distinct retrieved chunks when available.",
    )
    p_prep.add_argument("--sentence-window-max-per-chunk", type=int, default=2)
    p_prep.add_argument(
        "--sentence-window-merge-overlaps",
        action=BooleanOptionalAction,
        default=True,
    )
    p_prep.add_argument(
        "--sentence-window-duplicate-filter",
        action=BooleanOptionalAction,
        default=True,
    )
    p_prep.add_argument(
        "--sentence-window-include-title",
        action=BooleanOptionalAction,
        default=True,
    )
    p_prep.add_argument("--limit", type=int, default=None)
    p_prep.add_argument("--only-question-id", action="append", default=[])
    p_prep.add_argument("--completion-window", default="24h")
    p_prep.add_argument("--include-graph-provenance", action="store_true")
    p_prep.add_argument("--dry-run", action="store_true")
    p_prep.add_argument("--router-device", default="cpu")
    p_prep.add_argument("--router-input-mode", default="both")
    p_prep.add_argument(
        "--router-inference-batch-size",
        type=int,
        default=32,
        help="Mini-batch size for learned-router query embeddings + features.",
    )
    p_prep.set_defaults(func=cmd_prepare)

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
    p_ev.set_defaults(func=cmd_evaluate)

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
        elif args.cmd == "collect":
            merge_e2e_common_args(args, cfg)
        elif args.cmd == "evaluate":
            merge_e2e_evaluate_args(args, cfg)
        elif args.cmd == "print-config":
            merge_e2e_common_args(args, cfg)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

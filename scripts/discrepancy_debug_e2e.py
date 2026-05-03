#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

from surf_rag.config.argv import argv_provides
from surf_rag.evaluation.artifact_paths import (
    benchmark_jsonl_path,
    default_benchmark_base,
)
from surf_rag.evaluation.discrepancy_debug import (
    SelectionRule,
    assert_reranker_none_or_allow_ce,
    extract_interesting_rows,
    load_answers_by_qid,
    load_benchmark_index,
    load_retrieval_by_qid,
    load_split_test_question_ids,
    read_e2e_manifest_block,
    write_discrepancy_bundle,
)
from surf_rag.evaluation.e2e_runner import make_e2e_run_paths
from surf_rag.evaluation.e2e_policies import parse_routing_policy
from surf_rag.evaluation.router_dataset_artifacts import (
    make_router_dataset_paths_for_cli,
)

_FLAG_BY_DEST: tuple[tuple[str, str], ...] = (
    ("benchmark_path", "--benchmark-path"),
    ("benchmark_base", "--benchmark-base"),
    ("benchmark_name", "--benchmark-name"),
    ("benchmark_id", "--benchmark-id"),
    ("policy", "--policy"),
    ("policy_a", "--policy-a"),
    ("policy_b", "--policy-b"),
    ("run_id_a", "--run-id-a"),
    ("run_id_b", "--run-id-b"),
    ("run_root_a", "--run-root-a"),
    ("run_root_b", "--run-root-b"),
    ("restrict_router_test_split", "--restrict-router-test-split"),
    ("restrict_question_ids_path", "--restrict-question-ids-path"),
    ("router_base", "--router-base"),
    ("epsilon_ndcg", "--epsilon-ndcg"),
    ("delta_f1", "--delta-f1"),
    ("output_root", "--output-root"),
    ("test_id", "--test-id"),
    ("top_k_chunks", "--top-k-chunks"),
    ("chunk_preview_chars", "--chunk-preview-chars"),
    ("markdown_max_rows", "--markdown-max-rows"),
)


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.expanduser().resolve().read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise SystemExit(
            f"Invalid YAML root in {path}: expected mapping, got {type(raw)}"
        )
    inner = raw.get("discrepancy_debug")
    if isinstance(inner, dict):
        return inner
    return raw


def _maybe_path(val: Any) -> Path | None:
    if val is None:
        return None
    if isinstance(val, Path):
        return val
    if isinstance(val, str) and val.strip().lower() in ("", "~", "null"):
        return None
    return Path(str(val).strip()).expanduser()


def merge_discrepancy_yaml_into_args(
    args: argparse.Namespace, data: Mapping[str, Any]
) -> None:
    """YAML fills Namespace when the corresponding CLI flag was not passed."""
    argv = sys.argv
    if not isinstance(data, dict):
        return
    for dest, flag in _FLAG_BY_DEST:
        if argv_provides(argv, flag):
            continue
        key = dest
        if key not in data or data[key] is None:
            continue
        if dest in {
            "benchmark_path",
            "benchmark_base",
            "output_root",
            "run_root_a",
            "run_root_b",
            "router_base",
            "restrict_question_ids_path",
        }:
            setattr(args, dest, _maybe_path(data[key]))
            continue
        if dest in (
            "run_id_a",
            "run_id_b",
            "benchmark_name",
            "benchmark_id",
            "policy",
            "policy_a",
            "policy_b",
            "test_id",
            "restrict_router_test_split",
        ):
            setattr(args, dest, str(data[key]).strip())
            continue
        setattr(args, dest, data[key])
    # Booleans without opposite flags
    if (
        not argv_provides(argv, "--allow-cross-encoder")
        and data.get("allow_cross_encoder") is True
    ):
        args.allow_cross_encoder = True
    if (
        not argv_provides(argv, "-q")
        and not argv_provides(argv, "--quiet")
        and data.get("quiet") is True
    ):
        args.quiet = True


def finalize_benchmark_and_requirements(args: argparse.Namespace) -> None:
    """Set benchmark_path from bundle triple when needed; enforce required inputs."""
    if args.benchmark_path is None:
        if args.benchmark_name and args.benchmark_id:
            bb = (
                Path(args.benchmark_base).expanduser().resolve()
                if args.benchmark_base is not None
                else default_benchmark_base()
            )
            args.benchmark_path = benchmark_jsonl_path(
                bb,
                str(args.benchmark_name).strip(),
                str(args.benchmark_id).strip(),
            )
    if args.benchmark_path is None:
        raise SystemExit(
            "benchmark_path unset: pass --benchmark-path, use --config with benchmark_path "
            "or benchmark_name + benchmark_id (and optionally benchmark_base); "
            "default bundle layout is …/<name>/<id>/benchmark/benchmark.jsonl."
        )
    if args.test_id is None or not str(args.test_id).strip():
        raise SystemExit(
            "Missing test_id (use --test-id or set test_id in --config YAML)."
        )
    args.test_id = str(args.test_id).strip()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.split("\n\n")[3:]).strip(),
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML defaults (configs/gen-debug/example.yaml); CLI overrides file.",
    )
    p.add_argument(
        "--benchmark-path",
        type=Path,
        default=None,
        help="benchmark.jsonl (defaults from config or benchmark_base/name/id)",
    )

    grp = p.add_argument_group(
        "Run roots",
        "Either both --run-root-a/b or resolve under evaluations/ using bundle triple, "
        "run ids, and policy (shared or per-side).",
    )
    grp.add_argument("--run-root-a", type=Path, default=None)
    grp.add_argument("--run-root-b", type=Path, default=None)

    grp.add_argument(
        "--benchmark-base",
        type=Path,
        default=None,
        help="With --benchmark-name/--benchmark-id and run ids (/ per-side policy), "
        "defaults to BENCHMARK_BASE env or data/benchmarks.",
    )
    grp.add_argument("--benchmark-name", default=None)
    grp.add_argument("--benchmark-id", default=None)
    grp.add_argument(
        "--policy",
        default=None,
        help=(
            "Shared routing policy for both runs (evaluations/<policy>/… within the bundle). "
            "Omit when using distinct --policy-a / --policy-b."
        ),
    )
    grp.add_argument(
        "--policy-a",
        dest="policy_a",
        default=None,
        help="Run A only; falls back to --policy when omitted.",
    )
    grp.add_argument(
        "--policy-b",
        dest="policy_b",
        default=None,
        help="Run B only; falls back to --policy when omitted.",
    )
    grp.add_argument("--run-id-a", default=None)
    grp.add_argument("--run-id-b", default=None)

    r = p.add_argument_group("Restrict questions")
    r.add_argument(
        "--restrict-router-test-split",
        metavar="ROUTER_ID",
        default=None,
        help="Only benchmark rows whose question_id is in router dataset test split.",
    )
    r.add_argument(
        "--restrict-question-ids-path",
        type=Path,
        default=None,
        help="JSON with a top-level `test` list of question_id strings.",
    )
    r.add_argument(
        "--router-base",
        type=Path,
        default=None,
        help="Override router bundle root for --restrict-router-test-split.",
    )

    p.add_argument(
        "--allow-cross-encoder",
        action="store_true",
        help="Do not fail when manifest e2e.reranker is set (e.g. cross_encoder).",
    )
    p.add_argument(
        "--epsilon-ndcg",
        type=float,
        default=1e-9,
        help="Minimum ΔnDCG@10 (B minus A) to count as retrieval improved.",
    )
    p.add_argument(
        "--delta-f1",
        type=float,
        default=0.0,
        help="F1(B) must be ≤ F1(A) + this threshold.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("temp/discrepancy-debugging"),
        help="Parent directory for this report (default: temp/discrepancy-debugging).",
    )
    p.add_argument(
        "--test-id",
        default=None,
        help="Subdirectory under output-root (from config when omitted there).",
    )
    p.add_argument("--top-k-chunks", type=int, default=10)
    p.add_argument("--chunk-preview-chars", type=int, default=240)
    p.add_argument(
        "--markdown-max-rows",
        type=int,
        default=200,
        help="Max rows in interesting.md table.",
    )
    p.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress info logs (errors still print).",
    )
    return p


def _effective_policies(args: argparse.Namespace) -> tuple[str, str]:
    """Return raw policy CLI strings after applying shared ``policy`` fallback."""
    fb = str(getattr(args, "policy", "") or "").strip()
    pa = str(getattr(args, "policy_a", None) or "").strip() or fb
    pb = str(getattr(args, "policy_b", None) or "").strip() or fb
    return pa, pb


def _resolve_run_roots(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.run_root_a and args.run_root_b:
        return (
            args.run_root_a.expanduser().resolve(),
            args.run_root_b.expanduser().resolve(),
        )
    pa, pb = _effective_policies(args)
    need = [
        args.benchmark_base,
        args.benchmark_name,
        args.benchmark_id,
        pa,
        pb,
        args.run_id_a,
        args.run_id_b,
    ]
    if not all(need):
        raise SystemExit(
            "Provide --run-root-a and --run-root-b, or all of: "
            "--benchmark-base, --benchmark-name, --benchmark-id, "
            "--run-id-a, --run-id-b, and a routing policy "
            "(e.g. --policy for both runs, or --policy-a/--policy-b with fallback to --policy)."
        )
    bb = (
        Path(args.benchmark_base).expanduser().resolve()
        if args.benchmark_base is not None
        else default_benchmark_base()
    )
    canon_a = parse_routing_policy(pa)
    canon_b = parse_routing_policy(pb)
    ra = make_e2e_run_paths(
        benchmark_base=bb,
        benchmark_name=args.benchmark_name,
        benchmark_id=args.benchmark_id,
        policy=canon_a,
        run_id=args.run_id_a,
    ).run_root
    rb = make_e2e_run_paths(
        benchmark_base=bb,
        benchmark_name=args.benchmark_name,
        benchmark_id=args.benchmark_id,
        policy=canon_b,
        run_id=args.run_id_b,
    ).run_root
    return ra, rb


def _comparison_resolution_meta(args: argparse.Namespace) -> dict[str, Any]:
    """Metadata written to manifest.inputs.comparison_resolution."""
    if args.run_root_a and args.run_root_b:
        return {"mode": "explicit_run_roots"}
    pa, pb = _effective_policies(args)
    ca = parse_routing_policy(pa)
    cb = parse_routing_policy(pb)
    return {
        "mode": "evaluations_layout",
        "run_a": {"routing_policy": ca, "run_id": str(args.run_id_a)},
        "run_b": {"routing_policy": cb, "run_id": str(args.run_id_b)},
        "same_policy_folder": ca == cb,
    }


def _resolve_restrict_qids(args: argparse.Namespace) -> set[str] | None:
    if args.restrict_router_test_split and args.restrict_question_ids_path:
        raise SystemExit(
            "Use only one of --restrict-router-test-split or --restrict-question-ids-path."
        )
    if args.restrict_router_test_split:
        dsp = make_router_dataset_paths_for_cli(
            str(args.restrict_router_test_split).strip(),
            router_base=args.router_base,
        )
        sp = dsp.split_question_ids
        if not sp.is_file():
            raise SystemExit(f"Missing split file: {sp}")
        return load_split_test_question_ids(sp)
    if args.restrict_question_ids_path:
        p = args.restrict_question_ids_path.expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"Missing split file: {p}")
        return load_split_test_question_ids(p)
    return None


def main() -> int:
    args = build_parser().parse_args()
    if args.config is not None:
        cfg_path = Path(args.config).expanduser().resolve()
        if not cfg_path.is_file():
            raise SystemExit(f"Config not found: {cfg_path}")
        merge_discrepancy_yaml_into_args(args, _load_yaml_mapping(cfg_path))
    finalize_benchmark_and_requirements(args)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    bench_path = Path(args.benchmark_path).expanduser().resolve()
    if not bench_path.is_file():
        raise SystemExit(f"Benchmark not found: {bench_path}")

    try:
        root_a, root_b = _resolve_run_roots(args)
        cmp_meta = _comparison_resolution_meta(args)
    except ValueError as e:
        raise SystemExit(str(e)) from e

    try:
        assert_reranker_none_or_allow_ce(
            root_a,
            root_b,
            allow_cross_encoder=bool(args.allow_cross_encoder),
        )
    except ValueError as e:
        raise SystemExit(str(e)) from e

    restrict = _resolve_restrict_qids(args)

    rule = SelectionRule(
        epsilon_ndcg=float(args.epsilon_ndcg), delta_f1=float(args.delta_f1)
    )

    bench = load_benchmark_index(bench_path)
    retr_a = load_retrieval_by_qid(root_a / "retrieval" / "retrieval_results.jsonl")
    retr_b = load_retrieval_by_qid(root_b / "retrieval" / "retrieval_results.jsonl")
    ans_a = load_answers_by_qid(root_a / "generation" / "answers.jsonl")
    ans_b = load_answers_by_qid(root_b / "generation" / "answers.jsonl")

    interesting_rows, counts = extract_interesting_rows(
        bench_by_qid=bench,
        retr_a=retr_a,
        retr_b=retr_b,
        ans_a=ans_a,
        ans_b=ans_b,
        restrict_qids=restrict,
        rule=rule,
        top_k_chunks=int(args.top_k_chunks),
        chunk_preview_chars=int(args.chunk_preview_chars),
    )

    out_parent = (
        Path(args.output_root).expanduser().resolve() / str(args.test_id).strip()
    )

    write_discrepancy_bundle(
        out_parent,
        test_id=str(args.test_id),
        benchmark_path=bench_path,
        run_root_a=root_a,
        run_root_b=root_b,
        e2e_a=read_e2e_manifest_block(root_a),
        e2e_b=read_e2e_manifest_block(root_b),
        rule=rule,
        interesting_rows=interesting_rows,
        counts=counts,
        markdown_max_rows=int(args.markdown_max_rows),
        comparison_resolution=cmp_meta,
    )
    logging.info(
        "Wrote %s (interesting=%d, evaluated_rows=%d)",
        out_parent,
        counts.interesting,
        counts.joined_question_ids,
    )
    print(out_parent.as_posix())
    return 0


if __name__ == "__main__":
    sys.exit(main())

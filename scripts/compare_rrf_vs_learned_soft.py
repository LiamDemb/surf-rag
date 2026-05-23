#!/usr/bin/env python3
"""Export questions where RRF nDCG@k beats learned-soft, with top-k chunk previews.

Reads paths and run ids from a pipeline YAML (``paths``, ``e2e``, ``results``).
Optional overrides live under ``policy_retrieval_compare:`` in the same file.

Examples:

    poetry run python scripts/compare_rrf_vs_learned_soft.py \\
        --config configs/results/004.yaml

    poetry run python scripts/compare_rrf_vs_learned_soft.py \\
        --config configs/dev-may/rg-ep-balancing.yaml \\
        --run-id e2e-dev/rg-ep \\
        --restrict-split train \\
        --output-id rg-ep-train
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

from surf_rag.config.argv import argv_provides
from surf_rag.config.loader import load_pipeline_config, resolve_paths
from surf_rag.evaluation.policy_retrieval_compare import (
    RETRIEVAL_ARTIFACT_FINAL,
    RETRIEVAL_ARTIFACT_PRETRUNC,
    PolicyRetrievalCompareConfig,
    run_policy_retrieval_compare,
)

_FLAG_BY_DEST: tuple[tuple[str, str], ...] = (
    ("run_id", "--run-id"),
    ("run_id_rrf", "--run-id-rrf"),
    ("run_id_learned_soft", "--run-id-learned-soft"),
    ("restrict_split", "--restrict-split"),
    ("retrieval_artifact", "--retrieval-artifact"),
    ("metric_k", "--metric-k"),
    ("epsilon_ndcg", "--epsilon-ndcg"),
    ("top_k_chunks", "--top-k-chunks"),
    ("chunk_preview_chars", "--chunk-preview-chars"),
    ("output_root", "--output-root"),
    ("output_id", "--output-id"),
    ("split_question_ids", "--split-question-ids"),
)


def _load_compare_yaml_block(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.expanduser().resolve().read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise SystemExit(
            f"Invalid YAML root in {path}: expected mapping, got {type(raw)}"
        )
    inner = raw.get("policy_retrieval_compare")
    if isinstance(inner, dict):
        return inner
    return {}


def _maybe_path(val: Any) -> Path | None:
    if val is None:
        return None
    if isinstance(val, Path):
        return val
    if isinstance(val, str) and val.strip().lower() in ("", "~", "null"):
        return None
    return Path(str(val).strip()).expanduser()


def merge_compare_yaml_into_args(
    args: argparse.Namespace, data: Mapping[str, Any]
) -> None:
    argv = sys.argv
    if not isinstance(data, dict):
        return
    for dest, flag in _FLAG_BY_DEST:
        if argv_provides(argv, flag):
            continue
        if dest not in data or data[dest] is None:
            continue
        if dest in ("output_root", "split_question_ids"):
            setattr(args, dest, _maybe_path(data[dest]))
            continue
        if dest in ("run_id", "run_id_rrf", "run_id_learned_soft", "restrict_split"):
            setattr(args, dest, str(data[dest]).strip())
            continue
        setattr(args, dest, data[dest])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RRF vs learned-soft retrieval wins (nDCG@k) for qualitative review.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Pipeline YAML (paths, e2e.run_id, and/or results.policies).",
    )
    p.add_argument("--run-id", default=None, help="Shared run id for both policies.")
    p.add_argument("--run-id-rrf", default=None)
    p.add_argument("--run-id-learned-soft", default=None)
    p.add_argument(
        "--restrict-split",
        default=None,
        choices=("all", "train", "dev", "test"),
        help="Filter by router split_question_ids.json (default: results.split or e2e.split).",
    )
    p.add_argument(
        "--retrieval-artifact",
        default=RETRIEVAL_ARTIFACT_PRETRUNC,
        choices=(RETRIEVAL_ARTIFACT_PRETRUNC, RETRIEVAL_ARTIFACT_FINAL),
        help="pretrunc = before reranker; final = retrieval_results.jsonl.",
    )
    p.add_argument("--metric-k", type=int, default=10)
    p.add_argument("--epsilon-ndcg", type=float, default=1e-9)
    p.add_argument("--top-k-chunks", type=int, default=10)
    p.add_argument("--chunk-preview-chars", type=int, default=50)
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("temp/policy-retrieval-compare"),
    )
    p.add_argument(
        "--output-id",
        default=None,
        help="Subdirectory under output-root (default: experiment_id or 'default').",
    )
    p.add_argument(
        "--split-question-ids",
        type=Path,
        default=None,
        help="Override router dataset split_question_ids.json.",
    )
    p.add_argument(
        "--markdown-max-rows",
        type=int,
        default=200,
        help="Max rows in rrf_wins.md (full data always in rrf_wins.jsonl).",
    )
    p.add_argument("-q", "--quiet", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")

    compare_yaml = _load_compare_yaml_block(cfg_path)
    merge_compare_yaml_into_args(args, compare_yaml)

    cfg = load_pipeline_config(cfg_path)
    rp = resolve_paths(cfg)

    output_id = str(args.output_id or "").strip()
    if not output_id:
        output_id = str(cfg.experiment_id or "default").strip() or "default"

    compare = PolicyRetrievalCompareConfig(
        metric_k=int(args.metric_k),
        epsilon_ndcg=float(args.epsilon_ndcg),
        top_k_chunks=int(args.top_k_chunks),
        chunk_preview_chars=int(args.chunk_preview_chars),
        retrieval_artifact=str(args.retrieval_artifact),
        restrict_split=(
            str(args.restrict_split).strip().lower() if args.restrict_split else None
        ),
        output_root=Path(args.output_root),
        output_id=output_id,
    )

    split_path = args.split_question_ids
    if split_path is None:
        split_path = rp.router_dataset_dir / "split_question_ids.json"
    split_path = Path(split_path).expanduser().resolve()
    if not split_path.is_file():
        raise SystemExit(f"Split file not found: {split_path}")

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    try:
        out_dir, counts, n_wins = run_policy_retrieval_compare(
            cfg,
            benchmark_path=rp.benchmark_path,
            split_question_ids_path=split_path,
            run_id=args.run_id,
            run_id_rrf=args.run_id_rrf,
            run_id_learned_soft=args.run_id_learned_soft,
            compare=compare,
            compare_yaml=compare_yaml,
            markdown_max_rows=int(args.markdown_max_rows),
        )
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(str(e)) from e

    logging.info(
        "Wrote %s (rrf_wins=%d, evaluated=%d, ls_wins=%d, ties=%d)",
        out_dir,
        n_wins,
        counts.evaluated_question_ids,
        counts.learned_soft_wins,
        counts.ties,
    )
    print(out_dir.as_posix())
    return 0


if __name__ == "__main__":
    sys.exit(main())

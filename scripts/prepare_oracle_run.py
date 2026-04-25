"""Produce raw oracle weight scores."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from surf_rag.evaluation.oracle_artifacts import (
    DEFAULT_DENSE_WEIGHT_GRID,
    OracleRunPaths,
    build_oracle_run_root,
    default_oracle_base,
)
from surf_rag.evaluation.oracle_pipeline import (
    OracleRunConfig,
    prepare_oracle_run,
)
from surf_rag.evaluation.retrieval_metrics import (
    DEFAULT_NDCG_KS,
    PRIMARY_NDCG_K,
)
from surf_rag.strategies.factory import (
    build_dense_retriever,
    build_graph_retriever,
)

logger = logging.getLogger(__name__)


def _progress(label: str, idx: int, total: int) -> None:
    if total <= 0:
        return
    if idx == total or idx % max(1, total // 20) == 0:
        logger.info("%s: %d/%d", label, idx, total)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare (or resume) an oracle labeling run."
    )
    parser.add_argument(
        "--router-id", required=True, help="Router bundle id, e.g. v01 (oracle path key)."
    )
    parser.add_argument(
        "--benchmark-name",
        required=True,
        help="Source benchmark family name, e.g. 'mix' or 'hotpotqa'.",
    )
    parser.add_argument(
        "--benchmark-id",
        required=True,
        help="Source benchmark bundle id, e.g. v01 (for provenance).",
    )
    parser.add_argument(
        "--source-split",
        default=None,
        help="Optional upstream split label (metadata only; not used in paths).",
    )
    parser.add_argument(
        "--benchmark-path",
        required=True,
        type=Path,
        help="Path to benchmark JSONL containing question_id, question, dataset_source, gold_support_sentences.",
    )
    parser.add_argument(
        "--retrieval-asset-dir",
        required=True,
        type=Path,
        help="Directory containing corpus, FAISS index, and graph artifacts used by both retrievers.",
    )
    parser.add_argument(
        "--branch-top-k",
        type=int,
        default=int(os.getenv("ORACLE_BRANCH_TOP_K", "25")),
        help="Top-k for each branch retriever before fusion. Default 25.",
    )
    parser.add_argument(
        "--fusion-keep-k",
        type=int,
        default=int(os.getenv("ORACLE_FUSION_KEEP_K", "25")),
        help="Chunks retained after fusion re-scoring. Default 25.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on benchmark rows for quick smoke runs.",
    )
    parser.add_argument(
        "--router-base",
        type=Path,
        default=None,
        help="Override router bundle root (else $ROUTER_BASE, then $DATA_BASE/router).",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s: %(message)s")

    router_base = args.router_base if args.router_base else default_oracle_base()
    paths = OracleRunPaths(
        run_root=build_oracle_run_root(router_base, args.router_id)
    )

    cfg = OracleRunConfig(
        router_id=args.router_id,
        benchmark_name=args.benchmark_name,
        benchmark_id=args.benchmark_id,
        benchmark_path=args.benchmark_path,
        retrieval_asset_dir=args.retrieval_asset_dir,
        source_split=args.source_split,
        branch_top_k=args.branch_top_k,
        fusion_keep_k=args.fusion_keep_k,
        weight_grid=DEFAULT_DENSE_WEIGHT_GRID,
        oracle_metric="stateful_ndcg",
        oracle_metric_k=PRIMARY_NDCG_K,
        diagnostic_metric_ks=DEFAULT_NDCG_KS,
    )

    logger.info("Oracle run root: %s", paths.run_root)
    summary = prepare_oracle_run(
        cfg,
        paths,
        dense_retriever_factory=lambda: build_dense_retriever(
            str(args.retrieval_asset_dir),
            top_k=args.branch_top_k,
        ),
        graph_retriever_factory=lambda: build_graph_retriever(
            str(args.retrieval_asset_dir),
            top_k=args.branch_top_k,
        ),
        limit=args.limit,
        progress=_progress,
    )

    logger.info("questions: %d", summary["questions_snapshot"])
    logger.info("dense cached (total): %d", summary["dense_cached"])
    logger.info("graph cached (total): %d", summary["graph_cached"])
    logger.info("oracle rows (total): %d", summary["oracle_scored"])
    logger.info("newly retrieved dense: %d", summary["newly_retrieved_dense"])
    logger.info("newly retrieved graph: %d", summary["newly_retrieved_graph"])
    logger.info("newly scored: %d", summary["newly_scored"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

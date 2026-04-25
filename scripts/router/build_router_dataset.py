"""Build router training dataset (features, embeddings, soft labels) as Parquet.

Example::

    python -m scripts.router.build_router_dataset \\
        --router-benchmark mybench \\
        --router-dataset-id ds1 \\
        --benchmark-path data/processed/benchmark.jsonl \\
        --retrieval-asset-dir data/processed \\
        --oracle-benchmark mybench --oracle-split dev --oracle-run-id run1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from surf_rag.evaluation.oracle_artifacts import (
    OracleRunPaths,
    build_oracle_run_root,
    default_oracle_base,
    read_jsonl,
)
from surf_rag.evaluation.router_dataset_artifacts import (
    default_router_dataset_base,
    make_router_dataset_paths_for_cli,
    write_feature_stats,
    write_router_dataset_manifest,
    write_split_summary,
    write_parquet,
)
from surf_rag.router.dataset import build_router_dataframe
from surf_rag.router.query_features import QueryFeatureContext

logger = None


def _load_benchmark(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark not found: {path}")
    return read_jsonl(path)


def _resolve_entity_pipeline(
    retrieval_asset_dir: Optional[str],
) -> Any:
    if not retrieval_asset_dir:
        return None
    p = Path(retrieval_asset_dir)
    alias = p / "alias_map.json"
    if not alias.is_file():
        return None
    try:
        from surf_rag.entity_matching.pipeline import LexiconAliasEntityPipeline

        return LexiconAliasEntityPipeline.from_artifacts(str(p))
    except (OSError, FileNotFoundError, ValueError) as e:
        if logger:
            logger.warning("Entity pipeline unavailable: %s", e)
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build router Parquet dataset from oracle labels."
    )
    p.add_argument(
        "--router-benchmark",
        required=True,
        help="Benchmark tag for data/router/... path.",
    )
    p.add_argument(
        "--router-dataset-id",
        required=True,
        help="Dataset run id (directory under benchmark).",
    )
    p.add_argument(
        "--benchmark-path",
        type=Path,
        required=True,
        help="Full benchmark JSONL (same as oracle; single file).",
    )
    p.add_argument(
        "--retrieval-asset-dir",
        type=Path,
        default=None,
        help="Corpus/alias dir for Graph entity pipeline (optional but recommended).",
    )
    p.add_argument(
        "--oracle-benchmark", required=True, help="Oracle run benchmark tag."
    )
    p.add_argument("--oracle-split", required=True, help="Oracle split tag, e.g. dev.")
    p.add_argument("--oracle-run-id", required=True, help="Oracle run id folder.")
    p.add_argument(
        "--selected-beta",
        type=float,
        default=None,
        help="Beta used in labels/selected.jsonl; if omitted, read from run manifest/labels file.",
    )
    p.add_argument(
        "--router-base",
        type=Path,
        default=None,
        help="Override router dataset base (else ROUTER_DATASET_BASE or data/router).",
    )
    p.add_argument(
        "--oracle-base",
        type=Path,
        default=None,
        help="Override oracle base (else ORACLE_BASE or data/oracle).",
    )
    p.add_argument(
        "--embedding-model",
        default=None,
        help="SentenceTransformer model name (else EMBEDDING_MODEL / MODEL_NAME env).",
    )
    p.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Override SEED (default 42 from env).",
    )
    p.add_argument(
        "--train-ratio", type=float, default=None, help="Override TRAIN_RATIO (env)."
    )
    p.add_argument(
        "--dev-ratio", type=float, default=None, help="Override DEV_RATIO (env)."
    )
    p.add_argument(
        "--test-ratio", type=float, default=None, help="Override TEST_RATIO (env)."
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    global logger
    load_dotenv()
    import logging

    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    router_base = (
        args.router_base if args.router_base else default_router_dataset_base()
    )
    oracle_base = args.oracle_base if args.oracle_base else default_oracle_base()
    o_paths = OracleRunPaths(
        run_root=build_oracle_run_root(
            oracle_base,
            args.oracle_benchmark,
            args.oracle_split,
            args.oracle_run_id,
        )
    )
    if not o_paths.labels_selected.is_file():
        logger.error(
            "Missing %s. Run create_soft_labels / oracle flow first.",
            o_paths.labels_selected,
        )
        return 1
    label_rows = read_jsonl(o_paths.labels_selected)
    if not label_rows:
        logger.error("Empty labels: %s", o_paths.labels_selected)
        return 1

    selected_beta = (
        float(args.selected_beta)
        if args.selected_beta is not None
        else float(label_rows[0].get("beta", 0.0))
    )
    bench = _load_benchmark(args.benchmark_path)

    r_paths = make_router_dataset_paths_for_cli(
        args.router_benchmark, args.router_dataset_id, router_base=router_base
    )
    r_paths.ensure_dirs()

    ent_pipe = _resolve_entity_pipeline(
        str(args.retrieval_asset_dir) if args.retrieval_asset_dir else None
    )
    ctx = QueryFeatureContext(
        nlp=None,
        entity_pipeline=ent_pipe,
        retrieval_asset_dir=(
            str(args.retrieval_asset_dir) if args.retrieval_asset_dir else None
        ),
    )

    emb_model = (
        args.embedding_model
        or os.getenv("EMBEDDING_MODEL")
        or os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
    )
    split_seed = int(
        args.split_seed if args.split_seed is not None else os.getenv("SEED", "42")
    )
    tr = (
        float(args.train_ratio)
        if args.train_ratio is not None
        else float(os.getenv("TRAIN_RATIO", "0.8"))
    )
    dv = (
        float(args.dev_ratio)
        if args.dev_ratio is not None
        else float(os.getenv("DEV_RATIO", "0.1"))
    )
    te = (
        float(args.test_ratio)
        if args.test_ratio is not None
        else float(os.getenv("TEST_RATIO", "0.1"))
    )
    s = tr + dv + te
    if s <= 0:
        raise SystemExit("split ratios must be positive and sum to > 0")
    tr, dv, te = tr / s, dv / s, te / s

    df, normalizer, sum_meta = build_router_dataframe(
        bench,
        label_rows,
        feature_context=ctx,
        embedding_model=emb_model,
        train_ratio=tr,
        dev_ratio=dv,
        test_ratio=te,
        split_seed=split_seed,
        selected_beta=selected_beta,
        oracle_run_id=args.oracle_run_id,
    )

    write_parquet(r_paths.router_dataset_parquet, df)
    write_feature_stats(r_paths.feature_stats, normalizer.to_json())
    write_split_summary(r_paths.split_summary, sum_meta, run_root=r_paths.run_root)
    write_router_dataset_manifest(
        r_paths,
        dataset_id=args.router_dataset_id,
        benchmark=args.router_benchmark,
        benchmark_path=str(args.benchmark_path.resolve()),
        oracle_base=(
            str(oracle_base.resolve())
            if isinstance(oracle_base, Path)
            else str(oracle_base)
        ),
        oracle_benchmark=args.oracle_benchmark,
        oracle_split=args.oracle_split,
        oracle_run_id=args.oracle_run_id,
        oracle_run_root=str(o_paths.run_root.resolve()),
        labels_selected_path=str(o_paths.labels_selected.resolve()),
        selected_beta=selected_beta,
        feature_set_version=df["feature_set_version"].iloc[0] if len(df) else "1",
        embedding_model=emb_model,
        split_seed=split_seed,
        train_ratio=tr,
        dev_ratio=dv,
        test_ratio=te,
        extra={"row_count": int(len(df))},
    )
    logger.info("Wrote %s", r_paths.router_dataset_parquet)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Build router training dataset (features, embeddings, soft labels) as Parquet.

Example::

    python -m scripts.router.build_router_dataset \\
        --router-id v01 \\
        --benchmark-name mix \\
        --benchmark-id v01 \\
        --benchmark-path data/mix/v01/benchmark/benchmark.jsonl \\
        --retrieval-asset-dir data/mix/v01/corpus
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from surf_rag.config.env import load_app_env, apply_pipeline_env_from_config
from surf_rag.config.loader import load_pipeline_config, resolve_paths
from surf_rag.config.merge import merge_router_build_dataset_args
from surf_rag.config.resolved import write_resolved_config_yaml

from surf_rag.evaluation.oracle_artifacts import (
    OracleRunPaths,
    build_oracle_run_root,
    read_jsonl,
)
from surf_rag.evaluation.artifact_paths import default_router_base
from surf_rag.evaluation.router_dataset_artifacts import (
    build_split_question_ids_dict,
    make_router_dataset_paths_for_cli,
    write_feature_stats,
    write_router_dataset_manifest,
    write_split_question_ids,
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
        "--config",
        type=Path,
        default=None,
        help="YAML pipeline config (see configs/templates/pipeline.yaml).",
    )
    p.add_argument(
        "--router-id",
        default=None,
        help="Router bundle id (same as oracle + dataset directory).",
    )
    p.add_argument(
        "--benchmark-name",
        default=None,
        help="Source benchmark family (provenance).",
    )
    p.add_argument(
        "--benchmark-id",
        default=None,
        help="Source benchmark bundle id (provenance).",
    )
    p.add_argument(
        "--benchmark-path",
        type=Path,
        default=None,
        help="Full benchmark JSONL (same as oracle; single file).",
    )
    p.add_argument(
        "--retrieval-asset-dir",
        type=Path,
        default=None,
        help="Corpus/alias dir for Graph entity pipeline (optional but recommended).",
    )
    p.add_argument(
        "--selected-beta",
        type=float,
        default=None,
        help="Beta used in labels/selected.jsonl; if omitted, read from label rows.",
    )
    p.add_argument(
        "--router-base",
        type=Path,
        default=None,
        help="Override router bundle base (else $ROUTER_BASE, then $DATA_BASE/router).",
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
    load_app_env()
    load_dotenv()
    import logging

    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    args._pipeline_cfg = None
    if args.config:
        pcfg = load_pipeline_config(args.config.resolve())
        args._pipeline_cfg = pcfg
        apply_pipeline_env_from_config(pcfg)
        merge_router_build_dataset_args(args, pcfg)
    if (
        not args.router_id
        or not args.benchmark_name
        or not args.benchmark_id
        or not args.benchmark_path
    ):
        logger.error("Provide --config or all of router/benchmark id/name/path flags.")
        return 2

    router_base = args.router_base if args.router_base else default_router_base()
    o_paths = OracleRunPaths(
        run_root=build_oracle_run_root(router_base, args.router_id)
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

    r_paths = make_router_dataset_paths_for_cli(args.router_id, router_base=router_base)
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
        else float(os.getenv("TRAIN_RATIO", "0.6"))
    )
    dv = (
        float(args.dev_ratio)
        if args.dev_ratio is not None
        else float(os.getenv("DEV_RATIO", "0.2"))
    )
    te = (
        float(args.test_ratio)
        if args.test_ratio is not None
        else float(os.getenv("TEST_RATIO", "0.2"))
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
        router_id=args.router_id,
    )

    write_parquet(r_paths.router_dataset_parquet, df)
    write_feature_stats(r_paths.feature_stats, normalizer.to_json())
    write_split_summary(r_paths.split_summary, sum_meta, run_root=r_paths.run_root)
    split_payload = build_split_question_ids_dict(
        df,
        router_id=args.router_id,
        source_benchmark_name=args.benchmark_name,
        source_benchmark_id=args.benchmark_id,
        split_seed=split_seed,
    )
    write_split_question_ids(r_paths.split_question_ids, split_payload)
    write_router_dataset_manifest(
        r_paths,
        router_id=args.router_id,
        source_benchmark_name=args.benchmark_name,
        source_benchmark_id=args.benchmark_id,
        benchmark_path=str(args.benchmark_path.resolve()),
        retrieval_asset_dir=(
            str(args.retrieval_asset_dir.resolve()) if args.retrieval_asset_dir else ""
        ),
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
    if getattr(args, "_pipeline_cfg", None) is not None:
        write_resolved_config_yaml(
            r_paths.run_root / "resolved_config.yaml",
            args._pipeline_cfg,
            resolve_paths(args._pipeline_cfg),
        )
    logger.info("Wrote %s", r_paths.router_dataset_parquet)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Build router training dataset (features, embeddings, oracle curves) as Parquet."""

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
    build_ignored_router_questions_payload,
    build_split_question_ids_dict,
    make_router_dataset_paths_for_cli,
    write_feature_stats,
    write_ignored_router_questions,
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
        description="Build router Parquet dataset from oracle curve labels."
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
    p.add_argument(
        "--embedding-provider",
        default=None,
        help="sentence-transformers | openai (else config / ROUTER_EMBEDDING_PROVIDER).",
    )
    p.add_argument(
        "--embedding-cache-mode",
        default=None,
        help="auto | off | prefer | required | build (else config).",
    )
    p.add_argument(
        "--embedding-cache-id",
        default=None,
        help="Cache subdirectory id under query_embeddings/...",
    )
    p.add_argument(
        "--embedding-cache-path",
        default=None,
        help="Override cache root directory (advanced).",
    )
    p.add_argument(
        "--embedding-cache-writeback",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Append live-computed rows to benchmark cache in prefer/build modes.",
    )
    p.add_argument(
        "--openai-embedding-dimensions",
        type=int,
        default=None,
        help="OpenAI embeddings API dimensions (e.g. 256 for text-embedding-3-large).",
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
    if not o_paths.router_labels.is_file():
        logger.error(
            "Missing %s. Run router-label materialization first.",
            o_paths.router_labels,
        )
        return 1
    label_rows = read_jsonl(o_paths.router_labels)
    if not label_rows:
        logger.error("Empty labels: %s", o_paths.router_labels)
        return 1
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

    from surf_rag.router.embedding_config import (
        parse_embedding_provider,
        resolve_embedding_cache_mode_for_dataset,
        resolve_embedding_model_for_provider,
    )

    if args.config is not None:
        emb_prov = parse_embedding_provider(str(args.embedding_provider))
        emb_mode = str(args.embedding_cache_mode)
        emb_model = str(args.embedding_model)
    else:
        emb_prov = parse_embedding_provider(
            str(getattr(args, "embedding_provider", None) or "")
            or os.getenv("ROUTER_EMBEDDING_PROVIDER", "sentence-transformers")
        )
        emb_mode = resolve_embedding_cache_mode_for_dataset(
            str(emb_prov),
            str(getattr(args, "embedding_cache_mode", None) or "")
            or os.getenv("ROUTER_EMBEDDING_CACHE_MODE", "auto"),
        )
        emb_model = resolve_embedding_model_for_provider(
            str(emb_prov),
            str(
                args.embedding_model
                or os.getenv("EMBEDDING_MODEL_FOR_ROUTER")
                or os.getenv("EMBEDDING_MODEL")
                or os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
            ),
        )

    cache_id = (
        str(
            getattr(args, "embedding_cache_id", None)
            or os.getenv("ROUTER_EMBEDDING_CACHE_ID", "")
            or "default"
        ).strip()
        or "default"
    )
    cache_path_ov = getattr(args, "embedding_cache_path", None)
    if (
        cache_path_ov is not None
        and isinstance(cache_path_ov, str)
        and not str(cache_path_ov).strip()
    ):
        cache_path_ov = None
    if getattr(args, "embedding_cache_writeback", None) is None:
        emb_wb = True
    else:
        emb_wb = bool(args.embedding_cache_writeback)

    openai_ed = getattr(args, "openai_embedding_dimensions", None)
    if openai_ed is not None:
        openai_ed = int(openai_ed)
    df, normalizer, sum_meta = build_router_dataframe(
        bench,
        label_rows,
        feature_context=ctx,
        embedding_model=emb_model,
        embedding_provider=emb_prov,
        embedding_cache_mode=emb_mode,
        benchmark_path=args.benchmark_path.resolve(),
        embedding_cache_id=cache_id,
        embedding_cache_path=cache_path_ov,
        embedding_cache_writeback=emb_wb,
        train_ratio=tr,
        dev_ratio=dv,
        test_ratio=te,
        split_seed=split_seed,
        router_id=args.router_id,
        openai_embedding_dimensions=openai_ed,
    )

    write_parquet(r_paths.router_dataset_parquet, df)
    write_feature_stats(r_paths.feature_stats, normalizer.to_json())
    write_split_summary(r_paths.split_summary, sum_meta, run_root=r_paths.run_root)
    ignored_payload = build_ignored_router_questions_payload(
        df, router_id=args.router_id
    )
    write_ignored_router_questions(r_paths.ignored_router_questions, ignored_payload)
    split_payload = build_split_question_ids_dict(
        df,
        router_id=args.router_id,
        source_benchmark_name=args.benchmark_name,
        source_benchmark_id=args.benchmark_id,
        split_seed=split_seed,
    )
    write_split_question_ids(r_paths.split_question_ids, split_payload)
    emb_dim = None
    if len(df) and "embedding_dim" in df.columns:
        emb_dim = int(df["embedding_dim"].iloc[0])
    emb_cache = dict(sum_meta.get("embedding_resolution") or {})
    emb_cache["cache_id"] = cache_id
    emb_cache["writeback"] = emb_wb
    emb_cache["benchmark_path"] = str(args.benchmark_path.resolve())
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
        router_labels_path=str(o_paths.router_labels.resolve()),
        feature_set_version=df["feature_set_version"].iloc[0] if len(df) else "1",
        embedding_model=emb_model,
        embedding_provider=emb_prov,
        embedding_dim=emb_dim,
        embedding_cache=emb_cache,
        split_seed=split_seed,
        train_ratio=tr,
        dev_ratio=dv,
        test_ratio=te,
        openai_embedding_dimensions=openai_ed,
        extra={
            "row_count": int(len(df)),
            "row_count_total": int(ignored_payload["num_rows_total"]),
            "row_count_router_eligible": int(ignored_payload["eligible_count_total"]),
            "row_count_router_ignored_all_zero": int(
                ignored_payload["ignored_count_total"]
            ),
            "ignored_questions_report": r_paths.ignored_router_questions.name,
        },
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

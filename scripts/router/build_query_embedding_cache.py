"""Build or extend benchmark-local OpenAI / ST query embedding cache JSONL."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from surf_rag.config.argv import argv_provides
from surf_rag.config.env import apply_pipeline_env_from_config, load_app_env
from surf_rag.config.loader import load_pipeline_config, resolve_paths
from surf_rag.evaluation.oracle_artifacts import read_jsonl
from surf_rag.evaluation.query_embedding_cache_artifacts import (
    append_query_embedding_rows,
    fingerprint_benchmark_jsonl,
    load_query_embedding_map_by_qid,
    make_query_embedding_cache_paths,
    write_query_embedding_cache_manifest,
)
from surf_rag.router.embedding_config import (
    parse_embedding_provider,
    resolve_embedding_model_for_provider,
)
from surf_rag.router.embedding_providers import embed_texts
from surf_rag.router.question_text_hash import canonical_question_text_hash

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--benchmark-path", type=Path, default=None)
    p.add_argument("--embedding-provider", default=None)
    p.add_argument("--embedding-model", default=None)
    p.add_argument("--embedding-cache-id", default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete existing embeddings.jsonl before writing.",
    )
    p.add_argument(
        "--required-complete",
        action="store_true",
        help="Exit non-zero if any benchmark row is missing after run.",
    )
    p.add_argument(
        "--openai-embedding-dimensions",
        type=int,
        default=None,
        help="OpenAI embeddings API output dimensions (text-embedding-3*).",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    load_app_env()
    load_dotenv()
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s: %(message)s")

    cfg = None
    if args.config:
        cfg = load_pipeline_config(args.config.resolve())
        apply_pipeline_env_from_config(cfg)
        rp = resolve_paths(cfg)
        if not args.benchmark_path:
            args.benchmark_path = rp.benchmark_path
        if not args.embedding_provider:
            args.embedding_provider = cfg.router.dataset.embedding_provider
        if not args.embedding_model:
            args.embedding_model = cfg.router.dataset.embedding_model
        if not args.embedding_cache_id:
            args.embedding_cache_id = cfg.router.dataset.embedding_cache_id
        if not argv_provides(sys.argv, "--openai-embedding-dimensions"):
            args.openai_embedding_dimensions = (
                cfg.router.dataset.openai_embedding_dimensions
            )

    bp = args.benchmark_path
    if not bp or not bp.is_file():
        logger.error(
            "Provide --config or --benchmark-path to an existing benchmark.jsonl"
        )
        return 2

    provider = parse_embedding_provider(
        str(args.embedding_provider or "sentence-transformers")
    )
    model = resolve_embedding_model_for_provider(
        provider, str(args.embedding_model or "")
    )
    cache_id = str(args.embedding_cache_id or "default").strip() or "default"

    od = getattr(args, "openai_embedding_dimensions", None)
    if od is not None:
        od = int(od)

    paths = make_query_embedding_cache_paths(
        bp.resolve(),
        provider=provider,
        model=model,
        cache_id=cache_id,
        openai_dimensions=od,
    )
    paths.ensure_dirs()

    if args.rebuild and paths.embeddings_jsonl.is_file():
        paths.embeddings_jsonl.unlink()

    existing: Dict[str, Dict[str, Any]] = {}
    if paths.embeddings_jsonl.is_file():
        existing = load_query_embedding_map_by_qid(paths)

    rows_in = read_jsonl(bp)
    to_embed: List[Dict[str, Any]] = []
    for row in rows_in:
        qid = str(row.get("question_id", "") or "").strip()
        q = str(row.get("question", "") or "")
        if not qid:
            logger.error("Benchmark row missing question_id")
            return 1
        if qid in existing:
            continue
        to_embed.append({"question_id": qid, "question": q, "row": row})

    if not to_embed:
        logger.info("Cache already complete (%d rows).", len(existing))
        if args.required_complete and len(existing) < len(rows_in):
            logger.error("required-complete: not all benchmark rows cached")
            return 1
        return 0

    texts = [t["question"] for t in to_embed]
    vecs = embed_texts(
        texts,
        provider=provider,
        model=model,
        batch_size=int(args.batch_size or 64),
        openai_dimensions=od,
    )
    dim = int(vecs.shape[1]) if len(vecs) else 0

    out_rows: List[Dict[str, Any]] = []
    for i, item in enumerate(to_embed):
        qh = canonical_question_text_hash(item["question"])
        vec = vecs[i].tolist() if i < len(vecs) else []
        out_rows.append(
            {
                "question_id": item["question_id"],
                "question": item["question"],
                "question_hash": qh,
                "embedding": vec,
                "embedding_dim": len(vec),
                "embedding_provider": provider,
                "embedding_model": model,
            }
        )

    append_query_embedding_rows(paths, out_rows)

    merged = {**existing}
    for r in out_rows:
        merged[str(r["question_id"])] = r

    fp = fingerprint_benchmark_jsonl(bp.resolve())
    emb_block: Dict[str, Any] = {
        "provider": provider,
        "model": model,
        "dim": dim,
        "normalization": "l2_row",
    }
    if od is not None:
        emb_block["openai_dimensions"] = int(od)
    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "source_benchmark": {
            "path": str(bp.resolve()),
            "sha256": fp,
            "row_count": len(rows_in),
        },
        "embedding": emb_block,
        "cache_id": cache_id,
        "counts": {
            "cached_unique_qids": len(merged),
            "appended_rows": len(out_rows),
        },
    }
    write_query_embedding_cache_manifest(paths, manifest)

    if args.required_complete and len(merged) < len(rows_in):
        logger.error(
            "required-complete: cached %d unique qids but benchmark has %d rows",
            len(merged),
            len(rows_in),
        )
        return 1

    logger.info("Wrote %d new rows under %s", len(out_rows), paths.run_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())

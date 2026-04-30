#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from surf_rag.config.env import apply_pipeline_env_from_config
from surf_rag.config.loader import load_pipeline_config, resolve_paths
from surf_rag.config.resolved import write_resolved_config_yaml
from surf_rag.core.benchmark_samples import iter_jsonl
from surf_rag.core.corpus_finalize import (
    finalize_corpus_artifacts,
    load_corpus_chunks,
    write_corpus_finalize_manifest,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_samples(path: Path | None) -> list[dict] | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    return list(iter_jsonl(str(path)))


def _remaining_failed_chunk_ids(chunks: list[dict]) -> list[str]:
    out: list[str] = []
    for row in chunks:
        cid = str(row.get("chunk_id", "")).strip()
        if not cid:
            continue
        meta = row.get("metadata") or {}
        if str(meta.get("ie_status", "")).strip().lower() == "failed":
            out.append(cid)
    return sorted(set(out))


def _reconcile_ie_failure_artifacts(
    corpus_dir: Path, failed_chunk_ids: list[str]
) -> None:
    failures_path = corpus_dir / "ie_failures_final.json"
    pending_path = corpus_dir / "ie_pending_ids.txt"
    if failed_chunk_ids:
        payload = {
            "max_attempts": None,
            "remaining_unresolved_ids": failed_chunk_ids,
            "source": "corpus-finalize-reconciled",
        }
        failures_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        pending_path.write_text(
            "".join(f"{cid}\n" for cid in failed_chunk_ids), encoding="utf-8"
        )
        return
    if failures_path.exists():
        failures_path.unlink()
    if pending_path.exists():
        pending_path.unlink()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML pipeline config; sets corpus/benchmark/model defaults unless overridden.",
    )
    ap.add_argument(
        "--corpus-dir",
        type=Path,
        default=None,
        help="Directory containing corpus.jsonl.",
    )
    ap.add_argument(
        "--benchmark",
        type=Path,
        default=None,
        help="Optional benchmark JSONL for quality_report generation.",
    )
    ap.add_argument(
        "--model-name",
        default=None,
        help="Sentence-transformer model for dense/entity indexes.",
    )
    ap.add_argument(
        "--skip-quality-report",
        action="store_true",
        help="Skip quality report generation.",
    )
    args = ap.parse_args()

    if args.config:
        cfg = load_pipeline_config(args.config.resolve())
        apply_pipeline_env_from_config(cfg)
        rp = resolve_paths(cfg)
        if args.corpus_dir is None:
            args.corpus_dir = rp.corpus_dir
        if args.benchmark is None:
            args.benchmark = rp.benchmark_path
        if not args.model_name:
            args.model_name = cfg.model_setup.embedding_model

    if args.corpus_dir is None:
        ap.error("Pass --corpus-dir or --config with resolved corpus paths.")
    corpus_dir = args.corpus_dir.expanduser().resolve()
    corpus_path = corpus_dir / "corpus.jsonl"
    model_name = args.model_name or "all-MiniLM-L6-v2"

    try:
        chunks = load_corpus_chunks(corpus_path)
        samples = None if args.skip_quality_report else _load_samples(args.benchmark)
        artifacts = finalize_corpus_artifacts(
            chunks=chunks,
            output_dir=corpus_dir,
            model_name=model_name,
            samples=samples,
            quality_report=(not args.skip_quality_report and samples is not None),
        )
        manifest = write_corpus_finalize_manifest(
            output_dir=corpus_dir,
            corpus_path=corpus_path,
            chunks_count=len(chunks),
            produced_artifacts=artifacts,
        )
        _reconcile_ie_failure_artifacts(
            corpus_dir=corpus_dir,
            failed_chunk_ids=_remaining_failed_chunk_ids(chunks),
        )
        if args.config:
            write_resolved_config_yaml(corpus_dir / "resolved_config.yaml", cfg, rp)
    except (FileNotFoundError, ValueError) as e:
        logger.error("%s", e)
        return 1
    logger.info("Finalized corpus artifacts in %s", corpus_dir)
    logger.info("Manifest: %s", manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())

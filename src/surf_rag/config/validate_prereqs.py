"""Exit-code checks for Makefile validate-* targets (paths from resolved config)."""

from __future__ import annotations

import sys
from pathlib import Path

from surf_rag.config.loader import load_pipeline_config, resolve_paths


def _fail(msg: str) -> int:
    print(msg, file=sys.stderr)
    return 1


def validate_oracle(config_path: Path) -> int:
    cfg = load_pipeline_config(config_path)
    r = resolve_paths(cfg)
    if not r.benchmark_path.is_file():
        return _fail(f"Missing benchmark: {r.benchmark_path}")
    if not r.corpus_dir.is_dir():
        return _fail(f"Missing corpus dir: {r.corpus_dir}")
    return 0


def validate_router_dataset(config_path: Path) -> int:
    rc = validate_oracle(config_path)
    if rc != 0:
        return rc
    cfg = load_pipeline_config(config_path)
    r = resolve_paths(cfg)
    labels = r.router_oracle_dir / "router_labels.jsonl"
    if not labels.is_file():
        return _fail(
            f"Missing oracle labels {labels} — run make oracle-labels (CONFIG=...)"
        )
    from surf_rag.evaluation.query_embedding_cache_artifacts import (
        make_query_embedding_cache_paths,
    )
    from surf_rag.router.embedding_config import (
        EMBEDDING_CACHE_REQUIRED,
        EMBEDDING_PROVIDER_OPENAI,
        parse_embedding_provider,
        resolve_embedding_cache_mode_for_dataset,
        resolve_embedding_model_for_provider,
    )

    rd = cfg.router.dataset
    prov = parse_embedding_provider(str(rd.embedding_provider))
    mode = resolve_embedding_cache_mode_for_dataset(
        str(rd.embedding_provider), str(rd.embedding_cache_mode)
    )
    if prov == EMBEDDING_PROVIDER_OPENAI and mode == EMBEDDING_CACHE_REQUIRED:
        cid = (rd.embedding_cache_id or "").strip()
        if not cid:
            return _fail(
                "router.dataset uses OpenAI with required embedding cache but "
                "embedding_cache_id is empty. Set embedding_cache_id in CONFIG."
            )
        model = resolve_embedding_model_for_provider(
            str(rd.embedding_provider), str(rd.embedding_model)
        )
        cpaths = make_query_embedding_cache_paths(
            r.benchmark_path,
            provider=str(prov),
            model=model,
            cache_id=cid,
            openai_dimensions=rd.openai_embedding_dimensions,
        )
        if not cpaths.embeddings_jsonl.is_file():
            return _fail(
                "OpenAI + required cache: expected embedding cache JSONL at "
                f"{cpaths.embeddings_jsonl} — run "
                f"make router-build-query-embedding-cache CONFIG={config_path}"
            )
    return 0


def validate_router_train(config_path: Path) -> int:
    rc = validate_router_dataset(config_path)
    if rc != 0:
        return rc
    r = resolve_paths(load_pipeline_config(config_path))
    pq = r.router_dataset_dir / "router_dataset.parquet"
    if not pq.is_file():
        return _fail(f"Missing {pq} — run make router-build-dataset (CONFIG=...)")
    return 0

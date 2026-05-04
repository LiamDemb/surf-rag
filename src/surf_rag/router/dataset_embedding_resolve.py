"""Resolve query embeddings for router dataset rows (cache + live)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from surf_rag.evaluation.query_embedding_cache_artifacts import (
    append_query_embedding_rows,
    fingerprint_benchmark_jsonl,
    load_query_embedding_map_by_qid,
    make_query_embedding_cache_paths,
    read_query_embedding_cache_manifest,
    write_query_embedding_cache_manifest,
)
from surf_rag.router.embedding_config import (
    EMBEDDING_CACHE_BUILD,
    EMBEDDING_CACHE_OFF,
    EMBEDDING_CACHE_PREFER,
    EMBEDDING_CACHE_REQUIRED,
)
from surf_rag.router.embedding_providers import embed_texts
from surf_rag.router.question_text_hash import canonical_question_text_hash


def _cache_paths(
    benchmark_path: Path,
    *,
    provider: str,
    model: str,
    cache_id: str,
    cache_path_override: Optional[str],
    openai_embedding_dimensions: int | None = None,
) -> Any:
    from surf_rag.evaluation.query_embedding_cache_artifacts import (
        QueryEmbeddingCachePaths,
    )

    if cache_path_override and str(cache_path_override).strip():
        p = Path(cache_path_override).expanduser().resolve()
        return QueryEmbeddingCachePaths(run_root=p)
    return make_query_embedding_cache_paths(
        benchmark_path.resolve(),
        provider=provider,
        model=model,
        cache_id=cache_id,
        openai_dimensions=openai_embedding_dimensions,
    )


def _validate_cache_row(
    row: Mapping[str, Any],
    *,
    question: str,
    question_id: str,
    provider: str,
    model: str,
) -> np.ndarray:
    exp_hash = canonical_question_text_hash(question)
    got_hash = str(row.get("question_hash", "") or "")
    if got_hash and got_hash != exp_hash:
        raise ValueError(
            f"Embedding cache hash mismatch for question_id={question_id!r}: "
            f"expected {exp_hash!r}, cache has {got_hash!r}"
        )
    if (
        str(row.get("embedding_provider", "") or "").strip()
        and str(row.get("embedding_provider", "")).strip() != provider
    ):
        raise ValueError(
            f"Embedding cache provider mismatch for question_id={question_id!r}"
        )
    if (
        str(row.get("embedding_model", "") or "").strip()
        and str(row.get("embedding_model", "")).strip() != model
    ):
        raise ValueError(
            f"Embedding cache model mismatch for question_id={question_id!r}: "
            f"cache={row.get('embedding_model')!r} expected={model!r}"
        )
    emb = row.get("embedding")
    if emb is None:
        raise ValueError(
            f"Embedding cache row missing embedding for qid={question_id!r}"
        )
    vec = np.asarray(list(emb), dtype=np.float32)
    if vec.size == 0:
        raise ValueError(f"Empty embedding in cache for qid={question_id!r}")
    return vec


def resolve_aligned_query_embeddings(
    aligned_bench: Sequence[Mapping[str, Any]],
    *,
    embedding_provider: str,
    embedding_model: str,
    cache_mode: str,
    benchmark_path: Path,
    cache_id: str,
    cache_path_override: Optional[str],
    writeback: bool,
    batch_size: int = 32,
    openai_embedding_dimensions: int | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return ``(N, D)`` float32 embeddings aligned to ``aligned_bench`` order."""
    n = len(aligned_bench)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32), {
            "cache_mode": cache_mode,
            "hits": 0,
            "misses": 0,
            "live_computed": 0,
            "appended_rows": 0,
            "embedding_dim": 0,
        }

    meta: Dict[str, Any] = {
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "cache_mode": cache_mode,
        "hits": 0,
        "misses": 0,
        "live_computed": 0,
        "appended_rows": 0,
    }
    if openai_embedding_dimensions is not None:
        meta["openai_embedding_dimensions"] = int(openai_embedding_dimensions)
        meta["openai_dimensions"] = int(openai_embedding_dimensions)

    if cache_mode == EMBEDDING_CACHE_OFF:
        texts = [str(b.get("question", "") or "") for b in aligned_bench]
        emb = embed_texts(
            texts,
            provider=embedding_provider,
            model=embedding_model,
            batch_size=batch_size,
            openai_dimensions=openai_embedding_dimensions,
        )
        meta["live_computed"] = n
        meta["embedding_dim"] = int(emb.shape[1]) if emb.size else 0
        return emb, meta

    paths = _cache_paths(
        benchmark_path,
        provider=embedding_provider,
        model=embedding_model,
        cache_id=cache_id,
        cache_path_override=cache_path_override,
        openai_embedding_dimensions=openai_embedding_dimensions,
    )
    meta["cache_root"] = str(paths.run_root.resolve())
    if not paths.embeddings_jsonl.is_file():
        if cache_mode == EMBEDDING_CACHE_REQUIRED:
            raise FileNotFoundError(
                f"Required embedding cache missing: {paths.embeddings_jsonl}. "
                "Run: python -m scripts.router.build_query_embedding_cache --config <yaml>"
            )
        cache_map: Dict[str, Dict[str, Any]] = {}
    else:
        cache_map = load_query_embedding_map_by_qid(paths)

    dim_hint = 0
    for row in cache_map.values():
        e = row.get("embedding")
        if e is not None:
            dim_hint = len(list(e))
            break

    per_row: List[Optional[np.ndarray]] = [None] * n
    miss_idx: List[int] = []
    miss_texts: List[str] = []
    miss_qids: List[str] = []
    miss_questions: List[str] = []

    for i, b in enumerate(aligned_bench):
        qid = str(b.get("question_id", "") or "").strip()
        q = str(b.get("question", "") or "")
        if cache_mode in (
            EMBEDDING_CACHE_REQUIRED,
            EMBEDDING_CACHE_PREFER,
            EMBEDDING_CACHE_BUILD,
        ):
            if not qid:
                raise ValueError(
                    f"embedding_cache_mode={cache_mode!r} requires question_id on every row; "
                    f"missing at aligned index {i}"
                )
        row = cache_map.get(qid) if qid else None
        vec_cached: Optional[np.ndarray] = None
        if row is not None:
            try:
                vec_cached = _validate_cache_row(
                    row,
                    question=q,
                    question_id=qid,
                    provider=embedding_provider,
                    model=embedding_model,
                )
            except ValueError:
                if cache_mode == EMBEDDING_CACHE_REQUIRED:
                    raise
                vec_cached = None
        if vec_cached is not None:
            per_row[i] = vec_cached
            meta["hits"] += 1
        else:
            miss_idx.append(i)
            miss_texts.append(q)
            miss_qids.append(qid)
            miss_questions.append(q)
            meta["misses"] += 1

    if cache_mode == EMBEDDING_CACHE_BUILD and miss_idx and not writeback:
        raise ValueError(
            "embedding_cache_mode=build requires embedding_cache_writeback=true "
            "when cache has misses to fill"
        )

    if cache_mode == EMBEDDING_CACHE_REQUIRED and miss_idx:
        raise ValueError(
            f"Required embedding cache misses for question_ids: "
            f"{miss_qids[:20]}{'...' if len(miss_qids) > 20 else ''}"
        )

    if miss_idx:
        live = embed_texts(
            miss_texts,
            provider=embedding_provider,
            model=embedding_model,
            batch_size=batch_size,
            openai_dimensions=openai_embedding_dimensions,
        )
        meta["live_computed"] = len(miss_idx)
        d_live = int(live.shape[1]) if len(live) else 0
        if dim_hint and d_live and dim_hint != d_live:
            raise ValueError(
                f"Live embedding dim {d_live} != cached dim {dim_hint} for model {embedding_model!r}"
            )
        append_rows: List[Dict[str, Any]] = []
        for j, global_i in enumerate(miss_idx):
            vec = live[j]
            per_row[global_i] = vec.astype(np.float32, copy=False)
            qid = miss_qids[j]
            qtext = miss_questions[j]
            row_out: Dict[str, Any] = {
                "question_id": qid,
                "question": qtext,
                "question_hash": canonical_question_text_hash(qtext),
                "embedding": vec.tolist(),
                "embedding_dim": int(vec.shape[0]),
                "embedding_provider": embedding_provider,
                "embedding_model": embedding_model,
            }
            if openai_embedding_dimensions is not None:
                row_out["openai_dimensions"] = int(openai_embedding_dimensions)
            append_rows.append(row_out)
        if (
            append_rows
            and writeback
            and cache_mode
            in (
                EMBEDDING_CACHE_PREFER,
                EMBEDDING_CACHE_BUILD,
            )
        ):
            paths.ensure_dirs()
            append_query_embedding_rows(paths, append_rows)
            meta["appended_rows"] = len(append_rows)
            for r in append_rows:
                cache_map[str(r["question_id"])] = r
            try:
                if paths.manifest.is_file():
                    man = read_query_embedding_cache_manifest(paths)
                else:
                    emb_block: Dict[str, Any] = {
                        "provider": embedding_provider,
                        "model": embedding_model,
                        "dim": int(live.shape[1]) if len(live) else 0,
                        "normalization": "l2_row",
                    }
                    if openai_embedding_dimensions is not None:
                        emb_block["openai_dimensions"] = int(
                            openai_embedding_dimensions
                        )
                    man = {
                        "schema_version": 1,
                        "source_benchmark": {
                            "path": str(benchmark_path.resolve()),
                            "sha256": fingerprint_benchmark_jsonl(
                                benchmark_path.resolve()
                            ),
                        },
                        "embedding": emb_block,
                        "cache_id": cache_id,
                        "counts": {},
                    }
                if openai_embedding_dimensions is not None:
                    e_sub = man.setdefault("embedding", {})
                    if isinstance(e_sub, dict):
                        e_sub["openai_dimensions"] = int(openai_embedding_dimensions)
                man.setdefault("counts", {})
                man["counts"]["cached_unique_qids"] = len(cache_map)
                man["counts"]["appended_rows"] = int(
                    man["counts"].get("appended_rows", 0)
                ) + len(append_rows)
                write_query_embedding_cache_manifest(paths, man)
            except OSError:
                pass

    meta["embedding_source"] = (
        "cache" if meta["misses"] == 0 else ("live" if meta["hits"] == 0 else "mixed")
    )
    vecs = [v for v in per_row if v is not None]
    if not vecs:
        raise RuntimeError("resolve_aligned_query_embeddings: no vectors resolved")
    dim = int(vecs[0].shape[0])
    out = np.zeros((n, dim), dtype=np.float32)
    for i, v in enumerate(per_row):
        if v is None:
            raise RuntimeError(f"Unresolved embedding at index {i}")
        if int(v.shape[0]) != dim:
            raise ValueError(
                f"Inconsistent embedding dims at row {i}: {v.shape[0]} vs {dim}"
            )
        out[i] = v
    meta["embedding_dim"] = dim
    return out, meta

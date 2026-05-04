"""Benchmark-local query embedding cache paths, manifest, and JSONL I/O."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional

from surf_rag.evaluation.answerability_layout import bundle_root_from_benchmark_jsonl
from surf_rag.evaluation.oracle_artifacts import read_jsonl, utc_now_iso
from surf_rag.router.embedding_config import (
    EMBEDDING_PROVIDER_OPENAI,
    parse_embedding_provider,
)

_CACHE_SCHEMA_VERSION = 1
_HASH_FUNCTION_ID = "ingest_normalize_lower_then_sha256_text_v1"

_SAFE_SEGMENT = re.compile(r"[^a-zA-Z0-9._-]+")


def safe_cache_path_segment(segment: str) -> str:
    """Single filesystem component under the benchmark bundle."""
    t = _SAFE_SEGMENT.sub("-", str(segment).strip()).strip("-")
    return t or "unknown"


def build_query_embedding_cache_root(
    benchmark_path: Path,
    *,
    provider: str,
    model: str,
    cache_id: str,
    openai_dimensions: int | None = None,
) -> Path:
    bundle = bundle_root_from_benchmark_jsonl(benchmark_path)
    root = (
        bundle
        / "query_embeddings"
        / safe_cache_path_segment(provider)
        / safe_cache_path_segment(model)
    )
    if (
        parse_embedding_provider(provider) == EMBEDDING_PROVIDER_OPENAI
        and openai_dimensions is not None
    ):
        root = root / safe_cache_path_segment(f"dim-{int(openai_dimensions)}")
    return root / safe_cache_path_segment(cache_id)


@dataclass(frozen=True)
class QueryEmbeddingCachePaths:
    run_root: Path

    @property
    def manifest(self) -> Path:
        return self.run_root / "manifest.json"

    @property
    def embeddings_jsonl(self) -> Path:
        return self.run_root / "embeddings.jsonl"

    def ensure_dirs(self) -> None:
        self.run_root.mkdir(parents=True, exist_ok=True)


def make_query_embedding_cache_paths(
    benchmark_path: Path,
    *,
    provider: str,
    model: str,
    cache_id: str,
    openai_dimensions: int | None = None,
) -> QueryEmbeddingCachePaths:
    root = build_query_embedding_cache_root(
        benchmark_path,
        provider=provider,
        model=model,
        cache_id=cache_id,
        openai_dimensions=openai_dimensions,
    )
    return QueryEmbeddingCachePaths(run_root=root)


def read_query_embedding_cache_manifest(
    paths: QueryEmbeddingCachePaths,
) -> Dict[str, Any]:
    if not paths.manifest.is_file():
        raise FileNotFoundError(f"Missing embedding cache manifest: {paths.manifest}")
    return json.loads(paths.manifest.read_text(encoding="utf-8"))


def write_query_embedding_cache_manifest(
    paths: QueryEmbeddingCachePaths,
    payload: Dict[str, Any],
) -> None:
    paths.ensure_dirs()
    data = dict(payload)
    data.setdefault("schema_version", _CACHE_SCHEMA_VERSION)
    data.setdefault("created_at", utc_now_iso())
    data["updated_at"] = utc_now_iso()
    data.setdefault("hash_function", _HASH_FUNCTION_ID)
    paths.manifest.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def iter_query_embedding_rows(
    paths: QueryEmbeddingCachePaths,
) -> Iterator[Dict[str, Any]]:
    if not paths.embeddings_jsonl.is_file():
        return iter(())
    return iter(read_jsonl(paths.embeddings_jsonl))


def load_query_embedding_map_by_qid(
    paths: QueryEmbeddingCachePaths,
) -> Dict[str, Dict[str, Any]]:
    """Last row wins for duplicate question_id."""
    out: Dict[str, Dict[str, Any]] = {}
    if not paths.embeddings_jsonl.is_file():
        return out
    for row in read_jsonl(paths.embeddings_jsonl):
        qid = str(row.get("question_id", "") or "").strip()
        if qid:
            out[qid] = dict(row)
    return out


def append_query_embedding_rows(
    paths: QueryEmbeddingCachePaths,
    rows: List[Mapping[str, Any]],
) -> None:
    paths.ensure_dirs()
    with paths.embeddings_jsonl.open("a", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def fingerprint_benchmark_jsonl(benchmark_path: Path) -> str:
    """Stable fingerprint of benchmark file bytes (for manifest provenance)."""
    import hashlib

    data = benchmark_path.read_bytes()
    return hashlib.sha256(data).hexdigest()

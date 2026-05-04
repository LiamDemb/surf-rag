from __future__ import annotations

import json
from pathlib import Path

from surf_rag.evaluation.query_embedding_cache_artifacts import (
    append_query_embedding_rows,
    build_query_embedding_cache_root,
    load_query_embedding_map_by_qid,
    make_query_embedding_cache_paths,
    write_query_embedding_cache_manifest,
)


def _bench_path(tmp: Path) -> Path:
    bench = tmp / "bn" / "bid" / "benchmark" / "benchmark.jsonl"
    bench.parent.mkdir(parents=True, exist_ok=True)
    bench.write_text('{"question_id":"q1","question":"hello"}\n', encoding="utf-8")
    return bench


def test_build_root_layout(tmp_path: Path) -> None:
    bench = _bench_path(tmp_path)
    root = build_query_embedding_cache_root(
        bench, provider="openai", model="text-embedding-3-large", cache_id="v1"
    )
    assert root.name == "v1"
    assert root.parent.name == "text-embedding-3-large"
    assert root.parent.parent.name == "openai"
    assert root.parent.parent.parent.name == "query_embeddings"


def test_manifest_roundtrip_and_map(tmp_path: Path) -> None:
    bench = _bench_path(tmp_path)
    paths = make_query_embedding_cache_paths(
        bench, provider="openai", model="m", cache_id="c1"
    )
    paths.ensure_dirs()
    write_query_embedding_cache_manifest(
        paths,
        {
            "embedding": {"provider": "openai", "model": "m", "dim": 3},
            "source_benchmark": {"path": str(bench), "sha256": "abc"},
        },
    )
    man = json.loads(paths.manifest.read_text(encoding="utf-8"))
    assert man["schema_version"] == 1
    assert man["hash_function"]

    rows = [
        {
            "question_id": "q1",
            "question": "hello",
            "question_hash": "x",
            "embedding": [1.0, 0.0, 0.0],
            "embedding_dim": 3,
            "embedding_provider": "openai",
            "embedding_model": "m",
        },
        {
            "question_id": "q1",
            "question": "hello",
            "question_hash": "y",
            "embedding": [0.0, 1.0, 0.0],
            "embedding_dim": 3,
            "embedding_provider": "openai",
            "embedding_model": "m",
        },
    ]
    append_query_embedding_rows(paths, rows)
    m = load_query_embedding_map_by_qid(paths)
    assert len(m) == 1
    assert m["q1"]["question_hash"] == "y"


def test_safe_cache_path_segment() -> None:
    from surf_rag.evaluation.query_embedding_cache_artifacts import (
        safe_cache_path_segment,
    )

    assert "/" not in safe_cache_path_segment("a/b")
    assert safe_cache_path_segment("  ") == "unknown"


def test_build_root_layout_openai_dimensions_segment(tmp_path: Path) -> None:
    bench = _bench_path(tmp_path)
    root = build_query_embedding_cache_root(
        bench,
        provider="openai",
        model="text-embedding-3-large",
        cache_id="v1",
        openai_dimensions=256,
    )
    assert root.name == "v1"
    assert root.parent.name == "dim-256"
    assert root.parent.parent.name == "text-embedding-3-large"

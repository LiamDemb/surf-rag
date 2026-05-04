from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from surf_rag.evaluation.query_embedding_cache_artifacts import (
    append_query_embedding_rows,
    make_query_embedding_cache_paths,
)
from surf_rag.router.embedding_config import (
    EMBEDDING_CACHE_OFF,
    EMBEDDING_CACHE_REQUIRED,
)
from surf_rag.router.question_text_hash import canonical_question_text_hash
from surf_rag.router.dataset_embedding_resolve import resolve_aligned_query_embeddings


def _bench(tmp: Path) -> Path:
    bench = tmp / "n" / "id" / "benchmark" / "benchmark.jsonl"
    bench.parent.mkdir(parents=True, exist_ok=True)
    bench.write_text(
        '{"question_id":"a","question":"foo"}\n{"question_id":"b","question":"bar"}\n',
        encoding="utf-8",
    )
    return bench


def test_required_miss_raises(tmp_path: Path) -> None:
    bench = _bench(tmp_path)
    aligned = [{"question_id": "a", "question": "foo"}]
    with pytest.raises(FileNotFoundError):
        resolve_aligned_query_embeddings(
            aligned,
            embedding_provider="sentence-transformers",
            embedding_model="all-MiniLM-L6-v2",
            cache_mode=EMBEDDING_CACHE_REQUIRED,
            benchmark_path=bench,
            cache_id="missing",
            cache_path_override=None,
            writeback=False,
        )


def test_required_hit_roundtrip(tmp_path: Path) -> None:
    bench = _bench(tmp_path)
    paths = make_query_embedding_cache_paths(
        bench,
        provider="sentence-transformers",
        model="all-MiniLM-L6-v2",
        cache_id="t1",
    )
    paths.ensure_dirs()
    vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    append_query_embedding_rows(
        paths,
        [
            {
                "question_id": "a",
                "question": "foo",
                "question_hash": canonical_question_text_hash("foo"),
                "embedding": vec.tolist(),
                "embedding_dim": 3,
                "embedding_provider": "sentence-transformers",
                "embedding_model": "all-MiniLM-L6-v2",
            }
        ],
    )
    emb, meta = resolve_aligned_query_embeddings(
        [{"question_id": "a", "question": "foo"}],
        embedding_provider="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        cache_mode=EMBEDDING_CACHE_REQUIRED,
        benchmark_path=bench,
        cache_id="t1",
        cache_path_override=None,
        writeback=False,
    )
    assert emb.shape == (1, 3)
    assert meta["hits"] == 1
    assert meta["misses"] == 0


def test_off_live(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bench = _bench(tmp_path)
    calls: list[int] = []

    def fake_embed_texts(texts, **kwargs):
        calls.append(len(texts))
        return np.ones((len(texts), 2), dtype=np.float32)

    monkeypatch.setattr(
        "surf_rag.router.dataset_embedding_resolve.embed_texts", fake_embed_texts
    )
    emb, meta = resolve_aligned_query_embeddings(
        [{"question_id": "a", "question": "foo"}],
        embedding_provider="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        cache_mode=EMBEDDING_CACHE_OFF,
        benchmark_path=bench,
        cache_id="x",
        cache_path_override=None,
        writeback=False,
    )
    assert emb.shape == (1, 2)
    assert meta["live_computed"] == 1
    assert calls == [1]

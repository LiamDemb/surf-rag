from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import yaml

from surf_rag.core.corpus_finalize import (
    finalize_corpus_artifacts,
    load_corpus_chunks,
    write_corpus_finalize_manifest,
)


def _tiny_chunks() -> list[dict]:
    return [
        {
            "chunk_id": "c1",
            "source": "wiki",
            "text": "Einstein was a physicist.",
            "metadata": {
                "entities": [{"norm": "einstein", "surface": "Einstein", "qid": None}],
                "relations": [],
            },
        },
        {
            "chunk_id": "c2",
            "source": "wiki",
            "text": "Paris is in France.",
            "metadata": {
                "entities": [{"norm": "paris", "surface": "Paris", "qid": None}],
                "relations": [],
            },
        },
    ]


def _load_script_module(module_name: str, rel_path: str):
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / rel_path
    spec = importlib.util.spec_from_file_location(module_name, target)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_corpus_chunks_missing_or_empty(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"
    with pytest.raises(FileNotFoundError):
        load_corpus_chunks(missing)
    empty = tmp_path / "corpus.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        load_corpus_chunks(empty)


def test_finalize_corpus_artifacts_writes_expected_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _stub_faiss(chunks, output_index_path, output_meta_path, model_name):
        Path(output_index_path).write_bytes(b"faiss")
        rows = [
            {"row_id": i, "chunk_id": c.get("chunk_id")} for i, c in enumerate(chunks)
        ]
        pd.DataFrame(rows).to_parquet(output_meta_path, index=False)

    def _stub_graph(chunks):
        return {"nodes": len(chunks)}

    def _stub_entity_index(
        lexicon_path, output_index_path, output_meta_path, model_name
    ):
        Path(output_index_path).write_bytes(b"entity")
        pd.DataFrame([{"row_id": 0, "norm": "x"}]).to_parquet(
            output_meta_path, index=False
        )

    monkeypatch.setattr("surf_rag.core.corpus_finalize.build_faiss_index", _stub_faiss)
    monkeypatch.setattr("surf_rag.core.corpus_finalize.build_graph", _stub_graph)
    monkeypatch.setattr(
        "surf_rag.core.corpus_finalize.build_entity_index", _stub_entity_index
    )

    chunks = _tiny_chunks()
    artifacts = finalize_corpus_artifacts(
        chunks=chunks,
        output_dir=tmp_path,
        model_name="all-MiniLM-L6-v2",
        samples=[],
        quality_report=False,
    )
    expected = {
        "vector_index",
        "vector_meta",
        "graph",
        "entity_lexicon",
        "entity_index",
        "entity_index_meta",
    }
    assert expected.issubset(set(artifacts))
    for k in expected:
        assert artifacts[k].is_file()
    vm = pd.read_parquet(artifacts["vector_meta"])
    assert len(vm) == len(chunks)
    lex = pd.read_parquet(artifacts["entity_lexicon"])
    assert len(lex) > 0


def test_finalize_manifest_contains_counts(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "chunk_id": "c1",
                "text": "x",
                "metadata": {"entities": [{"norm": "x", "surface": "X", "qid": None}]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    vm = tmp_path / "vector_meta.parquet"
    lex = tmp_path / "entity_lexicon.parquet"
    pd.DataFrame([{"row_id": 0, "chunk_id": "c1"}]).to_parquet(vm, index=False)
    pd.DataFrame(
        [{"norm": "x", "surface_forms": ["x"], "qid_candidates": [], "df": 1}]
    ).to_parquet(lex, index=False)
    artifacts = {"vector_meta": vm, "entity_lexicon": lex}
    manifest = write_corpus_finalize_manifest(
        output_dir=tmp_path,
        corpus_path=corpus,
        chunks_count=1,
        produced_artifacts=artifacts,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["chunks_count"] == 1
    assert payload["vector_meta_rows"] == 1
    assert payload["entity_lexicon_rows"] == 1


def test_finalize_script_fails_on_missing_or_empty_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script_mod = _load_script_module(
        "finalize_corpus_artifacts_script",
        "scripts/corpus/finalize_corpus_artifacts.py",
    )

    monkeypatch.setattr(
        "sys.argv",
        ["finalize_corpus_artifacts.py", "--corpus-dir", str(tmp_path)],
    )
    assert script_mod.main() == 1

    (tmp_path / "corpus.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        ["finalize_corpus_artifacts.py", "--corpus-dir", str(tmp_path)],
    )
    assert script_mod.main() == 1


def test_finalize_script_writes_resolved_config_when_config_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script_mod = _load_script_module(
        "finalize_corpus_artifacts_script_config",
        "scripts/corpus/finalize_corpus_artifacts.py",
    )
    data_base = tmp_path / "data"
    corpus_dir = data_base / "benchmarks" / "b" / "dev" / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "corpus.jsonl").write_text(
        '{"chunk_id":"c1","text":"x","metadata":{"ie_status":"success"}}\n',
        encoding="utf-8",
    )
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "surf-rag/pipeline/v1",
                "paths": {
                    "data_base": str(data_base),
                    "benchmark_name": "b",
                    "benchmark_id": "dev",
                    "router_id": "r1",
                },
            }
        ),
        encoding="utf-8",
    )

    def _fake_finalize(**kwargs):
        out = kwargs["output_dir"]
        files = {
            "vector_index": out / "vector_index.faiss",
            "vector_meta": out / "vector_meta.parquet",
            "graph": out / "graph.pkl",
            "entity_lexicon": out / "entity_lexicon.parquet",
            "entity_index": out / "entity_index.faiss",
            "entity_index_meta": out / "entity_index_meta.parquet",
        }
        for p in files.values():
            if p.suffix == ".parquet":
                pd.DataFrame([{"x": 1}]).to_parquet(p, index=False)
            else:
                p.write_bytes(b"x")
        return files

    def _fake_manifest(**kwargs):
        p = kwargs["output_dir"] / "corpus_finalize_manifest.json"
        p.write_text("{}", encoding="utf-8")
        return p

    monkeypatch.setattr(script_mod, "finalize_corpus_artifacts", _fake_finalize)
    monkeypatch.setattr(script_mod, "write_corpus_finalize_manifest", _fake_manifest)
    monkeypatch.setattr(
        "sys.argv",
        [
            "finalize_corpus_artifacts.py",
            "--config",
            str(cfg_path),
            "--skip-quality-report",
        ],
    )
    assert script_mod.main() == 0
    assert (corpus_dir / "resolved_config.yaml").is_file()


def test_reconcile_ie_failure_artifacts_updates_stale_files(tmp_path: Path) -> None:
    script_mod = _load_script_module(
        "finalize_corpus_artifacts_script_reconcile",
        "scripts/corpus/finalize_corpus_artifacts.py",
    )
    fail_path = tmp_path / "ie_failures_final.json"
    pending_path = tmp_path / "ie_pending_ids.txt"
    fail_path.write_text(
        json.dumps({"remaining_unresolved_ids": ["old1", "old2"]}),
        encoding="utf-8",
    )
    pending_path.write_text("old1\nold2\n", encoding="utf-8")

    script_mod._reconcile_ie_failure_artifacts(tmp_path, ["new1"])
    payload = json.loads(fail_path.read_text(encoding="utf-8"))
    assert payload["remaining_unresolved_ids"] == ["new1"]
    assert pending_path.read_text(encoding="utf-8").strip() == "new1"

    script_mod._reconcile_ie_failure_artifacts(tmp_path, [])
    assert not fail_path.exists()
    assert not pending_path.exists()

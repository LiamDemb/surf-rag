from __future__ import annotations

import importlib.util
from pathlib import Path

from surf_rag.core.corpus_finalize import write_corpus_finalize_manifest


def _load_script_module(module_name: str, rel_path: str):
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / rel_path
    spec = importlib.util.spec_from_file_location(module_name, target)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_submit_normalize_metadata_legacy_success():
    submit_mod = _load_script_module(
        "submit_llm_ie_batch_test",
        "scripts/corpus/submit_llm_ie_batch.py",
    )
    meta = {"ie_extracted": True, "entities": [{"name": "x"}]}
    out = submit_mod._normalize_ie_metadata(meta)
    assert out["ie_status"] == "success"
    assert out["ie_extracted"] is True
    assert out["ie_attempts"] >= 0
    assert out["ie_last_error"] is None


def test_submit_pending_filter_honors_only_chunk_ids():
    submit_mod = _load_script_module(
        "submit_llm_ie_batch_test_filter",
        "scripts/corpus/submit_llm_ie_batch.py",
    )
    chunks = [
        {"chunk_id": "c1", "metadata": {"ie_status": "success"}},
        {"chunk_id": "c2", "metadata": {"ie_status": "failed"}},
        {"chunk_id": "c3", "metadata": {"ie_extracted": False}},
    ]
    pending = submit_mod._iter_pending_chunks(chunks, only_chunk_ids={"c2", "c3"})
    assert [c["chunk_id"] for c in pending] == ["c2", "c3"]


def test_collect_normalize_metadata_pending_when_empty():
    collect_mod = _load_script_module(
        "collect_llm_ie_batch_test",
        "scripts/corpus/collect_llm_ie_batch.py",
    )
    meta = {"ie_extracted": False, "entities": [], "relations": []}
    out = collect_mod._normalize_ie_metadata(meta)
    assert out["ie_status"] == "pending"
    assert out["ie_extracted"] is False
    assert out["ie_attempts"] == 0


def test_retry_report_with_unresolved_can_be_finalized_separately(tmp_path: Path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"chunk_id":"c1","text":"x","metadata":{"ie_status":"failed","entities":[],"relations":[]}}\n',
        encoding="utf-8",
    )
    vm = tmp_path / "vector_meta.parquet"
    lex = tmp_path / "entity_lexicon.parquet"
    import pandas as pd

    pd.DataFrame([{"row_id": 0, "chunk_id": "c1"}]).to_parquet(vm, index=False)
    pd.DataFrame(
        [{"norm": "x", "surface_forms": ["x"], "qid_candidates": [], "df": 1}]
    ).to_parquet(lex, index=False)
    manifest = write_corpus_finalize_manifest(
        output_dir=tmp_path,
        corpus_path=corpus,
        chunks_count=1,
        produced_artifacts={"vector_meta": vm, "entity_lexicon": lex},
    )
    assert manifest.is_file()


def test_parse_failure_rows_remain_failed_but_do_not_block_finalize_script():
    collect_mod = _load_script_module(
        "collect_llm_ie_batch_failure_test",
        "scripts/corpus/collect_llm_ie_batch.py",
    )
    out = collect_mod._normalize_ie_metadata(
        {
            "ie_status": "failed",
            "ie_last_error": "parse_error:json_decode_error",
            "entities": [],
            "relations": [],
            "ie_extracted": False,
        }
    )
    assert out["ie_status"] == "failed"
    assert out["ie_extracted"] is False
    assert "parse_error" in str(out["ie_last_error"])

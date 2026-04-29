from __future__ import annotations

import importlib.util
from pathlib import Path


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

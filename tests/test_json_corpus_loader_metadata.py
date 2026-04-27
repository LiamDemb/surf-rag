"""JsonCorpusLoader stores full rows for metadata."""

from __future__ import annotations

from pathlib import Path

from surf_rag.core.mapping import JsonCorpusLoader, metadata_from_corpus_record


def test_get_record_and_metadata_from_corpus_row(tmp_path: Path) -> None:
    p = tmp_path / "c.jsonl"
    p.write_text(
        '{"chunk_id":"c1","text":"Hello.","title":"T","doc_id":"d1","url":"http://x","source":"s","section_path":[]}\n',
        encoding="utf-8",
    )
    loader = JsonCorpusLoader(jsonl_path=str(p))
    assert loader.get_text("c1") == "Hello."
    rec = loader.get_record("c1")
    assert rec is not None
    assert rec.get("title") == "T"
    meta = metadata_from_corpus_record(rec)
    assert meta["title"] == "T"
    assert meta["doc_id"] == "d1"

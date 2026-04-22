"""Tests for split pipeline: benchmark samples, DocStore 2Wiki load, alignment drops."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from surf_rag.benchmark.align_2wiki import run_2wiki_support_alignment
from surf_rag.core.benchmark_samples import (
    build_samples,
    load_benchmark_by_source,
    resolve_raw_paths_for_benchmark_sources,
)
from surf_rag.core.corpus_acquisition import load_2wiki_docs_from_docstore
from surf_rag.core.docstore import DocRecord, DocStore
from surf_rag.core.docstore_sentence_index import (
    build_title_to_candidate_sentences_from_docstore,
)


def test_resolve_raw_paths_for_benchmark_sources() -> None:
    paths, missing = resolve_raw_paths_for_benchmark_sources(
        {"nq", "2wiki"},
        nq_path="a.jsonl",
        wiki2_path="b.jsonl",
    )
    assert paths == {"nq": "a.jsonl", "2wiki": "b.jsonl"}
    assert missing == []


def test_build_samples_joins_benchmark_to_raw(tmp_path: Path) -> None:
    raw = tmp_path / "2wiki.jsonl"
    from surf_rag.core.schemas import sha256_text

    qid = sha256_text("What is X?")
    bench2 = tmp_path / "bench2.jsonl"
    bench2.write_text(
        json.dumps(
            {
                "question_id": qid,
                "question": "What is X?",
                "gold_answers": ["Y"],
                "dataset_source": "2wiki",
                "gold_support_sentences": ["s1"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    raw.write_text(
        json.dumps(
            {
                "question": "What is X?",
                "supporting_facts": {"title": ["T"], "sent_id": [0]},
                "answer": ["Y"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    by_src = load_benchmark_by_source(bench2)
    samples = build_samples(by_src, {"2wiki": str(raw)})
    assert len(samples) == 1
    assert samples[0]["question_id"] == qid
    assert samples[0]["supporting_facts"]["title"] == ["T"]


def test_load_2wiki_docs_from_docstore_missing_without_fetch(tmp_path: Path) -> None:
    ds = DocStore(str(tmp_path / "d.sqlite"))
    sample = {
        "supporting_facts": {"title": ["MissingTitle"], "sent_id": [0]},
    }
    docs, missing = load_2wiki_docs_from_docstore(
        sample, ds, wiki=None, fetch_missing=False
    )
    ds.close()
    assert docs == []
    assert missing == ["MissingTitle"]


def test_build_title_index_from_docstore_matches_chunk_path(tmp_path: Path) -> None:
    ds_path = tmp_path / "d.sqlite"
    html = (
        '<div class="mw-parser-output">'
        "<p>First sentence. Second sentence.</p>"
        "</div>"
    )
    ds = DocStore(str(ds_path))
    ds.put(
        "title:Foo",
        DocRecord(
            title="Foo",
            page_id="1",
            revision_id="1",
            url="u",
            html=html,
            cleaned_text=None,
            anchors={"outgoing_titles": [], "incoming_stub": []},
            source="2wiki",
            dataset_origin="2wiki",
        ),
    )
    ds.close()
    ds2 = DocStore(str(ds_path))
    m = build_title_to_candidate_sentences_from_docstore(ds2, ["Foo"])
    ds2.close()
    assert "Foo" in m
    joined = " ".join(m["Foo"])
    assert "First sentence." in joined
    assert "Second sentence." in joined


def test_align_drops_unresolved_2wiki_row(tmp_path: Path) -> None:
    """No DocStore HTML for title -> row removed when drop_unresolved (default)."""
    bench = tmp_path / "benchmark.jsonl"
    ds_path = tmp_path / "docstore.sqlite"
    bench.write_text(
        json.dumps(
            {
                "question_id": "w1",
                "question": "Q?",
                "gold_answers": ["A"],
                "dataset_source": "2wiki",
                "gold_support_sentences": ["No match in corpus."],
                "gold_support_titles": ["T"],
                "gold_support_sent_ids": [0],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    DocStore(str(ds_path)).close()  # empty db
    emb = MagicMock()
    run_2wiki_support_alignment(
        bench,
        backup_path=tmp_path / "b.bak",
        report_path=tmp_path / "r.md",
        docstore_path=ds_path,
        corpus_path=None,
        embedder=emb,
        tau_sem=0.99,
        tau_lex=0.99,
        drop_unresolved=True,
    )
    out = [json.loads(l) for l in bench.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert out == []

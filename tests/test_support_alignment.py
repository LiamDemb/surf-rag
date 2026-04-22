"""Tests for 2Wiki title-localized support alignment."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from surf_rag.benchmark.support_alignment import (
    GoldSupportAnchor,
    align_one_support_fact,
    build_title_to_candidate_sentences,
    rouge_l_f1,
)
from surf_rag.core.schemas import BenchmarkItem, parse_benchmark_support_fields


def test_rouge_l_f1_identical() -> None:
    assert rouge_l_f1("The cat sat.", "The cat sat.") == pytest.approx(1.0)


def test_build_title_to_candidate_sentences_dedupes_by_normalized_sentence() -> None:
    rows = [
        {"title": "Foo", "text": "Hello world. Hello world."},
        {"title": "Foo", "text": "Hello world. Second sentence."},
    ]
    m = build_title_to_candidate_sentences(rows)
    assert "Foo" in m
    # "Hello world." appears twice in first chunk; should appear once in index
    joined = " ".join(m["Foo"])
    assert joined.count("Hello world.") == 1


def test_align_keeps_original_when_already_present() -> None:
    anchor = GoldSupportAnchor(
        title="T",
        sent_id=0,
        sentence="Exact sentence here.",
    )
    title_map = {"T": ["Exact sentence here.", "Other noise."]}
    emb = MagicMock()
    emb.model.encode = MagicMock(
        side_effect=AssertionError("encode should not run when already present")
    )
    from surf_rag.core.embedder import SentenceTransformersEmbedder

    stub = SentenceTransformersEmbedder.__new__(SentenceTransformersEmbedder)
    stub.model = emb
    out = align_one_support_fact(anchor, title_map, stub, tau_sem=0.9, tau_lex=0.9)
    assert out.replacement is None
    assert out.reason == "already_present"


def test_align_replaces_when_both_thresholds_met() -> None:
    """Controlled embeddings + shared wording so ROUGE-L and cosine pass."""
    anchor = GoldSupportAnchor(
        title="Article",
        sent_id=0,
        sentence="The director was born in Taiyuan.",
    )
    candidate = "The director was born in Taiyuan, Shanxi, and later moved to Beijing."
    title_map = {"Article": [candidate]}

    def encode_stub(texts, **kwargs):
        vecs = []
        for _ in texts:
            v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            vecs.append(v)
        return np.stack(vecs, axis=0)

    emb = MagicMock()
    emb.encode = encode_stub
    from surf_rag.core.embedder import SentenceTransformersEmbedder

    stub = SentenceTransformersEmbedder.__new__(SentenceTransformersEmbedder)
    stub.model = emb

    rl = rouge_l_f1(anchor.sentence, candidate)
    out = align_one_support_fact(
        anchor,
        title_map,
        stub,
        tau_sem=0.99,
        tau_lex=max(0.0, rl - 1e-6),
    )
    assert out.replaced
    assert out.replacement == candidate
    assert out.reason == "aligned"


def test_align_no_candidates_for_title() -> None:
    anchor = GoldSupportAnchor(title="Missing", sent_id=0, sentence="x")
    emb = MagicMock()
    emb.encode = MagicMock(side_effect=AssertionError)
    from surf_rag.core.embedder import SentenceTransformersEmbedder

    stub = SentenceTransformersEmbedder.__new__(SentenceTransformersEmbedder)
    stub.model = emb
    out = align_one_support_fact(anchor, {}, stub, tau_sem=0.0, tau_lex=0.0)
    assert out.reason == "no_corpus_chunks_for_title"
    assert out.nearest_candidate is None


def test_align_below_threshold_includes_nearest_candidate() -> None:
    """Same embedding for all texts → cosine 1.0; ROUGE-L fails high threshold → nearest still filled."""
    anchor = GoldSupportAnchor(
        title="Article",
        sent_id=0,
        sentence="The director was born in Taiyuan.",
    )
    title_map = {
        "Article": [
            "Unrelated sentence one about cats.",
            "The director was born in Taiyuan, Shanxi province.",
        ]
    }

    def encode_stub(texts, **kwargs):
        return np.ones((len(texts), 4), dtype=np.float32)

    emb = MagicMock()
    emb.encode = encode_stub
    from surf_rag.core.embedder import SentenceTransformersEmbedder

    stub = SentenceTransformersEmbedder.__new__(SentenceTransformersEmbedder)
    stub.model = emb
    out = align_one_support_fact(
        anchor,
        title_map,
        stub,
        tau_sem=0.99,
        tau_lex=0.95,
    )
    assert out.reason == "below_thresholds"
    assert out.replacement is None
    assert out.nearest_candidate is not None
    assert out.nearest_semantic_cosine is not None
    assert out.nearest_rouge_l is not None
    # Second candidate has higher ROUGE-L vs gold when cosine is tied.
    assert "Shanxi" in (out.nearest_candidate or "")


def test_benchmark_item_roundtrip_gold_support_metadata() -> None:
    item = BenchmarkItem(
        question_id="q",
        question="Q?",
        gold_answers=["a"],
        dataset_source="2wiki",
        gold_support_sentences=["hello"],
        gold_support_titles=["A"],
        gold_support_sent_ids=[1],
    )
    raw = item.to_json()
    assert raw["gold_support_titles"][0] == "A"
    assert raw["gold_support_sent_ids"][0] == 1
    sentences, titles, sids = parse_benchmark_support_fields(raw)
    assert sentences == ["hello"] and titles == ["A"] and sids == [1]


def test_parse_benchmark_support_fields_migrates_legacy_two_wiki_support_facts() -> (
    None
):
    row = {
        "gold_support_sentences": ["hello"],
        "two_wiki_support_facts": [
            {"title": "Legacy", "sent_id": 3, "sentence": "hello"}
        ],
    }
    sentences, titles, sids = parse_benchmark_support_fields(row)
    assert sentences == ["hello"]
    assert titles == ["Legacy"]
    assert sids == [3]


def test_run_alignment_backup_and_report_nq_only(tmp_path: Path) -> None:
    """NQ-only benchmark: alignment is a no-op; backup + report must still be written."""
    from surf_rag.benchmark.align_2wiki import run_2wiki_support_alignment

    bench = tmp_path / "benchmark.jsonl"
    corpus = tmp_path / "corpus.jsonl"
    backup = tmp_path / "bench.backup.jsonl"
    report = tmp_path / "rep.md"
    bench.write_text(
        json.dumps(
            {
                "question_id": "nq1",
                "question": "What?",
                "gold_answers": ["A"],
                "dataset_source": "nq",
                "gold_support_sentences": ["Evidence."],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    corpus.write_text(
        json.dumps({"title": "T", "text": "Evidence."}) + "\n",
        encoding="utf-8",
    )
    fake = MagicMock()
    run_2wiki_support_alignment(
        bench,
        backup_path=backup,
        report_path=report,
        corpus_path=corpus,
        docstore_path=None,
        embedder=fake,
    )

    assert backup.is_file()
    body = report.read_text(encoding="utf-8")
    assert "2Wiki gold support alignment report" in body
    assert "No sentences were replaced" in body


def test_run_alignment_full_report_has_per_line_sections(tmp_path: Path) -> None:
    from surf_rag.benchmark.align_2wiki import run_2wiki_support_alignment
    from surf_rag.core.docstore import DocRecord, DocStore

    bench = tmp_path / "benchmark.jsonl"
    ds_path = tmp_path / "docstore.sqlite"
    backup = tmp_path / "bench.backup.jsonl"
    report = tmp_path / "rep_full.md"
    bench.write_text(
        json.dumps(
            {
                "question_id": "w1",
                "question": "Q?",
                "gold_answers": ["A"],
                "dataset_source": "2wiki",
                "gold_support_sentences": ["Exact match sentence."],
                "gold_support_titles": ["T"],
                "gold_support_sent_ids": [0],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    html = (
        '<html><body><div class="mw-parser-output">'
        "<p>Exact match sentence. Other.</p>"
        "</div></body></html>"
    )
    ds = DocStore(str(ds_path))
    ds.put(
        "title:T",
        DocRecord(
            title="T",
            page_id="99",
            revision_id="1",
            url="http://example.com",
            html=html,
            cleaned_text=None,
            anchors={"outgoing_titles": [], "incoming_stub": []},
            source="2wiki",
            dataset_origin="2wiki",
        ),
    )
    ds.close()
    fake = MagicMock()
    run_2wiki_support_alignment(
        bench,
        backup_path=backup,
        report_path=report,
        docstore_path=ds_path,
        corpus_path=None,
        embedder=fake,
        full_report=True,
        tau_sem=0.99,
        tau_lex=0.99,
    )
    body = report.read_text(encoding="utf-8")
    assert "alignment report (full)" in body
    assert "Support line 1" in body
    assert "`w1`" in body

"""Unit tests for LLM one-pass extraction post-processing (no API calls)."""

from __future__ import annotations

import pytest

from surf_rag.core.llm_ie import (
    _find_evidence_span,
    _post_process_ie,
)


def test_find_evidence_span_exact_match() -> None:
    text = "Eva Busch was a German cabaret artist."
    evidence = "cabaret artist"
    start, end = _find_evidence_span(evidence, text)
    assert start == 23
    assert end == 23 + len(evidence)


def test_find_evidence_span_not_found() -> None:
    text = "Eva Busch was a singer."
    evidence = "cabaret artist"
    start, end = _find_evidence_span(evidence, text)
    assert start == -1
    assert end == -1


def test_find_evidence_span_normalized_whitespace() -> None:
    text = "Eva  Busch   was   a   cabaret   artist."
    evidence = "Eva Busch was a cabaret artist"
    start, end = _find_evidence_span(evidence, text)
    assert start >= 0
    assert end > start


def test_find_evidence_span_empty() -> None:
    assert _find_evidence_span("", "hello") == (-1, -1)
    assert _find_evidence_span("x", "") == (-1, -1)


def test_post_process_dedupes_entities_by_norm() -> None:
    raw_entities = [
        {"surface": "Eva Busch", "type": "PERSON"},
        {"surface": "EVA BUSCH", "type": "PERSON"},
    ]
    raw_triples = []
    entities, _ = _post_process_ie(raw_entities, raw_triples, "text", None)
    assert len(entities) == 1
    assert entities[0]["norm"] == "eva busch"


def test_post_process_filters_banned_predicates() -> None:
    raw_entities = [{"surface": "Eva Busch", "type": "PERSON"}]
    raw_triples = [
        {"subj_surface": "Eva Busch", "pred": "is", "obj_surface": "singer", "evidence": ""},
        {"subj_surface": "Eva Busch", "pred": "occupation", "obj_surface": "singer", "evidence": ""},
    ]
    _, relations = _post_process_ie(raw_entities, raw_triples, "Eva Busch was a singer.", None)
    assert len(relations) == 1
    assert relations[0]["pred"] == "occupation"


def test_post_process_injects_endpoint_entities() -> None:
    raw_entities = [{"surface": "Eva Busch", "type": "PERSON"}]
    raw_triples = [
        {"subj_surface": "Eva Busch", "pred": "occupation", "obj_surface": "singer", "evidence": ""},
    ]
    entities, relations = _post_process_ie(raw_entities, raw_triples, "text", None)
    norms = {e["norm"] for e in entities}
    assert "eva busch" in norms
    assert "singer" in norms


def test_post_process_shape() -> None:
    raw_entities = [{"surface": "Ra", "type": "PERSON"}, {"surface": "sun", "type": "NOUN_CHUNK"}]
    raw_triples = [
        {"subj_surface": "Ra", "pred": "deity_of", "obj_surface": "sun", "evidence": "Ra, the sun god"},
    ]
    entities, relations = _post_process_ie(
        raw_entities, raw_triples, "In Egyptian mythology, Ra, the sun god was central.", "chunk-1"
    )
    assert len(relations) == 1
    rec = relations[0]
    assert "subj_surface" in rec
    assert "obj_surface" in rec
    assert "subj_norm" in rec
    assert "pred" in rec
    assert "obj_norm" in rec
    assert rec["rule_id"] == "LLM_IE_V1"
    assert "confidence" in rec
    assert "match_text" in rec
    assert "start_char" in rec
    assert "end_char" in rec
    assert rec.get("chunk_id") == "chunk-1"
    assert rec["source"] == "llm"


def test_post_process_skips_incomplete_triple() -> None:
    raw_triples = [
        {"subj_surface": "Eva", "pred": "", "obj_surface": "singer"},
        {"subj_surface": "", "pred": "x", "obj_surface": "y"},
    ]
    _, relations = _post_process_ie([], raw_triples, "text", None)
    assert len(relations) == 0


def test_post_process_dedupes_triples_by_norm_key() -> None:
    raw_entities = [{"surface": "Eva Busch", "type": "PERSON"}]
    raw_triples = [
        {"subj_surface": "Eva Busch", "pred": "occupation", "obj_surface": "singer", "evidence": ""},
        {"subj_surface": "Eva Busch", "pred": "occupation", "obj_surface": "Singer", "evidence": ""},
    ]
    _, relations = _post_process_ie(raw_entities, raw_triples, "text", None)
    assert len(relations) == 1

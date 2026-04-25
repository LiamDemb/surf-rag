from __future__ import annotations

from dataclasses import dataclass

from surf_rag.core.enrich_entities import (
    extract_entities_capitalization,
    extract_entities_spacy,
)


@dataclass
class _Span:
    text: str
    label_: str = ""


@dataclass
class _Doc:
    ents: list[_Span]


class _NlpStub:
    def __init__(self, doc: _Doc):
        self._doc = doc

    def __call__(self, text: str):
        return self._doc


def test_extract_entities_spacy_no_entities_returns_empty():
    doc = _Doc(ents=[])
    assert extract_entities_spacy(text="dummy", nlp=_NlpStub(doc), alias_map={}) == []


def test_extract_entities_spacy_respects_allowed_types():
    doc = _Doc(ents=[_Span(text="Acme", label_="MISC")])
    assert extract_entities_spacy(
        text="dummy",
        nlp=_NlpStub(doc),
        alias_map={},
        allowed_types={"ORG"},
    ) == []


def test_extract_entities_spacy_dedupes_by_norm_and_sorts():
    doc = _Doc(
        ents=[
            _Span(text="Zeta Corp", label_="ORG"),
            _Span(text="Alpha Org", label_="ORG"),
        ]
    )
    ents = extract_entities_spacy(text="dummy", nlp=_NlpStub(doc), alias_map={})
    assert [e["norm"] for e in ents] == ["alpha org", "zeta corp"]


def test_extract_entities_capitalization_united_states():
    result = extract_entities_capitalization(
        "What happened in the United States?",
        alias_map={},
    )
    assert "united states" in result


def test_extract_entities_capitalization_multiple_entities():
    result = extract_entities_capitalization(
        "Which film is more recent? Tab Hunter Confidential or Louisiana Story?",
        alias_map={},
    )
    assert "tab hunter confidential" in result
    assert "louisiana story" in result


def test_extract_entities_capitalization_respects_stoplist():
    result = extract_entities_capitalization(
        "What is the capital of France?",
        alias_map={},
    )
    assert "what" not in result
    assert "france" in result


def test_extract_entities_capitalization_applies_alias_map():
    result = extract_entities_capitalization(
        "Who founded United States?",
        alias_map={"united states": "usa"},
    )
    assert "usa" in result


def test_extract_entities_capitalization_skips_short_matches():
    result = extract_entities_capitalization("A and I went to Paris", alias_map={})
    assert "a" not in result
    assert "i" not in result
    assert "paris" in result


def test_extract_entities_capitalization_empty_input():
    assert extract_entities_capitalization("", alias_map={}) == []


def test_extract_entities_capitalization_strips_parenthetical_content():
    """Parenthetical disambiguation like '(2006 Film)' is removed so 'Film' is not extracted."""
    result = extract_entities_capitalization(
        "Do both directors of films The Floating Dutchman and The Living And The Dead (2006 Film) have the same nationality?",
        alias_map={},
    )
    assert "film" not in result
    assert "floating dutchman" in result
    assert "living and the dead" in result

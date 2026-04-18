from __future__ import annotations

from dataclasses import dataclass

from surf_rag.core.enrich_entities import (
    extract_entities_capitalization,
    extract_entities_spacy,
)


@dataclass
class _Token:
    text: str
    is_stop: bool
    is_alpha: bool
    pos_: str


@dataclass
class _Span:
    text: str
    label_: str = ""
    tokens: list[_Token] | None = None

    def __iter__(self):
        return iter(self.tokens or [])

    def __len__(self):
        return len(self.tokens or [])


@dataclass
class _Doc:
    ents: list[_Span]
    noun_chunks: list[_Span]


class _NlpStub:
    def __init__(self, doc: _Doc):
        self._doc = doc

    def __call__(self, text: str):
        return self._doc


def _tok(text: str, *, is_stop: bool = False, pos_: str = "NOUN") -> _Token:
    return _Token(text=text, is_stop=is_stop, is_alpha=text.isalpha(), pos_=pos_)


def test_noun_chunks_disabled_returns_only_ner_entities():
    doc = _Doc(
        ents=[],
        noun_chunks=[
            _Span(text="United States", tokens=[_tok("United"), _tok("States")])
        ],
    )
    ents = extract_entities_spacy(
        text="dummy",
        nlp=_NlpStub(doc),
        alias_map={},
        use_noun_chunks=False,
    )
    assert ents == []


def test_noun_chunks_enabled_adds_np_entity_when_ner_misses():
    doc = _Doc(
        ents=[],
        noun_chunks=[
            _Span(
                text="The United States",
                tokens=[
                    _tok("The", is_stop=True, pos_="DET"),
                    _tok("United"),
                    _tok("States"),
                ],
            )
        ],
    )
    ents = extract_entities_spacy(
        text="dummy",
        nlp=_NlpStub(doc),
        alias_map={},
        use_noun_chunks=True,
    )
    assert len(ents) == 1
    assert ents[0]["norm"] == "united states"
    assert ents[0]["type"] == "NOUN_CHUNK"


def test_noun_chunk_filters_length_and_stopword_ratio():
    long_chunk = _Span(
        text="one two three four five six",
        tokens=[
            _tok("one"),
            _tok("two"),
            _tok("three"),
            _tok("four"),
            _tok("five"),
            _tok("six"),
        ],
    )
    stopword_heavy = _Span(
        text="the of and thing",
        tokens=[
            _tok("the", is_stop=True, pos_="DET"),
            _tok("of", is_stop=True, pos_="ADP"),
            _tok("and", is_stop=True),
            _tok("thing"),
        ],
    )
    doc = _Doc(ents=[], noun_chunks=[long_chunk, stopword_heavy])
    ents = extract_entities_spacy(
        text="dummy",
        nlp=_NlpStub(doc),
        alias_map={},
        use_noun_chunks=True,
        noun_chunk_max_tokens=5,
        noun_chunk_stopword_ratio_max=0.6,
    )
    norms = [ent["norm"] for ent in ents]
    assert "one two three four five six" not in norms
    assert "thing" in norms


def test_noun_chunk_entities_are_deduped_and_sorted_by_norm():
    doc = _Doc(
        ents=[
            _Span(text="Zeta Corp", label_="ORG"),
            _Span(text="Alpha Org", label_="ORG"),
        ],
        noun_chunks=[
            _Span(
                text="the zeta corp",
                tokens=[
                    _tok("the", is_stop=True, pos_="DET"),
                    _tok("zeta"),
                    _tok("corp"),
                ],
            ),
            _Span(text="alpha org", tokens=[_tok("alpha"), _tok("org")]),
        ],
    )
    ents = extract_entities_spacy(
        text="dummy",
        nlp=_NlpStub(doc),
        alias_map={},
        use_noun_chunks=True,
    )
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

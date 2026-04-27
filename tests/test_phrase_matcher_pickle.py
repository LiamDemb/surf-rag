"""PhraseMatcher pickles as flat records (no deep trie recursion)."""

from __future__ import annotations

import pickle

import pytest

from surf_rag.entity_matching.matcher import (
    PHRASE_MATCHER_PICKLE_VERSION,
    PhraseMatcher,
    records_to_matcher,
)
from surf_rag.entity_matching.types import PhraseRecord, PhraseSource


def test_pickle_roundtrip_very_long_match_key() -> None:
    """Deep trie paths used to exceed default recursion during pickle.dump."""
    long_key = "z" * 8000
    records = [
        PhraseRecord(
            match_key=long_key,
            canonical_norm="entity_norm",
            source=PhraseSource.CANONICAL,
            df=3,
        ),
        PhraseRecord(
            match_key="short",
            canonical_norm="other",
            source=PhraseSource.SURFACE,
            df=1,
        ),
    ]
    original = records_to_matcher(records)
    blob = pickle.dumps(original, protocol=4)
    restored = pickle.loads(blob)
    end, rec = restored.longest_from(long_key, 0)
    assert end == len(long_key)
    assert rec.match_key == long_key
    assert rec.canonical_norm == "entity_norm"
    hit_short = restored.longest_from("prefix short suffix", 7)
    assert hit_short is not None
    assert hit_short[1].match_key == "short"


def test_getstate_is_flat_record_tuples() -> None:
    m = records_to_matcher(
        [PhraseRecord("ab", "n", PhraseSource.CANONICAL, 1)],
    )
    state = m.__getstate__()
    assert state["v"] == PHRASE_MATCHER_PICKLE_VERSION
    assert state["records"] == [("ab", "n", PhraseSource.CANONICAL.value, 1)]


def test_unpickle_rejects_unknown_state() -> None:
    m = PhraseMatcher()
    with pytest.raises(ValueError, match="Unsupported PhraseMatcher pickle state"):
        m.__setstate__({"v": 999, "records": []})


def test_legacy_setstate_nested_root_still_loads() -> None:
    """Pickles from before flat format stored ``{"root": _TrieNode(...)}``."""
    built = records_to_matcher(
        [PhraseRecord("hi", "n", PhraseSource.CANONICAL, 1)],
    )
    restored = PhraseMatcher()
    restored.__setstate__({"root": built.root})
    assert restored.longest_from("hi", 0) is not None

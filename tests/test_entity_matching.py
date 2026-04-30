"""Tests for lexicon/alias exact entity matching pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from surf_rag.core.entity_alias_resolver import EntityAliasResolver
from surf_rag.entity_matching.matcher import (
    PhraseMatcher,
    build_phrase_records,
    greedy_nonoverlapping_matches,
    records_to_matcher,
)
from surf_rag.core.enrich_entities import normalize_key
from surf_rag.entity_matching.normalization import normalize_for_query_match
from surf_rag.entity_matching.pipeline import LexiconAliasEntityPipeline
from surf_rag.entity_matching.types import PhraseRecord, PhraseSource


def _write_min_artifacts(
    tmp_path: Path,
    *,
    lexicon_rows: list[dict],
    alias_map: dict | None = None,
) -> str:
    alias_map = alias_map or {}
    with (tmp_path / "alias_map.json").open("w", encoding="utf-8") as f:
        json.dump(alias_map, f, ensure_ascii=False, sort_keys=True)
    pd.DataFrame(lexicon_rows).to_parquet(
        tmp_path / "entity_lexicon.parquet", index=False
    )
    return tmp_path.as_posix()


def test_trie_prefers_longest_phrase():
    a = PhraseRecord(
        match_key="new york",
        canonical_norm="new york",
        source=PhraseSource.CANONICAL,
        df=1,
    )
    b = PhraseRecord(
        match_key="york",
        canonical_norm="york city",
        source=PhraseSource.CANONICAL,
        df=1,
    )
    m = PhraseMatcher()
    m.insert(a)
    m.insert(b)
    q = normalize_for_query_match("I live in new york today")
    raw = greedy_nonoverlapping_matches(q, m)
    assert len(raw) == 1
    assert raw[0].match_key == "new york"
    assert raw[0].canonical_norm == "new york"


def test_pipeline_matches_surface_and_respects_df(tmp_path: Path):
    odir = _write_min_artifacts(
        tmp_path,
        lexicon_rows=[
            {
                "norm": "very common",
                "surface_forms": ["very common"],
                "qid_candidates": [],
                "df": 50,
            },
            {
                "norm": "rare item",
                "surface_forms": ["rare item"],
                "qid_candidates": [],
                "df": 1,
            },
        ],
    )
    pl = LexiconAliasEntityPipeline.from_artifacts(
        odir, max_df=5, max_entities_per_query=10
    )
    out = pl.extract("Is rare item better than very common?")
    assert "rare item" in out
    assert "very common" not in out


def test_hunger_games_article_preserved(tmp_path: Path):
    odir = _write_min_artifacts(
        tmp_path,
        lexicon_rows=[
            {
                "norm": "the hunger games",
                "surface_forms": ["The Hunger Games"],
                "qid_candidates": [],
                "df": 2,
            }
        ],
    )
    pl = LexiconAliasEntityPipeline.from_artifacts(
        odir, max_df=8, max_entities_per_query=10
    )
    out = pl.extract("Who was in the film The Hunger Games?")
    # Resolver strips leading articles; graph ids use ``normalize_key``/alias form.
    assert "hunger games" in out


def test_build_records_includes_redirect_and_surfaces(tmp_path: Path):
    with (tmp_path / "alias_map.json").open("w", encoding="utf-8") as f:
        json.dump({"usa": "united states"}, f)
    rows = [
        {
            "norm": "united states",
            "surface_forms": ["United States of America"],
            "df": 4,
        }
    ]
    pd.DataFrame(rows).to_parquet(tmp_path / "entity_lexicon.parquet", index=False)
    _, recs = build_phrase_records(tmp_path.as_posix())
    norms = {r.canonical_norm for r in recs}
    assert "united states" in norms
    assert any(r.source == PhraseSource.REDIRECT_ALIAS for r in recs)


def test_curated_alias_present(tmp_path: Path):
    _write_min_artifacts(
        tmp_path,
        lexicon_rows=[
            {"norm": "united states", "surface_forms": [], "df": 1},
        ],
    )
    r = EntityAliasResolver.from_artifacts(output_dir=tmp_path.as_posix())
    assert r.normalize("U.S.") == "united states"


def test_no_match_inside_alnum_word():
    a = PhraseRecord(
        match_key="ate",
        canonical_norm="ate",
        source=PhraseSource.CANONICAL,
        df=1,
    )
    m = PhraseMatcher()
    m.insert(a)
    q = normalize_for_query_match("paternal line")
    raw = greedy_nonoverlapping_matches(q, m)
    assert raw == []


def test_normalization_folds_diacritics_and_modifier_marks():
    assert normalize_key("ʿAḍud") == "adud"
    assert normalize_for_query_match("ʿAḍud") == "adud"


def test_pipeline_matches_diacritic_variant(tmp_path: Path):
    odir = _write_min_artifacts(
        tmp_path,
        lexicon_rows=[
            {
                "norm": "adud",
                "surface_forms": ["ʿAḍud"],
                "qid_candidates": [],
                "df": 1,
            }
        ],
    )
    pl = LexiconAliasEntityPipeline.from_artifacts(
        odir, max_df=8, max_entities_per_query=5
    )
    assert "adud" in pl.extract("Tell me about adud")


def test_pipeline_no_match(tmp_path: Path):
    odir = _write_min_artifacts(
        tmp_path,
        lexicon_rows=[
            {"norm": "alpha only", "surface_forms": ["alpha only"], "df": 1},
        ],
    )
    pl = LexiconAliasEntityPipeline.from_artifacts(
        odir, max_df=8, max_entities_per_query=5
    )
    assert pl.extract("no known entities here please") == []

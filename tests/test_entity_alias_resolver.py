"""Tests for artifact-aware entity alias resolver behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from surf_rag.core.entity_alias_resolver import EntityAliasResolver


def _write_lexicon(path: Path) -> None:
    rows = [
        {
            "norm": "barack obama",
            "surface_forms": ["President Obama"],
            "qid_candidates": [],
            "df": 7,
        }
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_from_artifacts_requires_alias_map(tmp_path: Path):
    _write_lexicon(tmp_path / "entity_lexicon.parquet")
    with pytest.raises(FileNotFoundError):
        EntityAliasResolver.from_artifacts(output_dir=tmp_path.as_posix())


def test_from_artifacts_merges_alias_map_curated_and_lexicon(tmp_path: Path):
    alias_map = {
        "usa": "united states",
        "the united states": "united states",
    }
    with (tmp_path / "alias_map.json").open("w", encoding="utf-8") as handle:
        json.dump(alias_map, handle, ensure_ascii=False, sort_keys=True)

    _write_lexicon(tmp_path / "entity_lexicon.parquet")
    resolver = EntityAliasResolver.from_artifacts(output_dir=tmp_path.as_posix())

    assert resolver.normalize("USA") == "united states"
    assert resolver.normalize("U.S.") == "united states"
    assert resolver.normalize("President Obama") == "barack obama"


def test_load_df_map_from_lexicon(tmp_path: Path):
    _write_lexicon(tmp_path / "entity_lexicon.parquet")
    df_map = EntityAliasResolver.load_df_map_from_lexicon(
        lexicon_path=(tmp_path / "entity_lexicon.parquet").as_posix()
    )
    assert df_map["barack obama"] == 7

"""Precomputed entity phrase matcher artifacts."""

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from surf_rag.entity_matching.artifacts import (
    ENTITY_MATCHING_SCHEMA,
    MANIFEST_FILENAME,
    build_entity_matching_artifacts,
    try_load_precomputed_matcher,
)
from surf_rag.entity_matching.pipeline import LexiconAliasEntityPipeline


@pytest.fixture
def tiny_corpus(tmp_path: Path) -> Path:
    alias = {"einstein redirect": "albert einstein"}
    (tmp_path / "alias_map.json").write_text(
        json.dumps(alias, ensure_ascii=False), encoding="utf-8"
    )
    df = pd.DataFrame(
        [
            {
                "norm": "albert einstein",
                "surface_forms": ["Einstein", "Albert Einstein"],
                "df": 3,
            }
        ]
    )
    df.to_parquet(tmp_path / "entity_lexicon.parquet", index=False)
    return tmp_path


def test_build_writes_manifest_and_records(tiny_corpus: Path) -> None:
    build_entity_matching_artifacts(tiny_corpus, force=True)
    man = json.loads((tiny_corpus / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert man["schema_version"] == ENTITY_MATCHING_SCHEMA
    assert man["record_count"] > 0
    assert (tiny_corpus / "entity_phrase_records.parquet").is_file()
    assert (tiny_corpus / "entity_phrase_matcher.pkl").is_file()


def test_pipeline_skips_build_when_precomputed(tiny_corpus: Path) -> None:
    build_entity_matching_artifacts(tiny_corpus, force=True)
    with patch("surf_rag.entity_matching.pipeline.build_phrase_records") as mock_build:
        LexiconAliasEntityPipeline.from_artifacts(str(tiny_corpus))
    mock_build.assert_not_called()


def test_stale_alias_invalidates_matcher(tiny_corpus: Path) -> None:
    build_entity_matching_artifacts(tiny_corpus, force=True)
    alias = json.loads((tiny_corpus / "alias_map.json").read_text(encoding="utf-8"))
    alias["extra"] = "extra_norm"
    (tiny_corpus / "alias_map.json").write_text(
        json.dumps(alias, ensure_ascii=False), encoding="utf-8"
    )
    assert try_load_precomputed_matcher(tiny_corpus) is None

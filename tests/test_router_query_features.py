"""V1 query feature extraction."""

from __future__ import annotations

import pytest

from surf_rag.core.enrich_entities import load_spacy_syntactic_query_features
from surf_rag.router.query_features import (
    V1_FEATURE_NAMES,
    QueryFeatureContext,
    extract_features_v1,
    feature_vector_ordered,
)

pytest.importorskip("spacy")
import spacy  # noqa: E402

try:
    nlp = load_spacy_syntactic_query_features()
except (OSError, RuntimeError) as e:
    pytest.skip(
        f"spaCy model not available ({e!s}); run: make setup-models",
        allow_module_level=True,
    )


def test_v1_keys_complete() -> None:
    f = extract_features_v1(
        "Who won more titles than X in 2019?", QueryFeatureContext(nlp=nlp)
    )
    assert set(f) == set(V1_FEATURE_NAMES)
    assert len(feature_vector_ordered(f)) == len(V1_FEATURE_NAMES)


def test_empty_query() -> None:
    f = extract_features_v1("", QueryFeatureContext(nlp=nlp))
    assert all(f[k] == 0.0 for k in V1_FEATURE_NAMES)


def test_syntactic_loader_lemmatizes_inflected_verbs() -> None:
    """Lemmas must be set so relation_cue matches base-form _RELATION_CUE_LEMMAS."""
    doc = nlp("who owns a golf course in scotland")
    owns = next(t for t in doc if t.text.lower() == "owns")
    assert owns.lemma_ == "own"


def test_relation_cue_density_matches_own_lemma() -> None:
    f = extract_features_v1(
        "who owns st andrews golf course in scotland",
        QueryFeatureContext(nlp=nlp),
    )
    assert f["relation_cue_density"] > 0.0


def test_relation_cue_density_matches_die_lemma() -> None:
    f = extract_features_v1(
        "When did the president die?",
        QueryFeatureContext(nlp=nlp),
    )
    assert f["relation_cue_density"] > 0.0


def test_bridge_composition_with_related_lemma_and_two_ner() -> None:
    """Compositional: relate lemma + 2+ NERs + prep-style deps often yields bridge=1.0."""
    f = extract_features_v1(
        "How is Apple related to Microsoft?",
        QueryFeatureContext(nlp=nlp),
    )
    assert f["multi_entity_indicator"] == 1.0
    assert f["relation_cue_density"] > 0.0
    assert f["bridge_composition_indicator"] == 1.0

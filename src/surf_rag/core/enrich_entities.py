from __future__ import annotations

import os
import re
import unicodedata
from typing import Dict, List, Optional

import spacy

DEFAULT_ENTITY_TYPES = {"PERSON", "ORG", "GPE", "LOC", "EVENT", "WORK_OF_ART"}
DEFAULT_SPACY_MODEL = "en_core_web_sm"


def normalize_key(text: str) -> str:
    s = unicodedata.normalize("NFKC", text).lower()
    s = re.sub(r"\([^)]*\)", "", s)  # Remove anything in brackets
    s = re.sub(r"['’]s\b", "", s)
    s = re.sub(r"[^\w\s-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Keep alias/entity lookup deterministic across common title forms.
    while True:
        stripped = re.sub(r"^(the|a|an)\s+", "", s).strip()
        if stripped == s:
            break
        s = stripped
    return s


def norm_entity(text: str, alias_map: Optional[Dict[str, str]] = None) -> str:
    alias_map = alias_map or {}
    normalized = normalize_key(text)
    return alias_map.get(normalized, normalized)


def _env_bool(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def add_both_alias_and_raw_triples() -> bool:
    """Whether to add both aliased and raw-normalized triples when they differ.

    When enabled, relation extractors emit an extra triple using normalize_key()
    (no alias lookup) in addition to the aliased triple. Helps when the alias
    map is imperfect. Controlled by ADD_BOTH_ALIAS_AND_RAW_TRIPLES env var.
    """
    return _env_bool("ADD_BOTH_ALIAS_AND_RAW_TRIPLES", "0")


def load_spacy(model: str | None = None) -> "spacy.Language":
    """Load spaCy for **NER only** (tagger, parser, lemmatizer disabled) — fast default."""
    model = model or os.environ.get("SPACY_MODEL", DEFAULT_SPACY_MODEL)
    try:
        return spacy.load(model, disable=["tagger", "parser", "lemmatizer"])
    except OSError as e:
        raise RuntimeError(
            f"spaCy model '{model}' not found. Install it with:\n"
            f"  python -m spacy download {model}\n"
            "Or run: make setup-models"
        ) from e


def load_spacy_syntactic_query_features(model: str | None = None) -> "spacy.Language":
    """Load the full spaCy pipeline for router query features (tagger, parser, NER, lemmatizer).

    Router features use token lemmas (e.g. relation-cue matching); do not disable the lemmatizer.
    This path is only used for pre-retrieval query features, not the fast NER-only :func:`load_spacy`.
    """
    model = model or os.environ.get("SPACY_MODEL", DEFAULT_SPACY_MODEL)
    try:
        return spacy.load(model)
    except OSError as e:
        raise RuntimeError(
            f"spaCy model '{model}' not found. Install it with:\n"
            f"  python -m spacy download {model}\n"
            "Or run: make setup-models"
        ) from e


# Query-only heuristic: entities in questions often appear in Title Case.
# Matches: "United States", "WWE Talking Smack", "Fabio Fognini"
_CAPITALIZED_SPAN_PATTERN = re.compile(
    r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,}))*\b"
)
_QUERY_STOPLIST = frozenset(
    {"what", "which", "how", "where", "when", "who", "whose", "why"}
)


_PAREN_PATTERN = re.compile(r"\s*\([^)]*\)\s*")


def extract_entities_capitalization(
    text: str,
    alias_map: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Extract entity spans using capitalization heuristic (query-only).

    Entities in questions often appear in Title Case. This regex matches
    consecutive Title-Case words and 2+ character acronyms. Use with
    extract_entities_spacy for combined query extraction.

    Parenthetical content (e.g. "(2006 Film)", "(album)") is removed before
    extraction to avoid matching generic disambiguation words as entities.

    Note: [A-Z] is ASCII-only; accented uppercase (e.g. Čilić) may not match.
    """
    alias_map = alias_map or {}
    # Remove parenthetical content to avoid "Film", "album", etc. from "(2006 Film)"
    text_no_parens = _PAREN_PATTERN.sub(" ", text)
    text_no_parens = re.sub(r"\s+", " ", text_no_parens).strip()

    result: List[str] = []
    seen: set[str] = set()
    for match in _CAPITALIZED_SPAN_PATTERN.finditer(text_no_parens):
        surface = match.group(0).strip()
        if not surface:
            continue
        # Skip interrogative words when matched as single token
        if surface.lower() in _QUERY_STOPLIST:
            continue
        norm = norm_entity(surface, alias_map)
        if not norm or len(norm) < 2:
            continue
        if norm not in seen:
            seen.add(norm)
            result.append(norm)
    return result


def extract_entities_spacy(
    text: str,
    nlp,
    alias_map: Optional[Dict[str, str]] = None,
    allowed_types: Optional[set] = None,
) -> List[Dict[str, str]]:
    """Extract NER entities from *text* using *nlp*; allowed spaCy label types only."""
    allowed = allowed_types or DEFAULT_ENTITY_TYPES
    alias_map = alias_map or {}
    ents: Dict[str, Dict[str, str]] = {}
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ not in allowed:
            continue
        norm = norm_entity(ent.text, alias_map)
        if not norm:
            continue
        ents[norm] = {
            "surface": ent.text,
            "norm": norm,
            "type": ent.label_,
            "qid": None,
        }

    return [ents[key] for key in sorted(ents)]

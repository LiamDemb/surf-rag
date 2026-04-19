from __future__ import annotations

import os
import re
import unicodedata
from typing import Dict, Iterable, List, Optional

import spacy

DEFAULT_ENTITY_TYPES = {"PERSON", "ORG", "GPE", "LOC", "EVENT", "WORK_OF_ART"}
DEFAULT_SPACY_MODEL = "en_core_web_sm"
DEFAULT_USE_NOUN_CHUNKS = "0"
DEFAULT_NOUN_CHUNK_MAX_TOKENS = 5
DEFAULT_NOUN_CHUNK_STOPWORD_RATIO_MAX = 0.6


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


def should_use_noun_chunks(use_noun_chunks: Optional[bool] = None) -> bool:
    if use_noun_chunks is not None:
        return bool(use_noun_chunks)
    return _env_bool("ENTITY_USE_NOUN_CHUNKS", DEFAULT_USE_NOUN_CHUNKS)


def load_spacy(
    model: str | None = None,
    use_noun_chunks: Optional[bool] = None,
) -> "spacy.Language":
    model = model or os.environ.get("SPACY_MODEL", DEFAULT_SPACY_MODEL)
    noun_chunks_enabled = should_use_noun_chunks(use_noun_chunks)
    disable = (
        ["lemmatizer"] if noun_chunks_enabled else ["tagger", "parser", "lemmatizer"]
    )
    try:
        return spacy.load(model, disable=disable)
    except OSError as e:
        raise RuntimeError(
            f"spaCy model '{model}' not found. Install it with:\n"
            f"  python -m spacy download {model}\n"
            "Or run: make setup-models"
        ) from e


def _trim_np_tokens(tokens: Iterable[object]) -> List[object]:
    trim_pos = {"DET", "ADP", "PART"}
    trimmed = list(tokens)
    while trimmed and (
        getattr(trimmed[0], "is_stop", False)
        or getattr(trimmed[0], "pos_", "") in trim_pos
    ):
        trimmed.pop(0)
    while trimmed and (
        getattr(trimmed[-1], "is_stop", False)
        or getattr(trimmed[-1], "pos_", "") in trim_pos
    ):
        trimmed.pop()
    return trimmed


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


def _extract_noun_chunk_surfaces(
    doc,
    max_tokens: int,
    stopword_ratio_max: float,
) -> List[str]:
    if not hasattr(doc, "noun_chunks"):
        return []

    surfaces: List[str] = []
    for chunk in doc.noun_chunks:
        tokens = _trim_np_tokens(list(chunk))
        if not tokens:
            continue
        if len(tokens) > max_tokens:
            continue
        stop_count = sum(1 for tok in tokens if getattr(tok, "is_stop", False))
        ratio = stop_count / max(1, len(tokens))
        if ratio > stopword_ratio_max:
            continue
        has_alpha = any(getattr(tok, "is_alpha", False) for tok in tokens)
        has_non_stop = any(not getattr(tok, "is_stop", False) for tok in tokens)
        if not has_alpha or not has_non_stop:
            continue
        surface = " ".join(getattr(tok, "text", "") for tok in tokens).strip()
        if surface:
            surfaces.append(surface)
    return surfaces


def extract_entities_spacy(
    text: str,
    nlp,
    alias_map: Optional[Dict[str, str]] = None,
    allowed_types: Optional[set] = None,
    use_noun_chunks: Optional[bool] = None,
    noun_chunk_max_tokens: Optional[int] = None,
    noun_chunk_stopword_ratio_max: Optional[float] = None,
) -> List[Dict[str, str]]:
    allowed = allowed_types or DEFAULT_ENTITY_TYPES
    alias_map = alias_map or {}
    noun_chunks_enabled = should_use_noun_chunks(use_noun_chunks)
    max_tokens = noun_chunk_max_tokens or int(
        os.environ.get("NOUN_CHUNK_MAX_TOKENS", str(DEFAULT_NOUN_CHUNK_MAX_TOKENS))
    )
    stop_ratio_max = (
        noun_chunk_stopword_ratio_max
        if noun_chunk_stopword_ratio_max is not None
        else float(
            os.environ.get(
                "NOUN_CHUNK_STOPWORD_RATIO_MAX",
                str(DEFAULT_NOUN_CHUNK_STOPWORD_RATIO_MAX),
            )
        )
    )
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

    if noun_chunks_enabled:
        for surface in _extract_noun_chunk_surfaces(
            doc=doc,
            max_tokens=max_tokens,
            stopword_ratio_max=stop_ratio_max,
        ):
            norm = norm_entity(surface, alias_map)
            if not norm:
                continue
            ents.setdefault(
                norm,
                {
                    "surface": surface,
                    "norm": norm,
                    "type": "NOUN_CHUNK",
                    "qid": None,
                },
            )

    return [ents[key] for key in sorted(ents)]

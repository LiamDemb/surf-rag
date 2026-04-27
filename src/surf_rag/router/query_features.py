"""Modular V1 query features for the router (pre-retrieval, lightweight)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

# Bump when feature definitions change.
FEATURE_SET_VERSION = "1"

# Stable column order for vector export / classifiers.
V1_FEATURE_NAMES: Tuple[str, ...] = (
    "content_token_len",
    "quoted_span_count",
    "numeric_or_date_token_count",
    "named_entity_count",
    "multi_entity_indicator",
    "kg_linkable_entity_ratio",
    "noun_phrase_count",
    "relation_cue_density",
    "max_dependency_depth",
    "coord_subordination_count",
    "bridge_composition_indicator",
    "comparison_indicator",
    "temporal_indicator",
    "numeric_count_indicator",
)

# Lemmas and surface cues for relation-like questions (subset of draft + common KBQA).
_RELATION_CUE_LEMMAS: Set[str] = {
    "relate",
    "relation",
    "between",
    "connect",
    "influence",
    "cause",
    "found",
    "parent",
    "spouse",
    "marry",
    "head",
    "lead",
    "own",
    "part",
    "member",
    "locate",
    "born",
    "die",
    "happen",
    "award",
    "play",
    "work",
    "hold",
    "succeed",
    "predecessor",
}

# Dependency labels for compositional / subordinate structure
_COORD_SUBORD_DEPS: Set[str] = {
    "conj",
    "advcl",
    "ccomp",
    "xcomp",
    "acl",
    "relcl",
    "csubj",
    "csubjpass",
}

# Bridge-like attachments (prepositional / nominal modifier chains)
_BRIDGE_DEPS: Set[str] = {"nmod", "pobj", "prep", "agent", "dobj"}


@dataclass
class QueryFeatureContext:
    """Optional resources for feature extraction (lazy-initialized in CLI)."""

    nlp: Any = None
    """Full spaCy pipeline (``load_spacy_syntactic_query_features``): tagger, parser, NER, lemmatizer.

    Not the NER-only ``load_spacy`` shortcut used in fast corpus paths.
    """

    entity_pipeline: Any = None
    """Optional :class:`LexiconAliasEntityPipeline` for KG-linkable signals."""

    retrieval_asset_dir: Optional[str] = None


_QUOTED_RE = re.compile(r'"[^"]+"|\'[^\']+\'')
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_NUM_RE = re.compile(
    r"\b\d+([.,]\d+)?%?\b|"
    r"\b(first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th|six|ten)\b",
    re.IGNORECASE,
)
_COMPARISON_TOKENS: Set[str] = {
    "more",
    "less",
    "older",
    "larger",
    "smaller",
    "taller",
    "younger",
    "earlier",
    "later",
    "same",
    "different",
    "than",
    "compared",
    "compare",
    "vs",
    "versus",
    "unlike",
    "like",
}
_TEMPORAL_TOKENS: Set[str] = {
    "when",
    "before",
    "after",
    "during",
    "since",
    "until",
    "while",
    "latest",
    "current",
    "yesterday",
    "today",
    "century",
    "decade",
    "era",
    "ancient",
    "modern",
}
_NUMERIC_CUE_TOKENS: Set[str] = {
    "how many",
    "how much",
    "number of",
    "amount of",
    "count",
    "total",
    "sum",
    "percent",
    "percentage",
}


def _get_nlp(nlp: Any = None) -> Any:
    if nlp is not None:
        return nlp
    from surf_rag.core.enrich_entities import load_spacy_syntactic_query_features

    return load_spacy_syntactic_query_features()


def _content_tokens(doc: Any) -> List[Any]:
    return [
        t
        for t in doc
        if not (getattr(t, "is_space", False) or getattr(t, "is_punct", False))
        and not getattr(t, "is_stop", False)
    ]


def _max_tree_depth(sent: Any) -> int:
    """Max depth of token dependency tree in one sentence (root depth 0)."""
    roots = [t for t in sent if t.head == t or t.dep_ == "ROOT"]
    if not roots:
        return 0
    best = 0
    for root in roots:
        stack: List[Tuple[Any, int]] = [(root, 0)]
        while stack:
            tok, d = stack.pop()
            if d > best:
                best = d
            for c in tok.children:
                if c is not root:
                    stack.append((c, d + 1))
    return best


def _dependency_depths(doc: Any) -> int:
    return max((_max_tree_depth(s) for s in doc.sents), default=0)


def extract_features_v1(
    query: str,
    context: Optional[QueryFeatureContext] = None,
) -> Dict[str, float]:
    """Compute the full V1 feature dict for a single query string.

    All values are numeric (float) for z-score; binary features use 0.0/1.0.
    If ``retrieval_asset_dir`` is not available, KG features default to 0.0.
    """
    ctx = context or QueryFeatureContext()
    text = (query or "").strip()
    if not text:
        return {name: 0.0 for name in V1_FEATURE_NAMES}

    nlp = _get_nlp(ctx.nlp)
    ctx.nlp = nlp
    doc = nlp(text)
    content = _content_tokens(doc)
    content_len = max(len(content), 1)
    content_token_len = float(len(content))

    quoted = _QUOTED_RE.findall(text)
    quoted_span_count = float(len(quoted))

    n_date_num = 0
    for t in content:
        tx = t.text
        if _YEAR_RE.search(tx) or _NUM_RE.search(tx):
            n_date_num += 1
    n_date_num += len(_YEAR_RE.findall(text))
    numeric_or_date_token_count = float(n_date_num)

    from surf_rag.core.enrich_entities import DEFAULT_ENTITY_TYPES

    ents = [e for e in doc.ents if e.label_ in DEFAULT_ENTITY_TYPES]
    named_entity_count = float(len(ents))
    # Distinct norm surfaces
    seen_norm: Set[str] = set()
    for e in ents:
        s = e.text.strip().casefold()
        if s:
            seen_norm.add(s)
    multi_entity_indicator = 1.0 if len(seen_norm) >= 2 else 0.0

    kg_n = 0.0
    kg_ratio = 0.0
    if ctx.entity_pipeline is not None:
        try:
            linked = ctx.entity_pipeline.extract(text)
            kg_n = float(len(linked))
            denom = max(1.0, named_entity_count if named_entity_count > 0 else kg_n)
            kg_ratio = min(1.0, kg_n / denom)
        except (OSError, FileNotFoundError, ValueError, RuntimeError):
            kg_ratio = 0.0
            kg_n = 0.0

    noun_phrase_count = float(len(list(doc.noun_chunks)))

    lemmas = [getattr(t, "lemma_", "").lower() for t in content]
    n_rel = sum(1 for lem in lemmas if lem in _RELATION_CUE_LEMMAS)
    relation_cue_density = float(n_rel) / float(content_len)

    max_dep = float(_dependency_depths(doc))

    coord_sub = 0
    for t in doc:
        dep = getattr(t, "dep_", "")
        if dep in _COORD_SUBORD_DEPS:
            coord_sub += 1
        low = t.text.casefold()
        if low in {
            "and",
            "or",
            "but",
            "while",
            "after",
            "before",
            "although",
        }:
            coord_sub += 1
    coord_subordination_count = float(coord_sub)

    has_rel_cue = n_rel > 0
    has_ner2 = len(seen_norm) >= 2
    bridge_d = False
    for t in doc:
        if t.dep_ in _BRIDGE_DEPS and (has_rel_cue or has_ner2):
            bridge_d = True
            break
    bridge_composition_indicator = (
        1.0 if (has_rel_cue and has_ner2 and bridge_d) else 0.0
    )

    low_text = text.casefold()
    comparison_indicator = 0.0
    for w in _COMPARISON_TOKENS:
        if re.search(rf"\b{re.escape(w)}\b", low_text):
            comparison_indicator = 1.0
            break
    if comparison_indicator == 0.0:
        for t in content:
            lem = getattr(t, "lemma_", "") or ""
            if lem.endswith("er") or lem.endswith("est"):
                comparison_indicator = 1.0
                break

    temporal = 0.0
    for w in _TEMPORAL_TOKENS:
        if re.search(rf"\b{re.escape(w)}\b", low_text):
            temporal = 1.0
            break
    if _YEAR_RE.search(text) or re.search(
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
        low_text,
    ):
        temporal = 1.0
    temporal_indicator = temporal

    numeric_count = 0.0
    if re.search(r"\b(how many|how much|number of)\b", low_text):
        numeric_count = 1.0
    elif n_date_num > 0 and re.search(
        r"\b(how|what).*\b(many|much|number)\b", low_text
    ):
        numeric_count = 1.0
    for cue in _NUMERIC_CUE_TOKENS:
        if len(cue) > 3 and cue in low_text:
            numeric_count = 1.0
    numeric_count_indicator = numeric_count

    out: Dict[str, float] = {
        "content_token_len": content_token_len,
        "quoted_span_count": quoted_span_count,
        "numeric_or_date_token_count": numeric_or_date_token_count,
        "named_entity_count": named_entity_count,
        "multi_entity_indicator": multi_entity_indicator,
        "kg_linkable_entity_ratio": kg_ratio,
        "noun_phrase_count": noun_phrase_count,
        "relation_cue_density": relation_cue_density,
        "max_dependency_depth": max_dep,
        "coord_subordination_count": coord_subordination_count,
        "bridge_composition_indicator": bridge_composition_indicator,
        "comparison_indicator": comparison_indicator,
        "temporal_indicator": temporal_indicator,
        "numeric_count_indicator": numeric_count_indicator,
    }
    return out


def feature_vector_ordered(features: Dict[str, float]) -> List[float]:
    return [float(features.get(n, 0.0)) for n in V1_FEATURE_NAMES]

"""Deterministic query intent features: anchors vs operators vs leftover content tokens."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, FrozenSet, Iterable

if TYPE_CHECKING:
    from surf_rag.entity_matching.types import SeedCandidate

# Stopwords + structural QA tokens
_STOPWORDS: FrozenSet[str] = frozenset("""
    a an the and or but if in is it of on at to as by no we he be do me my up was were are
    for not with from that this when whom which how why her him his has had may out new can
    did its now one any all our she own too few per via etc age ago two six ten
    """.split())

# Tokens often acting as relation-like cues in questions (soft overlap only).
_OPERATOR_LEXICON: FrozenSet[str] = frozenset("""
    plays played playing directed directs director spouse married marry grandfather grandmother
    uncle aunt cousin president vice prime minister born died burial buried spouse husband wife
    child children father mother son daughter sibling team group album song film movie novel book
    author wrote written starring cause caused leads leading located situated headquartered based
    member founder founded acquired merged acquired opponent defeated won released dated date
    voiced voice voices cast starred starring plays plays plays
    """.split())

# Surface tokens that often appear as extracted ``E:{norm}`` seeds but act as query scaffolding,
# not answer-bearing anchors (used only for classification / damping; not a graph deny list).
_STRUCTURAL_QUERY_ENTITY_TERMS: FrozenSet[str] = frozenset("""
    name most team group plays played directed voicing voiced voice cast starring song film movie
    band album country city date year first second third team group
    """.split())

_WH_WORDS = ("who", "what", "where", "when", "which", "whose", "whom", "how")


def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"[^\w]+", text.lower()) if t]


def _mask_seed_spans(query_lower: str, seeds: Iterable["SeedCandidate"]) -> list[bool]:
    """Per-character mask for matched entity spans on the normalized query string."""
    n = len(query_lower)
    mask = [False] * n
    for s in seeds:
        a = max(0, min(s.start, n))
        b = max(0, min(s.end, n))
        for i in range(a, b):
            mask[i] = True
    return mask


def _answer_hint(query_lower: str) -> str | None:
    q = query_lower.strip()
    for w in _WH_WORDS:
        if re.search(rf"\b{re.escape(w)}\b", q):
            return w
    if "date" in q.split() or "year" in q.split():
        return "when"
    return None


@dataclass(frozen=True, slots=True)
class QueryIntent:
    """Soft lexical signals derived without brittle dependency parses."""

    operator_terms: tuple[str, ...]
    content_terms: tuple[str, ...]
    operator_token_set: frozenset[str]
    content_token_set: frozenset[str]
    query_token_set: frozenset[str]
    relation_overlap_tokens: frozenset[str]
    answer_hint: str | None
    operator_lexicon_hits_full_query: frozenset[str]


def operator_lexicon_hits_in_full_query(query: str) -> frozenset[str]:
    """Relation-like tokens anywhere in the raw query (before seed-span masking)."""
    q_lower = query.strip().lower()
    hits: set[str] = set()
    for m in re.finditer(r"\w+", q_lower):
        tok = m.group(0)
        if tok in _OPERATOR_LEXICON or tok in _STRUCTURAL_QUERY_ENTITY_TERMS:
            hits.add(tok)
    return frozenset(hits)


def intent_relation_embedding_text(intent: "QueryIntent", query: str) -> str:
    """Compact text for relation-centric query embeddings (v04 transitions / frontier)."""
    pieces = list(
        dict.fromkeys(list(intent.operator_terms) + list(intent.content_terms))
    )
    if intent.answer_hint:
        pieces.append(intent.answer_hint)
    tail = query.strip()
    if pieces:
        return " ".join(pieces) + " || " + tail
    return tail


def is_structural_query_entity_token(tok: str) -> bool:
    """True if ``tok`` is a single-token surface often used as scaffolding, not an answer anchor."""
    return tok.casefold() in _STRUCTURAL_QUERY_ENTITY_TERMS


def build_query_intent(
    query: str, seed_candidates: list["SeedCandidate"]
) -> QueryIntent:
    """Separate operator-like cues from leftover content words outside seed spans."""
    q_raw = query.strip()
    q_lower = q_raw.lower()
    lex_hits_full = operator_lexicon_hits_in_full_query(q_raw)
    mask = _mask_seed_spans(q_lower, seed_candidates)

    tokens_pos: list[tuple[str, int, int]] = []
    for m in re.finditer(r"\w+", q_lower):
        tokens_pos.append((m.group(0), m.start(), m.end()))

    operators_from_unmasked: list[str] = []
    content: list[str] = []
    for tok, a, b in tokens_pos:
        if tok in _STOPWORDS:
            continue
        under_seed = any(mask[i] for i in range(a, min(b, len(mask))))
        if under_seed:
            continue
        if tok in _OPERATOR_LEXICON:
            operators_from_unmasked.append(tok)
        else:
            content.append(tok)

    hint = _answer_hint(q_lower)
    merged_operators = sorted(set(operators_from_unmasked) | set(lex_hits_full))

    extra_for_rel = set(merged_operators) | set(content)
    if hint:
        extra_for_rel.add(hint)

    q_set = frozenset(t for t in _tokenize(q_lower) if t)

    return QueryIntent(
        operator_terms=tuple(merged_operators),
        content_terms=tuple(content),
        operator_token_set=frozenset(merged_operators),
        content_token_set=frozenset(content),
        query_token_set=q_set,
        relation_overlap_tokens=frozenset(extra_for_rel),
        answer_hint=hint,
        operator_lexicon_hits_full_query=lex_hits_full,
    )


def clean_relation_label(rel: str) -> str:
    """Normalize relation text for lexical overlap (matches ``graph_scoring.clean_rel`` spirit)."""
    return rel.replace("inv:", "").replace("_", " ").strip().lower()


def relation_lexical_compatibility(
    relation: str,
    intent: QueryIntent,
    *,
    floor: float,
    weight: float,
) -> float:
    """Bounded overlap between relation tokens and query residue tokens."""
    rel_text = clean_relation_label(relation)
    tokens = [t for t in rel_text.replace("-", " ").split() if t]
    if not tokens:
        return floor
    qtok = intent.relation_overlap_tokens | intent.query_token_set
    overlap = sum(1 for t in tokens if t in qtok)
    boost = min(1.0, weight * overlap / max(len(tokens), 1))
    return floor + (1.0 - floor) * boost


def operator_span_overlap_fraction(seed: "SeedCandidate", intent: QueryIntent) -> float:
    """How strongly matched span tokens overlap intent operator lexicon (soft penalty input)."""
    toks = [t for t in re.split(r"\W+", seed.matched_text.lower()) if t]
    if not toks:
        return 0.0
    hits = sum(
        1 for t in toks if t in intent.operator_token_set or t in _OPERATOR_LEXICON
    )
    return float(hits / max(len(toks), 1))

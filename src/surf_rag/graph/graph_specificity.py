"""Graph-derived entity specificity without curated generic-node blocklists.

Signals combine structural statistics (appearance frequency, relation density)
with lightweight label-shape features. Values are continuous and bounded so
high-document-frequency entities like ``real madrid`` or ``supreme court`` are
downweighted smoothly rather than hard-filtered by corpus DF caps.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any, Dict, List

# Corpus-independent token statistics only (not entity-specific deny lists).
_STOPWORDS_EN = frozenset("""
    a an the and or but if in is it of on at to as by no we he be do me my up was were are
    for not with from that this when who whom which what how why her him his has had may out
    new can did its now one any all our she own too few per via etc
    """.split())


def _strip_entity_prefix(node_id: str) -> str:
    if isinstance(node_id, str) and node_id.startswith("E:"):
        return node_id[2:]
    return node_id if isinstance(node_id, str) else ""


def count_appears_in_edges(graph: Any, node: str) -> int:
    """Number of ``appears_in`` edges from this entity node to chunk nodes."""
    if node not in graph:
        return 0
    return sum(
        1
        for _, _, d in graph.out_edges(node, data=True)
        if d.get("kind") == "appears_in"
    )


def count_rel_edges(graph: Any, node: str) -> int:
    """Undirected count of ``rel`` edges touching this entity (in + out)."""
    if node not in graph:
        return 0
    n_out = sum(
        1 for _, _, d in graph.out_edges(node, data=True) if d.get("kind") == "rel"
    )
    n_in = sum(
        1 for _, _, d in graph.in_edges(node, data=True) if d.get("kind") == "rel"
    )
    return int(n_out + n_in)


def total_graph_degree(graph: Any, node: str) -> int:
    if node not in graph:
        return 0
    return int(graph.degree(node))


def label_shape_score(norm: str) -> float:
    """Bounded [0, 1] score from surface form alone (no entity deny lists).

    Rewards moderate-length multi-token mentions; penalizes very short tokens,
    digit-only spans, and extremely long boilerplate-like strings.
    """
    raw = (norm or "").strip()
    if not raw:
        return 0.05

    text = unicodedata.normalize("NFKC", raw)
    tokens = [t for t in re.split(r"\s+", text.lower()) if t]
    if not tokens:
        single = text.lower()
        alnum = sum(1 for ch in single if ch.isalnum())
        ratio = alnum / max(len(single), 1)
        if ratio < 0.5:
            return 0.15
        return 0.55

    n_tok = len(tokens)
    alpha_chars = sum(1 for ch in text if ch.isalpha())
    digit_chars = sum(1 for ch in text if ch.isdigit())
    total_chars = max(len(text), 1)
    digit_ratio = digit_chars / total_chars

    stop_hits = sum(1 for t in tokens if t in _STOPWORDS_EN)
    stop_ratio = stop_hits / max(n_tok, 1)

    # Length term: peak around 2–5 tokens.
    if n_tok <= 1:
        len_term = 0.55
    elif n_tok <= 5:
        len_term = 0.75 + 0.05 * (n_tok - 2)
    else:
        len_term = max(0.45, 0.95 - 0.04 * (n_tok - 5))

    digit_penalty = 1.0 - min(0.85, digit_ratio * 2.0)
    stop_penalty = 1.0 - min(0.5, stop_ratio * 1.2)

    alpha_density = alpha_chars / total_chars
    script_term = 0.55 + 0.45 * min(1.0, alpha_density)

    combined = len_term * digit_penalty * stop_penalty * script_term
    return float(max(0.05, min(1.0, combined)))


def node_specificity_score(graph: Any, node_id: str) -> float:
    """Continuous specificity in (0, 1]: higher for informative, non-hub entities."""
    if not node_id or node_id not in graph:
        return 0.05

    app = count_appears_in_edges(graph, node_id)
    rel_ct = count_rel_edges(graph, node_id)
    deg = total_graph_degree(graph, node_id)
    shape = label_shape_score(_strip_entity_prefix(node_id))

    app_term = 1.0 / math.sqrt(1.0 + float(app))
    rel_term = 1.0 / math.log(2.0 + float(rel_ct))
    deg_term = 1.0 / math.log(2.0 + float(max(deg, rel_ct)))

    combined = app_term * rel_term * deg_term * (0.65 + 0.35 * shape)
    return float(max(0.02, min(1.0, combined)))


def query_coverage_for_seed(
    seed_node: str,
    extracted_norms: set[str],
    vector_matches: List[Dict[str, Any]],
) -> float:
    """Soft coverage signal [0, 1]: direct lexical seed hits beat fuzzy matches."""
    norm = seed_node[2:] if seed_node.startswith("E:") else seed_node
    if norm in extracted_norms:
        return 1.0
    best = 0.0
    for m in vector_matches:
        if m.get("matched_norm") == norm:
            best = max(best, float(m.get("score", 0.0)))
    if best > 0:
        return max(0.25, best)
    return 0.5


def seed_restart_mass_for_nodes(
    graph: Any,
    seed_nodes: set[str],
    extracted_norms: List[str],
    vector_matches: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Unnormalized restart masses combining specificity with seed/query overlap."""
    ext_set = set(extracted_norms)
    weights: Dict[str, float] = {}
    for node in seed_nodes:
        cov = query_coverage_for_seed(node, ext_set, vector_matches)
        spec = node_specificity_score(graph, node)
        weights[node] = float(spec) * float(max(cov, 0.05))
    return weights


def normalize_restart_masses(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        n = len(weights)
        if n == 0:
            return {}
        flat = 1.0 / float(n)
        return {k: flat for k in weights}
    inv = 1.0 / total
    return {k: float(v) * inv for k, v in weights.items()}


def specificity_seed_summary(graph: Any, seed_nodes: set[str]) -> Dict[str, Any]:
    """Min/median/max specificity stats plus lowest-specificity seeds (explainability)."""
    vals = sorted(
        float(node_specificity_score(graph, s)) for s in seed_nodes if s in graph
    )
    if not vals:
        return {
            "min": None,
            "median": None,
            "max": None,
            "lowest_specificity_seeds": [],
        }

    mid = vals[len(vals) // 2]
    paired = [
        (float(node_specificity_score(graph, s)), s)
        for s in sorted(seed_nodes)
        if s in graph
    ]
    paired.sort(key=lambda x: x[0])
    lowest = [{"seed": n, "specificity": v} for v, n in paired[: min(5, len(paired))]]
    return {
        "min": vals[0],
        "median": mid,
        "max": vals[-1],
        "lowest_specificity_seeds": lowest,
    }

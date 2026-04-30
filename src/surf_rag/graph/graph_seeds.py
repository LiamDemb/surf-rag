"""Canonical Personalized PageRank restart: semantic softmax × IDF over query-linked entities."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from surf_rag.entity_matching.types import SeedCandidate


def _entity_display(node_id: str) -> str:
    return node_id[2:] if node_id.startswith("E:") else node_id


def _cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _group_candidates_by_node(
    candidates: List[SeedCandidate],
) -> Dict[str, List[SeedCandidate]]:
    by_node: Dict[str, List[SeedCandidate]] = defaultdict(list)
    for c in candidates:
        if c.node_id and c.graph_present:
            by_node[c.node_id].append(c)
    return dict(by_node)


def compute_restart_distribution_canonical(
    _graph: Any,
    query: str,
    enriched_candidates: List[SeedCandidate],
    extracted_norms: List[str],
    embedder: Any,
    *,
    softmax_temperature: float,
    df_ref: float | None = None,
    cache: Dict[str, Any] | None = None,
) -> tuple[Dict[str, float], Dict[str, Any]]:
    """Restart masses ∝ ``exp(cos(q,e)/tau) × IDF``, L1-normalized over query-linked entities only.

    Softmax-like activation uses ``exp((phi - max_phi)/tau)`` per candidate entity node (stable),
    multiplied by ``log((Nref+1)/(df+1))``, then renormalized to sum to 1.

    ``extracted_norms`` is unused for weighting but kept for API compatibility with callers.
    """
    _ = extracted_norms  # retained for call-site compatibility

    dfs = [float(c.df) for c in enriched_candidates if c.df >= 0]
    n_ref = float(df_ref if df_ref is not None else (max(dfs) + 1.0 if dfs else 128.0))

    by_node = _group_candidates_by_node(enriched_candidates)
    if not by_node:
        return {}, {
            "mode": "canonical_semantic_softmax_idf",
            "df_reference": n_ref,
            "posterior": {},
            "per_node": {},
            "softmax_temperature": float(softmax_temperature),
        }

    tau = max(float(softmax_temperature), 1e-12)

    # Best DF per graph node (same span-resolution semantics as before).
    per_node_df: Dict[str, tuple[SeedCandidate, float, float]] = {}
    for node_id, cands in by_node.items():
        best_c = None
        best_df_pen = -1.0
        best_df = 0.0
        for c in cands:
            df_pen = math.log((n_ref + 1.0) / (float(c.df) + 1.0))
            if df_pen > best_df_pen:
                best_df_pen = df_pen
                best_c = c
                best_df = float(c.df)
        assert best_c is not None
        per_node_df[node_id] = (best_c, best_df, best_df_pen)

    node_ids = sorted(by_node.keys())
    texts = [_entity_display(n) for n in node_ids]

    cache = cache if cache is not None else {}
    q_key = f"seed_query_emb::{query}"
    if q_key in cache:
        q_emb = cache[q_key]
    else:
        q_emb = embedder.embed_query(query)
        cache[q_key] = q_emb

    phis: List[float] = []
    for t in texts:
        ek = f"seed_entity_emb::{t}"
        if ek in cache:
            e_emb = cache[ek]
        else:
            e_emb = embedder.embed_query(t)
            cache[ek] = e_emb
        phis.append(
            _cosine_np(
                np.asarray(q_emb, dtype=np.float64), np.asarray(e_emb, dtype=np.float64)
            )
        )

    # z_i = exp((phi_i - max_phi) / tau); w_i = z_i * IDF_i; normalize.
    max_phi = max(phis)
    z_list = [math.exp((phi - max_phi) / tau) for phi in phis]
    raw: Dict[str, float] = {}
    per_node_meta: Dict[str, Any] = {}

    for idx, node_id in enumerate(node_ids):
        _, df_used, idf_pen = per_node_df[node_id]
        z_i = z_list[idx]
        w_i = z_i * idf_pen
        raw[node_id] = float(w_i)
        per_node_meta[node_id] = {
            "canonical_norm": getattr(per_node_df[node_id][0], "canonical_norm", None),
            "cosine_query_entity": float(phis[idx]),
            "exp_activation": float(z_i),
            "df": df_used,
            "idf_log_ratio": float(idf_pen),
            "unnormalized_weight": float(w_i),
        }

    total = float(sum(raw.values()))
    if total <= 0:
        nn = max(len(raw), 1)
        posterior = {k: 1.0 / float(nn) for k in raw}
    else:
        posterior = {k: float(v) / total for k, v in raw.items()}

    diag: Dict[str, Any] = {
        "mode": "canonical_semantic_softmax_idf",
        "df_reference": n_ref,
        "softmax_temperature": float(tau),
        "unnormalized_mass": {k: float(v) for k, v in raw.items()},
        "posterior": {k: float(v) for k, v in posterior.items()},
        "per_node": per_node_meta,
    }
    return posterior, diag

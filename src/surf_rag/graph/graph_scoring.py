"""Graph scoring: bundle embedding score + canonical Personalized PageRank retrieval."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from surf_rag.graph.graph_beam_paths import edge_support_boost
from surf_rag.graph.graph_types import EvidenceBundle, GraphHop, GraphPath
from surf_rag.core.scoring_config import ScoringConfig


def cosine(a, b) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)

    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def clean_rel(rel: str) -> str:
    """Normalize relation text for embedding."""
    return rel.replace("inv:", "").replace("_", " ")


def node_text(x: str) -> str:
    """Strip entity prefix if present."""
    return x[2:] if isinstance(x, str) and x.startswith("E:") else x


def get_cached_embedding(text: str, embedder, cache: dict | None = None):
    """Embed text with optional caching."""
    if cache is not None and text in cache:
        return cache[text]

    emb = embedder.embed_query(text)

    if cache is not None:
        cache[text] = emb

    return emb


def pred_sim(
    query_emb,
    hop,
    embedder,
    cache: dict | None = None,
) -> float:
    """Predicate-only similarity."""
    rel_text = clean_rel(hop.relation)
    rel_emb = get_cached_embedding(rel_text, embedder, cache)
    return cosine(query_emb, rel_emb)


def triple_sim(
    query_emb,
    hop,
    embedder,
    cache: dict | None = None,
) -> float:
    """Full-hop similarity."""
    src = node_text(hop.source)
    tgt = node_text(hop.target)
    rel_text = clean_rel(hop.relation)

    hop_text = f"{src} {rel_text} {tgt}"
    hop_emb = get_cached_embedding(hop_text, embedder, cache)
    return cosine(query_emb, hop_emb)


def score_bundle(
    query: str,
    bundle: EvidenceBundle,
    graph,
    corpus,
    embedder,
    config: ScoringConfig,
    cache=None,
    debug: bool = False,
):
    """
    Score an EvidenceBundle using embedding similarity (used by tests / exploration).

    Tunable params come only from ``config.local_pred_weight``, ``bundle_pred_weight``,
    ``length_penalty`` — orthogonal to canonical PPR chunk retrieval.
    """
    grounded_hops = bundle.grounded_hops
    if not grounded_hops:
        return 0.0, {
            "s_pred": 0.0,
            "s_triple": 0.0,
            "s_len": 0.0,
            "hop_scores": [],
        }

    local_triple_weight = 1.0 - config.local_pred_weight
    bundle_triple_weight = 1.0 - config.bundle_pred_weight

    q_emb = get_cached_embedding(f"query::{query}", embedder, cache)

    pred_scores = []
    triple_scores = []
    local_scores = []
    hop_debug = []

    for grounded_hop in grounded_hops:
        hop = grounded_hop.hop

        s_pred = pred_sim(q_emb, hop, embedder, cache)
        s_triple = triple_sim(q_emb, hop, embedder, cache)

        s_local = config.local_pred_weight * s_pred + local_triple_weight * s_triple

        pred_scores.append(s_pred)
        triple_scores.append(s_triple)
        local_scores.append(s_local)

        if debug:
            hop_debug.append(
                {
                    "text": f"{node_text(hop.source)} {clean_rel(hop.relation)} {node_text(hop.target)}",
                    "s_pred": s_pred,
                    "s_triple": s_triple,
                    "s_local": s_local,
                }
            )

    mean_pred = float(np.mean(pred_scores))
    mean_triple = float(np.mean(triple_scores))

    if len(grounded_hops) > 1:
        weakest_hop = float(np.min(local_scores))
        length_bonus = config.length_penalty * (len(grounded_hops) - 1) * weakest_hop
    else:
        length_bonus = 0.0

    total_score = (
        config.bundle_pred_weight * mean_pred
        + bundle_triple_weight * mean_triple
        + length_bonus
    )

    details = {
        "s_pred": mean_pred,
        "s_triple": mean_triple,
        "s_len": length_bonus,
        "hop_scores": local_scores,
    }

    if debug:
        print("============")
        print(bundle)
        for i, h in enumerate(hop_debug):
            print(
                f"hop {i+1}: {h['text']}\n"
                f"  s_pred   = {h['s_pred']:.4f}\n"
                f"  s_triple = {h['s_triple']:.4f}\n"
                f"  s_local  = {h['s_local']:.4f}"
            )
        print(
            f"mean_pred   = {mean_pred:.4f}\n"
            f"mean_triple = {mean_triple:.4f}\n"
            f"len_bonus   = {length_bonus:.4f}\n"
            f"total_score = {total_score:.4f}\n"
        )

    return total_score, details


def _collect_entity_scope(paths: list[GraphPath], seeds: set[str]) -> list[str]:
    ents = set(seeds)
    for path in paths:
        ents.add(path.start_node)
        for hop in path.hops:
            ents.add(hop.source)
            ents.add(hop.target)
    return sorted(e for e in ents if isinstance(e, str) and e.startswith("E:"))


def _rel_degree(graph: Any, ent: str) -> int:
    """Undirected count of ``rel`` edges touching ``ent`` within ``graph``."""
    if ent not in graph:
        return 0
    n = 0
    for succ in graph.successors(ent):
        if isinstance(succ, str) and succ.startswith("E:"):
            data = graph[ent].get(succ) or {}
            if data.get("kind") == "rel":
                n += 1
    for pred in graph.predecessors(ent):
        if isinstance(pred, str) and pred.startswith("E:"):
            data = graph[pred].get(ent) or {}
            if data.get("kind") == "rel":
                n += 1
    return int(n)


def _collect_entity_scope_capped(
    paths: list[GraphPath],
    seeds: set[str],
    graph: Any,
    max_entities: int,
) -> list[str]:
    """Prefer seeds, then higher relational degree, then stable ``E:`` id tie-break."""
    base = _collect_entity_scope(paths, seeds)
    if len(base) <= max_entities:
        return sorted(base)
    ranked = sorted(
        base,
        key=lambda e: (-float(e in seeds), -float(_rel_degree(graph, e)), e),
    )
    return sorted(ranked[:max_entities])


def _edge_support_aggregate(graph: Any, u: str, v: str) -> float:
    """Maximum corpus support boost across relation labels on one directed ``rel`` edge."""
    if not graph.has_edge(u, v):
        return 0.0
    data = graph[u][v]
    if data.get("kind") != "rel":
        return 0.0
    labels = data.get("labels") or set()
    if labels == {"instance_of"}:
        return 0.0
    best = 0.0
    for pred in labels:
        gh = GraphHop(source=u, relation=str(pred), target=v)
        best = max(best, float(edge_support_boost(graph, gh)))
    return float(best)


def _normalize_sparse_row(weights: dict[int, float]) -> dict[int, float]:
    tot = float(sum(weights.values()))
    if tot <= 0:
        return {}
    inv = 1.0 / tot
    return {k: float(v) * inv for k, v in weights.items()}


def _entity_rel_out_distribution(
    graph: Any,
    ent: str,
    idx_map: dict[str, int],
    transition_mode: str,
) -> dict[int, float]:
    """Distribution over scoped entity column indices reachable via ``rel`` edges from ``ent``."""
    weights: dict[int, float] = defaultdict(float)
    if ent not in graph:
        return {}

    if transition_mode == "support":
        for succ in graph.successors(ent):
            if isinstance(succ, str) and succ.startswith("E:"):
                j = idx_map.get(succ)
                if j is None:
                    continue
                w = _edge_support_aggregate(graph, ent, succ)
                if w > 0:
                    weights[j] = max(weights[j], w)
        for pred in graph.predecessors(ent):
            if isinstance(pred, str) and pred.startswith("E:"):
                j = idx_map.get(pred)
                if j is None:
                    continue
                w = _edge_support_aggregate(graph, pred, ent)
                if w > 0:
                    weights[j] = max(weights[j], w)
        nw = _normalize_sparse_row(dict(weights))
        if nw:
            return nw
        # Fallback: uniform over rel neighbors in scope
        neigh_js: list[int] = []
        for succ in graph.successors(ent):
            if isinstance(succ, str) and succ.startswith("E:"):
                j = idx_map.get(succ)
                if j is None:
                    continue
                data = graph[ent].get(succ) or {}
                if data.get("kind") == "rel":
                    neigh_js.append(j)
        for pred in graph.predecessors(ent):
            if isinstance(pred, str) and pred.startswith("E:"):
                j = idx_map.get(pred)
                if j is None:
                    continue
                data = graph[pred].get(ent) or {}
                if data.get("kind") == "rel":
                    neigh_js.append(j)
        uniq = sorted(set(neigh_js))
        if not uniq:
            return {}
        u = 1.0 / float(len(uniq))
        return {j: u for j in uniq}

    # uniform over rel neighbors in scope
    neigh_js = []
    for succ in graph.successors(ent):
        if isinstance(succ, str) and succ.startswith("E:"):
            j = idx_map.get(succ)
            if j is None:
                continue
            data = graph[ent].get(succ) or {}
            if data.get("kind") == "rel":
                neigh_js.append(j)
    for pred in graph.predecessors(ent):
        if isinstance(pred, str) and pred.startswith("E:"):
            j = idx_map.get(pred)
            if j is None:
                continue
            data = graph[pred].get(ent) or {}
            if data.get("kind") == "rel":
                neigh_js.append(j)
    uniq = sorted(set(neigh_js))
    if not uniq:
        return {}
    u = 1.0 / float(len(uniq))
    return {j: u for j in uniq}


def _chunk_neighbors_scoped(
    graph: Any,
    ent: str,
    chunk_nodes: set[str],
) -> list[str]:
    """Chunk nodes adjacent to ``ent`` via ``appears_in`` (either direction)."""
    out: list[str] = []
    if ent not in graph:
        return out
    for succ in graph.successors(ent):
        if succ in chunk_nodes and graph[ent][succ].get("kind") == "appears_in":
            out.append(succ)
    for pred in graph.predecessors(ent):
        if pred in chunk_nodes and graph[pred][ent].get("kind") == "appears_in":
            out.append(pred)
    return sorted(set(out))


def _collect_adjacent_chunk_nodes(graph: Any, entity_nodes: list[str]) -> list[str]:
    s: set[str] = set()
    for ent in entity_nodes:
        if ent not in graph:
            continue
        for succ in graph.successors(ent):
            if isinstance(succ, str) and succ.startswith("C:"):
                if graph[ent][succ].get("kind") == "appears_in":
                    s.add(succ)
        for pred in graph.predecessors(ent):
            if isinstance(pred, str) and pred.startswith("C:"):
                if graph[pred][ent].get("kind") == "appears_in":
                    s.add(pred)
    return sorted(s)


def _build_heterogeneous_transition_matrix(
    graph: Any,
    ordered_nodes: list[str],
    idx_map: dict[str, int],
    config: ScoringConfig,
) -> np.ndarray:
    """Row-stochastic transitions over entity + chunk nodes with explicit E↔E vs E↔C mixing."""
    n = len(ordered_nodes)
    P = np.zeros((n, n), dtype=np.float64)
    lambda_ec = float(min(1.0, max(0.0, config.graph_entity_chunk_edge_weight)))
    lambda_ee = 1.0 - lambda_ec
    chunk_node_set = {x for x in ordered_nodes if x.startswith("C:")}
    mode = config.graph_transition_mode

    for i, node in enumerate(ordered_nodes):
        if node.startswith("E:"):
            rel_dist = _entity_rel_out_distribution(graph, node, idx_map, mode)
            ck_nodes = _chunk_neighbors_scoped(graph, node, chunk_node_set)
            ck_js = [idx_map[c] for c in ck_nodes if c in idx_map]

            has_rel = len(rel_dist) > 0
            has_ck = len(ck_js) > 0

            if has_rel and has_ck:
                for j, p in rel_dist.items():
                    P[i, j] += lambda_ee * float(p)
                uc = lambda_ec / float(len(ck_js))
                for j in ck_js:
                    P[i, j] += uc
            elif has_rel:
                for j, p in rel_dist.items():
                    P[i, j] += float(p)
            elif has_ck:
                uc = 1.0 / float(len(ck_js))
                for j in ck_js:
                    P[i, j] += uc
            else:
                P[i, i] = 1.0

        elif node.startswith("C:"):
            ent_js: list[int] = []
            if node in graph:
                for pred in graph.predecessors(node):
                    if isinstance(pred, str) and pred.startswith("E:"):
                        j = idx_map.get(pred)
                        if j is None:
                            continue
                        data = graph[pred].get(node) or {}
                        if data.get("kind") == "appears_in":
                            ent_js.append(j)
                for succ in graph.successors(node):
                    if isinstance(succ, str) and succ.startswith("E:"):
                        j = idx_map.get(succ)
                        if j is None:
                            continue
                        data = graph[node].get(succ) or {}
                        if data.get("kind") == "appears_in":
                            ent_js.append(j)
            ent_js = sorted(set(ent_js))
            if ent_js:
                u = 1.0 / float(len(ent_js))
                for j in ent_js:
                    P[i, j] += u
            else:
                P[i, i] = 1.0
        else:
            P[i, i] = 1.0

    row_sums = P.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-7):
        raise RuntimeError(
            "heterogeneous transition rows must be stochastic; "
            f"max deviation={float(np.max(np.abs(row_sums - 1.0)))}"
        )
    return P


def _build_entity_transition_matrix_uniform(
    graph: Any, entity_nodes: list[str]
) -> np.ndarray:
    """Row-stochastic uniform transitions over entity neighbors on ``rel`` edges."""
    n = len(entity_nodes)
    idx = {e: i for i, e in enumerate(entity_nodes)}
    P = np.zeros((n, n), dtype=np.float64)
    for i, ent in enumerate(entity_nodes):
        neigh_indices = []
        for succ in graph.successors(ent):
            if isinstance(succ, str) and succ.startswith("E:") and succ in idx:
                data = graph[ent][succ]
                if data.get("kind") == "rel":
                    neigh_indices.append(idx[succ])
        for pred in graph.predecessors(ent):
            if isinstance(pred, str) and pred.startswith("E:") and pred in idx:
                data = graph[pred][ent]
                if data.get("kind") == "rel":
                    neigh_indices.append(idx[pred])
        uniq = sorted(set(neigh_indices))
        if not uniq:
            P[i, i] = 1.0
        else:
            w = 1.0 / float(len(uniq))
            for j in uniq:
                P[i, j] += w
    return P


def _build_entity_transition_matrix_support_normalized(
    graph: Any,
    entity_nodes: list[str],
) -> np.ndarray:
    """Row-stochastic transitions proportional to corpus edge-support counts."""
    n = len(entity_nodes)
    idx = {e: i for i, e in enumerate(entity_nodes)}
    P = np.zeros((n, n), dtype=np.float64)

    for i, ent in enumerate(entity_nodes):
        weights: dict[int, float] = defaultdict(float)
        for succ in graph.successors(ent):
            if isinstance(succ, str) and succ.startswith("E:"):
                j = idx.get(succ)
                if j is None:
                    continue
                w = _edge_support_aggregate(graph, ent, succ)
                if w > 0:
                    weights[j] = max(weights[j], w)
        for pred in graph.predecessors(ent):
            if isinstance(pred, str) and pred.startswith("E:"):
                j = idx.get(pred)
                if j is None:
                    continue
                w = _edge_support_aggregate(graph, pred, ent)
                if w > 0:
                    weights[j] = max(weights[j], w)

        total = float(sum(weights.values()))
        if total <= 0:
            neigh_indices: list[int] = []
            for succ in graph.successors(ent):
                if isinstance(succ, str) and succ.startswith("E:"):
                    j = idx.get(succ)
                    if j is None:
                        continue
                    data = graph[ent].get(succ) or {}
                    if data.get("kind") == "rel":
                        neigh_indices.append(j)
            for pred in graph.predecessors(ent):
                if isinstance(pred, str) and pred.startswith("E:"):
                    j = idx.get(pred)
                    if j is None:
                        continue
                    data = graph[pred].get(ent) or {}
                    if data.get("kind") == "rel":
                        neigh_indices.append(j)
            uniq_js = sorted(set(neigh_indices))
            if not uniq_js:
                P[i, i] = 1.0
            else:
                u = 1.0 / float(len(uniq_js))
                for j in uniq_js:
                    P[i, j] = u
            continue
        inv = 1.0 / total
        for j, w in weights.items():
            P[i, j] = float(w) * inv
    return P


def _restart_row_vector(
    entity_nodes: list[str],
    restart: dict[str, float],
) -> np.ndarray:
    n = len(entity_nodes)
    idx = {e: i for i, e in enumerate(entity_nodes)}
    r = np.zeros((n,), dtype=np.float64)
    for ent, mass in restart.items():
        j = idx.get(ent)
        if j is not None:
            r[j] += float(max(mass, 0.0))
    s = float(np.sum(r))
    if s <= 0:
        r[:] = 1.0 / float(max(n, 1))
    else:
        r /= s
    return r


def run_local_ppr(
    graph,
    entity_nodes: list[str],
    restart: dict[str, float],
    *,
    alpha: float,
    max_iter: int,
    tol: float,
    transition_matrix: np.ndarray | None = None,
    transition_mode: str = "support",
) -> tuple[np.ndarray, dict]:
    """Personalized PageRank with teleport distribution ``restart`` (sums to 1)."""
    if not entity_nodes:
        return np.zeros((0,), dtype=np.float64), {"iterations": 0, "residual": None}

    if transition_matrix is None:
        if transition_mode == "uniform":
            P = _build_entity_transition_matrix_uniform(graph, entity_nodes)
        else:
            P = _build_entity_transition_matrix_support_normalized(graph, entity_nodes)
    else:
        P = transition_matrix
    v = _restart_row_vector(entity_nodes, restart)
    pi = v.copy()
    diag: dict = {
        "iterations": 0,
        "residual": None,
        "damping": alpha,
        "transition": transition_mode,
    }
    for it in range(max_iter):
        pi_new = alpha * (pi @ P) + (1.0 - alpha) * v
        residual = float(np.linalg.norm(pi_new - pi, ord=1))
        diag["iterations"] = it + 1
        diag["residual"] = residual
        if residual < tol:
            break
        pi = pi_new
    return pi, diag


def aggregate_chunk_scores_average_incidence(
    graph: Any,
    entity_nodes: list[str],
    pi: np.ndarray,
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, Any]]:
    """Average PPR mass over entities appearing in each chunk: πᵀ P̂ with row-normalized incidence."""
    chunk_to_entities: dict[str, dict[str, float]] = {}
    for i, ent in enumerate(entity_nodes):
        mass = float(pi[i])
        if mass <= 0:
            continue
        for _, succ, data in graph.out_edges(ent, data=True):
            if data.get("kind") != "appears_in":
                continue
            if not isinstance(succ, str) or not succ.startswith("C:"):
                continue
            cid = succ[2:]
            dmap = chunk_to_entities.setdefault(cid, {})
            dmap[ent] = dmap.get(ent, 0.0) + mass

    scores: dict[str, float] = {}
    breakdown: dict[str, Any] = {}
    for cid, ent_mass in chunk_to_entities.items():
        if not ent_mass:
            continue
        mass_sum = float(sum(ent_mass.values()))
        uniq = float(len(ent_mass))
        denom = max(1e-12, uniq)
        focus = mass_sum / denom
        scores[cid] = focus
        top_entities = sorted(
            ((float(v), k) for k, v in ent_mass.items()),
            reverse=True,
        )[:8]
        breakdown[cid] = {
            "sum_mass": mass_sum,
            "unique_entity_count": int(uniq),
            "avg_mass": focus,
            "top_entities": [{"entity": e, "mass": m} for m, e in top_entities],
        }

    return scores, chunk_to_entities, {"chunk_breakdown": breakdown}


def canonical_ppr_rank_chunks(
    graph: Any,
    paths: list[GraphPath],
    seeds: set[str],
    normalized_restart: dict[str, float],
    *,
    config: ScoringConfig,
) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    """Heterogeneous entity+chunk PPR; chunk scores are stationary masses on ``C:*`` nodes."""
    entity_nodes = _collect_entity_scope_capped(
        paths,
        seeds,
        graph,
        int(config.graph_max_entities),
    )
    empty_scores: dict[str, float] = {}

    if not entity_nodes:
        return empty_scores, {}, {"reason": "empty_entity_scope"}

    chunk_nodes = _collect_adjacent_chunk_nodes(graph, entity_nodes)
    ordered_nodes = list(entity_nodes) + list(chunk_nodes)
    idx_map = {n: i for i, n in enumerate(ordered_nodes)}

    P = _build_heterogeneous_transition_matrix(graph, ordered_nodes, idx_map, config)

    pi, ppr_diag = run_local_ppr(
        graph,
        ordered_nodes,
        normalized_restart,
        alpha=config.ppr_alpha,
        max_iter=config.ppr_max_iter,
        tol=config.ppr_tol,
        transition_matrix=P,
        transition_mode="heterogeneous_entity_chunk",
    )

    pi_dict = {e: float(pi[idx_map[e]]) for e in entity_nodes}
    top_entities = sorted(
        ((float(pi[idx_map[e]]), e) for e in entity_nodes if e in idx_map),
        reverse=True,
    )[:15]

    chunk_scores: dict[str, float] = {}
    chunk_mass_diag: dict[str, Any] = {}
    for cn in chunk_nodes:
        cid = cn[2:]
        mass = float(pi[idx_map[cn]])
        chunk_scores[cid] = mass
        chunk_mass_diag[cid] = {"ppr_mass": mass}

    extra_diag = {
        "entity_count": len(entity_nodes),
        "chunk_node_count": len(chunk_nodes),
        "mode": "canonical_heterogeneous_ppr",
        "ppr": {
            **ppr_diag,
            "top_entities": [
                {"entity": ent, "mass": mass} for mass, ent in top_entities
            ],
        },
        "chunk_scoring": {"chunk_ppr_mass": chunk_mass_diag},
    }

    return chunk_scores, pi_dict, extra_diag

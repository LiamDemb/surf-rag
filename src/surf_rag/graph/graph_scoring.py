import faiss
import numpy as np
from surf_rag.graph.graph_types import EvidenceBundle
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
    """
    Predicate-only similarity.

    Example:
        relation 'born_in' -> text 'born in'
    """
    rel_text = clean_rel(hop.relation)
    rel_emb = get_cached_embedding(rel_text, embedder, cache)
    return cosine(query_emb, rel_emb)


def triple_sim(
    query_emb,
    hop,
    embedder,
    cache: dict | None = None,
) -> float:
    """
    Full-hop similarity.

    Example:
        'chung mong koo born in korea'
    """
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
    Score an EvidenceBundle using:
      - mean predicate similarity across hops
      - mean triple similarity across hops
      - conditional length bonus based on weakest local hop

    Tunable params:
      - config.local_pred_weight
      - config.bundle_pred_weight
      - config.length_penalty

    Notes:
      - graph and corpus are unused, kept for drop-in compatibility
      - local triple weight = 1 - local_pred_weight
      - bundle triple weight = 1 - bundle_pred_weight
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

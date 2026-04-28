from __future__ import annotations

from typing import Iterable, List

import networkx as nx


def build_graph(chunks: Iterable[dict]) -> nx.DiGraph:
    G = nx.DiGraph()
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        if not chunk_id:
            continue

        chunk_node = f"C:{chunk_id}"
        G.add_node(chunk_node, kind="chunk")

        # Adding `appears_in` relations
        entities = chunk.get("metadata", {}).get("entities", [])
        for ent in entities:
            norm = ent.get("norm")
            if not norm:
                continue
            enode = f"E:{norm}"
            G.add_node(enode, kind="entity", type=ent.get("type"))
            G.add_edge(enode, chunk_node, kind="appears_in")
            G.add_edge(chunk_node, enode, kind="appears_in")

        # Adding graph triples
        relations = chunk.get("metadata", {}).get("relations", []) or []
        for rel in relations:
            # Getting the subject, object, and predicate
            subj = rel.get("subj_norm")
            obj = rel.get("obj_norm")
            pred = rel.get("pred") or rel.get("predicate")
            if not subj or not obj or not pred:
                continue

            subj = f"E:{subj}"
            obj = f"E:{obj}"

            if G.has_edge(subj, obj):
                # Enrich existing edge
                edge = G[subj][obj]
                edge["labels"].add(pred)
                edge.setdefault("chunk_ids_by_label", {}).setdefault(pred, set()).add(
                    chunk_id
                )
                support_counts = edge["support_count_by_label"]
                support_counts[pred] = support_counts.get(pred, 0) + 1
            else:
                # Create new edge
                G.add_node(subj, kind="entity")
                G.add_node(obj, kind="entity")
                G.add_edge(
                    subj,
                    obj,
                    kind="rel",
                    label=pred,
                    labels={pred},
                    chunk_ids_by_label={pred: {chunk_id}},
                    support_count_by_label={pred: 1},
                )
    return G

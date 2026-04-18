from __future__ import annotations

import json
from typing import Dict, Iterable, List


def coverage_gate(samples: List[dict], chunks: List[dict]) -> Dict[str, float]:
    matched = 0
    for sample in samples:
        answers = sample.get("gold_answers") or sample.get("answers") or []
        answers = [str(a).strip().lower() for a in answers if str(a).strip()]
        found = False
        if answers:
            for chunk in chunks:
                text = chunk.get("text", "").lower()
                if any(ans in text for ans in answers):
                    found = True
                    break
        if found:
            matched += 1
    return {"coverage_rate": matched / max(len(samples), 1)}


def graph_gate(chunks: List[dict]) -> Dict[str, float]:
    ent_counts = []
    rel_counts = []
    with_rel = 0
    for chunk in chunks:
        ents = chunk.get("metadata", {}).get("entities", []) or []
        ent_counts.append(len(ents))
        rels = chunk.get("metadata", {}).get("relations", []) or []
        rel_counts.append(len(rels))
        if rels:
            with_rel += 1
    avg_ents = sum(ent_counts) / max(len(ent_counts), 1)
    avg_rels = sum(rel_counts) / max(len(rel_counts), 1)
    return {
        "avg_entities_per_chunk": avg_ents,
        "avg_relations_per_chunk": avg_rels,
        "chunks_with_relation_rate": with_rel / max(len(chunks), 1),
    }


def run_quality_gates(
    samples: List[dict],
    chunks: List[dict],
    output_path: str,
    min_coverage: float = 0.2,
) -> Dict[str, float]:
    report = {}
    report.update(coverage_gate(samples, chunks))
    report.update(graph_gate(chunks))

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    if report["coverage_rate"] < min_coverage:
        raise ValueError("Coverage gate failed. Low answer evidence rate.")
    return report

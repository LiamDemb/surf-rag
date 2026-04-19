from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List

import pandas as pd


def build_entity_lexicon(chunks: Iterable[dict]) -> pd.DataFrame:
    surface_forms: Dict[str, Counter] = defaultdict(Counter)
    df_counts: Counter = Counter()
    qid_candidates: Dict[str, Counter] = defaultdict(Counter)

    for chunk in chunks:
        ents = chunk.get("metadata", {}).get("entities", [])
        seen_norms = set()
        for ent in ents:
            norm = ent.get("norm")
            surface = ent.get("surface")
            qid = ent.get("qid")
            if not norm:
                continue
            if surface:
                surface_forms[norm][surface] += 1
            if qid:
                qid_candidates[norm][qid] += 1
            seen_norms.add(norm)
        for norm in seen_norms:
            df_counts[norm] += 1

    rows: List[dict] = []
    for norm, counter in surface_forms.items():
        rows.append(
            {
                "norm": norm,
                "surface_forms": [s for s, _ in counter.most_common(5)],
                "qid_candidates": [
                    q for q, _ in qid_candidates.get(norm, Counter()).most_common(3)
                ],
                "df": df_counts.get(norm, 0),
            }
        )
    return pd.DataFrame(rows)

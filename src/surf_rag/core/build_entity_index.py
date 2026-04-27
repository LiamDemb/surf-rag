"""Build FAISS index over entity norms for vector similarity matching.

Reads entity_lexicon.parquet and produces entity_index.faiss + entity_meta.parquet.
Uses the same SentenceTransformer model as chunk embeddings.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
import pandas as pd
from surf_rag.core.model_cache import get_sentence_transformer


def build_entity_index(
    lexicon_path: str,
    output_index_path: str,
    output_meta_path: str,
    model_name: str = "all-MiniLM-L6-v2",
    norm_col: str = "norm",
    surface_forms_col: str = "surface_forms",
) -> None:
    """Build FAISS index from entity lexicon.

    Each entity is embedded as: "norm" or "norm surface1 surface2" to improve
    matching when query uses alternate phrasing.
    """
    df = pd.read_parquet(lexicon_path, columns=[norm_col, surface_forms_col])
    texts: List[str] = []
    meta_rows: List[dict] = []
    for _, row in df.iterrows():
        norm = str(row[norm_col] or "").strip()
        if not norm:
            continue
        surfaces = row.get(surface_forms_col)
        if isinstance(surfaces, list) and surfaces:
            combined = f"{norm} {' '.join(str(s) for s in surfaces[:3])}".strip()
        elif isinstance(surfaces, str) and surfaces:
            combined = f"{norm} {surfaces}".strip()
        else:
            combined = norm
        texts.append(combined)
        meta_rows.append({"row_id": len(meta_rows), "norm": norm})

    if not texts:
        index = faiss.IndexFlatIP(384)  # all-MiniLM-L6-v2 dim
        faiss.write_index(index, output_index_path)
        pd.DataFrame(meta_rows).to_parquet(output_meta_path, index=False)
        return

    model = get_sentence_transformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, output_index_path)
    pd.DataFrame(meta_rows).to_parquet(output_meta_path, index=False)

from __future__ import annotations

from typing import Iterable, List

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def build_faiss_index(
    chunks: Iterable[dict],
    output_index_path: str,
    output_meta_path: str,
    model_name: str = "all-MiniLM-L6-v2",
) -> None:
    chunks_list = list(chunks)
    texts = [chunk["text"] for chunk in chunks_list]
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, output_index_path)

    meta_rows = []
    for idx, chunk in enumerate(chunks_list):
        meta = chunk.get("metadata", {})
        meta_rows.append(
            {
                "row_id": idx,
                "chunk_id": chunk.get("chunk_id"),
                "year_min": meta.get("year_min"),
                "year_max": meta.get("year_max"),
                "years": meta.get("years", []),
                "source": chunk.get("source"),
            }
        )
    pd.DataFrame(meta_rows).to_parquet(output_meta_path, index=False)

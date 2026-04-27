"""Entity vector index for similarity-based graph node matching."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
from surf_rag.core.model_cache import get_sentence_transformer


class EntityIndexStore:
    """FAISS-backed entity index for vector similarity search.

    Given a query mention (e.g. "Einstein"), returns top-K entity norms
    that exist in the graph for use as traversal seeds.
    """

    def __init__(
        self,
        index_path: str,
        meta_path: str,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self._index = None
        self._meta_df = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._index is None:
            self._index = faiss.read_index(str(self.index_path))
        if self._meta_df is None:
            self._meta_df = pd.read_parquet(self.meta_path)
        if self._model is None:
            self._model = get_sentence_transformer(self.model_name)

    def search(
        self,
        mention: str,
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> List[Tuple[str, float]]:
        """Return (norm, score) for mentions similar to entity norms.

        Scores are cosine similarities in [0, 1] (assuming normalized embeddings).
        """
        if not mention or not mention.strip():
            return []
        self._ensure_loaded()
        if self._meta_df.empty:
            return []

        vec = self._model.encode([mention.strip()], normalize_embeddings=True)
        vec = np.array(vec, dtype="float32")
        k = min(top_k, self._index.ntotal)
        if k <= 0:
            return []
        scores, row_ids = self._index.search(vec, k)
        result: List[Tuple[str, float]] = []
        for s, rid in zip(scores[0].tolist(), row_ids[0].tolist()):
            if rid < 0 or rid >= len(self._meta_df):
                continue
            score = float(s)
            if score < threshold:
                continue
            norm = str(self._meta_df.iloc[int(rid)]["norm"])
            if norm:
                result.append((norm, score))
        return result

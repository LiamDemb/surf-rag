from __future__ import annotations
from dataclasses import dataclass
from typing import List, Protocol

import numpy as np

from surf_rag.core.model_cache import get_sentence_transformer


class Embedder(Protocol):
    def embed_query(self, text: str) -> np.ndarray: ...


@dataclass
class SentenceTransformersEmbedder:
    model_name: str

    def __post_init__(self) -> None:
        self.model = get_sentence_transformer(self.model_name)

    def embed_query(self, text: str) -> np.ndarray:
        vector = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        return vector.astype("float32")

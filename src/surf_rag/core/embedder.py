from __future__ import annotations
from dataclasses import dataclass
from typing import List, Protocol

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder(Protocol):
    def embed_query(self, text: str) -> np.ndarray: ...


@dataclass
class SentenceTransformersEmbedder:
    model_name: str

    def __post_init__(self) -> None:
        self.model = SentenceTransformer(self.model_name)

    def embed_query(self, text: str) -> np.ndarray:
        vector = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        return vector.astype("float32")

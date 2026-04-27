#!/usr/bin/env python3
"""
Download and warm required models (spaCy, SentenceTransformers).
Run once with network access. Models are cached for offline use.
Use HF_HOME / TRANSFORMERS_CACHE to control HuggingFace cache location.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SPACY_MODEL = os.environ.get("SPACY_MODEL", "en_core_web_sm")
MODEL_NAME = os.environ.get("MODEL_NAME", "all-MiniLM-L6-v2")
CROSS_ENCODER_MODEL = os.environ.get(
    "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def main() -> int:
    hf_home = os.environ.get("HF_HOME", "data/processed/hf_cache")
    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault(
        "TRANSFORMERS_CACHE",
        os.environ.get("TRANSFORMERS_CACHE") or os.path.join(hf_home, "transformers"),
    )
    os.makedirs(hf_home, exist_ok=True)
    logger.info("HF cache: %s", os.environ.get("HF_HOME"))

    # spaCy
    logger.info("Downloading spaCy model: %s", SPACY_MODEL)
    rc = subprocess.run(
        [sys.executable, "-m", "spacy", "download", SPACY_MODEL],
        capture_output=True,
        text=True,
    )
    if rc.returncode != 0:
        logger.error("spaCy download failed: %s", rc.stderr)
        return 1
    logger.info("spaCy model ready.")

    # SentenceTransformer + CrossEncoder (shared process cache after first load)
    from surf_rag.core.model_cache import get_cross_encoder, get_sentence_transformer

    logger.info("Downloading SentenceTransformer model: %s", MODEL_NAME)
    get_sentence_transformer(MODEL_NAME)
    logger.info("SentenceTransformer model ready.")

    logger.info("Downloading CrossEncoder model: %s", CROSS_ENCODER_MODEL)
    get_cross_encoder(CROSS_ENCODER_MODEL)
    logger.info("CrossEncoder model ready.")

    # tiktoken (for chunking - cl100k_base aligns with OpenAI models)
    logger.info("Downloading tiktoken encoding: cl100k_base")
    import tiktoken

    tiktoken.get_encoding("cl100k_base")
    logger.info("tiktoken encoding ready.")

    logger.info("All models downloaded. Caches can be reused offline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

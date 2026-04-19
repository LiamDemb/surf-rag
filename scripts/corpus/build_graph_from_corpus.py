"""Rebuild graph pickle from a corpus.jsonl file.

Usage:
    poetry run python scripts/corpus/build_graph_from_corpus.py --corpus data/processed/corpus_llm.jsonl --graph-out data/processed/graph_llm.pkl

Use after collect_llm_ie_batch.py or as part of build-corpus to produce graph.pkl from the IE-enriched corpus.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from surf_rag.core.build_graph import build_graph

logger = logging.getLogger(__name__)


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild graph pickle from corpus.jsonl.",
    )
    parser.add_argument(
        "--corpus",
        required=True,
        help="Path to corpus.jsonl.",
    )
    parser.add_argument(
        "--graph-out",
        required=True,
        help="Output path for graph.pkl.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    corpus_path = Path(args.corpus)
    if not corpus_path.is_file():
        logger.error("Corpus file not found: %s", corpus_path)
        return 1

    chunks = list(_iter_jsonl(corpus_path))
    logger.info("Loaded %d chunks from %s", len(chunks), corpus_path)

    graph = build_graph(chunks)
    graph_path = Path(args.graph_out)
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(graph, graph_path)
    logger.info(
        "Wrote graph to %s (%d nodes, %d edges)",
        graph_path,
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

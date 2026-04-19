"""Build wiki_titles.jsonl from corpus or docstore.

Produces a deterministic list of Wikipedia page titles for LLM seed anchoring.
Can be run standalone when regenerating the seed without a full corpus build.

Usage:
    poetry run python scripts/corpus/build_wiki_title_seed.py --corpus data/processed/corpus.jsonl
    poetry run python scripts/corpus/build_wiki_title_seed.py --output-dir data/processed
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from surf_rag.core.wiki_title_matcher import write_wiki_titles

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
        description="Build wiki_titles.jsonl from corpus or existing doc list.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        help="Path to corpus.jsonl (extract titles from chunks).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for wiki_titles.jsonl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Explicit output path (overrides --output-dir/wiki_titles.jsonl).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    titles: list[str] = []
    if args.corpus and args.corpus.is_file():
        for chunk in _iter_jsonl(args.corpus):
            t = chunk.get("title")
            if t and str(t).strip():
                titles.append(str(t).strip())
    else:
        logger.warning("No valid corpus path. Provide --corpus to extract titles.")
        return 1

    unique = list(dict.fromkeys(titles))
    output_path = args.output or (args.output_dir / "wiki_titles.jsonl")
    write_wiki_titles(unique, output_path, format="jsonl")
    logger.info("Wrote %d unique titles to %s", len(unique), output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

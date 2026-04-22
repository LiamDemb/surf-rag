"""Load benchmark JSONL and join raw NQ / 2Wiki datasets into corpus-build samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Tuple

from surf_rag.core.schemas import sha256_text


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_benchmark_by_source(path: str | Path) -> Dict[str, Dict[str, dict]]:
    """Load benchmark and group by dataset_source. Only sources with items are included."""
    by_source: Dict[str, Dict[str, dict]] = {}
    for item in iter_jsonl(path):
        source = item["dataset_source"]
        by_source.setdefault(source, {})[item["question_id"]] = item
    return by_source


def question_text_from_row(row: dict, source: str) -> str | None:
    if source == "nq":
        question_block = row.get("question")
        return (
            row.get("question_text")
            or row.get("questionText")
            or (
                question_block.get("text") if isinstance(question_block, dict) else None
            )
            or (question_block if isinstance(question_block, str) else None)
        )
    return row.get("question") or row.get("query")


def build_samples(
    benchmark_by_source: Mapping[str, Mapping[str, dict]],
    paths_by_source: Mapping[str, str],
) -> List[dict]:
    """Build samples from benchmark and raw dataset files. Only processes sources with paths."""
    samples: List[dict] = []

    nq_path = paths_by_source.get("nq")
    if nq_path:
        for row in iter_jsonl(nq_path):
            question = question_text_from_row(row, "nq")
            if not question:
                continue
            qid = sha256_text(question)
            bench = benchmark_by_source.get("nq", {}).get(qid)
            if not bench:
                continue
            samples.append(
                {
                    "source": "nq",
                    "question_id": qid,
                    "question": question,
                    "gold_answers": bench.get("gold_answers", []),
                    "gold_support_sentences": bench.get("gold_support_sentences", []),
                    "document": row.get("document"),
                    "document_html": row.get("document_html"),
                    "document_title": row.get("document_title"),
                    "title": row.get("title"),
                }
            )

    twowiki_path = paths_by_source.get("2wiki")
    if twowiki_path:
        for row in iter_jsonl(twowiki_path):
            question = question_text_from_row(row, "2wiki")
            if not question:
                continue
            qid = sha256_text(question)
            bench = benchmark_by_source.get("2wiki", {}).get(qid)
            if not bench:
                continue
            supporting_facts = row.get("supporting_facts") or {}
            if not supporting_facts:
                continue
            samples.append(
                {
                    "source": "2wiki",
                    "question_id": qid,
                    "question": question,
                    "gold_answers": bench.get("gold_answers", []),
                    "gold_support_sentences": bench.get("gold_support_sentences", []),
                    "supporting_facts": supporting_facts,
                }
            )

    return samples


def resolve_raw_paths_for_benchmark_sources(
    sources_in_benchmark: set[str],
    *,
    nq_path: str | None,
    wiki2_path: str | None,
) -> Tuple[Dict[str, str], List[str]]:
    """
    Map benchmark sources to raw JSONL paths.

    Returns (paths_by_source, missing_env_names) where missing lists required paths
    that were not provided for a source present in the benchmark.
    """
    path_env_map = {
        "nq": ("NQ_PATH", nq_path),
        "2wiki": ("2WIKI_PATH", wiki2_path),
    }
    paths_by_source: Dict[str, str] = {}
    missing: List[str] = []
    for source in sources_in_benchmark:
        env_name, path = path_env_map.get(source, (None, None))
        if path and str(path).strip():
            paths_by_source[source] = path.strip()
        else:
            missing.append(env_name or source)
    return paths_by_source, missing

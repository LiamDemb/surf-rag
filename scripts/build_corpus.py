from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm.auto import tqdm

from dotenv import load_dotenv
from surf_rag.core.build_entity_index import build_entity_index
from surf_rag.core.build_faiss import build_faiss_index
from surf_rag.core.build_graph import build_graph
from surf_rag.core.alias_map import (
    CURATED_ALIASES,
    build_alias_map_from_redirects,
    normalize_alias_map,
)
from surf_rag.core.canonical_clean import (
    clean_html_to_structured_doc,
    normalize_text_for_extraction,
)
from surf_rag.core.chunking import chunk_blocks
from surf_rag.core.corpus_acquisition import (
    Budgets,
    ingest_2wiki,
    ingest_nq,
)
from surf_rag.core.corpus_schemas import CorpusChunk
from surf_rag.core.docstore import DocStore
from surf_rag.core.wiki_title_matcher import build_wiki_title_matcher, write_wiki_titles
from surf_rag.core.entity_lexicon import build_entity_lexicon
from surf_rag.core.quality_gates import run_quality_gates
from surf_rag.core.schemas import sha256_text
from surf_rag.core.wikipedia_client import WikipediaClient
from surf_rag.core.wikidata_client import WikidataClient


logger = logging.getLogger(__name__)


def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_benchmark(path: str) -> Dict[str, Dict[str, dict]]:
    """Load benchmark and group by dataset_source. Only sources with items are included."""
    by_source: Dict[str, Dict[str, dict]] = {}
    for item in _iter_jsonl(path):
        source = item["dataset_source"]
        by_source.setdefault(source, {})[item["question_id"]] = item
    return by_source


def _question_text_from_row(row: dict, source: str) -> str | None:
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


def _build_samples(
    benchmark_by_source: Dict[str, Dict[str, dict]],
    paths_by_source: Dict[str, str],
) -> List[dict]:
    """Build samples from benchmark and raw dataset files. Only processes sources with paths."""
    samples: List[dict] = []

    nq_path = paths_by_source.get("nq")
    if nq_path:
        for row in _iter_jsonl(nq_path):
            question = _question_text_from_row(row, "nq")
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
                    "document": row.get("document"),
                    "document_html": row.get("document_html"),
                    "document_title": row.get("document_title"),
                    "title": row.get("title"),
                }
            )

    twowiki_path = paths_by_source.get("2wiki")
    if twowiki_path:
        for row in _iter_jsonl(twowiki_path):
            question = _question_text_from_row(row, "2wiki")
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
                    "supporting_facts": supporting_facts,
                }
            )

    return samples


def _write_jsonl(path: Path, items: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Build unified corpus + indexes.")
    parser.add_argument("--benchmark", default=os.getenv("BENCHMARK_PATH"))
    parser.add_argument("--nq", default=os.getenv("NQ_PATH"))
    parser.add_argument("--2wiki", dest="wiki2", default=os.getenv("2WIKI_PATH"))
    parser.add_argument(
        "--output-dir", default=os.getenv("OUTPUT_DIR", "data/processed")
    )
    parser.add_argument(
        "--docstore",
        default=os.getenv("DOCSTORE_PATH", "data/processed/docstore.sqlite"),
    )
    parser.add_argument(
        "--model-name", default=os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
    )
    parser.add_argument(
        "--max-pages", type=int, default=int(os.getenv("MAX_PAGES", "12"))
    )
    parser.add_argument("--max-hops", type=int, default=int(os.getenv("MAX_HOPS", "2")))
    parser.add_argument(
        "--max-list-pages", type=int, default=int(os.getenv("MAX_LIST_PAGES", "2"))
    )
    parser.add_argument(
        "--max-country-pages",
        type=int,
        default=int(os.getenv("MAX_COUNTRY_PAGES", "1")),
    )
    parser.add_argument(
        "--chunk-min-tokens",
        type=int,
        default=int(os.getenv("CHUNK_MIN_TOKENS", "500")),
    )
    parser.add_argument(
        "--chunk-max-tokens",
        type=int,
        default=int(os.getenv("CHUNK_MAX_TOKENS", "800")),
    )
    parser.add_argument(
        "--chunk-overlap-tokens",
        type=int,
        default=int(os.getenv("CHUNK_OVERLAP_TOKENS", "100")),
    )
    args = parser.parse_args()

    if not args.benchmark or not str(args.benchmark).strip():
        raise ValueError(
            "BENCHMARK_PATH is required. Provide --benchmark or set BENCHMARK_PATH."
        )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_by_source = _load_benchmark(args.benchmark)
    sources_in_benchmark = set(benchmark_by_source.keys())

    path_env_map = {
        "nq": ("NQ_PATH", args.nq),
        "2wiki": ("2WIKI_PATH", args.wiki2),
    }
    paths_by_source: Dict[str, str] = {}
    missing = []
    for source in sources_in_benchmark:
        env_name, path = path_env_map.get(source, (None, None))
        if path and str(path).strip():
            paths_by_source[source] = path.strip()
        else:
            missing.append(env_name or source)

    if missing:
        raise ValueError(
            f"Benchmark contains sources {sorted(sources_in_benchmark)} but paths are missing "
            f"for: {', '.join(missing)}. Provide the corresponding paths via CLI or env."
        )

    logger.info(
        "Building corpus from datasets: %s", ", ".join(sorted(paths_by_source.keys()))
    )

    samples = _build_samples(benchmark_by_source, paths_by_source)

    budgets = Budgets(
        max_pages_per_question=args.max_pages,
        max_hops=args.max_hops,
        max_list_pages=args.max_list_pages,
        max_country_pages=args.max_country_pages,
    )
    docstore = DocStore(args.docstore)
    wiki = WikipediaClient()
    if wiki.oauth2_authenticated:
        logger.info(
            "Wikipedia Action API requests use OAuth2 (authenticated rate limits)"
        )
    elif "2wiki" in paths_by_source:
        logger.warning(
            "WIKIMEDIA_OAUTH2_ACCESS_TOKEN is not set; unauthenticated Wikipedia API "
            "limits are low and large 2Wiki corpus builds may hit 429 errors"
        )
    wikidata = WikidataClient()

    logger.info(
        "Ingesting %d benchmark questions (docstore + Wikipedia fetch)...",
        len(samples),
    )
    all_docs = {}
    for sample in tqdm(
        samples,
        desc="Ingest questions",
        unit="question",
        dynamic_ncols=True,
    ):
        if sample["source"] == "2wiki":
            docs = ingest_2wiki(sample, budgets, docstore, wiki)
        else:
            docs = ingest_nq(sample, budgets, docstore, wiki)
        for doc in docs:
            all_docs[doc.doc_key] = doc

    logger.info(
        "Ingest finished: %d questions → %d unique docs",
        len(samples),
        len(all_docs),
    )

    alias_map_redirects = build_alias_map_from_redirects(
        titles=[doc.title for doc in all_docs.values() if doc.title],
        wiki=wiki,
        curated_aliases={},
    )
    alias_map = dict(alias_map_redirects)
    alias_map.update(normalize_alias_map(CURATED_ALIASES))
    alias_map_path = output_dir / "alias_map.json"
    with alias_map_path.open("w", encoding="utf-8") as handle:
        json.dump(alias_map_redirects, handle, ensure_ascii=False, sort_keys=True)

    wiki_titles = sorted({doc.title for doc in all_docs.values() if doc.title})
    wiki_titles_path = output_dir / "wiki_titles.jsonl"
    write_wiki_titles(wiki_titles, wiki_titles_path, format="jsonl")
    logger.info("Wrote wiki_titles.jsonl (%d titles)", len(wiki_titles))

    matcher = None
    try:
        matcher = build_wiki_title_matcher(wiki_titles, alias_map=alias_map_redirects)
    except ImportError:
        logger.warning(
            "FlashText not installed; seed titles will be empty for IE extraction."
        )

    # ── Load existing corpus as a chunk cache (keyed by chunk_id) ──
    corpus_path = output_dir / "corpus.jsonl"
    chunk_cache: Dict[str, dict] = {}
    if corpus_path.is_file():
        for cached in _iter_jsonl(str(corpus_path)):
            cid = cached.get("chunk_id")
            if cid:
                chunk_cache[cid] = cached
        logger.info(
            "Loaded chunk cache with %d existing chunks from %s",
            len(chunk_cache),
            corpus_path,
        )

    chunks: List[dict] = []
    chunk_texts: List[str] = []
    cached_count = 0

    for doc in all_docs.values():
        if not doc.html:
            continue
        structured = clean_html_to_structured_doc(
            html=doc.html,
            doc_id=doc.doc_key,
            title=doc.title,
            url=doc.url,
            anchors=doc.anchors,
            source=doc.source,
            dataset_origin=doc.dataset_origin,
            page_id=doc.page_id,
            revision_id=doc.revision_id,
        )
        for idx, piece in enumerate(
            chunk_blocks(
                structured.blocks,
                min_tokens=args.chunk_min_tokens,
                max_tokens=args.chunk_max_tokens,
                overlap_tokens=args.chunk_overlap_tokens,
            )
        ):
            chunk_id = sha256_text(f"{doc.doc_key}:{idx}:{piece.text}")
            text_for_extraction = normalize_text_for_extraction(piece.text)

            # Check cache: if this chunk was already extracted, carry over metadata
            prior = chunk_cache.get(chunk_id)
            if prior is not None:
                prior_meta = prior.get("metadata", {})
                has_extraction = bool(prior_meta.get("entities")) or bool(
                    prior_meta.get("relations")
                )
            else:
                has_extraction = False

            if prior is not None and has_extraction:
                # Reuse the fully-extracted chunk from cache
                prior.setdefault("metadata", {})["ie_extracted"] = True
                chunks.append(prior)
                chunk_texts.append(text_for_extraction)
                cached_count += 1
            else:
                # Build fresh chunk that needs LLM extraction
                entities: List[dict] = []
                relations: List[dict] = []
                metadata = {
                    "dataset_origin": structured.dataset_origin,
                    "page_id": structured.page_id,
                    "revision_id": structured.revision_id,
                    "entities": entities,
                    "anchors": structured.anchors,
                    "relations": relations,
                    "ie_extracted": False,
                }
                chunk = CorpusChunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_key,
                    source=doc.source,
                    title=doc.title,
                    url=doc.url,
                    text=piece.text,
                    section_path=piece.section_path,
                    char_span_in_doc=piece.char_span_in_doc,
                    metadata=metadata,
                )
                chunk_json = chunk.to_json()
                chunks.append(chunk_json)
                chunk_texts.append(text_for_extraction)

    logger.info(
        "Chunks: %d total (%d from cache, %d new)",
        len(chunks),
        cached_count,
        len(chunks) - cached_count,
    )

    _write_jsonl(corpus_path, chunks)

    # ── Clean up stale IE batch artifacts from prior runs ──
    stale_files = [
        "batch_state_ie.json",
        "corpus_llm_ie.jsonl",
    ]
    for name in stale_files:
        stale = output_dir / name
        if stale.is_file():
            stale.unlink()
            logger.info("Removed stale %s", name)

    script_dir = Path(__file__).resolve().parent
    ie_script = script_dir / "corpus" / "run_llm_ie_batch.py"
    if ie_script.is_file():
        logger.info(
            "Running LLM information extraction batch (submit -> wait -> collect -> replace corpus)..."
        )
        result = subprocess.run(
            [
                sys.executable,
                str(ie_script),
                "--corpus",
                str(corpus_path),
                "--output-dir",
                str(output_dir),
            ],
            cwd=str(script_dir.parent),
        )
        if result.returncode != 0:
            logger.error(
                "IE batch pipeline failed with exit code %d", result.returncode
            )
            return result.returncode
        chunks = list(_iter_jsonl(str(corpus_path)))
    else:
        logger.error("IE batch script not found: %s", ie_script)
        return 1

    build_faiss_index(
        chunks,
        output_index_path=(output_dir / "vector_index.faiss").as_posix(),
        output_meta_path=(output_dir / "vector_meta.parquet").as_posix(),
        model_name=args.model_name,
    )

    graph = build_graph(chunks)
    graph_path = output_dir / "graph.pkl"
    pd.to_pickle(graph, graph_path)

    lexicon = build_entity_lexicon(chunks)
    lexicon_path = output_dir / "entity_lexicon.parquet"
    lexicon.to_parquet(lexicon_path, index=False)

    build_entity_index(
        lexicon_path=str(lexicon_path),
        output_index_path=str(output_dir / "entity_index.faiss"),
        output_meta_path=str(output_dir / "entity_index_meta.parquet"),
        model_name=args.model_name,
    )
    logger.info("Built entity_index.faiss for vector similarity matching")

    run_quality_gates(
        samples,
        chunks,
        output_path=(output_dir / "quality_report.json").as_posix(),
    )

    logger.info("Wrote corpus and artifacts to %s", output_dir)
    docstore.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

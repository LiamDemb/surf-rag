from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List

import pandas as pd

from surf_rag.core.build_entity_index import build_entity_index
from surf_rag.core.build_faiss import build_faiss_index
from surf_rag.core.build_graph import build_graph
from surf_rag.core.entity_lexicon import build_entity_lexicon
from surf_rag.core.quality_gates import run_quality_gates


def load_corpus_chunks(corpus_path: Path) -> list[dict]:
    """Load corpus JSONL rows; fail for missing/empty inputs."""
    corpus_path = corpus_path.expanduser().resolve()
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Missing corpus file: {corpus_path}")
    chunks: list[dict] = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    if not chunks:
        raise ValueError(f"Corpus is empty: {corpus_path}")
    return chunks


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _count_parquet_rows(path: Path) -> int:
    df = pd.read_parquet(path)
    return int(len(df))


def write_corpus_finalize_manifest(
    *,
    output_dir: Path,
    corpus_path: Path,
    chunks_count: int,
    produced_artifacts: dict[str, Path],
) -> Path:
    payload: dict[str, Any] = {
        "schema_version": "surf-rag/corpus_finalize/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus_path": str(corpus_path.resolve()),
        "corpus_size_bytes": corpus_path.stat().st_size,
        "corpus_sha256": _sha256_file(corpus_path),
        "chunks_count": int(chunks_count),
        "artifacts": {},
    }
    for key, p in produced_artifacts.items():
        payload["artifacts"][key] = {
            "path": str(p.resolve()),
            "size_bytes": p.stat().st_size if p.exists() else 0,
        }
    vector_meta = produced_artifacts.get("vector_meta")
    if vector_meta and vector_meta.exists():
        payload["vector_meta_rows"] = _count_parquet_rows(vector_meta)
    lexicon = produced_artifacts.get("entity_lexicon")
    if lexicon and lexicon.exists():
        payload["entity_lexicon_rows"] = _count_parquet_rows(lexicon)

    out = output_dir.resolve() / "corpus_finalize_manifest.json"
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return out


def finalize_corpus_artifacts(
    *,
    chunks: Iterable[dict],
    output_dir: Path,
    model_name: str,
    samples: List[dict] | None = None,
    quality_report: bool = True,
) -> dict[str, Path]:
    """Build non-IE corpus artifacts from already-merged corpus chunks."""
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_list = list(chunks)
    if not chunks_list:
        raise ValueError("Cannot finalize corpus artifacts from empty chunks list.")

    vector_index = output_dir / "vector_index.faiss"
    vector_meta = output_dir / "vector_meta.parquet"
    build_faiss_index(
        chunks_list,
        output_index_path=str(vector_index),
        output_meta_path=str(vector_meta),
        model_name=model_name,
    )

    graph_path = output_dir / "graph.pkl"
    graph = build_graph(chunks_list)
    pd.to_pickle(graph, graph_path)

    lexicon_path = output_dir / "entity_lexicon.parquet"
    lexicon = build_entity_lexicon(chunks_list)
    lexicon.to_parquet(lexicon_path, index=False)

    entity_index = output_dir / "entity_index.faiss"
    entity_index_meta = output_dir / "entity_index_meta.parquet"
    build_entity_index(
        lexicon_path=str(lexicon_path),
        output_index_path=str(entity_index),
        output_meta_path=str(entity_index_meta),
        model_name=model_name,
    )

    artifacts: dict[str, Path] = {
        "vector_index": vector_index,
        "vector_meta": vector_meta,
        "graph": graph_path,
        "entity_lexicon": lexicon_path,
        "entity_index": entity_index,
        "entity_index_meta": entity_index_meta,
    }
    if quality_report and samples is not None:
        report_path = output_dir / "quality_report.json"
        run_quality_gates(samples, chunks_list, output_path=str(report_path))
        artifacts["quality_report"] = report_path
    return artifacts

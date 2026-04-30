#!/usr/bin/env python3

from __future__ import annotations

import argparse
import io
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import requests
from huggingface_hub import HfApi
from huggingface_hub.utils import build_hf_headers
from surf_rag.config.env import get_hf_hub_token, load_app_env
from tqdm import tqdm

_DATASETS_DIR = Path(__file__).resolve().parent
if str(_DATASETS_DIR) not in sys.path:
    sys.path.insert(0, str(_DATASETS_DIR))

import _common  # noqa: E402

logger = logging.getLogger(__name__)
DATASET_REPO = "framolfese/2WikiMultihopQA"


def titles_exist(row: dict, *, wiki_language: str) -> bool:
    return _common.titles_all_exist_head(
        _common.twowiki_supporting_titles(row),
        language=wiki_language,
    )


def _resolve_revision(api: HfApi, revision: str | None) -> str:
    if revision:
        return revision
    info = api.repo_info(repo_id=DATASET_REPO, repo_type="dataset")
    return str(info.sha)


def _parquet_relpaths(api: HfApi, *, split: str, revision: str) -> list[str]:
    files = api.list_repo_files(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        revision=revision,
    )
    # Common shard layouts used on Hub datasets.
    patterns = (
        f"{split}-",
        f"data/{split}-",
        f"{split}/",
        f"data/{split}/",
    )
    paths = sorted(
        f
        for f in files
        if f.endswith(".parquet") and any(f.startswith(prefix) for prefix in patterns)
    )
    if not paths:
        raise ValueError(
            f"No parquet shards found for split={split} in {DATASET_REPO} "
            f"at revision {revision[:12]}..."
        )
    return paths


def _parquet_https_url(revision: str, rel_path: str) -> str:
    return (
        f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/{revision}/{rel_path}"
    )


def iter_2wiki_examples(
    *,
    split: str,
    revision: str | None = None,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    load_app_env()
    token = get_hf_hub_token()
    api = HfApi(token=token)
    rev = _resolve_revision(api, revision)
    relpaths = _parquet_relpaths(api, split=split, revision=rev)
    urls = [_parquet_https_url(rev, p) for p in relpaths]
    if seed is not None:
        random.Random(seed).shuffle(urls)

    session = requests.Session()
    session.headers.update(
        build_hf_headers(
            token=token,
            library_name="surf-rag",
            user_agent="2wiki_download (research; +https://github.com)",
        )
    )
    logger.info(
        "Reading %d parquet shard(s) for %s (revision %s...)",
        len(urls),
        split,
        rev[:12],
    )

    rows: list[dict[str, Any]] = []
    for url in urls:
        resp = session.get(url, timeout=300)
        resp.raise_for_status()
        table = pq.read_table(io.BytesIO(resp.content))
        rows.extend(table.to_pylist())
    return rows


def main(
    split: str,
    n: int | None,
    output: str,
    types: str | None,
    wiki_language: str,
) -> None:
    load_app_env()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    rows = iter_2wiki_examples(split=split, seed=42)

    # Filter by question type
    if types is not None and types.strip():
        allowed = set(t.strip().lower() for t in types.split(",") if t.strip())
        if allowed:
            rows = [x for x in rows if (x.get("type") or "").lower() in allowed]

    random.Random(42).shuffle(rows)

    valid_rows = rows
    if n is not None:
        if n < 1:
            raise ValueError("n must be >= 1")
        valid_rows = []

        with tqdm(
            total=n,
            desc="Valid samples",
            unit="sample",
            mininterval=0.2,
        ) as pbar:
            rows_scanned = 0
            for row in rows:
                rows_scanned += 1
                pbar.set_postfix(rows_scanned=rows_scanned, refresh=True)
                if titles_exist(row, wiki_language=wiki_language):
                    valid_rows.append(row)
                    pbar.update(1)

                if len(valid_rows) >= n:
                    break

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for row in valid_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {len(valid_rows)} rows to {output} (split={split})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Download 2WikiMultihopQA from Hugging Face (framolfese/2WikiMultihopQA)"
    )
    p.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
        help="Split to download",
    )
    p.add_argument(
        "--n",
        type=int,
        default=None,
        help="Max samples to save (default: all). Uses fixed seed for reproducibility.",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL path (default: data/raw/2wikimultihop_{n}.jsonl)",
    )
    p.add_argument(
        "--types",
        type=str,
        default=None,
        help="Comma-separated question types to keep (e.g. bridge_comparison,comparison). "
        "Default: all types.",
    )
    p.add_argument(
        "--wiki-language",
        type=str,
        default="en",
        help="Wikipedia language for HEAD checks (default: en).",
    )
    args = p.parse_args()

    suffix = args.n if args.n is not None else args.split
    output = args.output or f"data/raw/2wikimultihop_{suffix}.jsonl"

    main(
        split=args.split,
        n=args.n,
        output=output,
        types=args.types,
        wiki_language=args.wiki_language,
    )

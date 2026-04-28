#!/usr/bin/env python3
"""Download n HotPotQA examples from Hugging Face with Wikipedia title pre-checks."""

from __future__ import annotations

import argparse
import io
import json
import logging
import random
import sys
from collections import defaultdict
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import requests
from huggingface_hub import HfApi
from huggingface_hub.utils import build_hf_headers

from surf_rag.config.env import get_hf_hub_token, load_app_env
from surf_rag.core.corpus_acquisition import (
    supporting_titles_from_supporting_facts_sample,
)
from surf_rag.core.wikipedia_client import WikipediaClient

logger = logging.getLogger(__name__)

DATASET_REPO = "hotpotqa/hotpot_qa"


def _jsonify(obj: Any) -> Any:
    """Make Hugging Face / NumPy row values JSON-serializable."""
    if obj is None:
        return None
    if hasattr(obj, "item"):
        try:
            return _jsonify(obj.item())
        except Exception:
            pass
    if hasattr(obj, "tolist"):
        return _jsonify(obj.tolist())
    if isinstance(obj, Mapping):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (str, bool, int, float)):
        return obj
    return str(obj)


def _resolve_revision(api: HfApi, revision: str | None) -> str:
    if revision:
        return revision
    info = api.repo_info(repo_id=DATASET_REPO, repo_type="dataset")
    return str(info.sha)


def _parquet_relpaths(
    api: HfApi, *, config_name: str, split: str, revision: str
) -> list[str]:
    files = api.list_repo_files(
        repo_id=DATASET_REPO, repo_type="dataset", revision=revision
    )
    prefix = f"{config_name}/{split}-"
    paths = sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))
    if not paths:
        raise ValueError(
            f"No parquet shards matching {prefix}*.parquet in {DATASET_REPO} "
            f"at revision {revision[:12]}…"
        )
    return paths


def _parquet_https_url(revision: str, rel_path: str) -> str:
    return (
        f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/{revision}/{rel_path}"
    )


def iter_hotpotqa_examples(
    *,
    config_name: str,
    split: str,
    revision: str | None = None,
    seed: int | None = None,
) -> Iterator[dict[str, Any]]:
    """
    Yield HotPotQA rows from Hub Parquet shards.

    This avoids ``datasets.load_dataset("hotpotqa/hotpot_qa", ...)``, which can
    fail with ``Feature type 'List' not found`` when the local dataset builder
    cache has legacy ``dataset_infos`` metadata incompatible with ``datasets`` 3.x.
    """
    load_app_env()
    token = get_hf_hub_token()
    api = HfApi(token=token)
    rev = _resolve_revision(api, revision)
    relpaths = _parquet_relpaths(
        api, config_name=config_name, split=split, revision=rev
    )
    urls = [_parquet_https_url(rev, p) for p in relpaths]
    if seed is not None:
        random.Random(seed).shuffle(urls)

    logger.info(
        "Reading %d parquet shard(s) for %s/%s (revision %s…)",
        len(urls),
        config_name,
        split,
        rev[:12],
    )

    session = requests.Session()
    session.headers.update(
        build_hf_headers(
            library_name="surf-rag",
            user_agent="download_hotpotqa (research; +https://github.com)",
        )
    )

    for url in urls:
        resp = session.get(url, timeout=300)
        resp.raise_for_status()
        table = pq.read_table(io.BytesIO(resp.content))
        yield from table.to_pylist()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stream HotPotQA from Hugging Face (Hub parquet shards) and write JSONL "
            "rows whose supporting-fact Wikipedia titles exist as direct mainspace "
            "pages (non-missing, non-redirect)."
        )
    )
    parser.add_argument("--n", type=int, required=True, help="Target number of rows.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL path (e.g. data/raw/hotpotqa_validation_100.jsonl).",
    )
    parser.add_argument(
        "--config-name",
        default="distractor",
        choices=("distractor", "fullwiki"),
        help="HotPotQA HF config (default: distractor).",
    )
    parser.add_argument(
        "--split",
        default="validation",
        help="Dataset split name (default: validation).",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Git revision (commit SHA) on the Hub; default = dataset repo HEAD.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional shuffle seed for shard order before scanning.",
    )
    parser.add_argument(
        "--max-scanned",
        type=int,
        default=None,
        help="Stop after scanning this many rows (default: max(10000, 50 * n)).",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional JSON summary path (default: <output>.summary.json).",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Deprecated: ignored. Loader uses HTTPS parquet only.",
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Deprecated: ignored. Each run downloads parquet shards over the network.",
    )
    args = parser.parse_args()

    load_app_env()

    if args.n < 1:
        parser.error("--n must be >= 1")
    if args.no_streaming or args.force_redownload:
        logger.info(
            "Note: --no-streaming / --force-redownload are ignored (parquet loader)."
        )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    max_scanned = args.max_scanned or max(10_000, 50 * args.n)

    wiki = WikipediaClient()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    skip_reasons: dict[str, int] = defaultdict(int)
    scanned = 0
    kept = 0

    try:
        row_iter = iter_hotpotqa_examples(
            config_name=args.config_name,
            split=args.split,
            revision=args.revision,
            seed=args.seed,
        )
    except ValueError as e:
        logger.error("%s", e)
        return 1

    with args.output.open("w", encoding="utf-8") as handle:
        for ex in row_iter:
            scanned += 1
            if scanned > max_scanned:
                logger.warning(
                    "Stopped at max_scanned=%d (kept=%d, target=%d)",
                    max_scanned,
                    kept,
                    args.n,
                )
                break

            sample = {"supporting_facts": ex.get("supporting_facts")}
            titles = supporting_titles_from_supporting_facts_sample(sample)
            ok, reason = wiki.titles_are_direct_mainspace_pages(titles)
            if not ok:
                prefix = reason.split(":", 1)[0] if reason else "skip"
                skip_reasons[prefix] += 1
                continue

            row = _jsonify(ex)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
            if kept >= args.n:
                break

    summary = {
        "target_n": args.n,
        "kept": kept,
        "scanned": scanned,
        "skip_reasons": dict(skip_reasons),
        "config_name": args.config_name,
        "split": args.split,
        "revision": args.revision,
        "loader": "parquet_https",
        "output": str(args.output.resolve()),
    }
    summary_path = args.summary or (
        args.output.parent / f"{args.output.stem}.summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote %d rows to %s (scanned=%d)", kept, args.output, scanned)
    logger.info("Summary: %s", summary_path)

    if kept < args.n:
        logger.error(
            "Only collected %d/%d rows; increase --max-scanned or relax filters.",
            kept,
            args.n,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

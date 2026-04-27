"""Precomputed lexicon phrase matcher artifacts under a corpus directory."""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from surf_rag.entity_matching.matcher import (
    PhraseMatcher,
    build_phrase_records,
    records_to_matcher,
)
from surf_rag.entity_matching.types import PhraseRecord, PhraseSource

logger = logging.getLogger(__name__)

ENTITY_MATCHING_SCHEMA = "surf-rag/entity_matching/v1"
RECORDS_FILENAME = "entity_phrase_records.parquet"
MATCHER_FILENAME = "entity_phrase_matcher.pkl"
MANIFEST_FILENAME = "entity_matching_manifest.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def phrase_records_to_dataframe(records: List[PhraseRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_key": r.match_key,
                "canonical_norm": r.canonical_norm,
                "source": r.source.value,
                "df": int(r.df),
            }
            for r in records
        ]
    )


def dataframe_to_phrase_records(df: pd.DataFrame) -> List[PhraseRecord]:
    out: List[PhraseRecord] = []
    for _, row in df.iterrows():
        out.append(
            PhraseRecord(
                match_key=str(row["match_key"]),
                canonical_norm=str(row["canonical_norm"]),
                source=PhraseSource(str(row["source"])),
                df=int(row["df"]),
            )
        )
    return out


def build_entity_matching_artifacts(
    corpus_dir: Path,
    *,
    force: bool = False,
) -> Path:
    """Write phrase records, pickled matcher, and manifest next to lexicon inputs."""
    corp = corpus_dir.resolve()
    alias_path = corp / "alias_map.json"
    lexicon_path = corp / "entity_lexicon.parquet"
    if not alias_path.is_file():
        raise FileNotFoundError(f"Missing {alias_path}")
    if not lexicon_path.is_file():
        raise FileNotFoundError(f"Missing {lexicon_path}")

    manifest_path = corp / MANIFEST_FILENAME
    records_path = corp / RECORDS_FILENAME
    matcher_path = corp / MATCHER_FILENAME

    if not force and manifest_path.is_file():
        logger.info("Entity matching manifest exists; use force=True to rebuild.")
        return manifest_path

    _, records = build_phrase_records(str(corp))
    df_rec = phrase_records_to_dataframe(records)
    matcher = records_to_matcher(records)

    df_rec.to_parquet(records_path, index=False)
    # PhraseMatcher.__getstate__ serializes flat record tuples (no deep trie pickling).
    with matcher_path.open("wb") as mf:
        pickle.dump(matcher, mf, protocol=4)

    manifest: Dict[str, Any] = {
        "schema_version": ENTITY_MATCHING_SCHEMA,
        "created_at": _utc_now_iso(),
        "corpus_dir": str(corp),
        "inputs": {
            "alias_map.json": {
                "sha256": _sha256_file(alias_path),
                "size_bytes": alias_path.stat().st_size,
            },
            "entity_lexicon.parquet": {
                "sha256": _sha256_file(lexicon_path),
                "size_bytes": lexicon_path.stat().st_size,
            },
        },
        "record_count": len(records),
        "artifacts": {
            "phrase_records": RECORDS_FILENAME,
            "phrase_matcher_pickle": MATCHER_FILENAME,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Wrote entity matching artifacts (%d records) under %s",
        len(records),
        corp,
    )
    return manifest_path


def _inputs_match_manifest(corpus_dir: Path, manifest: Dict[str, Any]) -> bool:
    inputs = manifest.get("inputs") or {}
    for rel, meta in inputs.items():
        p = corpus_dir / rel
        if not p.is_file():
            return False
        exp = meta.get("sha256")
        if exp and _sha256_file(p) != exp:
            return False
    return True


def try_load_precomputed_matcher(corpus_dir: Path) -> Optional[PhraseMatcher]:
    """Load pickled :class:`PhraseMatcher` if manifest matches current inputs."""
    corp = corpus_dir.resolve()
    manifest_path = corp / MANIFEST_FILENAME
    matcher_path = corp / MATCHER_FILENAME
    records_path = corp / RECORDS_FILENAME

    if not manifest_path.is_file() or not matcher_path.is_file():
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        warnings.warn(
            f"Invalid entity matching manifest at {manifest_path}; rebuilding at runtime.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    if manifest.get("schema_version") != ENTITY_MATCHING_SCHEMA:
        warnings.warn(
            f"Stale entity matching schema {manifest.get('schema_version')!r}; "
            "rebuilding matcher at runtime.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    if not _inputs_match_manifest(corp, manifest):
        warnings.warn(
            "entity_lexicon.parquet or alias_map.json changed since "
            f"{MANIFEST_FILENAME} was built; rebuilding matcher at runtime.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    if records_path.is_file():
        try:
            df = pd.read_parquet(records_path)
            expected = int(manifest.get("record_count", -1))
            if expected >= 0 and len(df) != expected:
                warnings.warn(
                    f"{RECORDS_FILENAME} row count mismatch; rebuilding matcher at runtime.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return None
        except OSError:
            pass

    try:
        with matcher_path.open("rb") as f:
            obj = pickle.load(f)
    except (OSError, pickle.UnpicklingError, AttributeError, EOFError) as e:
        warnings.warn(
            f"Could not load {matcher_path}: {e}; rebuilding matcher at runtime.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    if not isinstance(obj, PhraseMatcher):
        warnings.warn(
            f"Unexpected object in {matcher_path}; rebuilding matcher at runtime.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    logger.info("Loaded precomputed entity phrase matcher from %s", corp)
    return obj

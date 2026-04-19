from __future__ import annotations

import ast
import json
from collections.abc import Iterable as IterableABC
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from surf_rag.core.alias_map import CURATED_ALIASES, normalize_alias_map
from surf_rag.core.enrich_entities import normalize_key


def _iter_surface_forms(raw: object) -> Iterable[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, tuple):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except (SyntaxError, ValueError):
                return [text]
        return [text]
    if isinstance(raw, IterableABC):
        return [str(x) for x in raw]
    return []


@dataclass
class EntityAliasResolver:
    """Resolves entity aliases to canonical normalized keys."""

    alias_map: Dict[str, str]

    @staticmethod
    def _merge_aliases(
        base: Dict[str, str],
        incoming: Dict[str, str],
        overwrite: bool,
    ) -> None:
        for key, value in incoming.items():
            norm_key = normalize_key(str(key))
            norm_value = normalize_key(str(value))
            if not norm_key or not norm_value:
                continue
            if overwrite:
                base[norm_key] = norm_value
            else:
                base.setdefault(norm_key, norm_value)

    @classmethod
    def from_artifacts(
        cls,
        output_dir: str,
        alias_filename: str = "alias_map.json",
        lexicon_filename: str = "entity_lexicon.parquet",
        include_curated: bool = True,
        include_lexicon: bool = True,
    ) -> "EntityAliasResolver":
        output_path = Path(output_dir)
        alias_path = output_path / alias_filename
        if not alias_path.exists():
            raise FileNotFoundError(
                f"Required alias artifact not found: {alias_path.as_posix()}"
            )

        with alias_path.open("r", encoding="utf-8") as handle:
            alias_data = json.load(handle)
        if not isinstance(alias_data, dict):
            raise ValueError(f"Expected a JSON object at {alias_path.as_posix()}")

        alias_map: Dict[str, str] = {}
        cls._merge_aliases(alias_map, alias_data, overwrite=True)

        if include_curated:
            # Curated aliases are a safety net and should not override persisted redirects.
            cls._merge_aliases(
                alias_map, normalize_alias_map(CURATED_ALIASES), overwrite=False
            )

        if include_lexicon:
            lexicon_path = output_path / lexicon_filename
            if lexicon_path.exists():
                lexicon_resolver = cls.from_lexicon(
                    lexicon_path=lexicon_path.as_posix()
                )
                # Lexicon aliases are fallback only; they should not override persisted redirects.
                cls._merge_aliases(
                    alias_map, lexicon_resolver.alias_map, overwrite=False
                )

        return cls(alias_map=alias_map)

    @classmethod
    def from_lexicon(
        cls,
        lexicon_path: str,
        norm_col: str = "norm",
        surface_forms_col: str = "surface_forms",
    ) -> "EntityAliasResolver":
        alias_map = normalize_alias_map(CURATED_ALIASES)
        df = pd.read_parquet(lexicon_path, columns=[norm_col, surface_forms_col])
        for _, row in df.iterrows():
            norm_value = normalize_key(str(row[norm_col]))
            if not norm_value:
                continue
            for surface in _iter_surface_forms(row[surface_forms_col]):
                surface_norm = normalize_key(surface)
                if not surface_norm:
                    continue
                alias_map[surface_norm] = norm_value
        return cls(alias_map=alias_map)

    @staticmethod
    def load_df_map_from_lexicon(
        lexicon_path: str,
        norm_col: str = "norm",
        df_col: str = "df",
    ) -> Dict[str, int]:
        df = pd.read_parquet(lexicon_path, columns=[norm_col, df_col])
        result: Dict[str, int] = {}
        for _, row in df.iterrows():
            norm_value = normalize_key(str(row[norm_col]))
            if not norm_value:
                continue
            try:
                result[norm_value] = int(row[df_col])
            except (TypeError, ValueError):
                result[norm_value] = 0
        return result

    def normalize(self, text: str) -> str:
        key = normalize_key(text)
        return self.alias_map.get(key, key)

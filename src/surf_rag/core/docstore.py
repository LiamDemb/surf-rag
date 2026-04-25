from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional


@dataclass(frozen=True)
class DocRecord:
    title: str
    page_id: Optional[str]
    revision_id: Optional[str]
    url: Optional[str]
    html: Optional[str]
    cleaned_text: Optional[str]
    anchors: Dict[str, Any]
    source: str
    dataset_origin: str


class DocStore:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path.as_posix())
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS docstore (
                cache_key TEXT PRIMARY KEY,
                title TEXT,
                page_id TEXT,
                revision_id TEXT,
                url TEXT,
                html TEXT,
                cleaned_text TEXT,
                anchors_json TEXT,
                source TEXT,
                dataset_origin TEXT
            )
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def get(self, cache_key: str) -> Optional[DocRecord]:
        row = self._conn.execute(
            """
            SELECT title, page_id, revision_id, url, html, cleaned_text,
                   anchors_json, source, dataset_origin
            FROM docstore
            WHERE cache_key = ?
            """,
            (cache_key,),
        ).fetchone()
        if not row:
            return None
        anchors = json.loads(row[6]) if row[6] else {}
        return DocRecord(
            title=row[0],
            page_id=row[1],
            revision_id=row[2],
            url=row[3],
            html=row[4],
            cleaned_text=row[5],
            anchors=anchors,
            source=row[7],
            dataset_origin=row[8],
        )

    def put(self, cache_key: str, record: DocRecord) -> None:
        anchors_json = json.dumps(record.anchors, ensure_ascii=False)
        self._conn.execute(
            """
            INSERT OR REPLACE INTO docstore
                (cache_key, title, page_id, revision_id, url, html, cleaned_text,
                 anchors_json, source, dataset_origin)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cache_key,
                record.title,
                record.page_id,
                record.revision_id,
                record.url,
                record.html,
                record.cleaned_text,
                anchors_json,
                record.source,
                record.dataset_origin,
            ),
        )
        self._conn.commit()

    def get_or_fetch(
        self,
        cache_key: str,
        fetch_fn: Callable[[], DocRecord],
    ) -> DocRecord:
        cached = self.get(cache_key)
        if cached:
            return cached
        record = fetch_fn()
        self.put(cache_key, record)
        return record

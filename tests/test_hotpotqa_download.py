"""Tests for scripts/datasets/download_hotpotqa.py (mocked HF + Wikipedia)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_download_module(repo_root: Path):
    path = repo_root / "scripts" / "datasets" / "download_hotpotqa.py"
    spec = importlib.util.spec_from_file_location("download_hotpotqa", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["download_hotpotqa"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_download_hotpotqa_writes_filtered_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    mod = _load_download_module(repo_root)

    rows = [
        {
            "id": "1",
            "question": "Q one?",
            "answer": "A1",
            "supporting_facts": {"title": ["T1"], "sent_id": [0]},
            "context": {"title": ["T1"], "sentences": [["Hello world.", "x"]]},
        },
        {
            "id": "2",
            "question": "Q two?",
            "answer": "A2",
            "supporting_facts": {"title": ["Missing"], "sent_id": [0]},
            "context": {"title": ["Missing"], "sentences": [["y"]]},
        },
    ]

    def fake_iter_hotpotqa_examples(**_kwargs):
        yield from rows

    class _FakeWiki:
        def titles_are_direct_mainspace_pages(self, titles: list) -> tuple:
            if titles == ["T1"]:
                return True, "ok"
            return False, "missing:x"

    monkeypatch.setattr(mod, "iter_hotpotqa_examples", fake_iter_hotpotqa_examples)
    monkeypatch.setattr(mod, "WikipediaClient", _FakeWiki)
    out = tmp_path / "out.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_hotpotqa",
            "--n",
            "1",
            "--output",
            str(out),
            "--max-scanned",
            "10",
        ],
    )
    assert mod.main() == 0
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    saved = json.loads(lines[0])
    assert saved["question"] == "Q one?"
    summary_file = tmp_path / "out.summary.json"
    assert summary_file.is_file()
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    assert summary["kept"] == 1
    assert summary["scanned"] == 1
    assert summary.get("loader") == "parquet_https"

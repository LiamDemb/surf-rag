import json
import subprocess
import sys
from pathlib import Path

from surf_rag.benchmark.corpus_filter import (
    filter_benchmark_rows,
    normalize_for_matching,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def test_filter_rules_nq_2wiki_and_hotpotqa():
    benchmark_rows = [
        {
            "question_id": "nq-keep",
            "dataset_source": "nq",
            "gold_support_sentences": ["alpha", "missing"],
        },
        {
            "question_id": "nq-drop",
            "dataset_source": "nq",
            "gold_support_sentences": ["not present"],
        },
        {
            "question_id": "2wiki-keep",
            "dataset_source": "2wiki",
            "gold_support_sentences": ["beta", "gamma"],
        },
        {
            "question_id": "2wiki-drop",
            "dataset_source": "2wiki",
            "gold_support_sentences": ["beta", "not present"],
        },
        {
            "question_id": "hotpot-keep",
            "dataset_source": "hotpotqa",
            "gold_support_sentences": ["beta", "gamma"],
        },
        {
            "question_id": "hotpot-drop",
            "dataset_source": "hotpotqa",
            "gold_support_sentences": ["beta", "nope"],
        },
    ]
    corpus_rows = [
        {"text": "this corpus contains alpha and other content"},
        {"text": "table style beta | gamma values"},
    ]
    kept, stats = filter_benchmark_rows(benchmark_rows, corpus_rows)
    kept_ids = {row["question_id"] for row in kept}
    assert kept_ids == {"nq-keep", "2wiki-keep", "hotpot-keep"}
    assert stats.total == 6
    assert stats.kept == 3
    assert stats.dropped == 3
    assert stats.dropped_by_source == {"nq": 1, "2wiki": 1, "hotpotqa": 1}


def test_normalize_for_matching_cleans_bracket_spacing():
    left = (
        "He was the fourth (but third surviving) son of Ernest I, Prince of Anhalt-Dessau, "
        "by his wife Margarete, daughter of Henry I, Duke of Münsterberg-Oels and "
        "granddaughter of George of Poděbrady, King of Bohemia."
    )
    right = (
        "He was the fourth ( but third surviving ) son of Ernest I, Prince of Anhalt-Dessau , "
        "by his wife Margarete, daughter of Henry I, Duke of Münsterberg-Oels and "
        "granddaughter of George of Poděbrady , King of Bohemia."
    )
    assert normalize_for_matching(left) == normalize_for_matching(right)


def test_filter_script_replaces_benchmark_and_keeps_backup(tmp_path):
    benchmark_path = tmp_path / "benchmark.jsonl"
    corpus_path = tmp_path / "corpus.jsonl"
    backup_path = tmp_path / "backup.jsonl"

    benchmark_rows = [
        {
            "question_id": "nq-keep",
            "dataset_source": "nq",
            "gold_support_sentences": ["alpha"],
            "gold_answers": ["a"],
            "question": "q1",
        },
        {
            "question_id": "2wiki-drop",
            "dataset_source": "2wiki",
            "gold_support_sentences": ["alpha", "missing"],
            "gold_answers": ["b"],
            "question": "q2",
        },
    ]
    corpus_rows = [{"text": "alpha appears in corpus"}]
    _write_jsonl(benchmark_path, benchmark_rows)
    _write_jsonl(corpus_path, corpus_rows)

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "filter_benchmark_by_corpus.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--benchmark",
        str(benchmark_path),
        "--corpus",
        str(corpus_path),
        "--backup",
        str(backup_path),
    ]
    result = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    assert backup_path.is_file()
    backup_rows = _read_jsonl(backup_path)
    assert [r["question_id"] for r in backup_rows] == ["nq-keep", "2wiki-drop"]

    filtered_rows = _read_jsonl(benchmark_path)
    assert [r["question_id"] for r in filtered_rows] == ["nq-keep"]

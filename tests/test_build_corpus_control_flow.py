from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script_module(module_name: str, rel_path: str):
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / rel_path
    spec = importlib.util.spec_from_file_location(module_name, target)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_complete_post_ie_calls_finalize_after_ie_success(
    tmp_path: Path, monkeypatch
) -> None:
    build_corpus = _load_script_module(
        "build_corpus_control_1", "scripts/build_corpus.py"
    )
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"chunk_id":"c1","text":"x","metadata":{}}\n', encoding="utf-8")
    called = {"finalize": 0}

    monkeypatch.setattr(
        build_corpus,
        "_run_ie_batch_pipeline",
        lambda **kwargs: 0,
    )

    def _fake_finalize(**kwargs):
        called["finalize"] += 1

    monkeypatch.setattr(build_corpus, "_finalize_corpus_outputs", _fake_finalize)
    rc = build_corpus._complete_post_ie(
        corpus_path=corpus,
        output_dir=tmp_path,
        model_name="all-MiniLM-L6-v2",
        samples=[],
        script_dir=Path("/tmp"),
    )
    assert rc == 0
    assert called["finalize"] == 1


def test_complete_post_ie_does_not_finalize_after_ie_failure(
    tmp_path: Path, monkeypatch
) -> None:
    build_corpus = _load_script_module(
        "build_corpus_control_2", "scripts/build_corpus.py"
    )
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"chunk_id":"c1","text":"x","metadata":{}}\n', encoding="utf-8")
    called = {"finalize": 0}
    monkeypatch.setattr(
        build_corpus,
        "_run_ie_batch_pipeline",
        lambda **kwargs: 7,
    )

    def _fake_finalize(**kwargs):
        called["finalize"] += 1

    monkeypatch.setattr(build_corpus, "_finalize_corpus_outputs", _fake_finalize)
    rc = build_corpus._complete_post_ie(
        corpus_path=corpus,
        output_dir=tmp_path,
        model_name="all-MiniLM-L6-v2",
        samples=[],
        script_dir=Path("/tmp"),
    )
    assert rc == 7
    assert called["finalize"] == 0


def test_complete_post_ie_surfaces_ie_exit_code(tmp_path: Path, monkeypatch) -> None:
    build_corpus = _load_script_module(
        "build_corpus_control_3", "scripts/build_corpus.py"
    )
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"chunk_id":"c1","text":"x","metadata":{}}\n', encoding="utf-8")
    monkeypatch.setattr(
        build_corpus,
        "_run_ie_batch_pipeline",
        lambda **kwargs: 3,
    )
    monkeypatch.setattr(build_corpus, "_finalize_corpus_outputs", lambda **kwargs: None)
    rc = build_corpus._complete_post_ie(
        corpus_path=corpus,
        output_dir=tmp_path,
        model_name="all-MiniLM-L6-v2",
        samples=[],
        script_dir=Path("/tmp"),
    )
    assert rc == 3

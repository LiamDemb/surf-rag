"""Resume behavior for generation/answers.jsonl with parse errors."""

import json
from pathlib import Path

from surf_rag.generation.batch_orchestrator import (
    _load_completed_question_ids_from_answers,
    collect_batches,
)


def test_resume_excludes_rows_with_empty_answer_and_custom_id(tmp_path: Path) -> None:
    """Merged-without-batch-line placeholder rows must retry."""
    p = tmp_path / "answers.jsonl"
    rows = [
        {"question_id": "done", "answer": "yes", "custom_id": "cid"},
        {
            "question_id": "missing_batch_line",
            "answer": "",
            "custom_id": "",
            "generation_parse_error": None,
        },
    ]
    p.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    done = _load_completed_question_ids_from_answers(p)
    assert done == {"done"}


def test_resume_excludes_rows_with_generation_parse_error(tmp_path: Path) -> None:
    p = tmp_path / "answers.jsonl"
    rows = [
        {"question_id": "ok", "answer": "yes", "generation_parse_error": None},
        {"question_id": "bad", "answer": "", "generation_parse_error": "no tool call"},
        {"question_id": "legacy", "answer": "old"},
    ]
    p.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    done = _load_completed_question_ids_from_answers(p)
    assert done == {"ok", "legacy"}


def test_collect_ingests_cancelled_batch_with_output_file(
    tmp_path: Path, monkeypatch
) -> None:
    run_root = tmp_path / "run"
    state_path = run_root / "batch" / "batch_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "run_root": str(run_root),
        "samples": [
            {"question_id": "q1", "question": "What?", "gold_answers": ["a"]},
            {"question_id": "q2", "question": "Where?", "gold_answers": ["b"]},
        ],
        "shards": [{"batch_id": "b_cancelled"}],
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")

    class _Content:
        def __init__(self, text: str) -> None:
            self._text = text

        def read(self) -> str:
            return self._text

    class _Batches:
        @staticmethod
        def retrieve(batch_id: str):
            assert batch_id == "b_cancelled"
            return type(
                "Batch",
                (),
                {"status": "cancelled", "output_file_id": "out_cancelled"},
            )()

    class _Files:
        @staticmethod
        def content(file_id: str):
            assert file_id == "out_cancelled"
            line = {
                "custom_id": "rid::bench::test::dense::q1",
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [
                            {
                                "message": {
                                    "tool_calls": [
                                        {
                                            "type": "function",
                                            "function": {
                                                "name": "format_answer",
                                                "arguments": '{"reasoning":"r","answer":"a"}',
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                },
            }
            return _Content(json.dumps(line) + "\n")

    class _Client:
        batches = _Batches()
        files = _Files()

        def __init__(self, api_key: str) -> None:
            assert api_key == "test-key"

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "surf_rag.generation.batch_orchestrator.OpenAI",
        _Client,
    )

    rc = collect_batches(state_path=state_path)
    assert rc == 0

    answers_path = run_root / "generation" / "answers.jsonl"
    rows = [
        json.loads(line)
        for line in answers_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_qid = {str(r.get("question_id", "")): r for r in rows}
    assert by_qid["q1"]["answer"] == "a"
    assert by_qid["q1"]["custom_id"] == "rid::bench::test::dense::q1"
    assert by_qid["q2"]["answer"] == ""

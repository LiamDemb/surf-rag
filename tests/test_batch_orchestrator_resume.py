"""Resume behavior for generation/answers.jsonl with parse errors."""

import json
from pathlib import Path

from surf_rag.generation.batch_orchestrator import (
    _load_completed_question_ids_from_answers,
)


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

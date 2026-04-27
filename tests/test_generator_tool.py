"""Parsing of forced format_answer tool output (Batch lines + sync message dicts)."""

import json

import pytest

from surf_rag.generation.generator_tool import (
    FORMAT_ANSWER_FUNCTION_NAME,
    GENERATION_OUTPUT_FORMAT,
    parse_generation_output_line,
    parse_message_for_format_answer,
)


def _batch_line(
    *,
    custom_id: str = "run::b::test::dense::q1",
    status_code: int = 200,
    tool_name: str = FORMAT_ANSWER_FUNCTION_NAME,
    arguments: str | None = '{"reasoning": "r1", "answer": "a1"}',
    content: str = "ignored free text",
    error: object | None = None,
    choices: list | None = None,
) -> dict:
    if error is not None:
        return {"custom_id": custom_id, "error": error}
    msg: dict = {"role": "assistant", "content": content}
    if choices is None:
        if tool_name is not None:
            msg["tool_calls"] = [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": arguments,
                    },
                }
            ]
        body = {"choices": [{"message": msg}]}
    else:
        body = {"choices": choices}
    return {
        "custom_id": custom_id,
        "response": {"status_code": status_code, "body": body},
    }


def test_parse_valid_tool_call():
    line = _batch_line()
    r = parse_generation_output_line(line)
    assert r.custom_id == "run::b::test::dense::q1"
    assert r.answer == "a1"
    assert r.reasoning == "r1"
    assert r.generation_output_format == GENERATION_OUTPUT_FORMAT
    assert r.generation_parse_error is None


def test_parse_ignores_message_content():
    line = _batch_line(content="Answer: should not use this")
    r = parse_generation_output_line(line)
    assert r.answer == "a1"
    assert "should not" not in r.answer


def test_parse_top_level_error():
    r = parse_generation_output_line({"custom_id": "x", "error": {"message": "bad"}})
    assert r.custom_id == "x"
    assert r.generation_parse_error is not None
    assert r.answer == ""


def test_parse_non_200():
    line = {
        "custom_id": "y",
        "response": {"status_code": 500, "body": {"choices": []}},
    }
    r = parse_generation_output_line(line)
    assert r.generation_parse_error is not None


def test_parse_no_choices():
    line = {
        "custom_id": "z",
        "response": {"status_code": 200, "body": {"choices": []}},
    }
    r = parse_generation_output_line(line)
    assert "no choices" in (r.generation_parse_error or "")


def test_parse_wrong_tool_name():
    line = _batch_line(tool_name="other_fn")
    r = parse_generation_output_line(line)
    assert "no format_answer" in (r.generation_parse_error or "")


def test_parse_invalid_json_arguments():
    line = _batch_line(arguments="not json")
    r = parse_generation_output_line(line)
    assert r.generation_parse_error is not None


def test_parse_missing_answer_key():
    line = _batch_line(arguments=json.dumps({"reasoning": "only"}))
    r = parse_generation_output_line(line)
    assert "answer" in (r.generation_parse_error or "").lower()


def test_parse_message_direct():
    msg = {
        "tool_calls": [
            {
                "function": {
                    "name": FORMAT_ANSWER_FUNCTION_NAME,
                    "arguments": '{"reasoning": "x", "answer": "y"}',
                }
            }
        ]
    }
    r = parse_message_for_format_answer(msg, custom_id="cid")
    assert r.answer == "y" and r.reasoning == "x" and r.custom_id == "cid"


@pytest.mark.parametrize("empty", [None, {}, ""])
def test_parse_empty_line(empty):
    r = parse_generation_output_line(empty)  # type: ignore[arg-type]
    assert r.generation_parse_error is not None

"""OpenAIChatGenerator: tool-based output (mocked API)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from surf_rag.core.generator import OpenAIChatGenerator


class _FakeFunction:
    name = "format_answer"
    arguments = '{"reasoning": "ctx says Paris", "answer": "Paris"}'


class _FakeToolCall:
    id = "call_1"
    type = "function"
    function = _FakeFunction()


class _FakeMessage:
    content = None
    tool_calls = [_FakeToolCall()]

    def model_dump(self):
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "format_answer",
                        "arguments": '{"reasoning": "ctx says Paris", "answer": "Paris"}',
                    },
                }
            ],
        }


class _FakeChoice:
    message = _FakeMessage()


class _FakeResponse:
    choices = [_FakeChoice()]


@patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"})
@patch("surf_rag.core.generator.OpenAI")
def test_openai_chat_generator_uses_format_answer_tool(mock_openai):
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_client.chat.completions.create.return_value = _FakeResponse()

    gen = OpenAIChatGenerator(model_id="gpt-4o-mini", temperature=0.0, max_tokens=64)
    out = gen.generate(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "user text"},
        ]
    )

    assert out.text == "Paris"
    assert out.sampling.get("reasoning") == "ctx says Paris"
    assert out.sampling.get("generation_parse_error") is None

    call_kw = mock_client.chat.completions.create.call_args.kwargs
    assert call_kw["tools"][0]["function"]["name"] == "format_answer"
    assert call_kw["tool_choice"]["function"]["name"] == "format_answer"
    assert call_kw.get("parallel_tool_calls") is False

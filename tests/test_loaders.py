import json

from surf_rag.core.loaders import load_nq


def test_load_nq_parses_minimal_jsonl(tmp_path):
    path = tmp_path / "nq.jsonl"
    payload = {
        "question_text": "Who wrote The Hobbit?",
        "annotations": [{"short_answers": [{"text": "J.R.R. Tolkien"}]}],
        "context": "The Hobbit is a novel by J.R.R. Tolkien.",
        "document_title": "The Hobbit",
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    items = list(load_nq(str(path), dataset_version="v1", max_rows=5))
    assert len(items) == 1
    item = items[0]
    assert item.dataset_source == "nq"
    assert item.gold_answers == ["J.R.R. Tolkien"]


def test_load_nq_parses_nested_document(tmp_path):
    path = tmp_path / "nq_nested.jsonl"
    payload = {
        "id": "123",
        "document": {
            "title": "Google",
            "html": "<html><body><p>Google was founded in 1998.</p></body></html>",
            "tokens": [
                {"token": "<p>", "is_html": True},
                {"token": "Google", "is_html": False},
                {"token": "was", "is_html": False},
                {"token": "founded", "is_html": False},
                {"token": "in", "is_html": False},
                {"token": "1998", "is_html": False},
                {"token": "</p>", "is_html": True},
            ],
        },
        "question": {"text": "who founded google"},
        "annotations": [
            {
                "short_answers": [
                    {"text": "Larry Page"},
                    {"text": "Sergey Brin"},
                ]
            }
        ],
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    items = list(load_nq(str(path), dataset_version="v1", max_rows=5))
    assert len(items) == 1
    item = items[0]
    assert item.gold_answers == ["Larry Page", "Sergey Brin"]


def test_load_nq_parses_annotations_dict(tmp_path):
    path = tmp_path / "nq_annotations_dict.jsonl"
    payload = {
        "id": "123",
        "document": {
            "title": "Google",
            "html": "<html><body><p>Google was founded by Larry Page.</p></body></html>",
        },
        "question": {"text": "who founded google"},
        "annotations": {
            "id": "0",
            "short_answers": [{"text": "Larry Page"}],
        },
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    items = list(load_nq(str(path), dataset_version="v1", max_rows=5))
    assert len(items) == 1
    item = items[0]
    assert item.gold_answers == ["Larry Page"]


def test_load_nq_short_answer_text_as_list_like_hf_jsonl(tmp_path):
    """HF ``to_json`` uses ``text`` as a list of strings per short answer."""
    path = tmp_path / "nq_hf.jsonl"
    payload = {
        "document": {
            "html": "<html><body><p>Date: March 18, 2018</p></body></html>",
            "tokens": {
                "token": ["Date", ":", "March", "18", ",", "2018"],
                "is_html": [False] * 6,
            },
        },
        "question": {"text": "what date"},
        "annotations": {
            "short_answers": [
                {"text": ["March 18, 2018"], "start_token": [2], "end_token": [5]}
            ],
        },
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    items = list(load_nq(str(path), dataset_version="v1", max_rows=5))
    assert len(items) == 1
    assert items[0].gold_answers == ["March 18, 2018"]


def test_load_nq_short_answer_empty_text_uses_token_span(tmp_path):
    path = tmp_path / "nq_span.jsonl"
    payload = {
        "document": {
            "html": "<html><body>x</body></html>",
            "tokens": {
                "token": ["aa", "bb", "cc"],
                "is_html": [False, False, False],
            },
        },
        "question": {"text": "q"},
        "annotations": {
            "short_answers": [{"text": [], "start_token": [1], "end_token": [2]}],
        },
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    items = list(load_nq(str(path), dataset_version="v1", max_rows=5))
    assert len(items) == 1
    assert items[0].gold_answers == ["bb cc"]

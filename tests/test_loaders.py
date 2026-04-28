import json

from surf_rag.core.loaders import load_hotpotqa, load_nq


def test_load_nq_parses_minimal_jsonl(tmp_path):
    path = tmp_path / "nq.jsonl"
    payload = {
        "question_text": "Who wrote The Hobbit?",
        "annotations": [
            {
                "short_answers": [
                    {
                        "text": ["Tolkien"],
                        "start_token": [6],
                        "end_token": [6],
                    }
                ]
            }
        ],
        "document": {
            "title": "The Hobbit",
            "html": "<html><body><p>The Hobbit is a novel by Tolkien.</p></body></html>",
            "tokens": {
                "token": [
                    "The",
                    "Hobbit",
                    "is",
                    "a",
                    "novel",
                    "by",
                    "Tolkien",
                    ".",
                ],
                "is_html": [False] * 8,
            },
        },
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    items = list(load_nq(str(path), dataset_version="v1", max_rows=5))
    assert len(items) == 1
    item = items[0]
    assert item.dataset_source == "nq"
    assert item.gold_answers == ["Tolkien"]
    assert item.gold_support_sentences == ["The Hobbit is a novel by Tolkien."]
    assert item.gold_support_titles == ["The Hobbit"]
    assert item.gold_support_sent_ids == [-1]


def test_load_nq_parses_nested_document(tmp_path):
    path = tmp_path / "nq_nested.jsonl"
    payload = {
        "id": "123",
        "document": {
            "title": "Google",
            "html": "<html><body><p>Google was founded by Larry Page and Sergey Brin.</p></body></html>",
            "tokens": {
                "token": [
                    "Google",
                    "was",
                    "founded",
                    "by",
                    "Larry",
                    "Page",
                    "and",
                    "Sergey",
                    "Brin",
                    ".",
                ],
                "is_html": [False] * 10,
            },
        },
        "question": {"text": "who founded google"},
        "annotations": [
            {
                "short_answers": [
                    {"text": ["Larry Page"], "start_token": [4], "end_token": [5]},
                    {"text": ["Sergey Brin"], "start_token": [7], "end_token": [8]},
                ]
            }
        ],
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    items = list(load_nq(str(path), dataset_version="v1", max_rows=5))
    assert len(items) == 1
    item = items[0]
    assert item.gold_answers == ["Larry Page", "Sergey Brin"]
    assert item.gold_support_sentences == [
        "Google was founded by Larry Page and Sergey Brin."
    ]
    assert item.gold_support_titles == ["Google"]
    assert item.gold_support_sent_ids == [-1]


def test_load_nq_parses_annotations_dict(tmp_path):
    path = tmp_path / "nq_annotations_dict.jsonl"
    payload = {
        "id": "123",
        "document": {
            "title": "Google",
            "html": "<html><body><p>Google was founded by Larry Page.</p></body></html>",
            "tokens": {
                "token": ["Google", "was", "founded", "by", "Larry", "Page", "."],
                "is_html": [False] * 7,
            },
        },
        "question": {"text": "who founded google"},
        "annotations": {
            "id": "0",
            "short_answers": [
                {"text": ["Larry Page"], "start_token": [4], "end_token": [5]}
            ],
        },
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    items = list(load_nq(str(path), dataset_version="v1", max_rows=5))
    assert len(items) == 1
    item = items[0]
    assert item.gold_answers == ["Larry Page"]
    assert item.gold_support_sentences == ["Google was founded by Larry Page."]
    assert item.gold_support_titles == ["Google"]
    assert item.gold_support_sent_ids == [-1]


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
    assert items[0].gold_support_titles == [""]
    assert items[0].gold_support_sent_ids == [-1]


def test_load_nq_short_answer_empty_text_without_clean_match_is_dropped(tmp_path):
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
    assert len(items) == 0


def test_load_hotpotqa_minimal_jsonl(tmp_path):
    path = tmp_path / "hotpot.jsonl"
    payload = {
        "question": "Who founded X?",
        "answer": "Alice",
        "context": {
            "title": ["Article One", "Article Two"],
            "sentences": [
                ["Alice founded X in 2000.", "Other text."],
                ["Unrelated sentence."],
            ],
        },
        "supporting_facts": {
            "title": ["Article One", "Article One"],
            "sent_id": [0, 0],
        },
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    items = list(load_hotpotqa(str(path), dataset_version="hp-v1", max_rows=5))
    assert len(items) == 1
    item = items[0]
    assert item.dataset_source == "hotpotqa"
    assert item.gold_answers == ["Alice"]
    assert item.gold_support_sentences == ["Alice founded X in 2000."]
    assert item.gold_support_titles == ["Article One"]
    assert item.gold_support_sent_ids == [0]
    assert item.dataset_version == "hp-v1"

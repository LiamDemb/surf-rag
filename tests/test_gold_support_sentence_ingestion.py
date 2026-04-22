import json

from surf_rag.core.loaders import load_2wiki, load_nq


def test_load_2wiki_extracts_support_sentences_from_large_candidate_context(tmp_path):
    path = tmp_path / "2wiki.jsonl"
    payload = {
        "id": "sample-2wiki",
        "question": "Where was the director of film Breakup Buddies born?",
        "answer": "Taiyuan",
        "supporting_facts": {
            "title": ["Breakup Buddies", "Ning Hao"],
            "sent_id": [1, 2],
        },
        "context": {
            "title": ["Distractor Page", "Ning Hao", "Breakup Buddies"],
            "sentences": [
                [
                    "This distractor page contains many similar words but no answer.",
                    "A long candidate sentence appears here to make matching non-trivial.",
                    "Another unrelated fact is listed in this context block.",
                ],
                [
                    "Ning Hao is a Chinese film director.",
                    "He transferred to the Art Department of Peking University.",
                    "Ning studied at the Taiyuan Film School, where he majored in scenic design.",
                    "He graduated from the Beijing Film Academy in 2003.",
                ],
                [
                    "Breakup Buddies is a 2014 Chinese romantic comedy road film.",
                    "The film was directed by Ning Hao.",
                    "It follows a long cross-country trip from Beijing to Dali.",
                ],
            ],
        },
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    items = list(load_2wiki(str(path), dataset_version="v1", max_rows=5))
    assert len(items) == 1
    item = items[0]
    assert item.gold_answers == ["Taiyuan"]
    assert item.gold_support_sentences == [
        "The film was directed by Ning Hao.",
        "Ning studied at the Taiyuan Film School, where he majored in scenic design.",
    ]
    assert item.gold_support_titles == ["Breakup Buddies", "Ning Hao"]
    assert item.gold_support_sent_ids == [1, 2]


def test_load_nq_extracts_support_sentence_from_short_answer_span(tmp_path):
    path = tmp_path / "nq.jsonl"
    payload = {
        "id": "sample-nq",
        "question": {"text": "What is the capital of France?"},
        "document": {
            "title": "France",
            "html": "<html><body><p>The capital of France is Paris.</p></body></html>",
            "tokens": {
                "token": [
                    "<p>",
                    "The",
                    "capital",
                    "of",
                    "France",
                    "is",
                    "Paris",
                    ".",
                    "It",
                    "is",
                    "known",
                    "for",
                    "the",
                    "Eiffel",
                    "Tower",
                    ".",
                    "</p>",
                ],
                "is_html": [
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                ],
            },
        },
        "annotations": [
            {
                "short_answers": [
                    {"text": [], "start_token": [6], "end_token": [6]},
                ]
            }
        ],
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    items = list(load_nq(str(path), dataset_version="v1", max_rows=5))
    assert len(items) == 1
    item = items[0]
    assert item.gold_answers == ["Paris"]
    assert item.gold_support_sentences == ["The capital of France is Paris."]
    assert item.gold_support_titles == ["France"]
    assert item.gold_support_sent_ids == [-1]


def test_load_nq_prefers_clean_sentence_over_noisy_preface(tmp_path):
    path = tmp_path / "nq_noisy_preface.jsonl"
    payload = {
        "id": "sample-nq-noisy",
        "question": {"text": "in greek mythology who was the goddess of spring growth"},
        "document": {
            "title": "Persephone",
            "html": (
                "<html><body>"
                "<div class='navbox'>Part of a series on Ancient Greek religion ... Persephone ...</div>"
                "<p>In Greek mythology, Persephone, also called Kore, is the daughter of Zeus and Demeter.</p>"
                "<p>She is associated with spring growth and vegetation cycles.</p>"
                "</body></html>"
            ),
            "tokens": {
                "token": [
                    "Part",
                    "of",
                    "a",
                    "series",
                    "on",
                    "Ancient",
                    "Greek",
                    "religion",
                    "Persephone",
                    "In",
                    "Greek",
                    "mythology",
                    ",",
                    "Persephone",
                    ",",
                    "also",
                    "called",
                    "Kore",
                    ",",
                    "is",
                    "the",
                    "daughter",
                    "of",
                    "Zeus",
                    "and",
                    "Demeter",
                    ".",
                ],
                "is_html": [False] * 27,
            },
        },
        "annotations": [
            {
                "short_answers": [
                    {"text": ["Persephone"], "start_token": [13], "end_token": [13]}
                ]
            }
        ],
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    items = list(load_nq(str(path), dataset_version="v1", max_rows=5))
    assert len(items) == 1
    assert items[0].gold_support_sentences == [
        "In Greek mythology, Persephone, also called Kore, is the daughter of Zeus and Demeter."
    ]
    assert items[0].gold_support_titles == ["Persephone"]
    assert items[0].gold_support_sent_ids == [-1]


def test_load_nq_disambiguates_repeated_date_with_context(tmp_path):
    path = tmp_path / "nq_repeated_date.jsonl"
    payload = {
        "id": "sample-nq-date",
        "question": {"text": "when was the finale released"},
        "document": {
            "title": "Example Episode List",
            "html": (
                "<html><body>"
                "<p>The series premiered on March 18, 2018.</p>"
                "<p>The finale was released on March 18, 2018 and drew high ratings.</p>"
                "<p>A soundtrack album followed on March 18, 2018.</p>"
                "</body></html>"
            ),
            "tokens": {
                "token": [
                    "The",
                    "series",
                    "premiered",
                    "on",
                    "March",
                    "18",
                    ",",
                    "2018",
                    ".",
                    "The",
                    "finale",
                    "was",
                    "released",
                    "on",
                    "March",
                    "18",
                    ",",
                    "2018",
                    "and",
                    "drew",
                    "high",
                    "ratings",
                    ".",
                    "A",
                    "soundtrack",
                    "album",
                    "followed",
                    "on",
                    "March",
                    "18",
                    ",",
                    "2018",
                    ".",
                ],
                "is_html": [False] * 33,
            },
        },
        "annotations": [
            {
                "short_answers": [
                    {"text": ["March 18, 2018"], "start_token": [14], "end_token": [17]}
                ]
            }
        ],
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    items = list(load_nq(str(path), dataset_version="v1", max_rows=5))
    assert len(items) == 1
    assert items[0].gold_support_sentences == [
        "The finale was released on March 18, 2018 and drew high ratings."
    ]
    assert items[0].gold_support_titles == ["Example Episode List"]
    assert items[0].gold_support_sent_ids == [-1]

from surf_rag.evaluation.qa_metrics import (
    exact_match,
    max_f1_over_golds,
    normalize_answer,
)


def test_normalize_answer_strips_articles_and_punct() -> None:
    assert normalize_answer("The quick brown fox.") == "quick brown fox"


def test_exact_match_any_gold() -> None:
    assert exact_match("foo bar", ["Foo Bar!", "other"]) == 1.0
    assert exact_match("nope", ["foo"]) == 0.0


def test_max_f1_over_golds() -> None:
    assert max_f1_over_golds("the dog", ["dog"]) > 0.5

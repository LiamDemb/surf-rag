from __future__ import annotations

from surf_rag.evaluation.llm_judge_merge import merge_llm_judge_verdicts_into_metrics


def test_merge_llm_judge_adds_per_question_and_rollups() -> None:
    metrics = {
        "split_question_ids": None,
        "overlap_breakdown": {
            "all": {"count": 2, "latency_ms": {}, "qa": {"em": 0.5, "f1": 0.5}},
            "train": {"count": 0, "latency_ms": {}, "qa": {"em": 0.0, "f1": 0.0}},
            "dev": {"count": 0, "latency_ms": {}, "qa": {"em": 0.0, "f1": 0.0}},
            "test": {"count": 0, "latency_ms": {}, "qa": {"em": 0.0, "f1": 0.0}},
            "unseen": {"count": 2, "latency_ms": {}, "qa": {"em": 0.5, "f1": 0.5}},
        },
        "per_question": [
            {
                "question_id": "q1",
                "audit": {"in_primary_eval": True},
                "qa": {"em": 1.0, "f1": 1.0, "prediction": "a"},
            },
            {
                "question_id": "q2",
                "audit": {"in_primary_eval": False},
                "qa": {"em": 0.0, "f1": 0.0, "prediction": "b"},
            },
        ],
    }
    verdicts = {"q1": True, "q2": False}
    out = merge_llm_judge_verdicts_into_metrics(metrics, verdicts, split_sets=None)
    assert out["per_question"][0]["qa_llm_judge"]["correct"] is True
    assert out["per_question"][1]["qa_llm_judge"]["correct"] is False
    assert out["overlap_breakdown"]["all"]["qa_llm_judge"]["n"] == 2
    assert out["overlap_breakdown"]["all"]["qa_llm_judge"]["accuracy"] == 0.5
    assert out["llm_judge"]["merged"] is True

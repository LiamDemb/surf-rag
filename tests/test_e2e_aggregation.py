from surf_rag.evaluation.e2e_aggregation import (
    PerQuestionEval,
    aggregate_e2e_report,
    load_benchmark_index,
)
from surf_rag.evaluation.retrieval_metrics import RankedMetricSuite
from surf_rag.evaluation.router_overlap import RouterSplitSets


def test_aggregate_e2e_report_overlap_buckets() -> None:
    rows = [
        PerQuestionEval(
            "q_train",
            [RankedMetricSuite(k=5, ndcg=1.0, hit=1.0, recall=1.0)],
            em=1.0,
            f1=1.0,
        ),
        PerQuestionEval(
            "q_test",
            [RankedMetricSuite(k=5, ndcg=0.0, hit=0.0, recall=0.0)],
            em=0.0,
            f1=0.0,
        ),
        PerQuestionEval(
            "q_unseen",
            [RankedMetricSuite(k=5, ndcg=0.5, hit=0.5, recall=0.5)],
            em=0.5,
            f1=0.5,
        ),
    ]
    splits = RouterSplitSets(train={"q_train"}, dev=set(), test={"q_test"})
    rep = aggregate_e2e_report(rows, split_sets=splits, ks=[5])
    assert rep["all"]["count"] == 3
    assert rep["train"]["count"] == 1
    assert rep["test"]["count"] == 1
    assert rep["unseen"]["count"] == 1
    assert rep["train"]["qa"]["em"] == 1.0
    assert rep["test"]["qa"]["em"] == 0.0


def test_load_benchmark_index(tmp_path) -> None:
    p = tmp_path / "b.jsonl"
    p.write_text(
        '{"question_id":"a","question":"?","gold_answers":["x"]}\n'
        '{"question_id":"b","question":"?","gold_answers":["y"]}\n',
        encoding="utf-8",
    )
    idx = load_benchmark_index(p)
    assert set(idx) == {"a", "b"}

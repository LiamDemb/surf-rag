from __future__ import annotations

from surf_rag.results.oracle_diagnostics import (
    aggregate_oracle_stats,
    per_query_diagnostics,
)


def _oracle_row(qid: str, source: str, curve: list[float]) -> dict:
    grid = [i / 10.0 for i in range(len(curve))]
    scores = [
        {
            "diagnostic_ndcg": {str(k): v for k in (5, 10, 20)},
            "oracle_objective_value": v,
        }
        for v in curve
    ]
    for i, s in enumerate(scores):
        s["diagnostic_ndcg"] = {
            "5": curve[i],
            "10": curve[i],
            "20": curve[i],
        }
    return {
        "question_id": qid,
        "dataset_source": source,
        "weight_grid": grid,
        "scores": scores,
    }


def test_per_query_delta_and_dispersion() -> None:
    rows = [
        _oracle_row(
            "q1", "nq", [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
        ),
        _oracle_row("q2", "2wiki", [0.5] * 11),
    ]
    df = per_query_diagnostics(
        rows,
        metric="ndcg",
        k=5,
        diagnostic_ks=[5],
        plateau_tau=1e-6,
        qid_to_source={},
    )
    assert len(df) == 2
    assert df.iloc[0]["delta"] > 0


def test_aggregate_oracle_stats_nonempty() -> None:
    rows = [_oracle_row("q1", "nq", [0.0] * 11)]
    df = per_query_diagnostics(
        rows,
        metric="ndcg",
        k=5,
        diagnostic_ks=[5],
        plateau_tau=1e-6,
        qid_to_source={},
    )
    agg = aggregate_oracle_stats(df)
    assert not agg.empty
    assert "mean_dispersion" in agg["stat_name"].values

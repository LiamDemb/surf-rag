from surf_rag.evaluation.latency_metrics import (
    PIPE_TOTAL_EXCLUDES_ROUTER_PREDICT_KEY,
    bootstrap_mean_ci95,
    canonicalize_latency_ms,
    reported_latency_ms_from_question_row,
    summarize_latency,
)


def test_canonicalize_dense_only_no_graph_leakage() -> None:
    out = canonicalize_latency_ms(
        retriever_name="Dense",
        latency_ms={"total": 12.0, "total_ms": 13.0},
        routing_input_ms=0.0,
    )
    assert out["retrieval_stage_total_ms"] == 13.0
    assert out["retrieval_reported_total_ms"] == 13.0
    assert out["dense_branch_ms"] == 12.0
    assert "graph_branch_ms" not in out


def test_canonicalize_fused_includes_branches_and_fusion() -> None:
    out = canonicalize_latency_ms(
        retriever_name="Fused",
        latency_ms={
            "total": 20.0,
            "dense_total": 7.0,
            "graph_total": 11.0,
            "fusion": 2.0,
        },
        routing_input_ms=3.0,
    )
    assert out["retrieval_stage_total_ms"] == 23.0
    assert out["retrieval_reported_total_ms"] == 23.0
    assert out["dense_branch_ms"] == 7.0
    assert out["graph_branch_ms"] == 11.0
    assert out["fusion_ms"] == 2.0


def test_canonicalize_reported_adds_router_when_pipe_excludes_router_predict() -> None:
    out = canonicalize_latency_ms(
        retriever_name="Fused",
        latency_ms={
            "total_ms": 18.0,
            "dense_total": 7.0,
            "graph_total": 11.0,
            "fusion": 0.5,
            "routing_predict_ms": 4.0,
            PIPE_TOTAL_EXCLUDES_ROUTER_PREDICT_KEY: 1.0,
        },
        routing_input_ms=2.0,
    )
    assert PIPE_TOTAL_EXCLUDES_ROUTER_PREDICT_KEY not in out
    assert out["retrieval_stage_total_ms"] == 20.0
    assert out["router_predict_ms"] == 4.0
    assert out["retrieval_reported_total_ms"] == 24.0


def test_canonicalize_no_double_count_router_when_marker_absent() -> None:
    out = canonicalize_latency_ms(
        retriever_name="Fused",
        latency_ms={
            "total_ms": 30.0,
            "fusion": 1.0,
            "dense_total": 10.0,
            "graph_total": 10.0,
            "routing_predict_ms": 9.0,
        },
        routing_input_ms=0.0,
    )
    assert out["retrieval_stage_total_ms"] == 30.0
    assert out["retrieval_reported_total_ms"] == 30.0


def test_reported_latency_ms_from_question_row_prefers_reported() -> None:
    assert (
        reported_latency_ms_from_question_row(
            {"retrieval_reported_total_ms": 12.0, "retrieval_stage_total_ms": 5.0}
        )
        == 12.0
    )
    assert (
        reported_latency_ms_from_question_row({"retrieval_stage_total_ms": 7.0}) == 7.0
    )
    assert reported_latency_ms_from_question_row({}) is None
    assert reported_latency_ms_from_question_row(None) is None


def test_bootstrap_ci_is_deterministic() -> None:
    xs = [1.0, 2.0, 3.0, 4.0]
    a = bootstrap_mean_ci95(xs, samples=300, seed=7)
    b = bootstrap_mean_ci95(xs, samples=300, seed=7)
    assert a == b


def test_summarize_latency_with_missing() -> None:
    out = summarize_latency([10.0, 20.0], total_count=5, ci_samples=200, ci_seed=1)
    assert out["count"] == 5
    assert out["valid_count"] == 2
    assert out["missing_count"] == 3
    assert out["mean_ms"] == 15.0
    assert out["p95_ms"] >= out["median_ms"]

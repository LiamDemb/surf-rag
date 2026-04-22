"""Tests for soft-label math and beta-based label materialization."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from surf_rag.router.soft_labels import (
    BetaSweepStats,
    beta_scaled_softmax,
    entropy,
    expected_weight,
    kl_divergence,
    materialize_soft_labels,
    soft_label_from_scores,
    sweep_beta,
)


def test_softmax_is_uniform_for_beta_zero():
    p = beta_scaled_softmax([0.1, 0.5, 0.9, 0.0], beta=0.0)
    assert all(pi == pytest.approx(0.25) for pi in p)
    assert sum(p) == pytest.approx(1.0)


def test_softmax_is_uniform_when_all_scores_equal():
    p = beta_scaled_softmax([0.3, 0.3, 0.3], beta=5.0)
    assert all(pi == pytest.approx(1 / 3) for pi in p)


def test_softmax_peaks_as_beta_grows():
    scores = [0.1, 0.2, 0.9]
    flat = beta_scaled_softmax(scores, beta=1.0)
    peaked = beta_scaled_softmax(scores, beta=50.0)
    assert max(peaked) > max(flat)
    assert peaked[2] > 0.99
    assert sum(flat) == pytest.approx(1.0)
    assert sum(peaked) == pytest.approx(1.0)


def test_softmax_rejects_negative_beta():
    with pytest.raises(ValueError):
        beta_scaled_softmax([0.1, 0.2], beta=-1.0)


def test_softmax_is_numerically_stable_for_large_scores():
    p = beta_scaled_softmax([1000.0, 999.0, 998.0], beta=10.0)
    assert sum(p) == pytest.approx(1.0)
    assert p[0] > p[1] > p[2]


def test_entropy_uniform_equals_log_n():
    dist = [1 / 4] * 4
    assert entropy(dist) == pytest.approx(math.log(4))


def test_entropy_of_one_hot_is_zero():
    assert entropy([0.0, 1.0, 0.0]) == pytest.approx(0.0)


def test_expected_weight_matches_weighted_sum():
    dist = [0.2, 0.3, 0.5]
    grid = [0.0, 0.5, 1.0]
    assert expected_weight(dist, grid) == pytest.approx(0.2 * 0 + 0.3 * 0.5 + 0.5 * 1.0)


def test_expected_weight_requires_same_length():
    with pytest.raises(ValueError):
        expected_weight([0.5, 0.5], [0.0, 0.5, 1.0])


def test_kl_divergence_is_zero_for_identical_distributions():
    p = [0.1, 0.3, 0.6]
    assert kl_divergence(p, p) == pytest.approx(0.0)


def test_kl_divergence_infinite_when_q_is_zero_where_p_is_not():
    assert kl_divergence([0.5, 0.5], [1.0, 0.0]) == math.inf


def test_kl_divergence_skips_bins_where_p_is_zero():
    # p places no mass on q's zero bin -> finite
    val = kl_divergence([1.0, 0.0], [0.9, 0.1])
    assert math.isfinite(val)


def test_soft_label_from_scores_shape_and_content():
    scores = [0.0, 0.5, 1.0]
    grid = [0.0, 0.5, 1.0]
    label = soft_label_from_scores(scores, grid, beta=10.0)
    assert sum(label["distribution"]) == pytest.approx(1.0)
    assert label["argmax_index"] == 2
    assert label["argmax_weight"] == pytest.approx(1.0)
    # Expected weight is between lowest and highest grid value.
    assert 0.0 <= label["expected_weight"] <= 1.0
    # Large beta makes expected_weight pull toward the argmax weight.
    assert label["expected_weight"] > 0.9


def test_soft_label_argmax_tie_prefers_dense_weight_nearest_half():
    """Matches oracle sweep tie-break: max score, then ``-|w - 0.5|``."""
    scores = [1.0, 1.0, 0.0]
    grid = [0.0, 0.5, 1.0]
    label = soft_label_from_scores(scores, grid, beta=100.0)
    assert label["argmax_index"] == 1
    assert label["argmax_weight"] == pytest.approx(0.5)


def test_soft_label_beta_zero_is_uniform_with_midpoint_expected_weight():
    scores = [0.1, 0.5, 0.9]
    grid = [0.0, 0.5, 1.0]
    label = soft_label_from_scores(scores, grid, beta=0.0)
    assert label["distribution"] == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert label["expected_weight"] == pytest.approx(0.5)


def test_soft_label_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        soft_label_from_scores([0.0, 0.5], [0.0, 0.5, 1.0], beta=1.0)


def _oracle_row(qid: str, scores, grid) -> dict:
    return {
        "question_id": qid,
        "dataset_source": "nq",
        "weight_grid": list(grid),
        "scores": [
            {
                "dense_weight": float(w),
                "graph_weight": 1.0 - float(w),
                "ndcg_primary": float(s),
            }
            for w, s in zip(grid, scores)
        ],
    }


def test_materialize_soft_labels_writes_expected_jsonl(tmp_path: Path):
    grid = [0.0, 0.5, 1.0]
    rows = [
        _oracle_row("q1", [0.0, 0.2, 1.0], grid),
        _oracle_row("q2", [1.0, 0.5, 0.0], grid),
    ]
    out = tmp_path / "labels" / "beta_2.jsonl"
    n = materialize_soft_labels(rows, beta=2.0, output_path=out)
    assert n == 2

    records = [
        json.loads(line) for line in out.read_text().splitlines() if line.strip()
    ]
    assert [r["question_id"] for r in records] == ["q1", "q2"]
    for rec in records:
        assert rec["beta"] == 2.0
        assert len(rec["distribution"]) == 3
        assert sum(rec["distribution"]) == pytest.approx(1.0)
        assert rec["argmax_index"] in (0, 2)
        assert 0.0 <= rec["expected_weight"] <= 1.0


def test_sweep_beta_returns_per_beta_stats_and_peaks_entropy_at_low_beta():
    grid = [0.0, 0.5, 1.0]
    rows = [
        _oracle_row("q1", [0.0, 0.5, 1.0], grid),
        _oracle_row("q2", [1.0, 0.5, 0.0], grid),
    ]
    stats = sweep_beta(rows, betas=[0.0, 1.0, 20.0])
    assert [s.beta for s in stats] == [0.0, 1.0, 20.0]
    # At beta=0 the distribution is uniform -> entropy is log(3).
    assert stats[0].mean_entropy == pytest.approx(math.log(3))
    # Entropy must decrease monotonically as beta grows.
    assert stats[0].mean_entropy > stats[1].mean_entropy > stats[2].mean_entropy
    for s in stats:
        assert s.num_questions == 2


def test_beta_sweep_stats_json_round_trip():
    s = BetaSweepStats(
        beta=1.5,
        num_questions=10,
        mean_entropy=0.4,
        mean_expected_weight=0.5,
        mean_argmax_weight=0.6,
    )
    j = s.to_json()
    assert j["beta"] == 1.5
    assert j["num_questions"] == 10
    assert j["mean_entropy"] == 0.4

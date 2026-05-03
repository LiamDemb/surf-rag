from __future__ import annotations

from pathlib import Path

from surf_rag.evaluation.graph_retrieval_sweep import (
    TrialRecord,
    append_trial_row,
    combo_key,
    completed_combo_keys,
    default_sweep_dir,
    iter_grid_combos,
    normalize_sweep_grid,
    pick_best_trial,
    read_trial_rows,
    trial_run_id,
)


def _trial(
    *,
    idx: int,
    key: str,
    score: float | None,
    status: str = "ok",
) -> TrialRecord:
    return TrialRecord(
        trial_index=idx,
        run_id=trial_run_id(idx),
        combo_key=key,
        retrieval_overrides={"graph_ppr_alpha": 0.8 + idx * 0.01},
        objective_path=(
            "overlap_breakdown.all.retrieval_before_ce.retrieval_at_k.10.ndcg"
        ),
        objective_score=score,
        status=status,
        error=None if status == "ok" else "boom",
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        elapsed_ms=1000,
    )


def test_combo_key_stable_order_independent() -> None:
    a = {"graph_ppr_alpha": 0.8, "graph_seed_softmax_temperature": 0.1}
    b = {"graph_seed_softmax_temperature": 0.1, "graph_ppr_alpha": 0.8}
    assert combo_key(a) == combo_key(b)


def test_iter_grid_combos_cartesian_count() -> None:
    grid = normalize_sweep_grid(
        {
            "graph_ppr_alpha": [0.7, 0.8, 0.9],
            "graph_seed_softmax_temperature": [0.1, 0.2],
        }
    )
    combos = list(iter_grid_combos(grid))
    assert len(combos) == 6
    assert combos[0]["graph_ppr_alpha"] == 0.7


def test_trials_jsonl_append_and_read(tmp_path: Path) -> None:
    p = tmp_path / "trials.jsonl"
    t0 = _trial(idx=0, key="a", score=0.5)
    t1 = _trial(idx=1, key="b", score=None, status="error")
    append_trial_row(p, t0)
    append_trial_row(p, t1)
    rows = read_trial_rows(p)
    assert len(rows) == 2
    assert rows[0].combo_key == "a"
    assert rows[1].status == "error"


def test_completed_combo_keys_only_ok_rows() -> None:
    rows = [
        _trial(idx=0, key="done", score=0.5, status="ok"),
        _trial(idx=1, key="failed", score=None, status="error"),
    ]
    assert completed_combo_keys(rows) == {"done"}


def test_pick_best_trial_prefers_higher_then_lower_index() -> None:
    rows = [
        _trial(idx=2, key="a", score=0.7),
        _trial(idx=0, key="b", score=0.9),
        _trial(idx=1, key="c", score=0.9),
    ]
    best = pick_best_trial(rows)
    assert best is not None
    assert rows[best].trial_index == 0


def test_default_sweep_dir_under_benchmark_bundle() -> None:
    out = default_sweep_dir(
        benchmark_base=Path("data/benchmarks"),
        benchmark_name="surf-bench",
        benchmark_id="development",
        sweep_id="graph rag tuning",
    )
    assert out == Path("data/benchmarks/surf-bench/development/graph-rag-tuning")

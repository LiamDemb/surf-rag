"""Contract for evaluate_router metrics merge (see scripts/router/evaluate_router.py)."""


def test_metrics_merge_refreshes_splits_but_keeps_training_fields() -> None:
    """``{**existing, **payload}`` must keep train_router-only keys when re-evaluating."""
    existing = {
        "midpoint_balance_masking": {
            "enabled": True,
            "train_rows_after": 100,
        },
        "loss": "hinge_squared_regret",
        "best_epoch": 10,
        "splits": {"train": {"num_rows": 999.0}},
    }
    payload = {
        "router_id": "rid",
        "router_architecture_id": "aid",
        "architecture": "mlp-v1",
        "input_mode": "both",
        "splits": {"train": {"num_rows": 2067.0}},
        "router_quality_filtering": {},
    }
    merged = {**existing, **payload}
    assert merged["midpoint_balance_masking"]["train_rows_after"] == 100
    assert merged["best_epoch"] == 10
    assert merged["loss"] == "hinge_squared_regret"
    assert merged["splits"]["train"]["num_rows"] == 2067.0

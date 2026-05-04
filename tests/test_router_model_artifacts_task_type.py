import pytest

pytest.importorskip("torch")

from pathlib import Path

from surf_rag.evaluation.router_model_artifacts import (
    build_router_model_root,
    parse_router_task_type,
)


def test_parse_router_task_type_defaults_and_aliases() -> None:
    assert parse_router_task_type(None) == "regression"
    assert parse_router_task_type("reg") == "regression"
    assert parse_router_task_type("cls") == "classification"


def test_build_router_model_root_includes_task_for_arch_scoped_models() -> None:
    root = build_router_model_root(
        Path("/tmp/router"),
        "router-a",
        input_mode="both",
        router_architecture_id="mlp-main",
        router_task_type="classification",
    )
    assert str(root).endswith("router-a/models/mlp-main/classification/both")

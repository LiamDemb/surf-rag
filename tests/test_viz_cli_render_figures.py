from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest
import yaml

from surf_rag.evaluation.router_model_artifacts import make_router_model_paths_for_cli


def _write_minimal_config_yaml(path: Path, tmp_path: Path) -> None:
    rb = tmp_path / "router"
    mp = make_router_model_paths_for_cli(
        "rid",
        router_base=rb,
        input_mode="both",
        router_architecture_id="t",
    )
    mp.run_root.mkdir(parents=True, exist_ok=True)
    pred = mp.predictions("test")
    rows = [
        {
            "question_id": "q1",
            "target_oracle_best_weight": 0.2,
            "predicted_weight": 0.3,
            "is_valid_for_router_training": True,
        }
    ]
    pred.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    cfg_dict = {
        "schema_version": "surf-rag/pipeline/v1",
        "paths": {
            "data_base": str(tmp_path),
            "figures_base": str(tmp_path / "figures"),
            "router_base": str(rb),
            "benchmark_base": str(tmp_path / "bench"),
            "benchmark_name": "b",
            "benchmark_id": "id",
            "router_id": "rid",
            "router_architecture_id": "t",
        },
        "router": {"train": {"input_mode": "both"}},
        "figures": {
            "enabled": True,
            "plots": [{"kind": "router_pred_vs_oracle", "split": "test"}],
        },
    }
    path.write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")


def test_cli_main_happy_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfgp = tmp_path / "cfg.yaml"
    _write_minimal_config_yaml(cfgp, tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["render_figures.py", "--config", str(cfgp), "--force"],
    )
    import scripts.figures.render_figures as render_figures

    rc = render_figures.main()
    assert rc == 0
    expected_fig = (
        tmp_path / "figures" / "router" / "rid" / "t" / "both" / "benchmark" / "b__id"
    )
    assert expected_fig.is_dir()
    pngs = list(expected_fig.glob("*.png"))
    assert len(pngs) >= 1


def test_cli_exits_2_when_figures_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfgp = tmp_path / "cfg.yaml"
    cfgp.write_text(
        yaml.safe_dump(
            {
                "schema_version": "surf-rag/pipeline/v1",
                "paths": {
                    "data_base": str(tmp_path),
                    "router_base": str(tmp_path / "r"),
                    "router_id": "x",
                },
                "figures": {"enabled": False, "plots": []},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("sys.argv", ["render_figures.py", "--config", str(cfgp)])
    import scripts.figures.render_figures as render_figures

    rc = render_figures.main()
    assert rc == 2

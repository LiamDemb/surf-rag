"""argv_provides helper tests."""

from __future__ import annotations

from surf_rag.config.argv import argv_provides


def test_argv_provides_detects_flag() -> None:
    assert argv_provides(
        ["prog", "prepare", "--run-id", "x", "--config", "a.yaml"], "--run-id"
    )
    assert not argv_provides(["prog", "prepare", "--config", "a.yaml"], "--run-id")

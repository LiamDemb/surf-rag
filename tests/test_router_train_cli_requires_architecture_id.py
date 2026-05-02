from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from scripts.router import train_router


def test_train_router_requires_architecture_id(monkeypatch) -> None:
    monkeypatch.setattr(
        train_router,
        "parse_args",
        lambda: Namespace(
            config=None,
            router_id="rid",
            router_architecture_id=None,
            router_base=Path("/tmp"),
            epochs=None,
            batch_size=None,
            learning_rate=None,
            device=None,
            architecture=None,
            architecture_kwargs=None,
            input_mode=None,
            log_level="INFO",
        ),
    )
    assert train_router.main() == 2

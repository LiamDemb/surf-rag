from __future__ import annotations

import pytest

from surf_rag.router.architectures.registry import (
    get_architecture,
    list_architectures,
)


def test_registry_lists_expected_architectures() -> None:
    names = list_architectures()
    assert "mlp-v1" in names
    assert "mlp-v2" in names
    assert "logreg-v1" in names
    assert "polyreg-v1" in names
    assert "tower_v01" in names


def test_registry_unknown_architecture_errors() -> None:
    with pytest.raises(ValueError, match="Unknown router architecture"):
        get_architecture("missing-v0")

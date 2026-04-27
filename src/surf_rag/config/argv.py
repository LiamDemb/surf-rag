"""Detect whether a CLI flag was passed (so config file does not override it)."""

from __future__ import annotations

from collections.abc import Sequence


def argv_provides(argv: Sequence[str], flag: str) -> bool:
    """True if ``flag`` (e.g. ``--run-id``) or ``--run-id=value`` appears."""
    for a in argv:
        if a == flag or a.startswith(f"{flag}="):
            return True
    return False

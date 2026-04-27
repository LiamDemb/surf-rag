"""``python -m surf_rag.config`` — print merged YAML, or validate prereqs for Make."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from surf_rag.config.loader import (
    config_to_resolved_dict,
    load_pipeline_config,
    resolve_paths,
)
from surf_rag.config.validate_prereqs import (
    validate_oracle,
    validate_router_dataset,
    validate_router_train,
)

_VALIDATE = {
    "validate-oracle": validate_oracle,
    "validate-router": validate_router_dataset,
    "validate-router-train": validate_router_train,
}


def _usage() -> None:
    print(
        "Usage:\n"
        "  python -m surf_rag.config <config.yaml>   # print merged YAML\n"
        "  python -m surf_rag.config validate-oracle <config.yaml>\n"
        "  python -m surf_rag.config validate-router <config.yaml>\n"
        "  python -m surf_rag.config validate-router-train <config.yaml>\n",
        end="",
        file=sys.stderr,
    )


def main() -> int:
    if len(sys.argv) < 2:
        _usage()
        return 2
    cmd = sys.argv[1]
    if cmd in _VALIDATE:
        if len(sys.argv) < 3:
            _usage()
            return 2
        p = Path(sys.argv[2]).expanduser().resolve()
        return int(_VALIDATE[cmd](p))
    # Single-file print (backward compatible)
    p = Path(cmd).expanduser().resolve()
    cfg = load_pipeline_config(p)
    rp = resolve_paths(cfg)
    print(
        yaml.safe_dump(
            config_to_resolved_dict(cfg, rp),
            allow_unicode=True,
            sort_keys=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

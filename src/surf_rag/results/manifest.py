"""Build run manifest JSON."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from surf_rag.results.bundle import ResultsBundle


@dataclass
class ArtifactRecord:
    id: str
    kind: str
    status: str
    table_csv: str | None = None
    figure_image: str | None = None
    figure_meta: str | None = None
    error: str | None = None


@dataclass
class BuildManifest:
    bundle_id: str
    built_at: str
    config_path: str | None
    split: str
    oracle: dict[str, Any]
    policies: dict[str, dict[str, Any]]
    artifacts: list[ArtifactRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), indent=2) + "\n",
            encoding="utf-8",
        )


def _file_digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def manifest_from_bundle(
    bundle: ResultsBundle,
    *,
    config_path: Path | None,
    artifact_records: list[ArtifactRecord],
) -> BuildManifest:
    policies: dict[str, dict[str, Any]] = {}
    for name, pr in bundle.policies.items():
        policies[name] = {
            "run_id": pr.run_id,
            "router_role": pr.router_role,
            "metrics_path": str(pr.metrics_path),
            "metrics_sha256_prefix": _file_digest(pr.metrics_path),
            "resolved_config": (
                str(pr.resolved_config_path) if pr.resolved_config_path else None
            ),
        }
    oc = bundle.results.oracle
    return BuildManifest(
        bundle_id=bundle.results.bundle_id,
        built_at=datetime.now(timezone.utc).isoformat(),
        config_path=str(config_path.resolve()) if config_path else None,
        split=bundle.results.split,
        oracle={
            "metric": oc.metric,
            "k": oc.k,
            "diagnostic_ks": list(oc.diagnostic_ks),
            "plateau_tau": oc.plateau_tau,
        },
        policies=policies,
        artifacts=artifact_records,
        warnings=list(bundle.warnings),
    )

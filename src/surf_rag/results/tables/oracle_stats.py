"""Oracle curve summary statistics table."""

from __future__ import annotations

import pandas as pd

from surf_rag.config.schema import ResultsArtifactSpec
from surf_rag.results.bundle import ResultsBundle
from surf_rag.results.loaders import load_oracle_rows
from surf_rag.results.metric_fields import resolve_oracle_metric_k
from surf_rag.results.oracle_diagnostics import (
    aggregate_oracle_stats,
    per_query_diagnostics,
)
from surf_rag.results.tables.writer import write_table


def build_oracle_stats(
    bundle: ResultsBundle, spec: ResultsArtifactSpec
) -> tuple[pd.DataFrame, dict]:
    metric, k = resolve_oracle_metric_k(spec, bundle)
    oc = bundle.results.oracle
    rows = load_oracle_rows(bundle.oracle_scores_path)
    per_q = per_query_diagnostics(
        rows,
        metric=metric,
        k=k,
        diagnostic_ks=list(oc.diagnostic_ks),
        plateau_tau=oc.plateau_tau,
        qid_to_source=bundle.qid_to_source,
    )
    df = aggregate_oracle_stats(per_q)
    artifact_id = spec.id or "oracle_stats"
    paths = write_table(
        bundle,
        artifact_id,
        df,
        {"metric": metric, "k": k},
    )
    return df, {"csv": str(paths[0]), "meta": str(paths[1])}

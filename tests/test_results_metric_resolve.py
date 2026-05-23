from __future__ import annotations

from dataclasses import replace

from surf_rag.config.schema import (
    PipelineConfig,
    ResultsArtifactSpec,
    ResultsOracleConfig,
    ResultsSection,
)
from surf_rag.results.bundle import ResultsBundle
from surf_rag.results.metric_fields import (
    resolve_retrieval_ks,
    resolve_retrieval_metric_k,
)


def _bundle_with_oracle(**oracle_kw) -> ResultsBundle:
    oracle = ResultsOracleConfig(**oracle_kw)
    results = replace(ResultsSection(), oracle=oracle)
    cfg = replace(PipelineConfig(), results=results)
    bundle = ResultsBundle.__new__(ResultsBundle)
    bundle.cfg = cfg
    bundle.results = results
    return bundle


def test_resolve_metric_k_spec_overrides_oracle() -> None:
    bundle = _bundle_with_oracle(metric="recall", k=10)
    spec = ResultsArtifactSpec(id="x", metric="hit", k=5)
    assert resolve_retrieval_metric_k(spec, bundle) == ("hit", 5)


def test_resolve_metric_k_falls_back_to_oracle() -> None:
    bundle = _bundle_with_oracle(metric="recall", k=10)
    spec = ResultsArtifactSpec(id="x")
    assert resolve_retrieval_metric_k(spec, bundle) == ("recall", 10)


def test_resolve_ks_spec_ks() -> None:
    bundle = _bundle_with_oracle(diagnostic_ks=[5, 10, 20])
    spec = ResultsArtifactSpec(id="x", ks=[10, 20])
    assert resolve_retrieval_ks(spec, bundle) == [10, 20]


def test_resolve_ks_spec_k_only() -> None:
    bundle = _bundle_with_oracle(diagnostic_ks=[5, 10, 20])
    spec = ResultsArtifactSpec(id="x", k=10)
    assert resolve_retrieval_ks(spec, bundle) == [10]


def test_resolve_ks_diagnostic_ks_default() -> None:
    bundle = _bundle_with_oracle(diagnostic_ks=[5, 10, 20])
    spec = ResultsArtifactSpec(id="x")
    assert resolve_retrieval_ks(spec, bundle) == [5, 10, 20]

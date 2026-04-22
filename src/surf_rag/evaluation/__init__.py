"""Evaluation run directories, manifests, and artifact paths."""

from surf_rag.evaluation.oracle_artifacts import (
    DEFAULT_DENSE_WEIGHT_GRID,
    OracleRunPaths,
    OracleScoreRow,
    WeightBinScore,
    build_oracle_run_root,
    default_oracle_base,
    make_run_paths_for_cli as make_oracle_run_paths_for_cli,
    read_manifest as read_oracle_manifest,
    read_oracle_score_rows,
    read_retrieval_cache,
    read_summary as read_oracle_summary,
    write_manifest as write_oracle_manifest,
    write_questions_snapshot,
    write_summary as write_oracle_summary,
)
from surf_rag.evaluation.retrieval_metrics import (
    DEFAULT_NDCG_KS,
    PRIMARY_NDCG_K,
    RankedMetricSuite,
    compute_metric_suite,
    ndcg_at_k,
    score_retrieval_result,
    stateful_relevances,
)
from surf_rag.evaluation.run_artifacts import (
    RunArtifactPaths,
    build_run_root,
    default_evaluation_base,
    make_generation_custom_id,
    parse_generation_custom_id,
)

__all__ = [
    # Generation-run artifacts
    "RunArtifactPaths",
    "build_run_root",
    "default_evaluation_base",
    "make_generation_custom_id",
    "parse_generation_custom_id",
    # Oracle-run artifacts
    "OracleRunPaths",
    "OracleScoreRow",
    "WeightBinScore",
    "DEFAULT_DENSE_WEIGHT_GRID",
    "build_oracle_run_root",
    "default_oracle_base",
    "make_oracle_run_paths_for_cli",
    "write_oracle_manifest",
    "read_oracle_manifest",
    "write_oracle_summary",
    "read_oracle_summary",
    "write_questions_snapshot",
    "read_retrieval_cache",
    "read_oracle_score_rows",
    # Retrieval metrics
    "DEFAULT_NDCG_KS",
    "PRIMARY_NDCG_K",
    "RankedMetricSuite",
    "compute_metric_suite",
    "ndcg_at_k",
    "score_retrieval_result",
    "stateful_relevances",
]

"""Join benchmark, oracle curve labels, features, and embeddings into training rows."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from surf_rag.evaluation.oracle_artifacts import DEFAULT_DENSE_WEIGHT_GRID
from surf_rag.router.feature_normalization import (
    fit_normalizer_v1,
    prefix_raw_norm,
    transform_row,
    FeatureNormalizerV1,
)
from surf_rag.router.query_features import (
    V1_FEATURE_NAMES,
    extract_features_v1,
    feature_vector_ordered,
    QueryFeatureContext,
    FEATURE_SET_VERSION,
)
from surf_rag.router.splits import (
    assign_splits_stratified,
    split_summary,
    stratum_key,
    _quantiles,
)


def _std_quantiles(label_rows: Sequence[Mapping[str, Any]]) -> Tuple[float, float]:
    stds = [float(r["oracle_curve_std"]) for r in label_rows if "oracle_curve_std" in r]
    return _quantiles(stds)


def build_router_dataframe(
    benchmark_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    *,
    feature_context: QueryFeatureContext,
    embedding_model: str,
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    split_seed: int,
    router_id: str,
) -> Tuple[pd.DataFrame, FeatureNormalizerV1, Dict[str, Any]]:
    """Assemble a single dataframe with raw + norm features, embeddings, and splits.
    Skips benchmark rows with no matching label row.
    """
    by_q: Dict[str, Dict[str, Any]] = {}
    for r in label_rows:
        qid = str(r.get("question_id", "")).strip()
        if qid:
            by_q[qid] = dict(r)
    if not by_q:
        raise ValueError("No label rows with question_id")

    q1, q2 = _std_quantiles(list(by_q.values()))

    aligned_bench: List[Mapping[str, Any]] = []
    aligned_labels: List[Mapping[str, Any]] = []
    for row in benchmark_rows:
        qid = str(row.get("question_id", "")).strip()
        if not qid or qid not in by_q:
            continue
        aligned_bench.append(row)
        aligned_labels.append(by_q[qid])

    if not aligned_bench:
        raise ValueError("No benchmark rows matched label question_ids")

    qid_to_split = assign_splits_stratified(
        aligned_labels,
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
        test_ratio=test_ratio,
        seed=split_seed,
    )
    sum_meta = split_summary(qid_to_split, aligned_labels)

    raw_feature_rows: List[Dict[str, float]] = []
    for row in aligned_bench:
        q = str(row.get("question", ""))
        raw_feature_rows.append(extract_features_v1(q, feature_context))

    train_feats = [
        raw_feature_rows[i]
        for i in range(len(aligned_bench))
        if qid_to_split.get(str(aligned_bench[i].get("question_id", "")).strip(), "")
        == "train"
    ]
    if not train_feats:
        raise ValueError("Train split is empty; adjust ratios or data")
    normalizer = fit_normalizer_v1(train_feats)

    questions = [str(b.get("question", "")) for b in aligned_bench]
    emb = _embed_with_fallback(questions, embedding_model)

    weight_grid = list(
        map(float, aligned_labels[0].get("weight_grid") or DEFAULT_DENSE_WEIGHT_GRID)
    )

    records: List[Dict[str, Any]] = []
    for i, b in enumerate(aligned_bench):
        qid = str(b.get("question_id", "")).strip()
        lab = aligned_labels[i]
        raw = raw_feature_rows[i]
        norm = transform_row(raw, normalizer)
        prefixed = prefix_raw_norm(raw, norm)
        aw = float(lab.get("oracle_best_weight", 0.0))
        std = float(lab.get("oracle_curve_std", 0.0))
        stratum = stratum_key(aw, std, q1, q2)
        sp = qid_to_split.get(qid, "train")
        curve = [float(x) for x in (lab.get("oracle_curve") or [])]
        if len(curve) != len(weight_grid):
            raise ValueError(
                f"Oracle curve length {len(curve)} != weight grid {len(weight_grid)}"
            )
        best_score = float(lab.get("oracle_best_score", 0.0))
        rec: Dict[str, Any] = {
            "question_id": qid,
            "question": b.get("question", ""),
            "dataset_source": b.get("dataset_source", ""),
            "split": sp,
            "split_stratum": stratum,
            "split_seed": int(split_seed),
            "weight_grid": weight_grid,
            "oracle_curve": curve,
            "oracle_best_weight": aw,
            "oracle_best_score": best_score,
            "oracle_curve_std": std,
            "oracle_best_index": int(lab.get("oracle_best_index", 0)),
            "is_valid_for_router_training": bool(best_score > 0.0),
            "router_id": router_id,
            "oracle_run_id": router_id,
            "feature_set_version": FEATURE_SET_VERSION,
            "embedding_model": embedding_model,
            "embedding_dim": int(emb.shape[1]) if len(emb.shape) > 1 else 0,
        }
        rec.update(prefixed)
        # Fixed-order vector for convenience
        rec["feature_vector_raw"] = feature_vector_ordered(raw)
        rec["feature_vector_norm"] = [float(norm.get(n, 0.0)) for n in V1_FEATURE_NAMES]
        rec["query_embedding"] = emb[i].tolist() if i < len(emb) else []
        records.append(rec)

    df = pd.DataFrame.from_records(records)
    return df, normalizer, sum_meta


def _embed_with_fallback(questions: List[str], model_name: str) -> np.ndarray:
    from surf_rag.router.query_embeddings import embed_queries

    return embed_queries(questions, model_name=model_name)

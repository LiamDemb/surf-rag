"""Router dataframe join and pipeline (with mocked embeddings)."""

from __future__ import annotations

import numpy as np
import pytest

from surf_rag.evaluation.oracle_artifacts import DEFAULT_DENSE_WEIGHT_GRID
from surf_rag.router.dataset import build_router_dataframe
from surf_rag.router.query_features import QueryFeatureContext

pytest.importorskip("spacy")
import spacy  # noqa: E402

try:
    _nlp = spacy.load("en_core_web_sm")
except OSError:
    pytest.skip("en_core_web_sm", allow_module_level=True)


def _label_row(qid: str, *, std: float = 0.5, curve: list[float] | None = None) -> dict:
    w = [float(x) for x in DEFAULT_DENSE_WEIGHT_GRID]
    c = curve if curve is not None else [float(x) for x in DEFAULT_DENSE_WEIGHT_GRID]
    best_score = float(max(c))
    return {
        "question_id": qid,
        "dataset_source": "nq",
        "weight_grid": w,
        "oracle_curve": c,
        "oracle_best_score": best_score,
        "oracle_curve_std": std,
        "is_valid_for_router_training": bool(best_score > 0.0),
    }


def test_build_dataframe_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_resolve(aligned_bench, **kwargs):
        n = len(aligned_bench)
        return np.ones((n, 4), dtype=np.float32), {"hits": n, "misses": 0}

    monkeypatch.setattr(
        "surf_rag.router.dataset.resolve_aligned_query_embeddings", _fake_resolve
    )
    bench = [
        {
            "question_id": "a1",
            "question": "What is X?",
            "dataset_source": "nq",
        },
        {
            "question_id": "a2",
            "question": "How many are there?",
            "dataset_source": "nq",
        },
    ]
    lab = [
        _label_row("a1", std=0.1),
        _label_row("a2", std=0.9),
    ]
    ctx = QueryFeatureContext(nlp=_nlp)
    df, norm, _sum = build_router_dataframe(
        bench,
        lab,
        feature_context=ctx,
        embedding_model="fake",
        train_ratio=0.5,
        dev_ratio=0.25,
        test_ratio=0.25,
        split_seed=0,
        router_id="t1",
    )
    assert len(df) == 2
    assert "feature_raw__content_token_len" in df.columns
    assert "query_embedding" in df.columns
    assert len(df["query_embedding"].iloc[0]) == 4
    assert "oracle_best_weight" not in df.columns
    assert df["split_stratum"].tolist() == ["nq", "nq"]

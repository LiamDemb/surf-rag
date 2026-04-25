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


def _label_row(
    qid: str, *, ent: float = 0.5, aw: float = 0.5, dist: list[float] | None = None
) -> dict:
    w = [float(x) for x in DEFAULT_DENSE_WEIGHT_GRID]
    d = dist if dist is not None else [1.0 / 11.0] * 11
    return {
        "question_id": qid,
        "dataset_source": "nq",
        "beta": 2.0,
        "weight_grid": w,
        "scores": [0.1] * 11,
        "distribution": d,
        "expected_weight": 0.5,
        "argmax_weight": aw,
        "argmax_index": 5,
        "entropy": ent,
    }


def test_build_dataframe_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake(questions, model_name: str, batch_size: int = 32) -> np.ndarray:
        n = len(questions)
        return np.ones((n, 4), dtype=np.float32)

    monkeypatch.setattr("surf_rag.router.dataset._embed_with_fallback", _fake)
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
        _label_row("a1", ent=0.1, aw=0.0),
        _label_row("a2", ent=0.9, aw=1.0),
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
        selected_beta=2.0,
        oracle_run_id="t1",
    )
    assert len(df) == 2
    assert "feature_raw__content_token_len" in df.columns
    assert "query_embedding" in df.columns
    assert len(df["query_embedding"].iloc[0]) == 4

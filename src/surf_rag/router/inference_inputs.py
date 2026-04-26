"""Query-time embedding + normalized features for router prediction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from surf_rag.evaluation.artifact_paths import default_router_base
from surf_rag.evaluation.router_dataset_artifacts import (
    RouterDatasetPaths,
    build_router_dataset_root,
    read_router_dataset_manifest,
)
from surf_rag.evaluation.router_model_artifacts import make_router_model_paths_for_cli
from surf_rag.router.feature_normalization import FeatureNormalizerV1, transform_row
from surf_rag.router.inference import (
    LoadedRouter,
    load_router_checkpoint,
    predict_batch,
)
from surf_rag.router.model import parse_router_input_mode
from surf_rag.router.query_embeddings import embed_queries
from surf_rag.router.query_features import (
    QueryFeatureContext,
    extract_features_v1,
    feature_vector_ordered,
)


@dataclass
class RouterInferenceContext:
    """Checkpoint, normalizer, and query-side resources for one router bundle."""

    router: LoadedRouter
    normalizer: FeatureNormalizerV1
    embedding_model: str
    input_mode: str
    feature_context: Optional[QueryFeatureContext]


def _feature_context_for_corpus(corpus_dir: Path) -> QueryFeatureContext:
    alias = corpus_dir / "alias_map.json"
    if not alias.is_file():
        return QueryFeatureContext(retrieval_asset_dir=str(corpus_dir))
    from surf_rag.entity_matching.pipeline import LexiconAliasEntityPipeline

    return QueryFeatureContext(
        entity_pipeline=LexiconAliasEntityPipeline.from_artifacts(str(corpus_dir)),
        retrieval_asset_dir=str(corpus_dir),
    )


def load_router_inference_context(
    router_id: str,
    *,
    input_mode: str = "both",
    router_base: Optional[Path] = None,
    retrieval_asset_dir: Optional[Path] = None,
    device: str = "cpu",
) -> RouterInferenceContext:
    """Load checkpoint, train z-score stats, and corpus-linked feature context."""
    rb = router_base if router_base is not None else default_router_base()
    mode = parse_router_input_mode(input_mode)
    ds_paths = RouterDatasetPaths(run_root=build_router_dataset_root(rb, router_id))
    stats_path = ds_paths.feature_stats
    if not stats_path.is_file():
        raise FileNotFoundError(f"Missing feature stats: {stats_path}")
    normalizer = FeatureNormalizerV1.from_json(
        json.loads(stats_path.read_text(encoding="utf-8"))
    )
    dmanifest = read_router_dataset_manifest(ds_paths)
    embedding_model = str(
        dmanifest.get("embedding_model")
        or (dmanifest.get("model") or {}).get("embedding_model")
        or "all-MiniLM-L6-v2"
    )
    corp_raw = retrieval_asset_dir
    if corp_raw is None:
        sc = dmanifest.get("source_corpus") or {}
        corp_raw = Path(str(sc.get("retrieval_asset_dir") or ""))
    if not corp_raw.is_dir():
        raise FileNotFoundError(f"Corpus dir missing or not a directory: {corp_raw}")
    feat_ctx = _feature_context_for_corpus(corp_raw)

    mp = make_router_model_paths_for_cli(router_id, router_base=rb, input_mode=mode)
    if not mp.checkpoint.is_file():
        raise FileNotFoundError(f"Missing router checkpoint: {mp.checkpoint}")
    router = load_router_checkpoint(
        mp.checkpoint, device=device, manifest_path=mp.manifest
    )
    return RouterInferenceContext(
        router=router,
        normalizer=normalizer,
        embedding_model=embedding_model,
        input_mode=mode,
        feature_context=feat_ctx,
    )


def _dummy_embeddings(router: LoadedRouter) -> np.ndarray:
    d = int(router.config.embedding_dim)
    return np.zeros((1, d), dtype=np.float32)


def _dummy_features(router: LoadedRouter) -> np.ndarray:
    d = int(router.config.feature_dim)
    return np.zeros((1, d), dtype=np.float32)


def compute_query_tensors_for_router(
    query: str,
    ictx: RouterInferenceContext,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (query_embedding, feature_vector) batches for ``predict_batch``."""
    mode = parse_router_input_mode(ictx.input_mode)
    qe: Optional[np.ndarray] = None
    qf: Optional[np.ndarray] = None
    if mode in ("both", "embedding"):
        qe = embed_queries([query], model_name=ictx.embedding_model)
    if mode in ("both", "query-features"):
        raw = extract_features_v1(query, ictx.feature_context)
        norm = transform_row(raw, ictx.normalizer)
        qf = np.asarray([feature_vector_ordered(norm)], dtype=np.float32)
    r = ictx.router
    if mode == "embedding":
        qf = _dummy_features(r)
    elif mode == "query-features":
        qe = _dummy_embeddings(r)
    assert qe is not None and qf is not None
    return qe, qf


def predict_router_distribution(
    query: str,
    ictx: RouterInferenceContext,
) -> Tuple[np.ndarray, np.ndarray]:
    """Softmax distribution and expected dense weight for one query."""
    qe, qf = compute_query_tensors_for_router(query, ictx)
    return predict_batch(ictx.router, qe, qf)

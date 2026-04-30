"""Query-time embedding + normalized features for router prediction."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

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

if TYPE_CHECKING:
    from surf_rag.core.embedder import SentenceTransformersEmbedder


@dataclass
class RouterInferenceContext:
    """Checkpoint, normalizer, and query-side resources for one router bundle."""

    router: LoadedRouter
    normalizer: FeatureNormalizerV1
    embedding_model: str
    input_mode: str
    feature_context: Optional[QueryFeatureContext]
    _query_embedder: Optional["SentenceTransformersEmbedder"] = field(
        default=None, init=False, repr=False
    )

    def query_embedder(self) -> "SentenceTransformersEmbedder":
        """Lazy shared SentenceTransformer wrapper for batched query embeddings."""
        if self._query_embedder is None:
            from surf_rag.core.embedder import SentenceTransformersEmbedder

            self._query_embedder = SentenceTransformersEmbedder(
                model_name=self.embedding_model
            )
        return self._query_embedder


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


def compute_query_tensors_for_router_batch(
    queries: Sequence[str],
    ictx: RouterInferenceContext,
    *,
    st_batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Batched (query_embedding, feature_vector) arrays for ``predict_batch``."""
    mode = parse_router_input_mode(ictx.input_mode)
    r = ictx.router
    texts = [str(q or "").strip() for q in queries]
    n = len(texts)

    qe: Optional[np.ndarray] = None
    qf: Optional[np.ndarray] = None
    if mode in ("both", "embedding"):
        qe = embed_queries(
            texts,
            model_name=ictx.embedding_model,
            batch_size=st_batch_size,
            embedder_factory=ictx.query_embedder,
        )
    if mode in ("both", "query-features"):
        ctx = ictx.feature_context
        rows = [extract_features_v1(t, ctx) for t in texts]
        qf = np.stack(
            [
                np.asarray(
                    feature_vector_ordered(transform_row(raw, ictx.normalizer)),
                    dtype=np.float32,
                )
                for raw in rows
            ],
            axis=0,
        )
    if mode == "embedding":
        fd = int(r.config.feature_dim)
        qf = np.zeros((n, fd), dtype=np.float32)
    elif mode == "query-features":
        ed = int(r.config.embedding_dim)
        qe = np.zeros((n, ed), dtype=np.float32)

    assert qe is not None and qf is not None
    return qe, qf


def compute_query_tensors_for_router(
    query: str,
    ictx: RouterInferenceContext,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (query_embedding, feature_vector) batches for ``predict_batch``."""
    return compute_query_tensors_for_router_batch([query], ictx, st_batch_size=1)


def predict_router_weight(
    query: str,
    ictx: RouterInferenceContext,
) -> np.ndarray:
    """Predicted dense weight(s) for one query."""
    qe, qf = compute_query_tensors_for_router(query, ictx)
    return predict_batch(ictx.router, qe, qf)

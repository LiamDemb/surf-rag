"""Query-time embedding + normalized features for router prediction."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from surf_rag.evaluation.artifact_paths import default_router_base
from surf_rag.evaluation.query_embedding_cache_artifacts import (
    QueryEmbeddingCachePaths,
    make_query_embedding_cache_paths,
)
from surf_rag.evaluation.router_dataset_artifacts import (
    RouterDatasetPaths,
    build_router_dataset_root,
    read_router_dataset_manifest,
)
from surf_rag.evaluation.router_model_artifacts import make_router_model_paths_for_cli
from surf_rag.evaluation.router_model_artifacts import (
    ROUTER_TASK_REGRESSION,
    parse_router_task_type,
)
from surf_rag.router.dataset_embedding_resolve import resolve_aligned_query_embeddings
from surf_rag.router.embedding_config import (
    EMBEDDING_CACHE_OFF,
    EMBEDDING_CACHE_REQUIRED,
    EMBEDDING_PROVIDER_SENTENCE_TRANSFORMERS,
    parse_embedding_provider,
    resolve_embedding_model_for_provider,
)
from surf_rag.router.embedding_lock import (
    infer_embedding_provider_from_model,
    validate_router_embedding_compatibility,
)
from surf_rag.router.feature_normalization import FeatureNormalizerV1, transform_row
from surf_rag.router.inference import (
    LoadedRouter,
    load_router_checkpoint,
    predict_class_id_batch,
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
    embedding_provider: str = EMBEDDING_PROVIDER_SENTENCE_TRANSFORMERS
    embedding_cache_mode: str = EMBEDDING_CACHE_OFF
    embedding_cache_writeback: bool = True
    embedding_cache_id: str = ""
    embedding_cache_path_override: Optional[str] = None
    benchmark_jsonl_path: Optional[Path] = None
    embedding_cache_root: Optional[str] = None
    openai_embedding_dimensions: int | None = None
    router_cache_stats: Dict[str, int] = field(
        default_factory=lambda: {
            "hits": 0,
            "misses": 0,
            "live_computed": 0,
            "appended_rows": 0,
        }
    )
    _query_embedder: Optional["SentenceTransformersEmbedder"] = field(
        default=None, init=False, repr=False
    )

    def query_embedder(self) -> "SentenceTransformersEmbedder":
        """Lazy shared SentenceTransformer wrapper for batched query embeddings."""
        if parse_embedding_provider(self.embedding_provider) != (
            EMBEDDING_PROVIDER_SENTENCE_TRANSFORMERS
        ):
            raise RuntimeError(
                "query_embedder() is only valid for sentence-transformers provider; "
                f"got {self.embedding_provider!r}"
            )
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
    router_architecture_id: Optional[str] = None,
    router_base: Optional[Path] = None,
    retrieval_asset_dir: Optional[Path] = None,
    device: str = "cpu",
    router_task_type: str = ROUTER_TASK_REGRESSION,
    router_embedding_provider: Optional[str] = None,
    router_embedding_cache_mode: Optional[str] = None,
    router_embedding_cache_id: Optional[str] = None,
    router_embedding_cache_path: Optional[str] = None,
    router_embedding_cache_writeback: Optional[bool] = None,
    e2e_benchmark_path: Optional[Path] = None,
    router_openai_embedding_dimensions: Optional[int] = None,
) -> RouterInferenceContext:
    """Load checkpoint, train z-score stats, and corpus-linked feature context."""
    rb = router_base if router_base is not None else default_router_base()
    mode = parse_router_input_mode(input_mode)
    task_type = parse_router_task_type(router_task_type)
    ds_paths = RouterDatasetPaths(run_root=build_router_dataset_root(rb, router_id))
    stats_path = ds_paths.feature_stats
    if not stats_path.is_file():
        raise FileNotFoundError(f"Missing feature stats: {stats_path}")
    normalizer = FeatureNormalizerV1.from_json(
        json.loads(stats_path.read_text(encoding="utf-8"))
    )
    dmanifest = read_router_dataset_manifest(ds_paths)
    embedding_model_raw = str(
        dmanifest.get("embedding_model")
        or (dmanifest.get("model") or {}).get("embedding_model")
        or "all-MiniLM-L6-v2"
    )
    ds_prov_raw = dmanifest.get("embedding_provider")
    if str(ds_prov_raw or "").strip():
        ds_embedding_provider = parse_embedding_provider(str(ds_prov_raw))
    else:
        ds_embedding_provider = infer_embedding_provider_from_model(embedding_model_raw)

    eff_prov_raw = str(router_embedding_provider or "").strip()
    if eff_prov_raw:
        embedding_provider = parse_embedding_provider(eff_prov_raw)
    else:
        embedding_provider = ds_embedding_provider

    embedding_model = resolve_embedding_model_for_provider(
        embedding_provider, embedding_model_raw
    )

    ec = dmanifest.get("embedding_cache") or {}
    if str(router_embedding_cache_mode or "").strip():
        embedding_cache_mode = str(router_embedding_cache_mode).strip().lower()
    else:
        embedding_cache_mode = str(ec.get("cache_mode") or EMBEDDING_CACHE_OFF)

    cache_id = str(router_embedding_cache_id or ec.get("cache_id") or "").strip()
    cache_path_ov = str(router_embedding_cache_path or "").strip() or None
    if router_embedding_cache_writeback is not None:
        embedding_cache_writeback = bool(router_embedding_cache_writeback)
    else:
        embedding_cache_writeback = bool(ec.get("writeback", True))

    resolved_ode: Optional[int] = None
    if router_openai_embedding_dimensions is not None:
        resolved_ode = int(router_openai_embedding_dimensions)
    else:
        top_ode = dmanifest.get("openai_embedding_dimensions")
        if top_ode is not None:
            resolved_ode = int(top_ode)
        else:
            ec_ode = (dmanifest.get("embedding_cache") or {}).get(
                "openai_embedding_dimensions"
            )
            if ec_ode is not None:
                resolved_ode = int(ec_ode)

    if e2e_benchmark_path is not None:
        bench_jsonl: Optional[Path] = e2e_benchmark_path.resolve()
    else:
        p = str(
            (dmanifest.get("source_benchmark") or {}).get("benchmark_path") or ""
        ).strip()
        bench_jsonl = Path(p).expanduser().resolve() if p else None

    if embedding_cache_mode != EMBEDDING_CACHE_OFF:
        if bench_jsonl is None or not bench_jsonl.is_file():
            raise FileNotFoundError(
                f"Router dataset benchmark jsonl missing or invalid for embedding cache: {bench_jsonl}"
            )

    cache_paths: Optional[QueryEmbeddingCachePaths] = None
    cache_root_display: Optional[str] = None
    if embedding_cache_mode != EMBEDDING_CACHE_OFF:
        if not cache_id:
            raise ValueError(
                "Router embedding cache is enabled but embedding_cache_id is empty. "
                "Set router.dataset.embedding_cache_id (or e2e router_embedding_cache_id)."
            )
        cache_paths = make_query_embedding_cache_paths(
            bench_jsonl,
            provider=embedding_provider,
            model=embedding_model,
            cache_id=cache_id,
            openai_dimensions=resolved_ode,
        )
        if cache_path_ov:
            cache_paths = QueryEmbeddingCachePaths(
                run_root=Path(cache_path_ov).expanduser().resolve()
            )
        cache_root_display = str(cache_paths.run_root.resolve())
        if embedding_cache_mode == EMBEDDING_CACHE_REQUIRED and (
            not cache_paths.embeddings_jsonl.is_file()
        ):
            raise FileNotFoundError(
                f"Required query embedding cache missing: {cache_paths.embeddings_jsonl}. "
                "Run: make router-build-query-embedding-cache CONFIG=..."
            )

    corp_raw = retrieval_asset_dir
    if corp_raw is None:
        sc = dmanifest.get("source_corpus") or {}
        corp_raw = Path(str(sc.get("retrieval_asset_dir") or ""))
    if not corp_raw.is_dir():
        raise FileNotFoundError(f"Corpus dir missing or not a directory: {corp_raw}")
    feat_ctx = _feature_context_for_corpus(corp_raw)

    resolved_architecture_id: Optional[str] = (
        str(router_architecture_id).strip()
        if router_architecture_id and str(router_architecture_id).strip()
        else None
    )
    models_root = rb / str(router_id) / "models"
    if resolved_architecture_id is None and models_root.is_dir():
        candidates = sorted(
            [
                p.name
                for p in models_root.iterdir()
                if p.is_dir() and str(p.name).strip()
            ]
        )
        if len(candidates) == 1:
            resolved_architecture_id = candidates[0]
        elif len(candidates) > 1:
            raise ValueError(
                "router_architecture_id is required when multiple router models exist: "
                f"{candidates}"
            )

    candidate_paths = []
    if resolved_architecture_id is not None:
        candidate_paths.append(
            make_router_model_paths_for_cli(
                router_id,
                router_base=rb,
                input_mode=mode,
                router_architecture_id=resolved_architecture_id,
                router_task_type=task_type,
            )
        )
        candidate_paths.append(
            make_router_model_paths_for_cli(
                router_id,
                router_base=rb,
                input_mode=mode,
                router_architecture_id=resolved_architecture_id,
            )
        )
    candidate_paths.append(
        make_router_model_paths_for_cli(
            router_id, router_base=rb, input_mode=mode, router_architecture_id=None
        )
    )
    candidate_paths.append(
        make_router_model_paths_for_cli(
            router_id,
            router_base=rb,
            input_mode="both",
            router_architecture_id=None,
        )
    )

    selected_mp = None
    for mp in candidate_paths:
        if mp.checkpoint.is_file():
            selected_mp = mp
            break
    if selected_mp is None:
        raise FileNotFoundError(
            "Missing router checkpoint. Tried: "
            + ", ".join(str(p.checkpoint) for p in candidate_paths)
        )
    router = load_router_checkpoint(
        selected_mp.checkpoint,
        device=device,
        manifest_path=selected_mp.manifest,
        router_task_type=task_type,
    )
    validate_router_embedding_compatibility(
        model_manifest=router.manifest,
        dataset_manifest=dmanifest,
    )
    return RouterInferenceContext(
        router=router,
        normalizer=normalizer,
        embedding_model=embedding_model,
        input_mode=mode,
        feature_context=feat_ctx,
        embedding_provider=embedding_provider,
        embedding_cache_mode=embedding_cache_mode,
        embedding_cache_writeback=embedding_cache_writeback,
        embedding_cache_id=cache_id,
        embedding_cache_path_override=cache_path_ov,
        benchmark_jsonl_path=bench_jsonl,
        embedding_cache_root=cache_root_display,
        openai_embedding_dimensions=resolved_ode,
    )


def compute_query_tensors_for_router_batch(
    queries: Sequence[str],
    ictx: RouterInferenceContext,
    *,
    question_ids: Optional[Sequence[str]] = None,
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
        if ictx.embedding_cache_mode == EMBEDDING_CACHE_OFF:
            prov_parsed = parse_embedding_provider(ictx.embedding_provider)
            qe = embed_queries(
                texts,
                model_name=ictx.embedding_model,
                batch_size=st_batch_size,
                provider=ictx.embedding_provider,
                openai_dimensions=ictx.openai_embedding_dimensions,
                embedder_factory=(
                    ictx.query_embedder
                    if prov_parsed == EMBEDDING_PROVIDER_SENTENCE_TRANSFORMERS
                    else None
                ),
            )
        elif question_ids is None:
            if ictx.embedding_cache_mode == EMBEDDING_CACHE_REQUIRED:
                raise ValueError(
                    "embedding_cache_mode='required' needs question_id on each benchmark row; "
                    "pass question_ids= to compute_query_tensors_for_router_batch"
                )
            prov_parsed = parse_embedding_provider(ictx.embedding_provider)
            qe = embed_queries(
                texts,
                model_name=ictx.embedding_model,
                batch_size=st_batch_size,
                provider=ictx.embedding_provider,
                openai_dimensions=ictx.openai_embedding_dimensions,
                embedder_factory=(
                    ictx.query_embedder
                    if prov_parsed == EMBEDDING_PROVIDER_SENTENCE_TRANSFORMERS
                    else None
                ),
            )
        else:
            if len(question_ids) != n:
                raise ValueError(
                    f"question_ids length {len(question_ids)} != queries length {n}"
                )
            aligned: List[Dict[str, Any]] = []
            for i in range(n):
                aligned.append(
                    {
                        "question_id": str(question_ids[i] or "").strip(),
                        "question": texts[i],
                    }
                )
            if ictx.benchmark_jsonl_path is None:
                raise RuntimeError(
                    "benchmark_jsonl_path is unset; cannot resolve embedding cache. "
                    "Ensure router dataset manifest sets source_benchmark.benchmark_path."
                )
            emb, meta = resolve_aligned_query_embeddings(
                aligned,
                embedding_provider=ictx.embedding_provider,
                embedding_model=ictx.embedding_model,
                cache_mode=ictx.embedding_cache_mode,
                benchmark_path=ictx.benchmark_jsonl_path,
                cache_id=ictx.embedding_cache_id,
                cache_path_override=ictx.embedding_cache_path_override,
                writeback=ictx.embedding_cache_writeback,
                batch_size=st_batch_size,
                openai_embedding_dimensions=ictx.openai_embedding_dimensions,
            )
            qe = emb
            for k in ("hits", "misses", "live_computed", "appended_rows"):
                if k in meta:
                    ictx.router_cache_stats[k] = int(
                        ictx.router_cache_stats.get(k, 0)
                    ) + int(meta[k])
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
    *,
    question_id: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (query_embedding, feature_vector) batches for ``predict_batch``."""
    ids = [question_id] if question_id is not None else None
    return compute_query_tensors_for_router_batch(
        [query], ictx, question_ids=ids, st_batch_size=1
    )


def predict_router_weight(
    query: str,
    ictx: RouterInferenceContext,
    *,
    question_id: Optional[str] = None,
) -> np.ndarray:
    """Predicted dense weight(s) for one query."""
    qe, qf = compute_query_tensors_for_router(query, ictx, question_id=question_id)
    return predict_batch(ictx.router, qe, qf)


def predict_router_class_id(
    query: str,
    ictx: RouterInferenceContext,
    *,
    question_id: Optional[str] = None,
) -> np.ndarray:
    """Predicted class id(s) for one query."""
    qe, qf = compute_query_tensors_for_router(query, ictx, question_id=question_id)
    return predict_class_id_batch(ictx.router, qe, qf)

"""Load YAML into :class:`PipelineConfig` and resolve filesystem paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Type, TypeVar, get_type_hints

import yaml

import surf_rag.config.schema as _schema_module
from surf_rag.evaluation.artifact_paths import (
    benchmark_bundle_dir,
    router_dataset_dir,
    router_model_dir,
    router_oracle_dir,
)
from surf_rag.config.schema import (
    AlignmentSection,
    CorpusSection,
    E2ESection,
    EntityMatchingSection,
    FiguresSection,
    GenerationSection,
    GraphRetrievalSweepSection,
    ModelSetupSection,
    OracleSection,
    PathsSection,
    PipelineConfig,
    RawSourcesSection,
    RetrievalSection,
    RouterDatasetSection,
    RouterSection,
    RouterTrainSection,
    SecretsSection,
)

T = TypeVar("T")


def _merge_dataclass(cls: Type[T], data: dict[str, Any] | None, defaults: T) -> T:
    if not data:
        return defaults
    # ``schema`` uses ``from __future__ import annotations``; ``fields().type`` may be
    # a string. Resolve real types so nested dataclasses (e.g. ``FiguresSection.theme``)
    # merge instead of being left as raw dicts.
    try:
        hints = get_type_hints(
            cls,
            globalns=vars(_schema_module),
            localns=vars(_schema_module),
        )
    except Exception:
        hints = {}
    kwargs: dict[str, Any] = {}
    for f in fields(defaults):
        if f.name not in data:
            continue
        val = data[f.name]
        field_type = hints.get(f.name, f.type)
        if is_dataclass(field_type) and isinstance(val, dict):
            sub_default = getattr(defaults, f.name)
            kwargs[f.name] = _merge_dataclass(field_type, val, sub_default)  # type: ignore[arg-type]
        else:
            kwargs[f.name] = val
    return replace(defaults, **kwargs)


def pipeline_config_from_dict(raw: dict[str, Any]) -> PipelineConfig:
    """Build :class:`PipelineConfig` from a loaded YAML dict."""
    base = PipelineConfig()
    if not raw:
        return base
    out = PipelineConfig(
        schema_version=str(raw.get("schema_version", base.schema_version)),
        experiment_id=raw.get("experiment_id", base.experiment_id),
        seed=int(raw.get("seed", base.seed)),
        paths=_merge_dataclass(PathsSection, raw.get("paths"), base.paths),
        raw_sources=_merge_dataclass(
            RawSourcesSection, raw.get("raw_sources"), base.raw_sources
        ),
        model_setup=_merge_dataclass(
            ModelSetupSection, raw.get("model_setup"), base.model_setup
        ),
        corpus=_merge_dataclass(CorpusSection, raw.get("corpus"), base.corpus),
        alignment=_merge_dataclass(
            AlignmentSection, raw.get("alignment"), base.alignment
        ),
        entity_matching=_merge_dataclass(
            EntityMatchingSection, raw.get("entity_matching"), base.entity_matching
        ),
        oracle=_merge_dataclass(OracleSection, raw.get("oracle"), base.oracle),
        router=RouterSection(
            dataset=_merge_dataclass(
                RouterDatasetSection,
                (
                    (raw.get("router") or {}).get("dataset")
                    if isinstance(raw.get("router"), dict)
                    else None
                ),
                base.router.dataset,
            ),
            train=_merge_dataclass(
                RouterTrainSection,
                (
                    (raw.get("router") or {}).get("train")
                    if isinstance(raw.get("router"), dict)
                    else None
                ),
                base.router.train,
            ),
        ),
        retrieval=_merge_dataclass(
            RetrievalSection, raw.get("retrieval"), base.retrieval
        ),
        generation=_merge_dataclass(
            GenerationSection, raw.get("generation"), base.generation
        ),
        e2e=_merge_dataclass(E2ESection, raw.get("e2e"), base.e2e),
        secrets=_merge_dataclass(SecretsSection, raw.get("secrets"), base.secrets),
        graph_retrieval_sweep=_merge_dataclass(
            GraphRetrievalSweepSection,
            raw.get("graph_retrieval_sweep"),
            base.graph_retrieval_sweep,
        ),
        figures=_merge_dataclass(FiguresSection, raw.get("figures"), base.figures),
    )
    e2e = out.e2e
    if e2e.completion_window is None:
        e2e = replace(e2e, completion_window=out.generation.completion_window)
        out = replace(out, e2e=e2e)
    if not e2e.cross_encoder_model:
        e2e = replace(e2e, cross_encoder_model=out.model_setup.cross_encoder_model)
        out = replace(out, e2e=e2e)
    raw_e2e = raw.get("e2e") if isinstance(raw.get("e2e"), dict) else {}
    if not raw_e2e or "fusion_keep_k" not in raw_e2e:
        out = replace(
            out,
            e2e=replace(out.e2e, fusion_keep_k=out.oracle.fusion_keep_k),
        )
    if not raw_e2e or "branch_top_k" not in raw_e2e:
        out = replace(
            out,
            e2e=replace(out.e2e, branch_top_k=out.oracle.branch_top_k),
        )
    return _coerce_yaml_scalar_types(out)


def _opt_path_str(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _raw_dataset_path(v: Any) -> str | None:
    """Normalize NQ / 2Wiki raw file paths: empty YAML string → None."""
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _id_str(v: Any) -> str:
    """Normalize identifier scalars: ``None`` -> empty string, else ``str(v)``."""
    if v is None:
        return ""
    return str(v)


def _coerce_yaml_scalar_types(cfg: PipelineConfig) -> PipelineConfig:
    """YAML may parse unquoted numbers as int; path components must be strings."""
    p = cfg.paths
    paths = replace(
        p,
        data_base=str(p.data_base),
        benchmark_base=_opt_path_str(p.benchmark_base),
        router_base=_opt_path_str(p.router_base),
        figures_base=_opt_path_str(p.figures_base),
        benchmark_name=str(p.benchmark_name),
        benchmark_id=str(p.benchmark_id),
        router_id=_id_str(p.router_id),
        router_architecture_id=_opt_path_str(p.router_architecture_id),
        hf_home=_opt_path_str(p.hf_home),
        transformers_cache=_opt_path_str(p.transformers_cache),
    )
    rs = cfg.raw_sources
    raw_sources = replace(
        rs,
        nq_path=_raw_dataset_path(rs.nq_path),
        wiki2_path=_raw_dataset_path(rs.wiki2_path),
        hotpotqa_path=_raw_dataset_path(rs.hotpotqa_path),
        nq_version=_opt_path_str(rs.nq_version),
        wiki2_version=_opt_path_str(rs.wiki2_version),
        hotpotqa_version=_opt_path_str(rs.hotpotqa_version),
    )
    out = replace(cfg, paths=paths, raw_sources=raw_sources)
    if out.experiment_id is not None:
        out = replace(out, experiment_id=str(out.experiment_id))
    fg = out.figures
    out = replace(
        out,
        figures=replace(
            fg,
            image_format=str(fg.image_format or "png").strip().lower(),
        ),
    )
    return out


def load_pipeline_config(path: Path) -> PipelineConfig:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping, got {type(raw)}")
    return pipeline_config_from_dict(raw)


@dataclass(frozen=True)
class ResolvedPaths:
    """Absolute paths derived from a :class:`PipelineConfig`."""

    data_base: Path
    figures_base: Path
    benchmark_base: Path
    router_base: Path
    benchmark_name: str
    benchmark_id: str
    router_id: str
    router_architecture_id: str | None
    bundle: Path
    benchmark_dir: Path
    benchmark_path: Path
    corpus_dir: Path
    docstore_path: Path
    corpus_path: Path
    evaluations_dir: Path
    hf_home: Path
    transformations_cache: Path
    router_oracle_dir: Path
    router_dataset_dir: Path
    router_model_dir: Path


def resolve_paths(cfg: PipelineConfig) -> ResolvedPaths:
    p = cfg.paths
    data_base = Path(p.data_base).expanduser().resolve()
    bbase = (
        Path(p.benchmark_base).expanduser().resolve()
        if p.benchmark_base
        else data_base / "benchmarks"
    )
    rbase = (
        Path(p.router_base).expanduser().resolve()
        if p.router_base
        else data_base / "router"
    )
    name, bid = p.benchmark_name, p.benchmark_id
    rid = p.router_id
    bundle = benchmark_bundle_dir(bbase, name, bid)
    benchmark_dir = bundle / "benchmark"
    benchmark_path = benchmark_dir / "benchmark.jsonl"
    corpus_dir = bundle / "corpus"
    docstore_path = corpus_dir / "docstore.sqlite"
    corpus_path = corpus_dir / "corpus.jsonl"
    evaluations_dir = bundle / "evaluations"
    hf = (
        Path(p.hf_home).expanduser().resolve() if p.hf_home else corpus_dir / "hf_cache"
    )
    trans = (
        Path(p.transformers_cache).expanduser().resolve()
        if p.transformers_cache
        else hf / "transformers"
    )
    if p.figures_base and str(p.figures_base).strip():
        figures_base = Path(str(p.figures_base).strip()).expanduser().resolve()
    else:
        figures_base = data_base / "figures"
    return ResolvedPaths(
        data_base=data_base,
        figures_base=figures_base,
        benchmark_base=bbase,
        router_base=rbase,
        benchmark_name=name,
        benchmark_id=bid,
        router_id=rid,
        router_architecture_id=p.router_architecture_id,
        bundle=bundle,
        benchmark_dir=benchmark_dir,
        benchmark_path=benchmark_path,
        corpus_dir=corpus_dir,
        docstore_path=docstore_path,
        corpus_path=corpus_path,
        evaluations_dir=evaluations_dir,
        hf_home=hf,
        transformations_cache=trans,
        router_oracle_dir=router_oracle_dir(rbase, rid),
        router_dataset_dir=router_dataset_dir(rbase, rid),
        router_model_dir=router_model_dir(rbase, rid),
    )


def e2e_run_root(cfg: PipelineConfig, *, policy_value: str, run_id: str) -> Path:
    """``.../evaluations/<policy>/<run_id>/`` (same as ``e2e_policy_run_dir``)."""
    from surf_rag.evaluation.artifact_paths import e2e_policy_run_dir

    rp = resolve_paths(cfg)
    return e2e_policy_run_dir(
        rp.benchmark_base, rp.benchmark_name, rp.benchmark_id, policy_value, run_id
    )


def config_to_resolved_dict(cfg: PipelineConfig, rp: ResolvedPaths) -> dict[str, Any]:
    """Serializable dict for ``resolved_config.yaml`` (paths as strings)."""
    d = asdict(cfg)
    d["resolved_paths"] = {
        "data_base": str(rp.data_base),
        "figures_base": str(rp.figures_base),
        "benchmark_base": str(rp.benchmark_base),
        "router_base": str(rp.router_base),
        "bundle": str(rp.bundle),
        "benchmark_path": str(rp.benchmark_path),
        "corpus_dir": str(rp.corpus_dir),
        "docstore_path": str(rp.docstore_path),
        "corpus_path": str(rp.corpus_path),
        "evaluations_dir": str(rp.evaluations_dir),
        "hf_home": str(rp.hf_home),
        "transformers_cache": str(rp.transformations_cache),
        "router_oracle_dir": str(rp.router_oracle_dir),
        "router_dataset_dir": str(rp.router_dataset_dir),
        "router_model_dir": str(rp.router_model_dir),
        "router_architecture_id": rp.router_architecture_id,
    }
    return d


def validate_e2e_config(cfg: PipelineConfig) -> None:
    pol = (cfg.e2e.policy or "").strip().lower().replace("_", "-")
    if pol in ("learned-soft", "learned-hard", "oracle-upper-bound"):
        if not str(cfg.paths.router_id).strip():
            raise ValueError(
                "e2e policy learned-* and oracle-upper-bound require paths.router_id"
            )

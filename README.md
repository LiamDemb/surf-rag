# SuRF-RAG: Supervised Retrieval Fusion for Mixed-Reasoning QA

## Router training (architectures + input ablations)

The router is trained on a Parquet dataset built from oracle performance curves. One router dataset id (`ROUTER_ID`) shares one `dataset/`; trained checkpoints, metrics, and per-split predictions are stored under architecture + input-mode folders:

```text
$DATA_BASE/router/$ROUTER_ID/
  oracle/
  dataset/
    router_dataset.parquet
    manifest.json
  models/
    $ROUTER_ARCHITECTURE_ID/
      both/            # default: query embedding + normalized query features
      query-features/  # V1 query features only
      embedding/       # query embedding only
        model.pt
        manifest.json
        metrics.json
        training_history.json
        predictions_{train,dev,test}.jsonl
  model/               # legacy single-model location (read fallback)
    <input_mode>/
```

**Input modes**

| Mode             | Uses                                                                      |
| ---------------- | ------------------------------------------------------------------------- |
| `both`           | `query_embedding` and `feature_vector_norm` (14-d normalized V1 features) |
| `query-features` | `feature_vector_norm` only                                                |
| `embedding`      | `query_embedding` only                                                    |

**Makefile**

- One variant: set `ROUTER_INPUT_MODE` (default `both`), then `make router-train` / `make router-eval`.
- All ablations: `make router-train-ablations` and `make router-evaluate-ablations` (modes listed in `ROUTER_INPUT_MODES`).

**CLI** (see `poetry run python -m scripts.router.train_router --help`):

- `--router-architecture-id` is required for training and maps to `models/<id>/...`.
- `--architecture` selects implementation (`mlp-v1`, `logreg-v1`, `polyreg-v1`, or `tower_v01`).
- `--architecture-kwargs` accepts a JSON object (validated per architecture).
- `--input-mode` or env `ROUTER_INPUT_MODE` selects the branch-input ablation.

To compare runs, use `metrics.json` (and optional `predictions_*.jsonl`) under each `models/<router_architecture_id>/<input_mode>/` for the same `ROUTER_ID`.

**Config keys**

- `paths.router_architecture_id`: chosen architecture artifact id for downstream learned-router inference.
- `router.train.architecture`: architecture family (`mlp-v1`, `logreg-v1`, `polyreg-v1`, `tower_v01`).
- `polyreg-v1`: logistic regression on polynomial features up to **`architecture_kwargs.degree`** (default `2`). Optional **`excluded_features`**: drop named V1 columns (router feature order) before expanding monomials. Optional **`max_expanded_features`** caps the monomial count (large `degree` × wide inputs will error until you lower degree or shrink inputs).
- `router.train.architecture_kwargs`: per-architecture validated kwargs.

When `paths.router_architecture_id` is omitted in e2e:

- if exactly one child directory exists under `.../models/`, it is auto-selected;
- if multiple exist, the run fails with a disambiguation error;
- if no `models/` bundle exists, inference falls back to legacy `model/<input_mode>/`.

## End-to-end benchmark & evaluation

Benchmark bundles are defined in your pipeline YAML (`paths.benchmark_base`, `benchmark_name`, `benchmark_id`). Routed retrieval, optional cross-encoder reranking, OpenAI Batch generation, and overlap-split metrics are documented in **[docs/dev/end-to-end-system-and-evaluation.md](docs/dev/end-to-end-system-and-evaluation.md)**. Make targets: `e2e-prepare`, `e2e-submit`, `e2e-collect`, `e2e-evaluate`, `e2e-smoke-test-v01` (all use `CONFIG`, default `configs/pipelines/surf-bench-200.yaml`).

`e2e.policy` supports `dense-only`, `graph-only`, `50-50`, `learned-soft`, `learned-hard`, and `oracle-upper-bound`.
For `oracle-upper-bound`, retrieval is oracle-soft (per-question best fusion bin from `oracle_scores.jsonl`), is benchmark-scoped under `evaluations/oracle-upper-bound/<run_id>/`, is test-only, and fails fast if router test QIDs are missing in oracle artifacts.

**Config-driven runs:** every stage uses `--config "$(CONFIG)"`. Override with `CONFIG=...` or `make print-resolved-config`. See **[docs/config-driven-workflows.md](docs/config-driven-workflows.md)** and `configs/templates/`.

**LLM QA generation** always uses a forced OpenAI **`format_answer`** tool call (`reasoning` plus short `answer`). Batch collect writes `answer` (for EM/F1), `generation_reasoning`, and optional `generation_parse_error` into `generation/answers.jsonl`.

### Oracle retrieval upper-bound report

You can aggregate retrieval-only oracle upper-bound metrics on the router test split without rerunning retrieval:

- `poetry run python -m scripts.router.report_oracle_upper_bound --config <pipeline.yaml>`
- Output defaults to `data/router/<router_id>/oracle/reports/oracle_upper_bound_test.json`
- Metrics include NDCG/Hit/Recall at `@5`, `@10`, and `@20`

### Model cache and corpus entity artifacts

- **`make setup-models`** — downloads/warms `sentence-transformers/all-MiniLM-L6-v2` and `cross-encoder/ms-marco-MiniLM-L-6-v2` using `HF_HOME` / `TRANSFORMERS_CACHE` (exported by the Makefile; see `.env.example`).
- **Shared in-process models** — `src/surf_rag/core/model_cache.py` deduplicates `SentenceTransformer` / `CrossEncoder` construction across dense, graph, reranking, and router paths.
- **`make build-entity-matching-artifacts`** — builds `entity_phrase_records.parquet`, `entity_phrase_matcher.pkl`, and `entity_matching_manifest.json` under `CORPUS_DIR` for faster, reproducible `kg_linkable` features (use `ENTITY_MATCHING_FORCE=1` to rebuild).
- **E2E router batching** — `E2E_ROUTER_INFERENCE_BATCH_SIZE` (default `32`) batches learned-router query tensors; see the e2e dev doc for details.

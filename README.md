# SuRF-RAG: Supervised Retrieval Fusion for Mixed-Reasoning QA

## Router training (MLP) and input ablations

The router is trained on a Parquet dataset built from oracle soft labels. One run id (`ROUTER_ID`) shares a single `dataset/`; trained checkpoints, metrics, and per-split predictions are stored under a **per-input ablation** directory:

```text
$DATA_BASE/router/$ROUTER_ID/
  oracle/
  dataset/
    router_dataset.parquet
    manifest.json
  model/
    both/                 # default: query embedding + normalized query features
    query-features/     # V1 query features only
    embedding/          # query embedding only
      model.pt
      manifest.json
      metrics.json
      training_history.json
      predictions_{train,dev,test}.jsonl
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

- `--input-mode` or env `ROUTER_INPUT_MODE` selects the output folder and architecture.

To compare runs, use `metrics.json` (and optional `predictions_*.jsonl`) under each `model/<input_mode>/` for the same `ROUTER_ID`.

## End-to-end benchmark & evaluation

Benchmark bundles are defined in your pipeline YAML (`paths.benchmark_base`, `benchmark_name`, `benchmark_id`). Routed retrieval, optional cross-encoder reranking, OpenAI Batch generation, and overlap-split metrics are documented in **[docs/dev/end-to-end-system-and-evaluation.md](docs/dev/end-to-end-system-and-evaluation.md)**. Make targets: `e2e-prepare`, `e2e-submit`, `e2e-collect`, `e2e-evaluate`, `e2e-smoke-test-v01` (all use `CONFIG`, default `configs/pipelines/surf-bench-200.yaml`).

**Config-driven runs:** every stage uses `--config "$(CONFIG)"`. Override with `CONFIG=...` or `make print-resolved-config`. See **[docs/config-driven-workflows.md](docs/config-driven-workflows.md)** and `configs/templates/`.

**LLM QA generation** always uses a forced OpenAI **`format_answer`** tool call (`reasoning` plus short `answer`). Batch collect writes `answer` (for EM/F1), `generation_reasoning`, and optional `generation_parse_error` into `generation/answers.jsonl`.

### Model cache and corpus entity artifacts

- **`make setup-models`** — downloads/warms `sentence-transformers/all-MiniLM-L6-v2` and `cross-encoder/ms-marco-MiniLM-L-6-v2` using `HF_HOME` / `TRANSFORMERS_CACHE` (exported by the Makefile; see `.env.example`).
- **Shared in-process models** — `src/surf_rag/core/model_cache.py` deduplicates `SentenceTransformer` / `CrossEncoder` construction across dense, graph, reranking, and router paths.
- **`make build-entity-matching-artifacts`** — builds `entity_phrase_records.parquet`, `entity_phrase_matcher.pkl`, and `entity_matching_manifest.json` under `CORPUS_DIR` for faster, reproducible `kg_linkable` features (use `ENTITY_MATCHING_FORCE=1` to rebuild).
- **E2E router batching** — `E2E_ROUTER_INFERENCE_BATCH_SIZE` (default `32`) batches learned-router query tensors; see the e2e dev doc for details.

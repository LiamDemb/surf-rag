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

| Mode | Uses |
|------|------|
| `both` | `query_embedding` and `feature_vector_norm` (14-d normalized V1 features) |
| `query-features` | `feature_vector_norm` only |
| `embedding` | `query_embedding` only |

**Makefile**

- One variant: set `ROUTER_INPUT_MODE` (default `both`), then `make router-train` / `make router-eval`.
- All ablations: `make router-train-ablations` and `make router-evaluate-ablations` (modes listed in `ROUTER_INPUT_MODES`).

**CLI** (see `poetry run python -m scripts.router.train_router --help`):

- `--input-mode` or env `ROUTER_INPUT_MODE` selects the output folder and architecture.

To compare runs, use `metrics.json` (and optional `predictions_*.jsonl`) under each `model/<input_mode>/` for the same `ROUTER_ID`.

## End-to-end benchmark & evaluation

Benchmark bundles live under **`BENCHMARK_BASE`/`BENCHMARK_NAME`/`BENCHMARK_ID`** (default `data/benchmarks/...`). Routed retrieval, optional cross-encoder reranking, OpenAI Batch generation, and overlap-split metrics are documented in **[docs/dev/end-to-end-system-and-evaluation.md](docs/dev/end-to-end-system-and-evaluation.md)**. Make targets: `e2e-prepare`, `e2e-submit`, `e2e-collect`, `e2e-evaluate`, `e2e-smoke-test-v01`.
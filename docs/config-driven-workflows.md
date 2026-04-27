# Config-driven workflows

This guide explains how to run the SuRF-RAG research pipeline using **versioned YAML configs** under `configs/` instead of editing `.env` and `Makefile` variables for every experiment.

## Mental model

| Layer                      | Role                                                                                                                                                                       |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`.env`**                 | Secrets (`OPENAI_API_KEY`) and optional local machine paths (caches) only. Not for experiment design.                                                                      |
| **`configs/*.yaml`**       | Experiment recipes: benchmark identity, corpus settings, oracle, router, E2E, generation.                                                                                  |
| **`Makefile` targets**     | Choose the **stage** (ingest, build-corpus, oracle, router, e2e).                                                                                                          |
| **`CONFIG=path.yaml`**     | Choose the **recipe** (default in `Makefile`: `configs/pipelines/surf-bench-200.yaml`). |
| **`resolved_config.yaml`** | Written into major output dirs so runs are **auditable**; linked from `manifest.json` where applicable.                                                                    |

**Precedence (same everywhere):** explicit **CLI** overrides config; **config** defines experiment semantics; **defaults** in `surf_rag.config.schema` are last. Environment variables are still set from config when you pass `--config` (`apply_pipeline_env_from_config`) so code paths that read `os.environ` stay consistent.

## Schema

- **Version string:** `schema_version: surf-rag/pipeline/v1` (default if omitted in loader).
- **Typed model:** `PipelineConfig` in `src/surf_rag/config/schema.py`.
- **Resolved paths:** `resolve_paths()` in `src/surf_rag/config/loader.py` — bundle dir, `benchmark/`, `corpus/`, `docstore`, router oracle/dataset/model dirs, HF cache roots.
- **Inspect without running a stage:**  
  `make print-resolved-config CONFIG=configs/pipelines/surf-bench-200.yaml`  
  or `poetry run python -m surf_rag.config <file.yaml>`

Field-level inventory of targets vs sections: [CONFIG_INVENTORY.md](CONFIG_INVENTORY.md).

## Config file layout

| Location                                                | Use                                                                                             |
| ------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `configs/templates/`                                    | Copy-paste starters with comments (`pipeline.yaml`, `e2e_run.yaml`, `router_training.yaml`, …). |
| `configs/examples/`                                     | Committed **example** full pipelines.                                                           |
| `configs/pipelines/`                                    | Your primary **full** pipeline configs.                                                         |
| `configs/e2e/.../`                                      | **Named** E2E runs (policy tests, prompt tests).                                                |
| `configs/router/`, `configs/corpus/`, `configs/oracle/` | Stage-focused configs.                                                                          |

**Rule of thumb:** if the run is worth citing in a paper, **commit a named YAML**; for a disposable local check, CLI overrides on top of a template are fine.

**`raw_sources.nq_path` / `wiki2_path`:** With `--config`, these come only from YAML. Use `null` or an empty string to **omit** a dataset (no fallback to `NQ_PATH` / `2WIKI_PATH` in the environment). Omitting the key keeps the schema default paths.

## Workflows (commands)

**Print merged config (paths + sections):**

```bash
make print-resolved-config CONFIG=configs/pipelines/surf-bench-200.yaml
```

**Data → corpus (sequential):**

```bash
make ingest CONFIG=configs/pipelines/surf-bench-200.yaml
make fetch-wikipedia-articles CONFIG=...
make align-2wiki-support CONFIG=...
make build-corpus CONFIG=...
make build-entity-matching-artifacts CONFIG=...
```

Or `make pipeline` (uses default `CONFIG` unless you override).

**Oracle → soft labels:**

```bash
make oracle-labels CONFIG=configs/oracle/surf-bench-200-router-4000-test.yaml
# or stepwise: oracle-prepare, oracle-sweep-beta, oracle-create-soft-labels
```

**Router Parquet + training:**

```bash
make router-build-dataset CONFIG=configs/router/4000-test.yaml
make router-train CONFIG=...
make router-eval CONFIG=...
# Ablations: make router-train-ablations CONFIG=...  (overrides --input-mode per iter)
```

**E2E (one policy):**

```bash
# run_id / policy in YAML, or override from Make (CLI wins):
make e2e-submit CONFIG=configs/e2e/surf-bench/200-test/learned-soft-prompt-test-4.yaml
make e2e-collect CONFIG=...
make e2e-evaluate CONFIG=...
```

**E2E (all policies):** same `E2E_RUN_ID` prefix; optional `CONFIG` is forwarded to sub-makes.

```bash
export E2E_RUN_ID=e2e-20250101-abc
make e2e-run-all-policies CONFIG=configs/templates/e2e_multi_policy.yaml
```

**Smoke (dry-run, 1 question):**  
With CONFIG, use `configs/templates/smoke_test.yaml` or pass `CONFIG=...` and **also** set `E2E_RUN_ID` for uniqueness:

```bash
make e2e-smoke-test-v01 CONFIG=configs/templates/smoke_test.yaml
```

**Model download (HF + spaCy):**

```bash
make setup-models CONFIG=configs/pipelines/surf-bench-200.yaml
```

## Target-by-target Makefile reference

Default `CONFIG` is `configs/pipelines/surf-bench-200.yaml`. Every stage target runs the corresponding script with `--config "$(CONFIG)"`. Optional Make-only overrides: `E2E_RUN_ID`, `E2E_POLICY`, `E2E_SPLIT`, `E2E_POLICIES`, `SELECTED_BETA`, `ROUTER_INPUT_MODES`, `ENTITY_MATCHING_FORCE`, `ALIGN_2WIKI_EXTRA`.

| Target | Notes |
|--------|--------|
| `help` | Short usage |
| `print-resolved-config`, `print-paths`, `print-oracle-config`, `print-router-config` | Same: `python -m surf_rag.config "$(CONFIG)"` |
| `validate-oracle-config` | Checks benchmark JSONL + corpus dir exist (from YAML paths) |
| `validate-router-config` | Oracle checks + `labels/selected.jsonl` |
| `validate-router-train` | Router checks + `router_dataset.parquet` |
| `install`, `lock`, `test`, `install-hooks` | No `CONFIG` |
| `setup-models` … `pipeline` | Full `CONFIG` recipe |
| `oracle-*`, `router-*`, `e2e-*` | `CONFIG` + optional `E2E_*` / `SELECTED_BETA` overrides |
| `e2e-run-all-policies` | Uses `E2E_RUN_ID` as prefix if set, else `e2e-<policy>` |

`make pipeline` runs the four data stages in one invocation; the same `CONFIG` applies to all prerequisites.

## Migration: Make / `.env` → YAML

| Old knob                                 | Config section / field                                             |
| ---------------------------------------- | ------------------------------------------------------------------ |
| `BENCHMARK_NAME`, `BENCHMARK_ID`         | `paths.benchmark_name`, `paths.benchmark_id`                       |
| `BENCHMARK_BASE`, `DATA_BASE`            | `paths.benchmark_base`, `paths.data_base`                          |
| `ROUTER_ID`, `ROUTER_BASE`               | `paths.router_id`, `paths.router_base`                             |
| `NQ_PATH`, `2WIKI_PATH`                  | `raw_sources.nq_path`, `raw_sources.wiki2_path`                    |
| `EMBEDDING_MODEL`, `CROSS_ENCODER_MODEL` | `model_setup`                                                      |
| `ORACLE_*`                               | `oracle`                                                           |
| `TRAIN_RATIO`, `DEV_RATIO`, `TEST_RATIO` | `router.dataset`                                                   |
| `ROUTER_EPOCHS`, `ROUTER_*`              | `router.train` (and `apply_pipeline_env_from_config` mirrors many) |
| `E2E_*`                                  | `e2e`                                                              |
| `OPENAI_API_KEY`                         | Stays in `.env`; `secrets.openai_api_key_env` only names the var   |

## Artifacts

- **E2E run dir:** `resolved_config.yaml` + manifest entry `resolved_config` (see `e2e_runner.py`).
- **Build corpus, oracle, router dataset/model:** scripts call `write_resolved_config_yaml` where implemented in your branch.

## Tests

`tests/test_config_*.py` cover loading, path resolution, merge/CLI behavior, and resolved artifacts for E2E.

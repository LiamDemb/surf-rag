# Configuration inventory (Makefile targets and inputs)

This document maps each `Makefile` phony target to its inputs: recommended config file kind, `CONFIG=...` usage after migration, and common fields to change.

| Target                                                              | Purpose                  | Config                                | Common fields to change                   |
| ------------------------------------------------------------------- | ------------------------ | ------------------------------------- | ----------------------------------------- |
| `install`                                                           | poetry install           | no                                    | -                                         |
| `install-hooks`                                                     | pre-commit               | no                                    | -                                         |
| `setup-models`                                                      | download HF/spaCy models | pipeline or `model_setup` in template | `model_setup.*`, `paths.hf_home`          |
| `lock`                                                              | poetry lock              | no                                    | -                                         |
| `test`                                                              | pytest                   | no                                    | -                                         |
| `print-paths` (and `print-oracle-config` / `print-router-config`)    | merged YAML + paths      | any pipeline config                   | `paths.*`                                 |
| `ingest`                                                            | build benchmark JSONL    | full pipeline or corpus               | `raw_sources`, `paths.benchmark_name/id`  |
| `fetch-wikipedia-articles`                                          | fetch wiki pages         | full pipeline or corpus               | `raw_sources`, `paths`                    |
| `align-2wiki-support`                                               | align 2wiki              | full pipeline or corpus               | `alignment`, chunk env mirrors            |
| `align-2wiki-support-full`                                          | align with full report   | same                                  | set `full_report: true` or use Make extra |
| `build-corpus`                                                      | FAISS, graph, indexes    | full pipeline or corpus               | `corpus`, `model_setup`, `paths`          |
| `build-entity-matching-artifacts`                                   | phrase matcher           | corpus                                | `entity_matching`, `paths.corpus`         |
| `filter-benchmark`                                                  | filter by corpus         | pipeline                              | `paths`                                   |
| `pipeline`                                                          | ingest→fetch→align→build | full pipeline                         | all above                                 |
| `print-oracle-config` / `validate-oracle-config`                    | oracle checks            | pipeline + oracle                     | `oracle`, `router_id` in paths            |
| `oracle-prepare` / `sweep` / `create-soft-labels` / `oracle-labels` | oracle labels            | pipeline or oracle                    | `oracle.*`                                |
| `print-router-config` / `validate-router-config`                    | router checks            | pipeline                              | `router`                                  |
| `router-build-dataset` / `router-pipeline`                          | Parquet + splits         | pipeline or router                    | `router.dataset`                          |
| `validate-router-train`                                             | check Parquet            | pipeline                              | -                                         |
| `router-train` / `eval` / `*-ablations`                             | MLP                      | pipeline or router                    | `router.train`                            |
| `e2e-*`                                                             | E2E eval                 | e2e or full pipeline                  | `e2e.*`, `generation`                     |

Runtime `os.getenv` is still read by many modules; when you pass `--config`, `apply_pipeline_env_from_config` mirrors the YAML into the environment. `.env` should hold secrets (e.g. `OPENAI_API_KEY`) and optional local cache overrides only.

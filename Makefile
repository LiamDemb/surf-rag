.PHONY: install install-hooks setup-models lock test ingest fetch-wikipedia-articles build-corpus align-2wiki-support align-2wiki-support-full filter-benchmark pipeline \
	oracle-prepare oracle-sweep-beta oracle-create-soft-labels oracle-labels \
	router-build-dataset router-pipeline \
	print-oracle-config print-router-config validate-oracle-config validate-router-config

-include .env

NQ_PATH ?= data/raw/nq_100.jsonl
2WIKI_PATH ?= data/raw/2wikimultihop_100.jsonl
BENCHMARK_PATH ?= data/processed/benchmark.jsonl
OUTPUT_DIR ?= data/processed
CORPUS_PATH ?= $(OUTPUT_DIR)/corpus.jsonl
DOCSTORE_PATH ?= $(OUTPUT_DIR)/docstore.sqlite
HF_HOME ?= $(OUTPUT_DIR)/hf_cache

# --- Oracle run (soft labels) — set in .env for a consistent end-to-end run -----------------
ORACLE_BASE ?= data/oracle
ORACLE_BENCHMARK ?= mix
ORACLE_SPLIT ?= dev
ORACLE_RUN_ID ?= run1
ORACLE_BENCHMARK_PATH ?= $(BENCHMARK_PATH)
ORACLE_RETRIEVAL_ASSET_DIR ?= $(OUTPUT_DIR)
ORACLE_BRANCH_TOP_K ?= 25
ORACLE_FUSION_KEEP_K ?= 25
# Space-separated; must include the value you will select (recommended or manual)
ORACLE_BETAS ?= 0.5 1.0 2.0 5.0 10.0 20.0
# For create_soft_labels: argparse needs one --beta per value (sweep_beta uses --betas w/ nargs+)
ORACLE_BETA_FLAGS = $(foreach b,$(ORACLE_BETAS),--beta $(b))
ORACLE_MIN_ENTROPY_NATS ?= 0.1
# If empty, oracle-create-soft-labels reads recommended beta from the oracle run directory
SELECTED_BETA ?=

# --- Router dataset — defaults align with the oracle run above (override per experiment) ----
ROUTER_DATASET_BASE ?= data/router
ROUTER_BENCHMARK ?= $(ORACLE_BENCHMARK)
ROUTER_DATASET_ID ?= $(ORACLE_RUN_ID)
ROUTER_BENCHMARK_PATH ?= $(ORACLE_BENCHMARK_PATH)
ROUTER_ORACLE_BENCHMARK ?= $(ORACLE_BENCHMARK)
ROUTER_ORACLE_SPLIT ?= $(ORACLE_SPLIT)
ROUTER_ORACLE_RUN_ID ?= $(ORACLE_RUN_ID)
ORACLE_BASE_FOR_ROUTER ?= $(ORACLE_BASE)
SEED ?= 42
EMBEDDING_MODEL ?= all-MiniLM-L6-v2
EMBEDDING_MODEL_FOR_ROUTER ?= $(EMBEDDING_MODEL)

# Poetry Python for consistent CLI
PY = poetry run python

install:
	poetry install

install-hooks:
	poetry run pre-commit install

setup-models:
	HF_HOME="$(HF_HOME)" poetry run python scripts/setup_models.py

lock:
	poetry lock

test:
	poetry run pytest

ingest:
	poetry run python scripts/ingest_data.py \
		--nq "$(NQ_PATH)" \
		--2wiki "$(2WIKI_PATH)" \
		--output-dir "$(OUTPUT_DIR)"

fetch-wikipedia-articles:
	poetry run python scripts/fetch_wikipedia_articles.py \
		--benchmark "$(BENCHMARK_PATH)" \
		--nq "$(NQ_PATH)" \
		--2wiki "$(2WIKI_PATH)" \
		--output-dir "$(OUTPUT_DIR)" \
		--docstore "$(DOCSTORE_PATH)"

ALIGN_2WIKI_EXTRA ?=
align-2wiki-support:
	poetry run python scripts/align_2wiki_support.py \
		--benchmark "$(BENCHMARK_PATH)" \
		--docstore "$(DOCSTORE_PATH)" \
		$(ALIGN_2WIKI_EXTRA)

# Print full markdown report
align-2wiki-support-full:
	$(MAKE) align-2wiki-support ALIGN_2WIKI_EXTRA=--full-report

build-corpus:
	poetry run python scripts/build_corpus.py \
		--benchmark "$(BENCHMARK_PATH)" \
		--nq "$(NQ_PATH)" \
		--2wiki "$(2WIKI_PATH)" \
		--output-dir "$(OUTPUT_DIR)" \
		--docstore "$(DOCSTORE_PATH)"


filter-benchmark:
	poetry run python scripts/filter_benchmark_by_corpus.py \
		--benchmark "$(BENCHMARK_PATH)" \
		--corpus "$(CORPUS_PATH)"

# Full offline refresh: ingest → fetch → align (drop unresolved 2Wiki) → corpus + indexes
pipeline: ingest fetch-wikipedia-articles align-2wiki-support build-corpus

# ---------------------------------------------------------------------------
# Oracle: prepare → beta sweep → soft labels (canonical order)
# ---------------------------------------------------------------------------

print-oracle-config:
	@echo "ORACLE_BASE=$(ORACLE_BASE)"
	@echo "ORACLE_BENCHMARK=$(ORACLE_BENCHMARK) ORACLE_SPLIT=$(ORACLE_SPLIT) ORACLE_RUN_ID=$(ORACLE_RUN_ID)"
	@echo "ORACLE_BENCHMARK_PATH=$(ORACLE_BENCHMARK_PATH)"
	@echo "ORACLE_RETRIEVAL_ASSET_DIR=$(ORACLE_RETRIEVAL_ASSET_DIR)"
	@echo "ORACLE_BRANCH_TOP_K=$(ORACLE_BRANCH_TOP_K) ORACLE_FUSION_KEEP_K=$(ORACLE_FUSION_KEEP_K)"
	@echo "ORACLE_BETAS=$(ORACLE_BETAS)"
	@echo "ORACLE_BETA_FLAGS=$(ORACLE_BETA_FLAGS)"
	@echo "SELECTED_BETA=$(SELECTED_BETA) (empty => use recommended_beta.json for soft labels)"

validate-oracle-config:
	@test -n "$(ORACLE_BENCHMARK)" && test -n "$(ORACLE_SPLIT)" && test -n "$(ORACLE_RUN_ID)" && test -f "$(ORACLE_BENCHMARK_PATH)" && test -d "$(ORACLE_RETRIEVAL_ASSET_DIR)" || (echo "Missing oracle env/paths. Set ORACLE_BENCHMARK, ORACLE_SPLIT, ORACLE_RUN_ID, ORACLE_BENCHMARK_PATH, ORACLE_RETRIEVAL_ASSET_DIR"; exit 1)

oracle-prepare: validate-oracle-config
	$(PY) -m scripts.prepare_oracle_run \
		--benchmark "$(ORACLE_BENCHMARK)" \
		--split "$(ORACLE_SPLIT)" \
		--oracle-run-id "$(ORACLE_RUN_ID)" \
		--benchmark-path "$(ORACLE_BENCHMARK_PATH)" \
		--retrieval-asset-dir "$(ORACLE_RETRIEVAL_ASSET_DIR)" \
		--branch-top-k $(ORACLE_BRANCH_TOP_K) \
		--fusion-keep-k $(ORACLE_FUSION_KEEP_K) \
		--oracle-base "$(ORACLE_BASE)"

oracle-sweep-beta: validate-oracle-config
	$(PY) -m scripts.sweep_beta \
		--benchmark "$(ORACLE_BENCHMARK)" \
		--split "$(ORACLE_SPLIT)" \
		--oracle-run-id "$(ORACLE_RUN_ID)" \
		--min-entropy-nats $(ORACLE_MIN_ENTROPY_NATS) \
		--betas $(ORACLE_BETAS) \
		--oracle-base "$(ORACLE_BASE)"

# SELECTED_BETA: if empty, read recommended_beta.json; fallback 2.0
oracle-create-soft-labels: validate-oracle-config
	@BETA_SEL=""; \
	if [ -n "$(SELECTED_BETA)" ]; then BETA_SEL="$(SELECTED_BETA)"; \
	else BETA_SEL=$$($(PY) -c "import json, pathlib; p=pathlib.Path('$(ORACLE_BASE)')/'$(ORACLE_BENCHMARK)'/'$(ORACLE_SPLIT)'/'$(ORACLE_RUN_ID)'/'recommended_beta.json'; \
	print(json.loads(p.read_text(encoding='utf-8'))['recommended_beta'])" 2>/dev/null || echo "2.0"); fi; \
	echo "create_soft_labels: using --selected-beta $$BETA_SEL"; \
	$(PY) -m scripts.create_soft_labels \
		--benchmark "$(ORACLE_BENCHMARK)" \
		--split "$(ORACLE_SPLIT)" \
		--oracle-run-id "$(ORACLE_RUN_ID)" \
		$(ORACLE_BETA_FLAGS) \
		--selected-beta $$BETA_SEL \
		--oracle-base "$(ORACLE_BASE)"

oracle-labels: oracle-prepare oracle-sweep-beta oracle-create-soft-labels

# ---------------------------------------------------------------------------
# Router dataset (Parquet) — uses labels/selected.jsonl for the same oracle id
# ---------------------------------------------------------------------------

print-router-config:
	@echo "ROUTER_DATASET_BASE=$(ROUTER_DATASET_BASE) ROUTER_BENCHMARK=$(ROUTER_BENCHMARK) ROUTER_DATASET_ID=$(ROUTER_DATASET_ID)"
	@echo "ROUTER_BENCHMARK_PATH=$(ROUTER_BENCHMARK_PATH) ORACLE_BASE_FOR_ROUTER=$(ORACLE_BASE_FOR_ROUTER)"
	@echo "ROUTER_ORACLE_BENCHMARK=$(ROUTER_ORACLE_BENCHMARK) ROUTER_ORACLE_SPLIT=$(ROUTER_ORACLE_SPLIT) ROUTER_ORACLE_RUN_ID=$(ROUTER_ORACLE_RUN_ID)"
	@echo "ORACLE_BASE=$(ORACLE_BASE)"

validate-router-config:
	@test -n "$(ROUTER_BENCHMARK)" && test -n "$(ROUTER_DATASET_ID)" && test -f "$(ROUTER_BENCHMARK_PATH)" && test -n "$(ROUTER_ORACLE_BENCHMARK)" && test -n "$(ROUTER_ORACLE_SPLIT)" && test -n "$(ROUTER_ORACLE_RUN_ID)" || (echo "Set ROUTER_BENCHMARK, ROUTER_DATASET_ID, ROUTER_BENCHMARK_PATH, ROUTER_ORACLE_*"; exit 1)
	@test -f "$(ORACLE_BASE_FOR_ROUTER)/$(ROUTER_ORACLE_BENCHMARK)/$(ROUTER_ORACLE_SPLIT)/$(ROUTER_ORACLE_RUN_ID)/labels/selected.jsonl" || (echo "Missing oracle labels at $$(echo $(ORACLE_BASE_FOR_ROUTER))/$(ROUTER_ORACLE_BENCHMARK)/$(ROUTER_ORACLE_SPLIT)/$(ROUTER_ORACLE_RUN_ID)/labels/selected.jsonl — run make oracle-labels first"; exit 1)

router-build-dataset: validate-router-config
	$(PY) -m scripts.router.build_router_dataset \
		--router-benchmark "$(ROUTER_BENCHMARK)" \
		--router-dataset-id "$(ROUTER_DATASET_ID)" \
		--benchmark-path "$(ROUTER_BENCHMARK_PATH)" \
		--retrieval-asset-dir "$(ORACLE_RETRIEVAL_ASSET_DIR)" \
		--oracle-benchmark "$(ROUTER_ORACLE_BENCHMARK)" \
		--oracle-split "$(ROUTER_ORACLE_SPLIT)" \
		--oracle-run-id "$(ROUTER_ORACLE_RUN_ID)" \
		--router-base "$(ROUTER_DATASET_BASE)" \
		--oracle-base "$(ORACLE_BASE_FOR_ROUTER)" \
		--embedding-model "$(EMBEDDING_MODEL_FOR_ROUTER)" \
		--split-seed $(SEED) \
		--train-ratio $(TRAIN_RATIO) \
		--dev-ratio $(DEV_RATIO) \
		--test-ratio $(TEST_RATIO)

router-pipeline: oracle-labels router-build-dataset
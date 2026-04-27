.PHONY: install install-hooks setup-models lock test ingest fetch-wikipedia-articles build-corpus align-2wiki-support align-2wiki-support-full filter-benchmark pipeline \
	oracle-prepare oracle-sweep-beta oracle-create-soft-labels oracle-labels \
	router-build-dataset router-pipeline \
	router-train router-eval router-train-ablations router-evaluate-ablations \
	print-oracle-config print-router-config print-paths validate-oracle-config validate-router-config \
	validate-router-train \
	e2e-print-config e2e-prepare e2e-submit e2e-collect e2e-evaluate e2e-run \
	e2e-run-all-policies e2e-collect-all-policies e2e-evaluate-all-policies e2e-smoke-test-v01 \
	build-entity-matching-artifacts

-include .env
export OPENAI_API_KEY

# Data storage organisation
DATA_BASE ?= data
# Benchmark bundles live under $BENCHMARK_BASE/$BENCHMARK_NAME/$BENCHMARK_ID
BENCHMARK_BASE ?= $(DATA_BASE)/benchmarks
BENCHMARK_NAME ?= benchmark-name
BENCHMARK_ID ?= v01
ROUTER_ID ?= v01
SEED ?= 42

# Raw source inputs
NQ_PATH ?= data/raw/nq_100.jsonl
2WIKI_PATH ?= data/raw/2wikimultihop_100.jsonl

# Training / split knobs
EMBEDDING_MODEL ?= all-MiniLM-L6-v2
TRAIN_RATIO ?= 0.6
DEV_RATIO ?= 0.2
TEST_RATIO ?= 0.2

# Derived paths
BENCHMARK_BUNDLE_DIR := $(BENCHMARK_BASE)/$(BENCHMARK_NAME)/$(BENCHMARK_ID)
BENCHMARK_DIR := $(BENCHMARK_BUNDLE_DIR)/benchmark
CORPUS_DIR := $(BENCHMARK_BUNDLE_DIR)/corpus
EVALUATIONS_DIR := $(BENCHMARK_BUNDLE_DIR)/evaluations
BENCHMARK_PATH := $(BENCHMARK_DIR)/benchmark.jsonl
DOCSTORE_PATH := $(CORPUS_DIR)/docstore.sqlite
CORPUS_PATH := $(CORPUS_DIR)/corpus.jsonl
OUTPUT_DIR := $(CORPUS_DIR)
HF_HOME ?= $(CORPUS_DIR)/hf_cache
TRANSFORMERS_CACHE ?= $(HF_HOME)/transformers
export HF_HOME
export TRANSFORMERS_CACHE

# Router bundle (oracle + dataset + model)
ROUTER_BASE ?= $(DATA_BASE)/router
ROUTER_DIR := $(ROUTER_BASE)/$(ROUTER_ID)
ROUTER_ORACLE_DIR := $(ROUTER_DIR)/oracle
ROUTER_DATASET_DIR := $(ROUTER_DIR)/dataset
ROUTER_MODEL_DIR := $(ROUTER_DIR)/model

# Oracle run
ORACLE_RETRIEVAL_ASSET_DIR ?= $(CORPUS_DIR)
ORACLE_BENCHMARK_PATH ?= $(BENCHMARK_PATH)
ORACLE_BRANCH_TOP_K ?= 25
ORACLE_FUSION_KEEP_K ?= 25
ORACLE_BETAS ?= 0.5 1.0 2.0 5.0 10.0 20.0
ORACLE_BETA_FLAGS = $(foreach b,$(ORACLE_BETAS),--beta $(b))
ORACLE_MIN_ENTROPY_NATS ?= 0.1
SELECTED_BETA ?=

EMBEDDING_MODEL_FOR_ROUTER ?= $(EMBEDDING_MODEL)

# Router MLP training
ROUTER_EPOCHS ?= 100
ROUTER_BATCH_SIZE ?= 32
ROUTER_LEARNING_RATE ?= 0.001
ROUTER_TRAIN_DEVICE ?= cpu
# one of: both | query-features | embedding
ROUTER_INPUT_MODE ?= both
# space-separated; used by router-train-ablations / router-evaluate-ablations
ROUTER_INPUT_MODES ?= both query-features embedding

# End-to-end SuRF-RAG eval (routed retrieval → batch gen → metrics)
E2E_SPLIT ?= test
E2E_RUN_ID ?= e2e-$(shell date +%Y%m%d-%H%M%S)
E2E_POLICY ?= learned-soft
E2E_POLICIES ?= dense-only graph-only 50-50 learned-soft learned-hard
E2E_FUSION_KEEP_K ?= $(ORACLE_FUSION_KEEP_K)
# Reranking: none | cross_encoder (aliases: cross-encoder, ce)
E2E_RERANKER ?= cross_encoder
E2E_RERANK_TOP_K ?= 5
E2E_CROSS_ENCODER_MODEL ?= cross-encoder/ms-marco-MiniLM-L-6-v2
E2E_ROUTER_DEVICE ?= cpu
E2E_ROUTER_INFERENCE_BATCH_SIZE ?= 32

# Poetry Python for consistent CLI
PY = poetry run python

install:
	poetry install

install-hooks:
	poetry run pre-commit install

setup-models:
	HF_HOME="$(HF_HOME)" TRANSFORMERS_CACHE="$(TRANSFORMERS_CACHE)" poetry run python scripts/setup_models.py

lock:
	poetry lock

test:
	poetry run pytest

print-paths:
	@echo "DATA_BASE=$(DATA_BASE) BENCHMARK_BASE=$(BENCHMARK_BASE) BENCHMARK_NAME=$(BENCHMARK_NAME) BENCHMARK_ID=$(BENCHMARK_ID)"
	@echo "BUNDLE=$(BENCHMARK_BUNDLE_DIR) BENCHMARK_DIR=$(BENCHMARK_DIR) CORPUS_DIR=$(CORPUS_DIR)"
	@echo "BENCHMARK_PATH=$(BENCHMARK_PATH) ORACLE_RETRIEVAL_ASSET_DIR=$(ORACLE_RETRIEVAL_ASSET_DIR)"
	@echo "ROUTER_ID=$(ROUTER_ID) ROUTER_ORACLE_DIR=$(ROUTER_ORACLE_DIR) ROUTER_DATASET_DIR=$(ROUTER_DATASET_DIR) ROUTER_MODEL_DIR=$(ROUTER_MODEL_DIR)"

ingest:
	poetry run python scripts/ingest_data.py \
		--nq "$(NQ_PATH)" \
		--2wiki "$(2WIKI_PATH)" \
		--output-dir "$(BENCHMARK_DIR)"

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

# Precompute lexicon phrase matcher (optional; speeds graph + learned-router feature paths)
# Rebuild: make build-entity-matching-artifacts ENTITY_MATCHING_FORCE=1
build-entity-matching-artifacts:
	$(PY) -m scripts.build_entity_matching_artifacts \
		--corpus-dir "$(CORPUS_DIR)" \
		$(if $(ENTITY_MATCHING_FORCE),--force,)

filter-benchmark:
	poetry run python scripts/filter_benchmark_by_corpus.py \
		--benchmark "$(BENCHMARK_PATH)" \
		--corpus "$(CORPUS_PATH)"

# Full offline refresh: ingest → fetch → align (drop unresolved 2Wiki) → corpus + indexes
pipeline: ingest fetch-wikipedia-articles align-2wiki-support build-corpus

# Oracle: prepare -> beta sweep -> soft labels

print-oracle-config:
	@echo "ROUTER_ID=$(ROUTER_ID) ROUTER_BASE=$(ROUTER_BASE)"
	@echo "BENCHMARK_NAME=$(BENCHMARK_NAME) BENCHMARK_ID=$(BENCHMARK_ID)"
	@echo "ORACLE_BENCHMARK_PATH=$(ORACLE_BENCHMARK_PATH)"
	@echo "ORACLE_RETRIEVAL_ASSET_DIR=$(ORACLE_RETRIEVAL_ASSET_DIR)"
	@echo "ORACLE_BRANCH_TOP_K=$(ORACLE_BRANCH_TOP_K) ORACLE_FUSION_KEEP_K=$(ORACLE_FUSION_KEEP_K)"
	@echo "ORACLE_BETAS=$(ORACLE_BETAS) ORACLE_BETA_FLAGS=$(ORACLE_BETA_FLAGS)"
	@echo "SELECTED_BETA=$(SELECTED_BETA) (empty => use recommended_beta.json for soft labels)"

validate-oracle-config:
	@test -n "$(ROUTER_ID)" && test -f "$(ORACLE_BENCHMARK_PATH)" && test -d "$(ORACLE_RETRIEVAL_ASSET_DIR)" || (echo "Missing: ROUTER_ID, ORACLE_BENCHMARK_PATH, ORACLE_RETRIEVAL_ASSET_DIR"; exit 1)

oracle-prepare: validate-oracle-config
	$(PY) -m scripts.prepare_oracle_run \
		--router-id "$(ROUTER_ID)" \
		--benchmark-name "$(BENCHMARK_NAME)" \
		--benchmark-id "$(BENCHMARK_ID)" \
		--benchmark-path "$(ORACLE_BENCHMARK_PATH)" \
		--retrieval-asset-dir "$(ORACLE_RETRIEVAL_ASSET_DIR)" \
		--branch-top-k $(ORACLE_BRANCH_TOP_K) \
		--fusion-keep-k $(ORACLE_FUSION_KEEP_K) \
		--router-base "$(ROUTER_BASE)"

oracle-sweep-beta: validate-oracle-config
	$(PY) -m scripts.sweep_beta \
		--router-id "$(ROUTER_ID)" \
		--min-entropy-nats $(ORACLE_MIN_ENTROPY_NATS) \
		--betas $(ORACLE_BETAS) \
		--router-base "$(ROUTER_BASE)"

# SELECTED_BETA: if empty, read recommended beta from router oracle dir; fallback 2.0
oracle-create-soft-labels: validate-oracle-config
	@BETA_SEL=""; \
	if [ -n "$(SELECTED_BETA)" ]; then BETA_SEL="$(SELECTED_BETA)"; \
	else BETA_SEL=$$($(PY) -c "import json, pathlib; p=pathlib.Path('$(ROUTER_ORACLE_DIR)')/'recommended_beta.json'; \
	print(json.loads(p.read_text(encoding='utf-8'))['recommended_beta'])" 2>/dev/null || echo "2.0"); fi; \
	echo "create_soft_labels: using --selected-beta $$BETA_SEL"; \
	$(PY) -m scripts.create_soft_labels \
		--router-id "$(ROUTER_ID)" \
		$(ORACLE_BETA_FLAGS) \
		--selected-beta $$BETA_SEL \
		--router-base "$(ROUTER_BASE)"

oracle-labels: oracle-prepare oracle-sweep-beta oracle-create-soft-labels

# Router dataset (Parquet) — same ROUTER_ID as oracle labels

print-router-config:
	@echo "ROUTER_ID=$(ROUTER_ID) ROUTER_BASE=$(ROUTER_BASE)"
	@echo "BENCHMARK_NAME=$(BENCHMARK_NAME) BENCHMARK_ID=$(BENCHMARK_ID)"
	@echo "BENCHMARK_PATH=$(BENCHMARK_PATH) (router dataset input)"
	@echo "ORACLE_RETRIEVAL_ASSET_DIR=$(ORACLE_RETRIEVAL_ASSET_DIR)"

validate-router-config:
	@test -n "$(ROUTER_ID)" && test -f "$(BENCHMARK_PATH)" && test -d "$(ORACLE_RETRIEVAL_ASSET_DIR)" || (echo "Set ROUTER_ID, BENCHMARK_PATH, ORACLE_RETRIEVAL_ASSET_DIR"; exit 1)
	@test -f "$(ROUTER_ORACLE_DIR)/labels/selected.jsonl" || (echo "Missing $(ROUTER_ORACLE_DIR)/labels/selected.jsonl — run make oracle-labels first"; exit 1)

router-build-dataset: validate-router-config
	$(PY) -m scripts.router.build_router_dataset \
		--router-id "$(ROUTER_ID)" \
		--benchmark-name "$(BENCHMARK_NAME)" \
		--benchmark-id "$(BENCHMARK_ID)" \
		--benchmark-path "$(BENCHMARK_PATH)" \
		--retrieval-asset-dir "$(ORACLE_RETRIEVAL_ASSET_DIR)" \
		--router-base "$(ROUTER_BASE)" \
		--embedding-model "$(EMBEDDING_MODEL_FOR_ROUTER)" \
		--split-seed $(SEED) \
		--train-ratio $(TRAIN_RATIO) \
		--dev-ratio $(DEV_RATIO) \
		--test-ratio $(TEST_RATIO)

router-pipeline: oracle-labels router-build-dataset

# Router MLP: requires Parquet at $(ROUTER_DATASET_DIR)/router_dataset.parquet

validate-router-train:
	@test -n "$(ROUTER_ID)" && test -f "$(ROUTER_DATASET_DIR)/router_dataset.parquet" || (echo "Missing $(ROUTER_DATASET_DIR)/router_dataset.parquet — run make router-build-dataset first"; exit 1)

router-train: validate-router-train
	$(PY) -m scripts.router.train_router \
		--router-id "$(ROUTER_ID)" \
		--router-base "$(ROUTER_BASE)" \
		--epochs $(ROUTER_EPOCHS) \
		--batch-size $(ROUTER_BATCH_SIZE) \
		--learning-rate $(ROUTER_LEARNING_RATE) \
		--device "$(ROUTER_TRAIN_DEVICE)" \
		--input-mode "$(ROUTER_INPUT_MODE)"

# Train all input ablations; each run writes to $(ROUTER_MODEL_DIR)/<input_mode>/
router-train-ablations: validate-router-train
	@for m in $(ROUTER_INPUT_MODES); do \
		echo "=== router-train --input-mode $$m ==="; \
		$(PY) -m scripts.router.train_router \
			--router-id "$(ROUTER_ID)" \
			--router-base "$(ROUTER_BASE)" \
			--epochs $(ROUTER_EPOCHS) \
			--batch-size $(ROUTER_BATCH_SIZE) \
			--learning-rate $(ROUTER_LEARNING_RATE) \
			--device "$(ROUTER_TRAIN_DEVICE)" \
			--input-mode "$$m" || exit 1; \
	done

router-eval: validate-router-train
	$(PY) -m scripts.router.evaluate_router \
		--router-id "$(ROUTER_ID)" \
		--router-base "$(ROUTER_BASE)" \
		--device "$(ROUTER_TRAIN_DEVICE)" \
		--input-mode "$(ROUTER_INPUT_MODE)"

# Re-eval each ablation variant (requires checkpoints under each $(ROUTER_MODEL_DIR)/<input_mode>/)
router-evaluate-ablations: validate-router-train
	@for m in $(ROUTER_INPUT_MODES); do \
		echo "=== router-eval --input-mode $$m ==="; \
		$(PY) -m scripts.router.evaluate_router \
			--router-id "$(ROUTER_ID)" \
			--router-base "$(ROUTER_BASE)" \
			--device "$(ROUTER_TRAIN_DEVICE)" \
			--input-mode "$$m" || exit 1; \
	done

# --- E2E benchmark (see docs/dev/end-to-end-system-and-evaluation.md)

e2e-print-config:
	$(PY) -m scripts.e2e_benchmark print-config \
		--benchmark-base "$(BENCHMARK_BASE)" \
		--benchmark-name "$(BENCHMARK_NAME)" \
		--benchmark-id "$(BENCHMARK_ID)" \
		--benchmark-path "$(BENCHMARK_PATH)" \
		--split "$(E2E_SPLIT)" \
		--run-id "$(E2E_RUN_ID)" \
		--policy "$(E2E_POLICY)" \
		--retrieval-asset-dir "$(CORPUS_DIR)"

e2e-prepare:
	$(PY) -m scripts.e2e_benchmark prepare \
		--benchmark-base "$(BENCHMARK_BASE)" \
		--benchmark-name "$(BENCHMARK_NAME)" \
		--benchmark-id "$(BENCHMARK_ID)" \
		--benchmark-path "$(BENCHMARK_PATH)" \
		--split "$(E2E_SPLIT)" \
		--run-id "$(E2E_RUN_ID)" \
		--policy "$(E2E_POLICY)" \
		--retrieval-asset-dir "$(CORPUS_DIR)" \
		--router-id "$(ROUTER_ID)" \
		--router-base "$(ROUTER_BASE)" \
		--fusion-keep-k $(E2E_FUSION_KEEP_K) \
		--reranker "$(E2E_RERANKER)" \
		--rerank-top-k $(E2E_RERANK_TOP_K) \
		--cross-encoder-model "$(E2E_CROSS_ENCODER_MODEL)" \
		--router-device "$(E2E_ROUTER_DEVICE)" \
		--router-input-mode "$(ROUTER_INPUT_MODE)" \
		--router-inference-batch-size $(E2E_ROUTER_INFERENCE_BATCH_SIZE) \
		--dry-run

e2e-submit:
	$(PY) -m scripts.e2e_benchmark prepare \
		--benchmark-base "$(BENCHMARK_BASE)" \
		--benchmark-name "$(BENCHMARK_NAME)" \
		--benchmark-id "$(BENCHMARK_ID)" \
		--benchmark-path "$(BENCHMARK_PATH)" \
		--split "$(E2E_SPLIT)" \
		--run-id "$(E2E_RUN_ID)" \
		--policy "$(E2E_POLICY)" \
		--retrieval-asset-dir "$(CORPUS_DIR)" \
		--router-id "$(ROUTER_ID)" \
		--router-base "$(ROUTER_BASE)" \
		--fusion-keep-k $(E2E_FUSION_KEEP_K) \
		--reranker "$(E2E_RERANKER)" \
		--rerank-top-k $(E2E_RERANK_TOP_K) \
		--cross-encoder-model "$(E2E_CROSS_ENCODER_MODEL)" \
		--router-device "$(E2E_ROUTER_DEVICE)" \
		--router-input-mode "$(ROUTER_INPUT_MODE)" \
		--router-inference-batch-size $(E2E_ROUTER_INFERENCE_BATCH_SIZE)

e2e-collect:
	$(PY) -m scripts.e2e_benchmark collect \
		--benchmark-base "$(BENCHMARK_BASE)" \
		--benchmark-name "$(BENCHMARK_NAME)" \
		--benchmark-id "$(BENCHMARK_ID)" \
		--benchmark-path "$(BENCHMARK_PATH)" \
		--split "$(E2E_SPLIT)" \
		--run-id "$(E2E_RUN_ID)" \
		--policy "$(E2E_POLICY)" \
		--retrieval-asset-dir "$(CORPUS_DIR)"

e2e-evaluate:
	$(PY) -m scripts.e2e_benchmark evaluate \
		--benchmark-base "$(BENCHMARK_BASE)" \
		--benchmark-name "$(BENCHMARK_NAME)" \
		--benchmark-id "$(BENCHMARK_ID)" \
		--benchmark-path "$(BENCHMARK_PATH)" \
		--split "$(E2E_SPLIT)" \
		--run-id "$(E2E_RUN_ID)" \
		--policy "$(E2E_POLICY)" \
		--retrieval-asset-dir "$(CORPUS_DIR)" \
		--router-id "$(ROUTER_ID)" \
		--router-base "$(ROUTER_BASE)"

e2e-run:
	@echo "1) make e2e-submit   (or e2e-prepare for local dry-run)"
	@echo "   All policies: make e2e-run-all-policies  (same E2E_RUN_ID prefix as collect/eval-all)"
	@echo "2) Wait for OpenAI batch(es) to complete."
	@echo "3) make e2e-collect && make e2e-evaluate"
	@echo "   All policies: make e2e-collect-all-policies && make e2e-evaluate-all-policies"

# Shared prefix: set E2E_RUN_ID to the same value used for e2e-run-all-policies (each policy uses $(E2E_RUN_ID)-<policy>).
e2e-run-all-policies:
	@for pol in $(E2E_POLICIES); do \
		echo "=== E2E policy=$$pol run=$(E2E_RUN_ID)-$$pol ==="; \
		$(MAKE) e2e-submit E2E_POLICY=$$pol E2E_RUN_ID="$(E2E_RUN_ID)-$$pol" || exit 1; \
	done

e2e-collect-all-policies:
	@for pol in $(E2E_POLICIES); do \
		echo "=== E2E collect policy=$$pol run=$(E2E_RUN_ID)-$$pol ==="; \
		$(MAKE) e2e-collect E2E_POLICY=$$pol E2E_RUN_ID="$(E2E_RUN_ID)-$$pol" || exit 1; \
	done

e2e-evaluate-all-policies:
	@for pol in $(E2E_POLICIES); do \
		echo "=== E2E evaluate policy=$$pol run=$(E2E_RUN_ID)-$$pol ==="; \
		$(MAKE) e2e-evaluate E2E_POLICY=$$pol E2E_RUN_ID="$(E2E_RUN_ID)-$$pol" || exit 1; \
	done

e2e-smoke-test-v01: E2E_RUN_ID := smoke-$(shell date +%Y%m%d-%H%M%S)
e2e-smoke-test-v01:
	$(PY) -m scripts.e2e_benchmark prepare \
		--benchmark-base "$(BENCHMARK_BASE)" \
		--benchmark-name "$(BENCHMARK_NAME)" \
		--benchmark-id "$(BENCHMARK_ID)" \
		--benchmark-path "$(BENCHMARK_PATH)" \
		--split "$(E2E_SPLIT)" \
		--run-id "$(E2E_RUN_ID)" \
		--policy dense-only \
		--retrieval-asset-dir "$(CORPUS_DIR)" \
		--limit 1 \
		--router-inference-batch-size $(E2E_ROUTER_INFERENCE_BATCH_SIZE) \
		--dry-run


.PHONY: install setup-models lock test ingest build-corpus filter-benchmark

-include .env

NQ_PATH ?= data/raw/nq_100.jsonl
2WIKI_PATH ?= data/raw/2wikimultihop_100.jsonl
BENCHMARK_PATH ?= data/processed/benchmark.jsonl
OUTPUT_DIR ?= data/processed
CORPUS_PATH ?= $(OUTPUT_DIR)/corpus.jsonl
HF_HOME ?= $(OUTPUT_DIR)/hf_cache

install:
	poetry install

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

build-corpus:
	poetry run python scripts/build_corpus.py \
		--benchmark "$(BENCHMARK_PATH)" \
		--nq "$(NQ_PATH)" \
		--2wiki "$(2WIKI_PATH)" \
		--output-dir "$(OUTPUT_DIR)"

filter-benchmark:
	poetry run python scripts/filter_benchmark_by_corpus.py \
		--benchmark "$(BENCHMARK_PATH)" \
		--corpus "$(CORPUS_PATH)"
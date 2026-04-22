.PHONY: install install-hooks setup-models lock test ingest fetch-wikipedia-articles build-corpus align-2wiki-support align-2wiki-support-full filter-benchmark pipeline

-include .env

NQ_PATH ?= data/raw/nq_100.jsonl
2WIKI_PATH ?= data/raw/2wikimultihop_100.jsonl
BENCHMARK_PATH ?= data/processed/benchmark.jsonl
OUTPUT_DIR ?= data/processed
CORPUS_PATH ?= $(OUTPUT_DIR)/corpus.jsonl
DOCSTORE_PATH ?= $(OUTPUT_DIR)/docstore.sqlite
HF_HOME ?= $(OUTPUT_DIR)/hf_cache

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
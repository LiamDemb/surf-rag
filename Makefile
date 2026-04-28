.PHONY: help print-resolved-config print-paths print-oracle-config print-router-config \
	install install-hooks setup-models lock test ingest fetch-wikipedia-articles build-corpus align-2wiki-support align-2wiki-support-full filter-benchmark pipeline \
	oracle-prepare oracle-sweep-beta oracle-create-soft-labels oracle-labels \
	router-build-dataset router-pipeline \
	router-train router-eval router-train-ablations router-evaluate-ablations \
	validate-oracle-config validate-router-config validate-router-train \
	e2e-print-config e2e-prepare e2e-submit e2e-collect e2e-evaluate e2e-run \
	e2e-run-all-policies e2e-collect-all-policies e2e-evaluate-all-policies e2e-smoke-test-v01 \
	build-entity-matching-artifacts

# Default experiment recipe (override per run: make build-corpus CONFIG=configs/e2e/.../x.yaml)
CONFIG ?= configs/pipelines/surf-bench-200.yaml

-include .env
export OPENAI_API_KEY

# Optional Make overrides
E2E_RUN_ID ?=
E2E_POLICY ?=
E2E_SPLIT ?=
E2E_DEV_SYNC ?=
E2E_POLICIES ?= dense-only graph-only 50-50 learned-soft learned-hard
SELECTED_BETA ?=
ROUTER_INPUT_MODES ?= both query-features embedding
ENTITY_MATCHING_FORCE ?=
ALIGN_2WIKI_EXTRA ?=

PY = poetry run python

install:
	poetry install

install-hooks:
	poetry run pre-commit install

setup-models:
	$(PY) scripts/setup_models.py --config "$(CONFIG)"

lock:
	poetry lock

test:
	poetry run pytest

help:
	@echo "SuRF-RAG: set CONFIG to a YAML recipe (default: $(CONFIG))."
	@echo "Secrets: .env (OPENAI_API_KEY). All benchmark/oracle/router/E2E fields live in the YAML."
	@echo ""
	@echo "  make print-resolved-config   — merged config + absolute paths"
	@echo "  make pipeline                — ingest → fetch → align → build-corpus"
	@echo "  make oracle-labels            — oracle + sweep + soft labels"
	@echo "  make router-pipeline         — oracle-labels + router-build-dataset"
	@echo "  make e2e-submit / e2e-collect / e2e-evaluate   (+ optional E2E_RUN_ID= E2E_POLICY=)"
	@echo ""
	@echo "Full reference: docs/config-driven-workflows.md"

print-resolved-config print-paths print-oracle-config print-router-config:
	$(PY) -m surf_rag.config "$(CONFIG)"

validate-oracle-config:
	$(PY) -m surf_rag.config validate-oracle "$(CONFIG)"

validate-router-config:
	$(PY) -m surf_rag.config validate-router "$(CONFIG)"

validate-router-train:
	$(PY) -m surf_rag.config validate-router-train "$(CONFIG)"

ingest:
	$(PY) scripts/ingest_data.py --config "$(CONFIG)"

fetch-wikipedia-articles:
	$(PY) scripts/fetch_wikipedia_articles.py --config "$(CONFIG)"

align-2wiki-support:
	$(PY) scripts/align_2wiki_support.py --config "$(CONFIG)" $(ALIGN_2WIKI_EXTRA)

align-2wiki-support-full:
	$(MAKE) align-2wiki-support CONFIG="$(CONFIG)" ALIGN_2WIKI_EXTRA=--full-report

build-corpus:
	$(PY) scripts/build_corpus.py --config "$(CONFIG)"

build-entity-matching-artifacts:
	$(PY) -m scripts.build_entity_matching_artifacts --config "$(CONFIG)" \
		$(if $(ENTITY_MATCHING_FORCE),--force,)

filter-benchmark:
	$(PY) scripts/filter_benchmark_by_corpus.py --config "$(CONFIG)"

pipeline: ingest fetch-wikipedia-articles align-2wiki-support build-corpus

oracle-prepare: validate-oracle-config
	$(PY) -m scripts.prepare_oracle_run --config "$(CONFIG)"

oracle-sweep-beta: validate-oracle-config
	$(PY) -m scripts.sweep_beta --config "$(CONFIG)"

oracle-create-soft-labels: validate-oracle-config
	$(PY) -m scripts.create_soft_labels --config "$(CONFIG)" \
		$(if $(SELECTED_BETA),--selected-beta $(SELECTED_BETA),)

oracle-labels: oracle-prepare oracle-sweep-beta oracle-create-soft-labels

router-build-dataset: validate-router-config
	$(PY) -m scripts.router.build_router_dataset --config "$(CONFIG)"

router-pipeline: oracle-labels router-build-dataset

router-train: validate-router-train
	$(PY) -m scripts.router.train_router --config "$(CONFIG)"

router-train-ablations: validate-router-train
	@for m in $(ROUTER_INPUT_MODES); do \
		echo "=== router-train --config $(CONFIG) --input-mode $$m ==="; \
		$(PY) -m scripts.router.train_router --config "$(CONFIG)" --input-mode "$$m" || exit 1; \
	done

router-eval: validate-router-train
	$(PY) -m scripts.router.evaluate_router --config "$(CONFIG)"

router-evaluate-ablations: validate-router-train
	@for m in $(ROUTER_INPUT_MODES); do \
		echo "=== router-eval --config $(CONFIG) --input-mode $$m ==="; \
		$(PY) -m scripts.router.evaluate_router --config "$(CONFIG)" --input-mode "$$m" || exit 1; \
	done

e2e-print-config:
	$(PY) -m scripts.e2e_benchmark --config "$(CONFIG)" print-config \
		$(if $(E2E_RUN_ID),--run-id "$(E2E_RUN_ID)",) \
		$(if $(E2E_POLICY),--policy "$(E2E_POLICY)",) \
		$(if $(E2E_SPLIT),--split "$(E2E_SPLIT)",)

e2e-prepare:
	$(PY) -m scripts.e2e_benchmark --config "$(CONFIG)" prepare --dry-run \
		$(if $(E2E_RUN_ID),--run-id "$(E2E_RUN_ID)",) \
		$(if $(E2E_POLICY),--policy "$(E2E_POLICY)",) \
		$(if $(E2E_SPLIT),--split "$(E2E_SPLIT)",)

e2e-submit:
	$(PY) -m scripts.e2e_benchmark --config "$(CONFIG)" prepare \
		$(if $(E2E_RUN_ID),--run-id "$(E2E_RUN_ID)",) \
		$(if $(E2E_POLICY),--policy "$(E2E_POLICY)",) \
		$(if $(E2E_SPLIT),--split "$(E2E_SPLIT)",) \
		$(if $(E2E_DEV_SYNC),--dev-sync,)

e2e-collect:
	$(PY) -m scripts.e2e_benchmark --config "$(CONFIG)" collect \
		$(if $(E2E_RUN_ID),--run-id "$(E2E_RUN_ID)",) \
		$(if $(E2E_POLICY),--policy "$(E2E_POLICY)",) \
		$(if $(E2E_SPLIT),--split "$(E2E_SPLIT)",)

e2e-evaluate:
	$(PY) -m scripts.e2e_benchmark --config "$(CONFIG)" evaluate \
		$(if $(E2E_RUN_ID),--run-id "$(E2E_RUN_ID)",) \
		$(if $(E2E_POLICY),--policy "$(E2E_POLICY)",) \
		$(if $(E2E_SPLIT),--split "$(E2E_SPLIT)",)

e2e-run:
	@echo "1) make e2e-submit   (or e2e-prepare for local dry-run)"
	@echo "   All policies: make e2e-run-all-policies  (E2E_RUN_ID prefix; CONFIG=$(CONFIG))"
	@echo "2) Wait for OpenAI batch(es) to complete."
	@echo "3) make e2e-collect && make e2e-evaluate"

e2e-run-all-policies:
	@for pol in $(E2E_POLICIES); do \
		base="$(E2E_RUN_ID)"; \
		[ -n "$$base" ] || base=e2e; \
		rid="$(E2E_RUN_ID)"; \
		echo "=== E2E policy=$$pol run=$$rid ==="; \
		$(MAKE) e2e-submit E2E_POLICY=$$pol E2E_RUN_ID="$$rid" CONFIG="$(CONFIG)" || exit 1; \
	done

e2e-collect-all-policies:
	@for pol in $(E2E_POLICIES); do \
		base="$(E2E_RUN_ID)"; \
		[ -n "$$base" ] || base=e2e; \
		rid="$(E2E_RUN_ID)"; \
		echo "=== E2E collect policy=$$pol run=$$rid ==="; \
		$(MAKE) e2e-collect E2E_POLICY=$$pol E2E_RUN_ID="$$rid" CONFIG="$(CONFIG)" || exit 1; \
	done

e2e-evaluate-all-policies:
	@for pol in $(E2E_POLICIES); do \
		base="$(E2E_RUN_ID)"; \
		[ -n "$$base" ] || base=e2e; \
		rid="$(E2E_RUN_ID)"; \
		echo "=== E2E evaluate policy=$$pol run=$$rid ==="; \
		$(MAKE) e2e-evaluate E2E_POLICY=$$pol E2E_RUN_ID="$$rid" CONFIG="$(CONFIG)" || exit 1; \
	done

# Smoke: one question, dense-only, dry-run; override E2E_RUN_ID or rely on a smoke-oriented CONFIG
E2E_SMOKE_RUN_ID ?= smoke-$(shell date +%Y%m%d-%H%M%S)
e2e-smoke-test-v01:
	$(PY) -m scripts.e2e_benchmark --config "$(CONFIG)" prepare --dry-run --limit 1 \
		--run-id "$(E2E_SMOKE_RUN_ID)" --policy dense-only

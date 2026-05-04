.PHONY: help print-resolved-config print-paths print-oracle-config print-router-config \
	install install-hooks setup-models lock test ingest fetch-wikipedia-articles build-corpus align-2wiki-support align-2wiki-support-full filter-benchmark pipeline \
	oracle-prepare oracle-create-router-labels oracle-labels \
	router-build-dataset router-build-query-embedding-cache router-pipeline \
	router-train router-eval router-train-ablations router-evaluate-ablations \
	router-calibrate-threshold \
	figures-render \
	validate-oracle-config validate-router-config validate-router-train \
	e2e-print-config e2e-prepare e2e-submit e2e-collect e2e-evaluate e2e-run \
	e2e-run-all-policies e2e-collect-all-policies e2e-evaluate-all-policies e2e-smoke-test-v01 \
	answerability-submit answerability-collect answerability-balance \
	llm-judge-submit llm-judge-collect llm-judge-merge \
	llm-judge-submit-all-policies llm-judge-collect-all-policies llm-judge-merge-all-policies \
	gen-debug discrepancy-debug-e2e \
	build-entity-matching-artifacts corpus-ie-run corpus-ie-retry corpus-finalize corpus-ie-retry-and-finalize

# Default experiment recipe (override per run: make build-corpus CONFIG=configs/e2e/.../x.yaml)
CONFIG ?= configs/pipelines/surf-bench-200.yaml

-include .env
export OPENAI_API_KEY

# Optional Make overrides
E2E_RUN_ID ?=
E2E_POLICY ?=
E2E_SPLIT ?=
E2E_DEV_SYNC ?=
E2E_POLICIES ?= dense-only graph-only 50-50 learned-soft hard-routing hybrid oracle-upper-bound
ROUTER_INPUT_MODES ?= both query-features embedding
ROUTER_TASK_TYPE ?=
ENTITY_MATCHING_FORCE ?=
ALIGN_2WIKI_EXTRA ?=
PIPELINE_RUN_ID ?=

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
	@echo "  make oracle-labels            — oracle + router labels"
	@echo "  make router-pipeline         — oracle-labels + router-build-dataset"
	@echo "  make router-build-query-embedding-cache — benchmark query-embedding JSONL cache (oracle labels not required; runs validate-oracle-config)"
	@echo "  make e2e-submit / e2e-collect / e2e-evaluate   (+ optional E2E_RUN_ID= E2E_POLICY=)"
	@echo "  make answerability-submit / answerability-collect / answerability-balance (CONFIG=configs/audit/....yaml)"
	@echo "  make llm-judge-submit / collect / merge (optional E2E_RUN_ID= E2E_POLICY=; else from CONFIG)"
	@echo "  make gen-debug          (CONFIG=configs/gen-debug/….yaml)"
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
	$(PY) scripts/ingest_data.py --config "$(CONFIG)" \
		$(if $(PIPELINE_RUN_ID),--pipeline-run-id "$(PIPELINE_RUN_ID)",)

fetch-wikipedia-articles:
	$(PY) scripts/fetch_wikipedia_articles.py --config "$(CONFIG)"

align-2wiki-support:
	$(PY) scripts/align_2wiki_support.py --config "$(CONFIG)" \
		$(if $(PIPELINE_RUN_ID),--pipeline-run-id "$(PIPELINE_RUN_ID)",) \
		$(ALIGN_2WIKI_EXTRA)

align-2wiki-support-full:
	$(MAKE) align-2wiki-support CONFIG="$(CONFIG)" ALIGN_2WIKI_EXTRA=--full-report

build-corpus:
	$(PY) scripts/build_corpus.py --config "$(CONFIG)"

corpus-ie-run:
	$(PY) scripts/corpus/run_llm_ie_batch.py --config "$(CONFIG)"

corpus-ie-retry:
	$(PY) scripts/corpus/run_llm_ie_batch.py --config "$(CONFIG)"

corpus-finalize:
	$(PY) scripts/corpus/finalize_corpus_artifacts.py --config "$(CONFIG)"

corpus-ie-retry-and-finalize:
	$(MAKE) corpus-ie-retry CONFIG="$(CONFIG)" && $(MAKE) corpus-finalize CONFIG="$(CONFIG)"

build-entity-matching-artifacts:
	$(PY) -m scripts.build_entity_matching_artifacts --config "$(CONFIG)" \
		$(if $(ENTITY_MATCHING_FORCE),--force,)

filter-benchmark:
	$(PY) scripts/filter_benchmark_by_corpus.py --config "$(CONFIG)" \
		$(if $(PIPELINE_RUN_ID),--pipeline-run-id "$(PIPELINE_RUN_ID)",)

pipeline:
	@rid="$(PIPELINE_RUN_ID)"; \
	[ -n "$$rid" ] || rid=pipeline-$$(date +%Y%m%dT%H%M%SZ); \
	echo "=== pipeline run-id: $$rid ==="; \
	$(MAKE) ingest CONFIG="$(CONFIG)" PIPELINE_RUN_ID="$$rid" || exit 1; \
	$(MAKE) fetch-wikipedia-articles CONFIG="$(CONFIG)" PIPELINE_RUN_ID="$$rid" || exit 1; \
	$(MAKE) align-2wiki-support CONFIG="$(CONFIG)" ALIGN_2WIKI_EXTRA="$(ALIGN_2WIKI_EXTRA)" PIPELINE_RUN_ID="$$rid" || exit 1; \
	$(MAKE) build-corpus CONFIG="$(CONFIG)" PIPELINE_RUN_ID="$$rid" || exit 1; \
	$(MAKE) filter-benchmark CONFIG="$(CONFIG)" PIPELINE_RUN_ID="$$rid" || exit 1

oracle-prepare: validate-oracle-config
	$(PY) -m scripts.prepare_oracle_run --config "$(CONFIG)"

oracle-create-router-labels: validate-oracle-config
	$(PY) -m scripts.create_soft_labels --config "$(CONFIG)"

oracle-labels: oracle-prepare oracle-create-router-labels

router-build-dataset: validate-router-config
	$(PY) -m scripts.router.build_router_dataset --config "$(CONFIG)"

router-build-query-embedding-cache: validate-oracle-config
	$(PY) -m scripts.router.build_query_embedding_cache --config "$(CONFIG)"

router-pipeline: oracle-labels router-build-dataset

router-train: validate-router-train
	$(PY) -m scripts.router.train_router --config "$(CONFIG)" \
		$(if $(ROUTER_TASK_TYPE),--router-task-type "$(ROUTER_TASK_TYPE)",)

router-train-ablations: validate-router-train
	@for m in $(ROUTER_INPUT_MODES); do \
		echo "=== router-train --config $(CONFIG) --input-mode $$m ==="; \
		$(PY) -m scripts.router.train_router --config "$(CONFIG)" --input-mode "$$m" || exit 1; \
	done

router-evaluate: validate-router-train
	$(PY) -m scripts.router.evaluate_router --config "$(CONFIG)" \
		$(if $(ROUTER_TASK_TYPE),--router-task-type "$(ROUTER_TASK_TYPE)",)

# Regenerates figures in-place; pass FIGURES_EXTRA= to omit --force if you need overwrite protection.
FIGURES_EXTRA ?= --force
figures-render:
	$(PY) -m scripts.figures.render_figures --config "$(CONFIG)" $(FIGURES_EXTRA)

router-evaluate-ablations: validate-router-train
	@for m in $(ROUTER_INPUT_MODES); do \
		echo "=== router-eval --config $(CONFIG) --input-mode $$m ==="; \
		$(PY) -m scripts.router.evaluate_router --config "$(CONFIG)" --input-mode "$$m" || exit 1; \
	done

router-calibrate-threshold:
	$(PY) -m scripts.router.calibrate_confidence_threshold \
		--router-id "$${ROUTER_ID:?set ROUTER_ID}" \
		--regressor-router-id "$${REGRESSOR_ROUTER_ID:?set REGRESSOR_ROUTER_ID}" \
		$(if $(ROUTER_ARCHITECTURE_ID),--classifier-architecture-id "$(ROUTER_ARCHITECTURE_ID)",) \
		$(if $(REGRESSOR_ROUTER_ARCHITECTURE_ID),--regressor-architecture-id "$(REGRESSOR_ROUTER_ARCHITECTURE_ID)",)

e2e-print-config:
	$(PY) -m scripts.e2e_benchmark --config "$(CONFIG)" print-config \
		$(if $(E2E_RUN_ID),--run-id "$(E2E_RUN_ID)",) \
		$(if $(E2E_POLICY),--policy "$(E2E_POLICY)",) \
		$(if $(E2E_SPLIT),--split "$(E2E_SPLIT)",) \
		$(if $(ROUTER_TASK_TYPE),--router-task-type "$(ROUTER_TASK_TYPE)",)

e2e-prepare:
	$(PY) -m scripts.e2e_benchmark --config "$(CONFIG)" prepare --dry-run \
		$(if $(E2E_RUN_ID),--run-id "$(E2E_RUN_ID)",) \
		$(if $(E2E_POLICY),--policy "$(E2E_POLICY)",) \
		$(if $(E2E_SPLIT),--split "$(E2E_SPLIT)",) \
		$(if $(ROUTER_TASK_TYPE),--router-task-type "$(ROUTER_TASK_TYPE)",)

e2e-submit:
	$(PY) -m scripts.e2e_benchmark --config "$(CONFIG)" prepare \
		$(if $(E2E_RUN_ID),--run-id "$(E2E_RUN_ID)",) \
		$(if $(E2E_POLICY),--policy "$(E2E_POLICY)",) \
		$(if $(E2E_SPLIT),--split "$(E2E_SPLIT)",) \
		$(if $(ROUTER_TASK_TYPE),--router-task-type "$(ROUTER_TASK_TYPE)",) \
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
	@echo "   All policies: make e2e-run-all-policies  (optional E2E_RUN_ID=…; else e2e.run_id from CONFIG)"
	@echo "2) Wait for OpenAI batch(es) to complete."
	@echo "3) make e2e-collect && make e2e-evaluate"

e2e-run-all-policies:
	@for pol in $(E2E_POLICIES); do \
		echo "=== E2E policy=$$pol$(if $(E2E_RUN_ID), run=$(E2E_RUN_ID),) ==="; \
		$(MAKE) e2e-submit E2E_POLICY=$$pol CONFIG="$(CONFIG)" $(if $(E2E_RUN_ID),E2E_RUN_ID="$(E2E_RUN_ID)",) || exit 1; \
	done

e2e-collect-all-policies:
	@for pol in $(E2E_POLICIES); do \
		echo "=== E2E collect policy=$$pol$(if $(E2E_RUN_ID), run=$(E2E_RUN_ID),) ==="; \
		$(MAKE) e2e-collect E2E_POLICY=$$pol CONFIG="$(CONFIG)" $(if $(E2E_RUN_ID),E2E_RUN_ID="$(E2E_RUN_ID)",) || exit 1; \
	done

e2e-evaluate-all-policies:
	@for pol in $(E2E_POLICIES); do \
		echo "=== E2E evaluate policy=$$pol$(if $(E2E_RUN_ID), run=$(E2E_RUN_ID),) ==="; \
		$(MAKE) e2e-evaluate E2E_POLICY=$$pol CONFIG="$(CONFIG)" $(if $(E2E_RUN_ID),E2E_RUN_ID="$(E2E_RUN_ID)",) || exit 1; \
	done

answerability-submit:
	$(PY) -m scripts.audit.benchmark_answerability --config "$(CONFIG)" submit

answerability-collect:
	$(PY) -m scripts.audit.benchmark_answerability --config "$(CONFIG)" collect

answerability-balance:
	$(PY) -m scripts.audit.benchmark_answerability --config "$(CONFIG)" balance

llm-judge-submit:
	$(PY) -m scripts.evaluation.llm_judge --config "$(CONFIG)" \
		$(if $(E2E_RUN_ID),--run-id "$(E2E_RUN_ID)",) \
		$(if $(E2E_POLICY),--policy "$(E2E_POLICY)",) \
		submit

llm-judge-collect:
	$(PY) -m scripts.evaluation.llm_judge --config "$(CONFIG)" \
		$(if $(E2E_RUN_ID),--run-id "$(E2E_RUN_ID)",) \
		$(if $(E2E_POLICY),--policy "$(E2E_POLICY)",) \
		collect

llm-judge-merge:
	$(PY) -m scripts.evaluation.llm_judge --config "$(CONFIG)" \
		$(if $(E2E_RUN_ID),--run-id "$(E2E_RUN_ID)",) \
		$(if $(E2E_POLICY),--policy "$(E2E_POLICY)",) \
		merge

llm-judge-submit-all-policies:
	@for pol in $(E2E_POLICIES); do \
		echo "=== llm-judge submit policy=$$pol$(if $(E2E_RUN_ID), run=$(E2E_RUN_ID),) ==="; \
		$(MAKE) llm-judge-submit E2E_POLICY=$$pol CONFIG="$(CONFIG)" $(if $(E2E_RUN_ID),E2E_RUN_ID="$(E2E_RUN_ID)",) || exit 1; \
	done

llm-judge-collect-all-policies:
	@for pol in $(E2E_POLICIES); do \
		echo "=== llm-judge collect policy=$$pol$(if $(E2E_RUN_ID), run=$(E2E_RUN_ID),) ==="; \
		$(MAKE) llm-judge-collect E2E_POLICY=$$pol CONFIG="$(CONFIG)" $(if $(E2E_RUN_ID),E2E_RUN_ID="$(E2E_RUN_ID)",) || exit 1; \
	done

llm-judge-merge-all-policies:
	@for pol in $(E2E_POLICIES); do \
		echo "=== llm-judge merge policy=$$pol$(if $(E2E_RUN_ID), run=$(E2E_RUN_ID),) ==="; \
		$(MAKE) llm-judge-merge E2E_POLICY=$$pol CONFIG="$(CONFIG)" $(if $(E2E_RUN_ID),E2E_RUN_ID="$(E2E_RUN_ID)",) || exit 1; \
	done

# Smoke: one question, dense-only, dry-run; override E2E_RUN_ID or rely on a smoke-oriented CONFIG
E2E_SMOKE_RUN_ID ?= smoke-$(shell date +%Y%m%d-%H%M%S)
e2e-smoke-test-v01:
	$(PY) -m scripts.e2e_benchmark --config "$(CONFIG)" prepare --dry-run --limit 1 \
		--run-id "$(E2E_SMOKE_RUN_ID)" --policy dense-only

# Testing cases where retrieval performance increases and QA decreases to understand the LLM bottleneck
gen-debug:
	$(PY) scripts/discrepancy_debug_e2e.py --config "$(CONFIG)"

# Back-compat alias
discrepancy-debug-e2e: gen-debug

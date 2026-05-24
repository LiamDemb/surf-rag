# SuRF-RAG: Supervised Retrieval Fusion of DenseRAG and GraphRAG for Mixed-Reasoning QA

SuRF-RAG is a research codebase for query-adaptive fusion of **dense** and **graph** retrieval over a Wikipedia corpus. A lightweight supervised router predicts, per question, how much weight to give each branch. Retrieved evidence is rescored and reranked according to this weight and passed to an LLM for answer generation. The goal is to improve mixed-reasoning QA relative to dense-only, graph-only, and fixed-fusion baselines.

## How it works

```text
Benchmark ingest → Wikipedia corpus + knowledge graph
        ↓
Oracle fusion sweep (dense/graph branch cache + weight grid)
        ↓
Orchestrator training (predict fusion weight or branch class)
        ↓
End-to-end evaluation (retrieval → generation → QA metrics)
```

| Component        | Role                                                                         |
| ---------------- | ---------------------------------------------------------------------------- |
| **DenseRAG**     | Sentence-transformer embeddings + FAISS over corpus chunks                   |
| **GraphRAG**     | Entity–relation graph from corpus IE; query-linked seeds + heterogeneous PPR |
| **Orchestrator** | Supervised MLP trained on oracle labels                                      |
| **E2E**          | Routed retrieval, OpenAI batch QA generation                                 |

All stages are driven by YAML configs and wrapped by the [Makefile](Makefile).

## Requirements

- Python **3.12** (see `pyproject.toml` for supported patch exclusions)
- [Poetry](https://python-poetry.org/)
- **OpenAI API key** for corpus IE, optional OpenAI embeddings, and E2E generation (set in `.env`)
- Hugging Face model cache (`HF_HOME` / `TRANSFORMERS_CACHE`, optional in `.env`)
- **spaCy** English model for query features: `python -m spacy download en_core_web_sm`
- Disk space for Wikipedia articles, indexes, graphs, and evaluation artifacts under `data/` (not shipped in git)

## Installation

```bash
git clone <repo-url>
cd surf-rag
cp .env.example .env   # add OPENAI_API_KEY=...
poetry install
poetry run pre-commit install   # optional
make setup-models               # warm embedding + cross-encoder weights
make test
```

## Configuration

Point every `make` target at a pipeline YAML:

```bash
export CONFIG=configs/run-005/orchestrator-dataset.yaml
make print-resolved-config
```

The config files used in the final testing presented in this project's dissertation live under [`configs/run-005/`](configs/run-005/) (orchestrator dataset, regressor/classifier training, E2E regression/classification, results).

## Typical workflow

Adjust `CONFIG` for each stage. Order assumes benchmarks, corpus, and indexes already exist (see [Data](#data)).

**1. Corpus (one-time per benchmark slice)**

```bash
CONFIG=configs/your-pipeline.yaml make pipeline
# ingest → fetch Wikipedia → align 2Wiki support → build corpus → filter benchmark
```

**Note:** this pipeline was created for our specific research purposes. Running this corpus creation with a real API key can be **very expensive**. Please make sure you understand the cost of the run you are about to do before executing these scripts.

**2. Oracle labels**

```bash
CONFIG=configs/your-pipeline.yaml make oracle-labels
```

**3. Router dataset and training**

```bash
CONFIG=configs/run-005/orchestrator-dataset.yaml make router-pipeline
CONFIG=configs/run-005/regressor.yaml make router-train
CONFIG=configs/run-005/regressor.yaml make router-evaluate
```

**4. End-to-end benchmark**

```bash
CONFIG=configs/run-005/e2e-rg.yaml make e2e-submit      # submits OpenAI batch
# wait for batch completion
CONFIG=configs/run-005/e2e-rg.yaml make e2e-collect
CONFIG=configs/run-005/e2e-rg.yaml make e2e-evaluate
```

Use `make help` for the full target list (ablations, figures, LLM judge, answerability audits, `results-build`, etc.).

**Dry run:** `make e2e-prepare` runs retrieval locally without submitting a generation batch.

**All policies:** `make e2e-run-all-policies` (optional `E2E_RUN_ID=…`).

## Routing policies

| Policy         | Description                                               |
| -------------- | --------------------------------------------------------- |
| `dense-only`   | Dense branch only                                         |
| `graph-only`   | Graph branch only                                         |
| `50-50`        | Fixed equal fusion                                        |
| `rrf`          | Reciprocal rank fusion of both branches                   |
| `learned-soft` | Router predicts continuous dense weight (regression)      |
| `hard-routing` | Router picks dense **or** graph endpoint (classification) |

Regression vs classification training and policy naming are documented in [docs/router_dual_task_guide.md](docs/router_dual_task_guide.md).

## Project layout and contents

```text
src/surf_rag/       Core library (retrieval, graph, router, evaluation, generation)
scripts/            CLI entry points (corpus, oracle, router, e2e, results)
configs/            Pipeline YAML recipes (run-005 committed; more may be local)
prompts/            LLM prompts (generation, corpus IE)
tests/              Pytest suite
data/               Artifacts at runtime
```

**Note:** This repositorty contains several extra files, modules and technologies that were created throughout the development of this project and were NOT used in the final experimentation and results of the Honours thesis. Only what was reported in the paper is considered part of the final system and experimentation. At this stage, this repository is designed solely for this purpose.

## Data

Source benchmarks and corpora are **not** included in the repository. Download helpers:

```bash
poetry run python scripts/datasets/nq_download.py --help
poetry run python scripts/datasets/2wiki_download.py --help
```

After ingest, artifacts are written under `data/` (benchmarks, processed corpus, FAISS indexes, graphs, router checkpoints, E2E run outputs). Router checkpoints follow:

```text
data/router/<router_id>/
  oracle/           Oracle branch caches and soft labels
  dataset/          router_dataset.parquet
  models/<arch>/    Trained checkpoints (per input mode / task type)
```

## Citation

If you use this code in academic work, please cite the associated thesis.

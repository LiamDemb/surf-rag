import yaml
import json
import os
import sys
import argparse
from pathlib import Path

# Add src to path so we can import surf_rag
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))
from surf_rag.evaluation.retrieval_metrics import recall_at_k
from surf_rag.retrieval.types import RetrievedChunk


def main():
    parser = argparse.ArgumentParser(
        description="Calculate % of train questions that retrieved ALL needed gold sentences in top-k"
    )
    parser.add_argument(
        "config_path", help="Path to config file (e.g. configs/router-rg.yaml)"
    )
    parser.add_argument(
        "--run-id",
        help="Override the run ID (default is e2e-{router_architecture_id} from config)",
    )

    args = parser.parse_args()

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    paths = config.get("paths", {})
    benchmark_base = paths.get("benchmark_base", "data/benchmarks")
    benchmark_name = paths.get("benchmark_name", "surf-bench")
    benchmark_id = paths.get("benchmark_id", "main")
    router_base = paths.get("router_base", "data/router")
    router_id = paths.get("router_id", "final")

    if args.run_id:
        relevant_run_id = args.run_id
    else:
        router_arch_id = paths.get("router_architecture_id", "rg-001")
        relevant_run_id = f"e2e-{router_arch_id}"

    # Calculate relative to the config file's directory project root
    project_root = Path(args.config_path).resolve().parent.parent

    benchmark_dir = project_root / benchmark_base / benchmark_name / benchmark_id
    router_dir = project_root / router_base / router_id

    # Load train split IDs
    split_path = router_dir / "dataset" / "split_question_ids.json"
    if not split_path.exists():
        print(f"Split file not found at {split_path}")
        sys.exit(1)

    with open(split_path) as f:
        splits = json.load(f)
    train_ids = set(splits.get("train", []))

    # Load benchmark dataset to get the gold support sentences
    benchmark_path = benchmark_dir / "benchmark" / "benchmark.jsonl"
    questions = {}
    with open(benchmark_path) as f:
        for line in f:
            data = json.loads(line)
            qid = data["question_id"]
            if qid in train_ids:
                questions[qid] = {
                    "gold_support_sentences": data.get("gold_support_sentences", []),
                    "dataset_source": data.get("dataset_source", ""),
                }

    # Evaluate across all policies in the evaluations directory
    evals_dir = benchmark_dir / "evaluations"
    policies = sorted([d.name for d in evals_dir.iterdir() if d.is_dir()])

    print(f"Total train questions loaded: {len(questions)}")
    print(f"Evaluating across policies for run {relevant_run_id}...\n")

    for policy in policies:
        retrieval_path = (
            evals_dir
            / policy
            / relevant_run_id
            / "retrieval"
            / "retrieval_results_pretrunc.jsonl"
        )
        if not retrieval_path.exists():
            continue

        all_found = {5: 0, 10: 0, 20: 0}

        # Read file and dedup by qid (keep last occurrence in case of retries)
        q_data = {}
        with open(retrieval_path) as f:
            for line in f:
                data = json.loads(line)
                qid = data["question_id"]
                if qid in questions:
                    q_data[qid] = data

        total = len(q_data)
        if total == 0:
            continue

        for qid, data in q_data.items():
            chunks_data = data.get("chunks", [])

            # Reconstruct RetrievedChunk items
            chunks = []
            for c in chunks_data:
                chunks.append(
                    RetrievedChunk(
                        chunk_id=c.get("chunk_id", ""),
                        text=c.get("text", ""),
                        score=c.get("score", 0.0),
                        rank=c.get("rank", 0),
                        metadata=c.get("metadata", {}),
                    )
                )

            gold = questions[qid]["gold_support_sentences"]
            ds = questions[qid]["dataset_source"]

            for k in [5, 10, 20]:
                rec = recall_at_k(chunks, gold, k, dataset_source=ds)
                # If recall_at_k is 1.0, ALL required chunks were found
                if rec >= 0.999:
                    all_found[k] += 1

        print(f"Policy: {policy}")
        for k in [5, 10, 20]:
            pct = (all_found[k] / total) * 100
            print(f"  Top-{k}: {pct:.2f}% ({all_found[k]}/{total})")
        print("-" * 40)


if __name__ == "__main__":
    main()

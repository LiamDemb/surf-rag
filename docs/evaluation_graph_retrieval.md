# Graph Retrieval Evaluation (Canonical PPR)

This document describes the **current canonical graph retriever implementation** only.
It intentionally excludes historical/versioned implementations.

## Canonical Behavior

Graph retrieval uses a single canonical path in `GraphRetriever`:

1. Extract query entity candidates (lexicon+alias pipeline by default; optional vector fallback).
2. Resolve candidates to graph nodes `E:*`.
3. Build restart masses over resolved query-linked entities using:
   - semantic activation from `cos(query, entity_text)` (entity text is `E:` stripped),
   - stable softmax temperature `τ = graph_seed_softmax_temperature`,
   - IDF factor `log((Nref+1)/(df+1))`,
   - L1 normalization over **resolved candidate entity nodes only**.
4. Enumerate global frontier paths from restart-weighted seeds.
5. Ground paths for explanation diagnostics (support-threshold filtered), but **final ranking does not require grounded paths**.
6. Run heterogeneous entity+chunk Personalized PageRank over scoped nodes:
   - node set = capped entity scope + adjacent chunk nodes,
   - entity rows split between entity neighbors and chunk neighbors via `graph_entity_chunk_edge_weight`,
   - chunk rows distribute uniformly to incident entities,
   - isolated rows self-loop.
7. Rank chunks directly by stationary PPR mass on `C:*` nodes.

No versioned modes are used in canonical evaluation.

## Transition / PPR Details

PPR update follows:

`pi_new = alpha * (pi @ P) + (1 - alpha) * restart`

with:

- `alpha = graph_ppr_alpha`
- convergence by `graph_ppr_tol` or `graph_ppr_max_iter`
- transition mode recorded as `heterogeneous_entity_chunk` for the final walk

Entity transition split:

- `lambda_ec = clamp(graph_entity_chunk_edge_weight, 0, 1)` = entity→chunk share
- `lambda_ee = 1 - lambda_ec` = entity→entity share
- if both relation neighbors and chunk neighbors exist:
  - relation side gets `lambda_ee` (support-normalized when `graph_transition_mode=support`, else uniform)
  - chunk side gets `lambda_ec` uniformly over chunk neighbors
- if only one neighbor class exists, that class gets full row mass

Chunk edges are treated bidirectionally for transition construction (supports both new bidirectional graphs and older one-way artifacts).

## Retrieval Parameters (Complete)

These are the graph retriever knobs from `retrieval:` (or matching env vars) that affect canonical graph retrieval.

| YAML key | Env var | Used for | Default |
|---|---|---|---|
| `graph_max_hops` | `GRAPH_MAX_HOPS` | Max hop depth for frontier expansion | `2` |
| `graph_bidirectional` | `GRAPH_BIDIRECTIONAL` | Whether frontier expansion traverses reverse direction | `true` |
| `graph_entity_vector_top_k` | `GRAPH_ENTITY_VECTOR_TOP_K` | Top-k candidates for vector fallback when exact entity node missing | `3` |
| `graph_entity_vector_threshold` | `GRAPH_ENTITY_VECTOR_THRESHOLD` | Minimum similarity for vector fallback match | `0.5` |
| `graph_hop_support_threshold` | `GRAPH_HOP_SUPPORT_THRESHOLD` | Grounding filter threshold for path support diagnostics/explanations | `0.5` |
| `graph_ppr_alpha` | `GRAPH_PPR_ALPHA` | PPR continuation/diffusion probability | `0.85` |
| `graph_ppr_max_iter` | `GRAPH_PPR_MAX_ITER` | Max PPR power iterations | `64` |
| `graph_ppr_tol` | `GRAPH_PPR_TOL` | PPR L1 convergence tolerance | `1e-6` |
| `graph_transition_mode` | `GRAPH_TRANSITION_MODE` | Relation-neighbor distribution: `support` or `uniform` | `support` |
| `graph_seed_softmax_temperature` | `GRAPH_SEED_SOFTMAX_TEMPERATURE` | Seed semantic softmax temperature `τ` | `0.1` |
| `graph_entity_chunk_edge_weight` | `GRAPH_ENTITY_CHUNK_EDGE_WEIGHT` | Entity→chunk transition share `lambda_ec` | `0.5` |
| `graph_max_entities` | `GRAPH_MAX_ENTITIES` | Cap on entity nodes in PPR scope | `256` |
| `graph_max_paths` | `GRAPH_MAX_PATHS` | Global frontier path budget | `500` |
| `graph_max_frontier_pops` | `GRAPH_MAX_FRONTIER_POPS` | Global frontier pop budget | `50000` |

## Diagnostics Contract (`debug_info["graph_diagnostics"]`)

Canonical retrieval emits:

- `schema_version`
- `retriever_config` (effective retriever/scoring settings)
- `seed`:
  - counts and candidates (`extracted_candidates`, `resolved_candidates`)
  - `restart_mass`
  - `idf_components`
  - `df_reference`
  - `semantic_restart_diag`:
    - `mode` (`canonical_semantic_softmax_idf`)
    - `softmax_temperature`
    - `unnormalized_mass`
    - `posterior`
    - `per_node` (`cosine_query_entity`, `exp_activation`, `df`, `idf_log_ratio`, `unnormalized_weight`, canonical norm)
- `enumeration` (global frontier stats)
- `grounding` (`candidate_paths`, `grounded_paths_ok`, failures by kind, weak-support sample)
- `local_subgraph` (`entity_count`, relation-edge count, appears_in-edge count)
- `ppr` (`iterations`, `residual`, `damping`, `transition`, `top_entities`)
- `chunk_projection`:
  - `top_chunks`
  - `chunk_scoring.chunk_ppr_mass` (per-chunk stationary mass)
- `explanations.paths_by_chunk`
- `ranking` (`chunk_scores_nonzero`, `unique_chunk_ids_considered`, `chunks_with_text`, `top_chunk_scores`)

Optional under `debug=True`:

- `candidate_paths`
- `grounded_bundle_count`
- `bundle_trace` — per grounded bundle: canonical PPR-based rank score and per-hop entity masses (when `debug=True`)

## NO_CONTEXT / Status Behavior

`RetrievalResult.status` behavior:

- `NO_CONTEXT` when:
  - no start nodes (`no_context_reason: no_start_nodes`), or
  - ranked chunks have no corpus text (`no_context_reason: no_chunk_text_in_corpus`)
- Retrieval can continue with chunk-PPR ranking even when no grounded paths are found (`grounding.note` set accordingly)
- `ERROR` on unexpected exception
- `OK` when at least one chunk is returned

## Graph Construction Assumption

Canonical graph build adds `appears_in` edges in both directions:

- `E:* -> C:*`
- `C:* -> E:*`

Transition logic tolerates older artifacts that only had one directed side.

## Suggested Validation

1. Focused tests:
   - `poetry run pytest tests/test_graph_ppr_scoring.py tests/test_graph_strategy.py tests/test_graph_paths.py -q`
2. Broader suite:
   - `poetry run pytest -q`
3. Trace inspection:
   - `scripts/dev/debug_graph.py --config <your-config>`
   - verify non-zero seed masses, sensible top entities, and non-zero `chunk_ppr_mass`.

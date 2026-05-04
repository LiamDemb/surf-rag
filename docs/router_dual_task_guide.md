# Router Dual-Task Guide

This guide describes how to run router training and E2E evaluation in two modes:

- `regression`: predict dense fusion weight (continuous)
- `classification`: predict one of `dense`, `graph`, `fusion`

The router dataset now carries both supervision targets in one parquet.

## Conceptual Model

- Oracle preparation selects best bins using `oracle.oracle_metric` and `oracle.oracle_metric_k`.
- Soft-label materialization emits:
  - regression target: `oracle_curve`
  - classification target: `oracle_branch_class_id` (dense=0, graph=1, fusion=2)
- Classification tie-break is deterministic: `dense > graph > fusion`.
- Router training chooses which target to use with `router.train.task_type`.

## Configuration Fields

- `router.train.task_type`: `regression` (default) or `classification`
- `e2e.router_task_type`: `regression` (default) or `classification`
- `e2e.policy`: use `learned-*-cls` policies for classification checkpoints

Regression defaults remain unchanged when task fields are absent.

## Artifacts and Paths

Canonical model path:

- `data/router/<router_id>/models/<router_architecture_id>/<task_type>/<input_mode>/`

Loader fallback order is deterministic:

1. `models/<arch>/<task>/<input_mode>`
2. `models/<arch>/<input_mode>`
3. `model/<input_mode>`
4. `model/`

Missing manifest `task_type` defaults to `regression`.

## Policy Naming

- Regression policies:
  - `learned-soft`, `learned-hard`, `learned-hybrid`
- Classification policies:
  - `learned-soft-cls`, `learned-hard-cls`, `learned-hybrid-cls`

`*-cls` policies require a classification model. Non-`*-cls` learned policies require regression.

## End-to-End Examples

Train regression:

```bash
python -m scripts.router.train_router --config configs/pipelines/surf-bench-200.yaml --router-task-type regression
```

Train classification:

```bash
python -m scripts.router.train_router --config configs/pipelines/surf-bench-200.yaml --router-task-type classification
```

E2E regression:

```bash
python -m scripts.e2e_benchmark --config configs/pipelines/surf-bench-200.yaml prepare --policy learned-soft --router-task-type regression
```

E2E classification:

```bash
python -m scripts.e2e_benchmark --config configs/pipelines/surf-bench-200.yaml prepare --policy learned-soft-cls --router-task-type classification
```

## Migration Notes

- Existing regression bundles continue to load.
- Old manifests without `task_type` are interpreted as regression.
- Old model locations are still discoverable through fallback resolution.
- You can gradually add classification checkpoints next to existing regression models.

## Troubleshooting

- **Policy/task mismatch error**
  - Use `learned-*-cls` with `router_task_type=classification`.
  - Use `learned-*` with `router_task_type=regression`.
- **Checkpoint not found**
  - Verify `router_architecture_id`, `input_mode`, and `task_type`.
  - Confirm fallback paths exist for legacy bundles.
- **Classification predicts wrong branch semantics**
  - Check class mapping in manifest: `dense=1.0`, `graph=0.0`, `fusion=0.5`.
- **Unexpected regression behavior changes**
  - Remove task fields to confirm fallback behavior; defaults remain regression.

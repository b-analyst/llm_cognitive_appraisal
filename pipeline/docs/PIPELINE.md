# Pipeline overview

This document is the **end-to-end map** of the emotion–appraisal analysis pipeline. Implementation code is grouped by **stage** under `pipeline/stage_*` so you can open a folder and read its `README.md` (and optional `docs/`) for that slice of the workflow.

Shared infrastructure lives outside the stage tree:

- **`pipeline/core/`** — configuration, model metadata, research contracts, logging, CPU thread limits.

Benchmark-only helpers (probe latent scoring, runtime readout, adaptive appraisal targets, appraisal steering vectors) live under **`pipeline/stage_06_benchmarks/utils/`**, not in a global `lib/` package.

Top-level files such as `pipeline/train_probes.py` are thin **shims**: `python -m pipeline.train_probes` still works and delegates into the right `stage_*` module. Aside from `__init__.py` and those shims/aliases (`pipeline/config.py`, etc.), implementation code is under `core/` and `stage_*`.

## Stage order and output directories

| Order | Stage package | Primary outputs under `outputs/<model_id>/` |
|------|----------------|-----------------------------------------------|
| 1 | [`stage_01_probes`](../stage_01_probes/README.md) | `01_probe_robustness/`, `01_probes/`, optional `01_probes_grid_ablation/` |
| 2 | [`stage_02_circuit`](../stage_02_circuit/README.md) | `02_circuit/` (selection, hidden-state caches, `circuits.json`) |
| 3 | [`stage_03_appraisal_structure`](../stage_03_appraisal_structure/README.md) | `03_appraisal_structure/` (z-scores, clustering, `label_coupling/`, PCA EDA) |
| 4 | [`stage_04_appraisal_in_circuit`](../stage_04_appraisal_in_circuit/README.md) | `04_appraisal_in_circuit/` (geometry, correlation, ablations) |
| 5 | [`stage_05_appraisal_theory`](../stage_05_appraisal_theory/README.md) | `04_appraisal_theory/` (theory-facing analyses and summaries) |
| 6 | [`stage_06_benchmarks`](../stage_06_benchmarks/README.md) | `05_steering/`, `05_generation_behavior/`, `05_mental_health_steering/`, judges, baseline study |
| 7 | [`stage_07_orchestration`](../stage_07_orchestration/README.md) | `06_synthesis/`, multi-model aggregation, export utilities |

The automated runner (`python -m pipeline.run_pipeline_for_models`) executes steps in this order; see `stage_07_orchestration/runner.py` for the exact step list and completion markers.

## Where to read more

- **[README.md](../README.md)** — directory map and import conventions.
- **[../../docs/PIPELINE_MAP.md](../../docs/PIPELINE_MAP.md)** — detailed narrative per stage.
- **[../../docs/RUNBOOK.md](../../docs/RUNBOOK.md)** — environment setup and run commands.
- **[RUN_FULL_PIPELINE.md](RUN_FULL_PIPELINE.md)** — manual step-by-step checklist.

# How the `pipeline/` directory is organized

Use this file when you are **reviewing code** or onboarding. The layout is intentional: stable CLI names at the package root, real logic in subpackages.

## Top-level layout

| Path | What it is |
|------|----------------|
| [`core/`](core/) | Config, model registry, research contracts, logging, runtime env — **import as** `pipeline.core.*` or root aliases below. |
| [`stage_01_probes/`](stage_01_probes/) … [`stage_07_orchestration/`](stage_07_orchestration/) | **Implementation** of each pipeline stage. Start with each folder’s `README.md`. |
| [`shims/`](shims/) | Shared **`bind_shim`** helper used by root CLI modules (not run directly). |
| [`tests/`](tests/) | Pytest. |
| [`docs/`](docs/) | In-package markdown: master map, run order, standalone notes ([`docs/README.md`](docs/README.md)). |
| [`notebooks/`](notebooks/) | Jupyter notebooks (run from repo root with `PYTHONPATH` set). |
| `input_data/`, `input_probes/`, `outputs/` | Data and run artifacts (not source code). |

## Root `pipeline/*.py` files (why so many?)

Python’s `python -m pipeline.<name>` requires a **module** `pipeline.<name>`, so each supported command has a small file at the **package root**.

- **CLI shims** (most files): one-liners that call [`shims.redirect.bind_shim`](shims/redirect.py) and delegate to `pipeline.stage_*....` (see docstring on each file for the exact target).
- **Import aliases** (no `bind_shim`): `config.py`, `model_config.py`, `research_contracts.py`, `logger.py`, `logutil.py`, `runtime_env.py` — thin re-exports of [`core/`](core/) for `import pipeline.config` etc.

**To read implementation code**, open the **`stage_*`** module named in the shim docstring, not the root file.

## Quick map: root filename → implementation

| Root CLI module | Delegates to |
|-----------------|--------------|
| `probe_training_robustness`, `train_probes`, `train_appraisal_*`, `run_probe_grid_ablation`, `analyze_probe_grid_ablation` | `pipeline.stage_01_probes.*` |
| `circuit_evidence`, `phase1_circuits` | `pipeline.stage_02_circuit.*` |
| `appraisal_structure`, `appraisal_label_coupling`, `representation_analysis` | `pipeline.stage_03_appraisal_structure.*` |
| `phase2_*` | `pipeline.stage_04_appraisal_in_circuit.*` |
| `appraisal_theory`, `theory_analysis`, `appraisal_overlap_diagnostic` | `pipeline.stage_05_appraisal_theory.*` |
| `steering_benchmark`, `generation_behavior_benchmark`, `mental_health_steering_benchmark`, `baseline_probe_steering_study`, `build_generation_behavior_benchmark`, `run_behavior_judges` | `pipeline.stage_06_benchmarks.*` |
| `run_pipeline_for_models`, `synthesis`, `export_standalone`, `capture_requirements` | `pipeline.stage_07_orchestration.*` |

Stage 6 benchmark helpers live under [`stage_06_benchmarks/utils/`](stage_06_benchmarks/utils/).

## Docs entry points

- **This tree:** [`README.md`](README.md) (short map), [`docs/PIPELINE.md`](docs/PIPELINE.md) (end-to-end narrative).
- **Repository:** [`../docs/PIPELINE_MAP.md`](../docs/PIPELINE_MAP.md), [`../docs/RUNBOOK.md`](../docs/RUNBOOK.md).

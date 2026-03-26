# Stage 7 — Orchestration

Run the **full per-model pipeline**, **multi-model synthesis**, environment export, and dependency capture.

## Modules

- `runner` — step list, resume/skip logic, VRAM filtering (`run_pipeline_for_models` delegates here).
- `synthesis` — per-model and aggregate summaries (`06_synthesis/`).
- `export_standalone` — bundle a minimal reproducible tree.
- `capture_requirements` — record installed packages for a run.

CLI: `python -m pipeline.run_pipeline_for_models`, `python -m pipeline.synthesis`, `python -m pipeline.export_standalone`, `python -m pipeline.capture_requirements`.

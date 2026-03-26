# Stage 6 — Benchmarks and judges

**Intervention** studies: latent steering, behavioral generation benchmarks, mental-health post steering, optional LLM judges, and baseline readout studies.

## `utils/`

Shared **benchmark-only** Python helpers (probe latent scoring, runtime emotion readout, adaptive contrastive targets, appraisal-profile steering vectors). Import as `pipeline.stage_06_benchmarks.utils.<module>`.

## Modules

- `steering_benchmark` — appraisal vs emotion steering at circuit sites (`05_steering/`).
- `generation_behavior_benchmark` — generated text under interventions (`05_generation_behavior/`).
- `mental_health_steering_benchmark` — forum-style prompts and dose curves (`05_mental_health_steering/`).
- `run_behavior_judges` — optional scoring pass over generation outputs.
- `baseline_probe_steering_study` — prompted vs unprompted readouts and controls.
- `build_generation_behavior_benchmark` — dataset build helper for generation benchmark CSVs.

Repo docs: `docs/GENERATION_BEHAVIOR_BENCHMARK.md`, `docs/MENTAL_HEALTH_STEERING_BENCHMARK.md`, `docs/BASELINE_PROBE_STEERING_STUDY.md`.

CLI examples: `python -m pipeline.steering_benchmark`, `python -m pipeline.generation_behavior_benchmark`, `python -m pipeline.run_behavior_judges`.

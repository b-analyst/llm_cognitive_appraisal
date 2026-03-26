# Pipeline TLDR — For research partners

High-level narrative: see [START_HERE.md](START_HERE.md) and [RESEARCH_QUESTIONS.md](RESEARCH_QUESTIONS.md). Stage-by-stage technical detail: [PIPELINE_MAP.md](PIPELINE_MAP.md).

**Interface:** This repo is a **CLI** pipeline (`python -m pipeline.<module>`). There is no in-repo GUI.

## Stages in one glance

1. **Probe robustness** — Pick prompt formatting and supported emotion set.  
2. **Probes** — Train emotion classifiers and appraisal regressors (plus binary appraisal probes) over layers/locations.  
3. **Circuit** — Select multi-site combinations for emotion readout; cache hidden states.  
4. **Appraisal structure** — Describe appraisal profiles by emotion; optional label coupling.  
5. **Appraisal in circuit + theory** — Geometry, ablations, theory-facing timing/reconstruction analyses.  
6. **Steering & benchmarks** — Interventions, generation behavior, mental-health steering.  
7. **Synthesis** — Aggregate figures and `SUMMARY.md` per model (and optionally across models).

## Operational notes (for comparing runs)

- **Steering:** `run_pipeline_for_models` does not enable **behavioral** steering unless you pass **`--steering_behavioral`**. Cache-only vs forward-pass numbers differ.  
- **Skipping appraisal training:** **`--skip_appraisal_probes`** requires pre-existing appraisal probe artifacts under `input_probes` or from an earlier run.  
- **PCA plots:** Built when **`synthesis`** runs and `pca_eda/` is missing; otherwise run `python -m pipeline.representation_analysis`.  
- **Multi-model table:** Default aggregation uses only **pipeline-complete** output dirs (`06_synthesis/SUMMARY.md`). Use **`--aggregate_all_output_dirs`** for the old “every folder under outputs/” behavior.  

Full flag list: [RUNBOOK.md](RUNBOOK.md) (section *Runner flags, synthesis aggregation, and logging*).

# Runbook

This runbook explains how to execute the pipeline in order, what each stage depends on, and what to inspect before moving to the next stage.

## Goal Of This Runbook

Use this file when you want to:

- run the full pipeline from scratch
- rerun a single stage without losing track of dependencies
- sanity-check outputs after each stage
- inspect whether a run looks healthy before trusting the results

For the conceptual overview, start with `docs/START_HERE.md`. For method details, use `docs/EXPERIMENTAL_SETUP.md`. For stage definitions, use `docs/PIPELINE_MAP.md`.

## Before You Run Anything

Run from the repository root so `pipeline/`, `utils.py`, `prompt_manager.py`, and `experiments/` are importable.

Recommended setup:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements_standalone.txt
$env:PYTHONPATH = (Get-Location).Path
```

Minimum input expectation:

- `pipeline/input_data/emotion_appraisal_train_combined.csv` exists for standalone usage, or the default dataset path resolves through `pipeline/config.py`.
- if you want to run actual-generation behavior benchmarking, build or provide `pipeline/input_data/generation_behavior/behavior_benchmark.csv`

## Console / logging (mental health + generation behavior)

Both stages emit **one INFO line at start** with effective flags (`model_id`, approximate `~steps`, `unit_norm`, `gen_intervention_during_decode`, and for MH: `framings`, `appraisal_elicitation`; for generation behavior: `elicitation_steer`; both: **`adaptive_appraisal_target`** when `ADAPTIVE_APPRAISAL_TARGET_ENABLED` is relevant; both: **`RUNTIME_READOUT_EMOTION_MODE`** when runtime readout is relevant). Generation behavior also logs **`benchmark_family`** when the family changes. Progress is via **tqdm**; there is no per-row INFO spam.

**Legacy steering targets (generation + mental health):** keep **`ADAPTIVE_APPRAISAL_TARGET_ENABLED = False`** in `pipeline/config.py` to match pre-adaptive behavior: fixed taxonomy/config `target_emotion` or MH `contrastive_emotion` as the contrastive target, with no extra adaptive columns beyond the documented defaults.

**Runtime rank-1 default:** **`RUNTIME_READOUT_EMOTION_MODE = "circuit_sigmoid_mean"`** (aligned with `circuit_evidence` / `_circuit_logits`). For **legacy linear-only** single-site ranks, use **`single_site`** with the understanding that ranking is still on **σ(linear)** at that site (see `docs/RUNTIME_READOUT.md`). **`circuit_linear_mean`** or turning off aux changes rank-1 / source resolution vs σ-mean runs — compare flip rates if you switch mid-study.

**Runtime / row volume:** default **mental health** uses **three** framings and adds **`appraisal_elicitation`** for \(\alpha > 0\) (more rows than older two-framing runs). Default **generation behavior** adds **`appraisal_elicitation_steer`** (~one extra condition per eligible prompt vs four conditions). Set `GENERATION_BEHAVIOR_INCLUDE_ELICITATION_STEER = False` in `config.py` or use `run_generation_behavior_benchmark(..., include_elicitation_steer=False)` to opt out. For MH, use `--framings counselor introspective` to match legacy framing count.

Optional speed-up:

- prefilled probe files in `pipeline/input_probes/` can let you skip some expensive stages when appropriate

## Runner flags, synthesis aggregation, and logging

**`run_pipeline_for_models`:**

- **`--steering_behavioral`** — Runs real forward-pass steering during `steering_benchmark` (slower). Writes `steering_benchmark_behavioral.csv` and `steering_curves_behavioral.csv`. Without this flag, steering metrics are cache-based only; compare results carefully when reconciling with collaborators who used behavioral steering.
- **`--skip_appraisal_probes`** — Skips training appraisal Ridge probes. You must already have `pipeline/input_probes/appraisal_regression_probes.pt` (or equivalent from a prior run) so later stages can load directions. `train_appraisal_binary_probes` still runs unless you have binary outputs already.
- **`--aggregate_all_output_dirs`** — By default, the end-of-run **multi-model synthesis** includes only models with a complete pipeline (`outputs/<model_id>/06_synthesis/SUMMARY.md`). This flag restores the legacy behavior of aggregating **every** subdirectory of `outputs/` (except `synthesis`). **`--run_judges_after`** uses the same rule: completed models only unless this flag is set.
- **`PIPELINE_LOG_LEVEL`** — e.g. `DEBUG` (default `INFO`) for stderr logging from the runner.
- **`PIPELINE_STRICT_SYNTHESIS`** — Set to `1` or `true` so a failure in `representation_analysis` during `synthesis` aborts instead of logging a warning.

**PCA / EDA (`03_appraisal_structure/pca_eda/`):**

- Invoked from **`run_synthesis`** when `pca_eda` is missing for a model. If you never run synthesis (or skip it), run `python -m pipeline.representation_analysis --model_id ...` manually to generate PCA artifacts.

**CPU thread cap (optional):**

- Set **`PIPELINE_CPU_THREADS`** to a positive integer before running (e.g. `4`) to set `OMP_NUM_THREADS` / MKL / OpenBLAS when unset and to call `torch.set_num_threads`. See `README.md`.

## Full Stage Order

Optional pre-step for the generation benchmark:

```powershell
.venv\Scripts\python -m pipeline.build_generation_behavior_benchmark
```

This builds `pipeline/input_data/generation_behavior/behavior_benchmark.csv` from a hybrid mix of synthetic prompts and any accessible open-source benchmark sources.

Run the stages in this order for a full single-model run:

```powershell
.venv\Scripts\python -m pipeline.probe_training_robustness
.venv\Scripts\python -m pipeline.train_probes
.venv\Scripts\python -m pipeline.train_appraisal_probes
.venv\Scripts\python -m pipeline.train_appraisal_binary_probes
.venv\Scripts\python -m pipeline.circuit_evidence
.venv\Scripts\python -m pipeline.phase1_circuits
.venv\Scripts\python -m pipeline.appraisal_structure
.venv\Scripts\python -m pipeline.phase2_summary
.venv\Scripts\python -m pipeline.appraisal_theory
.venv\Scripts\python -m pipeline.steering_benchmark
.venv\Scripts\python -m pipeline.generation_behavior_benchmark
.venv\Scripts\python -m pipeline.mental_health_steering_benchmark
.venv\Scripts\python -m pipeline.synthesis
```

**Optional — baseline probe readouts + top-k appraisal steering** (extra GPU/time; not part of the default runner):

```powershell
.venv\Scripts\python -m pipeline.baseline_probe_steering_study --model_id Llama3.2_1B --max_rows 10
```

See `docs/BASELINE_PROBE_STEERING_STUDY.md` for design, outputs, and `config.py` constants (`BASELINE_PROBE_STUDY_*`).

**Mental health steering (design & interpretation):** `docs/MENTAL_HEALTH_STEERING_BENCHMARK.md` — unit normalization, alpha grid, prefill vs decode steering, how to read dose curves.

Use the same `--model_id` throughout if you are not using the default model.

**LLM judge (deferred — run after all models finish):**

The LLM judge is disabled by default (`GENERATION_BENCHMARK_RUN_JUDGE = False`) so the main pipeline
runs fast. After all target models have completed, run one judge pass with a stronger model:

```powershell
# Auto-discover all models with 06_synthesis/SUMMARY.md and grade them:
.venv\Scripts\python -m pipeline.run_behavior_judges `
    --auto `
    --judge_model_id Llama3.1_8B `
    --re_run_synthesis

# Or specify models explicitly:
.venv\Scripts\python -m pipeline.run_behavior_judges `
    --model_ids Llama3.2_1B Gemma2_2B `
    --judge_model_id Llama3.1_8B `
    --re_run_synthesis
```

Alternatively, trigger the judge automatically at the end of the pipeline runner:

```powershell
.venv\Scripts\python -m pipeline.run_pipeline_for_models `
    --model_ids Llama3.2_1B Gemma2_2B `
    --run_judges_after `
    --judge_model_id Llama3.1_8B
```

Optional probe grid ablation (slow; runs after canonical `train_probes`; writes `01_probes_grid_ablation/` only):

```powershell
.venv\Scripts\python -m pipeline.run_pipeline_for_models `
    --model_ids Llama3.2_1B `
    --run_grid_ablation
```

Judge outputs written alongside existing pipeline outputs:
- `05_generation_behavior/generation_behavior_judge_scores.csv`
- `05_generation_behavior/generation_behavior_judge_summary_by_condition.csv`
- `05_mental_health_steering/mental_health_judge_scores.csv`
- `05_mental_health_steering/mental_health_judge_summary.csv`

## Stage-By-Stage Checklist

## Stage 0: Probe Robustness

Command:

```powershell
.venv\Scripts\python -m pipeline.probe_training_robustness
```

Purpose:
- compare text variants
- define the supported emotion set
- create the split-aware OVA training artifacts used by the main probe stage

Check these outputs:
- `pipeline/outputs/<model_id>/01_probe_robustness/best_variant_selection.json`
- `pipeline/outputs/<model_id>/01_probe_robustness/supported_emotions.json`
- `pipeline/outputs/<model_id>/01_probe_robustness/probe_robustness_comparison.csv`

Healthy signs:
- the chosen variant is recorded
- supported emotions are not implausibly sparse
- the comparison file is not empty

Things to question:
- nearly all emotions dropped from the supported set
- suspiciously identical variant scores
- missing split manifests or missing counts files

## Stage 1A: Emotion Probes

Command:

```powershell
.venv\Scripts\python -m pipeline.train_probes
```

Purpose:
- train binary OVA emotion probes across the configured grid of layers, locs, and tokens

Check these outputs:
- `pipeline/outputs/<model_id>/01_probes/binary_ova_probes/probe_summary.csv`
- `pipeline/outputs/<model_id>/01_probes/binary_ova_probes/probe_manifest.json`
- matching `binary_ova_probes_*.pt`

Healthy signs:
- rows exist for all supported emotions across many sites
- metrics vary across sites rather than being unnaturally flat
- the probe manifest matches the intended model

Things to question:
- only one layer or loc appears
- all metrics are nearly identical
- a mismatched or missing probe bundle accompanies the summary CSV

## Stage 1B: Appraisal Probes

Command:

```powershell
.venv\Scripts\python -m pipeline.train_appraisal_probes
```

Purpose:
- train one Ridge regressor per appraisal dimension and per site

Check these outputs:
- `pipeline/outputs/<model_id>/01_probes/appraisal_regression_probes.pt`
- `pipeline/outputs/<model_id>/01_probes/appraisal_probe_validation_detail.csv`
- `pipeline/outputs/<model_id>/01_probes/appraisal_regression_probes_manifest.json`

Healthy signs:
- all expected appraisal dimensions appear
- metrics differ across layers and locs
- no obvious evidence that only one site was trained

Things to question:
- very few dimensions survive because of missing values
- every layer and loc shows the same metric pattern
- the output file exists but the validation CSV is empty

## Stage 2: Circuit Evidence

Command:

```powershell
.venv\Scripts\python -m pipeline.circuit_evidence
```

Purpose:
- compare single-site and multi-site circuit readout
- choose top-k on the selection split
- report final metrics on the test split

Check these outputs:
- `pipeline/outputs/<model_id>/02_circuit/circuit_evidence_classification.csv`
- `pipeline/outputs/<model_id>/02_circuit/circuit_top_k_selection.json`
- `pipeline/outputs/<model_id>/02_circuit/selection_hidden_states.pt`
- `pipeline/outputs/<model_id>/02_circuit/test_hidden_states.pt`

Healthy signs:
- `k` is selected by a real sweep rather than defaulting trivially without evidence
- selected pairs are present in the JSON
- multi-site methods are not mysteriously identical to single-site methods unless the CSV explains why

Things to question:
- the selected `k` is always the same across models without any sweep variation
- all comparison bars are exactly the same
- caches exist but manifest conditions do not match the current run

## Stage 2B: Circuit Packaging

Command:

```powershell
.venv\Scripts\python -m pipeline.phase1_circuits
```

Purpose:
- package selected circuit sites into reusable JSON files

Check these outputs:
- `pipeline/outputs/<model_id>/02_circuit/circuits.json`
- `pipeline/outputs/<model_id>/02_circuit/circuit_sites.json`

Healthy signs:
- selected sites match the stage-2 selection JSON

Things to question:
- fallback heuristic files appear when proper stage-2 selection should have been available

## Stage 3: Appraisal Structure

Command:

```powershell
.venv\Scripts\python -m pipeline.appraisal_structure
```

Purpose:
- describe the held-out emotional landscape in appraisal terms

Check these outputs:
- `pipeline/outputs/<model_id>/03_appraisal_structure/baseline_metrics.csv`
- `pipeline/outputs/<model_id>/03_appraisal_structure/appraisal_zscore_by_emotion.csv`
- `pipeline/outputs/<model_id>/03_appraisal_structure/appraisal_zscore_heatmap.png`

Healthy signs:
- appraisal profiles differ across emotions in interpretable ways
- baseline metrics are plausible relative to the earlier probe quality

Things to question:
- all emotions have nearly identical appraisal profiles
- the stage silently reuses stale artifacts from an incompatible run

## Stage 3B: Representation Analysis

This stage is usually triggered by synthesis if outputs are missing, but it can also be run directly through the representation stage entry point if needed.

Purpose:
- inspect hidden-state geometry with PCA and EDA

Check these outputs:
- `pipeline/outputs/<model_id>/03_appraisal_structure/pca_eda/pca_explained_variance.csv`
- `pipeline/outputs/<model_id>/03_appraisal_structure/pca_eda/pca_site_summary.csv`
- PCA HTML and PNG files under `pca/`

Healthy signs:
- explained variance changes across sites
- hoverable PCA plots contain meaningful metadata

Things to question:
- all PCA outputs are missing even though synthesis completed
- all sites report identical explained-variance structure

## Stage 4: Appraisal In Circuit

Command:

```powershell
.venv\Scripts\python -m pipeline.phase2_summary
```

Purpose:
- compute or summarize circuit geometry, correlation, and cache-based appraisal ablations

Check these outputs:
- `pipeline/outputs/<model_id>/04_appraisal_in_circuit/geometry_circuit_layers.csv`
- `pipeline/outputs/<model_id>/04_appraisal_in_circuit/correlation_circuit_vs_default.csv`
- `pipeline/outputs/<model_id>/04_appraisal_in_circuit/appraisal_ablation_summary.csv`
- `pipeline/outputs/<model_id>/04_appraisal_in_circuit/SUMMARY.md`

Healthy signs:
- geometry and ablation files both exist
- outputs vary by emotion, dimension, or site rather than collapsing to a flat pattern

Things to question:
- ablation output exists but selected circuit-site files are missing
- geometry values look copied across unrelated dimensions or emotions

## Stage 5: Steering Benchmark

Command:

```powershell
.venv\Scripts\python -m pipeline.steering_benchmark
```

Purpose:
- compare cache-based and behavioral interventions using emotion and appraisal directions

Check these outputs:
- `pipeline/outputs/<model_id>/05_steering/steering_benchmark.csv`
- `pipeline/outputs/<model_id>/05_steering/steering_benchmark_behavioral.csv`
- `pipeline/outputs/<model_id>/05_steering/steering_benchmark_by_pair.csv`
- `pipeline/outputs/<model_id>/05_steering/behavioral_appraisal_ablation.csv`

Healthy signs:
- pair-level variation is visible
- behavioral results exist in addition to cache-based results

Things to question:
- only average scores exist with no per-pair breakdown
- behavioral outputs are missing while cache-based outputs exist

## Stage 6: Synthesis

Command:

```powershell
.venv\Scripts\python -m pipeline.synthesis
```

Purpose:
- collect all stage outputs into a final story for the model

Check these outputs:
- `pipeline/outputs/<model_id>/06_synthesis/synthesis_metrics.csv`
- `pipeline/outputs/<model_id>/06_synthesis/SUMMARY.md`

Healthy signs:
- synthesis includes references to stage outputs that actually exist
- representation-analysis outputs are present if synthesis claims they were used

Things to question:
- synthesis completes while several upstream outputs are missing

## Stage 5B: Generation Behavior Benchmark

Command:

```powershell
.venv\Scripts\python -m pipeline.generation_behavior_benchmark
```

Purpose:
- test actual generated assistant behavior under baseline and steering conditions

Check these outputs:
- `pipeline/outputs/<model_id>/05_generation_behavior/generation_behavior_outputs.csv`
- `pipeline/outputs/<model_id>/05_generation_behavior/generation_behavior_latent_readouts.csv`
- `pipeline/outputs/<model_id>/05_generation_behavior/generation_behavior_scores.csv`
- `pipeline/outputs/<model_id>/05_generation_behavior/summary.md`

Healthy signs:
- each prompt has rows for baseline and at least one steering condition; with default config you should also see **`appraisal_elicitation_steer`** rows when `target_emotion` / circuit data support steering
- generated text is present and non-empty
- latent readouts align with the prompt/condition rows

Things to question:
- only baseline rows appear despite source and target emotions being provided
- generated outputs are empty or truncated in a suspiciously uniform way
- behavior scores are identical across all conditions for all prompts
- the stage silently skips because the benchmark CSV is missing

## Fast Validation Pass After A Run

If you want a compact sanity pass after the full pipeline:

1. Check `best_variant_selection.json` and `supported_emotions.json`.
2. Check `probe_summary.csv` for broad site coverage and non-flat metrics.
3. Check `circuit_top_k_selection.json` for a real top-k sweep result.
4. Check `appraisal_probe_validation_detail.csv` for dimension coverage.
5. Check `appraisal_ablation_summary.csv` and steering outputs for nontrivial intervention effects.
6. Read `06_synthesis/SUMMARY.md` last, not first.

## Re-Run Rules

If you change any of the following, do not blindly trust cached downstream artifacts:

- `model_id`
- dataset contents
- split fractions
- extraction layers, locs, or tokens
- supported emotion thresholds
- probe hyperparameters

At minimum, re-run the earliest stage affected by the change and verify manifests and caches downstream.

## Related Docs

- `docs/START_HERE.md`
- `docs/RESEARCH_QUESTIONS.md`
- `docs/EXPERIMENTAL_SETUP.md`
- `docs/PIPELINE_MAP.md`
- `docs/OUTPUTS_GUIDE.md`
- `docs/GLOSSARY.md`
- `docs/GENERATION_BEHAVIOR_BENCHMARK.md`
- `docs/BENCHMARK_DATASETS.md`
- `pipeline/docs/RUN_FULL_PIPELINE.md`

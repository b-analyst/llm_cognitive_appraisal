# Pipeline Map

**Code layout:** modules live under `pipeline/core/`, `pipeline/shims/` (shared shim helper), numbered stage packages `pipeline/stage_01_probes/` … `pipeline/stage_07_orchestration/`, benchmark helpers under `pipeline/stage_06_benchmarks/utils/`, and narrative docs under `pipeline/docs/` (see `pipeline/README.md`, `pipeline/STRUCTURE.md`, and `pipeline/docs/PIPELINE.md`). CLI module names (`python -m pipeline.train_probes`, etc.) are unchanged — they use root shims.

This file explains the pipeline stage by stage, focusing on what each stage does, what it produces, and how its outputs feed later stages.

See also:
- `docs/EXPERIMENTAL_SETUP.md` for the methodological rules behind the stages
- `docs/OUTPUTS_GUIDE.md` for output-by-output interpretation
- `docs/RUNBOOK.md` for execution order and stage checkpoints
- `docs/GLOSSARY.md` for definitions of circuit, probe, steering, and key metrics

## Pipeline Order

1. `01_probe_robustness`
2. `01_probes` (emotion probes)
3. `01_probes` (appraisal regression probes)
4. `01_probes` (appraisal binary OVA probes)
5. `02_circuit` (circuit evidence)
6. `02_circuit` packaging via `phase1_circuits.py`
7. `03_appraisal_structure`
8. `03_appraisal_structure/pca_eda`
9. `04_appraisal_in_circuit`
10. `04_appraisal_theory`
11. `05_steering`
12. `05_generation_behavior`
13. *(optional)* `05_baseline_probe_steering` — see `docs/BASELINE_PROBE_STEERING_STUDY.md`; not run by `run_pipeline_for_models` by default
14. `05_mental_health_steering`
15. `06_synthesis`

## Stage 0: Probe Robustness

Main file:
- `pipeline/probe_training_robustness.py`

Purpose:
Compare alternative text representations and define the supported emotion benchmark.

Inputs:
- canonical combined dataset
- prompt-generation policy

Core computation:
- build alternative training variants
- apply rigorous scenario-level splitting
- summarize per-emotion support
- optionally run a quick comparison probe
- choose the best variant using selection performance

Main outputs:
- `best_variant_selection.json`
- `supported_emotions.json`
- `supported_emotion_counts.csv`
- `probe_robustness_comparison.csv`

Downstream use:
- `train_probes.py` uses the winning variant by default
- later stages inherit the supported emotion set indirectly through the trained probe outputs

Why it exists:
It prevents the main probe stage from silently depending on an arbitrary text-format choice.

## Stage 1A: Emotion Probes

Main file:
- `pipeline/train_probes.py`

Backend:
- `experiments/utils/training_utils.py`

Purpose:
Train binary OVA emotion probes across the configured grid of internal model sites.

Inputs:
- supported emotion set
- best text variant from Stage 0
- configured extraction layers, locs, and tokens

Core computation:
- extract hidden states
- fit one binary classifier per emotion at each site
- tune `C` on the selection split
- write summary metrics and serialized probe bundles

Main outputs:
- `01_probes/binary_ova_probes/probe_summary.csv`
- `01_probes/binary_ova_probes/binary_ova_probes_*.pt`
- `01_probes/binary_ova_probes/probe_manifest.json`

Downstream use:
- `circuit_evidence.py` ranks sites from `probe_summary.csv`
- `appraisal_structure.py` uses emotion probes for descriptive analyses
- `steering_benchmark.py` uses probe directions for interventions

Why it exists:
This stage answers where emotion information is linearly readable.

## Stage 1B: Appraisal Probes

Main file:
- `pipeline/train_appraisal_probes.py`

Purpose:
Train appraisal regressors across internal model sites.

Inputs:
- canonical combined dataset
- train and selection splits
- available appraisal target columns

Core computation:
- optionally emotion-balance the training split
- extract hidden states
- fit one Ridge regressor per appraisal dimension and per site
- evaluate on the selection split

Main outputs:
- `01_probes/appraisal_regression_probes.pt`
- `01_probes/appraisal_probe_validation_detail.csv`
- `01_probes/appraisal_regression_probes_manifest.json`

Downstream use:
- `phase2_compute.py` compares appraisal and emotion directions in raw hidden-state space
- `steering_benchmark.py` builds appraisal steering vectors

Why it exists:
This stage answers where appraisal information is linearly readable.

## Stage 1C: Appraisal Binary Probes

Main file:
- `pipeline/train_appraisal_binary_probes.py`

Purpose:
Train binary OVA logistic probes for each appraisal dimension (high vs low), using median-split binarization on the train set.

Inputs:
- canonical combined dataset
- train and selection splits

Core computation:
- binarize each appraisal dimension at the train-set median
- fit one LogisticRegression per dimension per site
- evaluate on the selection split

Main outputs:
- `01_probes/appraisal_binary_ova_probes/appraisal_binary_ova_probes_*.pt`
- `01_probes/appraisal_binary_ova_probes/appraisal_binary_summary.csv`
- `01_probes/appraisal_binary_ova_probes/appraisal_binary_manifest.json`

Downstream use:
- `appraisal_theory.py` Analysis E compares Ridge vs binary probe directions
- potential sharper intervention directions for future steering experiments

Why it exists:
Binary probes provide categorical decision boundaries that may be more causally meaningful for interventions than regression gradients.

## Stage 2: Circuit Evidence

Main file:
- `pipeline/circuit_evidence.py`

Purpose:
Test whether a selected multi-site circuit outperforms single-site baselines.

Inputs:
- emotion probe summary and serialized probe bundle
- selection and test splits from the canonical dataset
- hidden-state caches or fresh extractions

Core computation:
- load or extract hidden states for selection and test
- compare `single_best`, `topk_fusion`, and `topk_fusion_global`
- sweep top-k values up to the configured maximum
- choose the best `k` on selection
- report final metrics on the held-out test split

Decision rule:
- `single_best` predicts from one site's full emotion score vector.
- `topk_fusion` builds one fused score per emotion by averaging that emotion's probe score across its own selected `(layer, loc)` circuit sites, then predicts with argmax over the fused emotion scores.
- `topk_fusion_global` averages full emotion score vectors across one shared global top-k site list, then predicts with argmax.

Main outputs:
- `02_circuit/circuit_evidence_classification.csv`
- `02_circuit/circuit_evidence_classification.png`
- `02_circuit/circuit_top_k_selection.json`
- `02_circuit/selection_hidden_states.pt`
- `02_circuit/test_hidden_states.pt`

Downstream use:
- `phase1_circuits.py` packages selected sites into reusable circuit files
- `phase2_compute.py`, `appraisal_structure.py`, and `steering_benchmark.py` reuse cached test hidden states

Why it exists:
This is the core stage for the circuit claim.

## Stage 2B: Circuit Packaging

Main file:
- `pipeline/phase1_circuits.py`

Purpose:
Convert circuit-selection outputs into reusable circuit definitions.

Inputs:
- `circuit_top_k_selection.json`
- probe summary fallback if strict circuit selection output is unavailable

Core computation:
- read the chosen site set
- write reusable summaries for downstream stages

Main outputs:
- `02_circuit/circuits.json`
- `02_circuit/circuit_sites.json`

Downstream use:
- phase 2 analyses
- steering
- synthesis

Why it exists:
Later stages need a stable, file-based circuit definition rather than rerunning site selection logic.

## Stage 3: Appraisal Structure

Main file:
- `pipeline/appraisal_structure.py`

Purpose:
Describe the emotional landscape in appraisal terms on held-out data.

Inputs:
- trained emotion probes
- canonical train and test splits
- appraisal columns

Core computation:
- evaluate baseline emotion classification on held-out data
- summarize cluster-level patterns
- compute appraisal z-scores by emotion using train-derived normalization

Main outputs:
- `03_appraisal_structure/baseline_metrics.csv`
- `03_appraisal_structure/cluster_emotion_mapping.csv`
- `03_appraisal_structure/appraisal_zscore_by_emotion.csv`
- `03_appraisal_structure/appraisal_zscore_heatmap.png`
- `03_appraisal_structure/summary.md`

Downstream use:
- interpretation of emotion relationships
- theory-facing context for phase 2 and steering results

Why it exists:
This stage turns raw model signals into a descriptive appraisal map.

## Stage 3B: Representation Analysis

Main file:
- `pipeline/representation_analysis.py`

Purpose:
Provide EDA and PCA-based views of hidden-state structure for each layer and loc.

Inputs:
- held-out test frame
- hidden-state features for configured sites
- emotion labels and appraisal metadata

Core computation:
- summarize data support and appraisal distributions
- run PCA per site
- save explained-variance tables
- create interactive and static visualizations

Main outputs:
- `03_appraisal_structure/pca_eda/pca_explained_variance.csv`
- `03_appraisal_structure/pca_eda/pca_site_summary.csv`
- `03_appraisal_structure/pca_eda/pca/layer_<L>_loc_<LOC>.html`
- `03_appraisal_structure/pca_eda/pca/layer_<L>_loc_<LOC>.png`
- `03_appraisal_structure/pca_eda/eda/*.csv`

Downstream use:
- theory-facing interpretation
- synthesis outputs

Why it exists:
It shows how emotion and appraisal information is distributed geometrically across the model rather than only through scalar metrics.

## Stage 4: Appraisal In Circuit

Main files:
- `pipeline/phase2_compute.py`
- `pipeline/phase2_summary.py`

Purpose:
Test whether appraisal structure is present and functionally relevant inside the selected emotion circuit.

Inputs:
- circuit definition files
- emotion probes
- appraisal probes
- cached held-out hidden states

Core computation:
- compare emotion and appraisal directions in raw hidden-state space
- compare circuit-site appraisal estimates with default-layer appraisal estimates
- erase full appraisal signatures or individual appraisal dimensions
- generate theory-facing comparison figures

Main outputs:
- `04_appraisal_in_circuit/geometry_circuit_layers.csv`
- `04_appraisal_in_circuit/correlation_circuit_vs_default.csv`
- `04_appraisal_in_circuit/appraisal_ablation_summary.csv`
- `04_appraisal_in_circuit/SUMMARY.md`

Why it exists:
This stage directly addresses the claim that appraisal information lives inside emotion circuits.

## Stage 4B: Appraisal Theory Analysis

Main file:
- `pipeline/appraisal_theory.py`

Purpose:
Test whether appraisal information precedes and builds emotion representations, consistent with cognitive appraisal theory. See `docs/APPRAISAL_THEORY.md` for full details.

Inputs:
- emotion and appraisal probe summaries from `01_probes`
- circuit selection from `02_circuit`
- cached hidden states from `02_circuit`
- appraisal z-scores from `03_appraisal_structure`
- binary appraisal probes from `01_probes` (optional, for Analysis E)

Core computation:
- A: Circuit structure characterization (location entropy, overlap matrix)
- B: Layer onset comparison (appraisal vs emotion)
- C: Within-layer location ordering
- D: Appraisal-to-emotion reconstruction experiment
- E: Ridge vs binary probe direction comparison
- F: Cross-layer appraisal-to-emotion prediction asymmetry
- G: Emotions in appraisal space (PCA biplot + interactive 3D)

Main outputs:
- `04_appraisal_theory/circuit_structure_summary.csv`
- `04_appraisal_theory/onset_comparison.csv`
- `04_appraisal_theory/reconstruction_by_layer_loc.csv`
- `04_appraisal_theory/cross_layer_prediction.csv`
- `04_appraisal_theory/appraisal_space_3d.html`
- `04_appraisal_theory/SUMMARY.md`
- multiple PNG figures

Downstream use:
- synthesis
- theory-facing evidence for cognitive appraisal theory claims

Why it exists:
This stage directly tests the four predictions of cognitive appraisal theory: temporal precedence, sufficiency, directional flow, and structural specificity.

## Stage 5: Steering Benchmark

Main file:
- `pipeline/steering_benchmark.py`

Purpose:
Test whether moving or erasing internal emotion or appraisal directions changes the model's emotion behavior.

Inputs:
- emotion probe directions
- appraisal probe directions
- circuit-site definitions
- held-out texts and hidden states

Core computation:
- default appraisal **α** and emotion **β** sweeps use **`STEERING_BENCHMARK_ALPHAS`** (same grid as `MENTAL_HEALTH_ALPHAS`); per-pair summary rows use the dose nearest **`MENTAL_HEALTH_REPORT_ALPHA`**
- run cache-based steering
- run behavioral forward-pass steering
- compare prompted and unprompted settings where relevant
- run behavioral appraisal ablations
- aggregate results by pair and by pair category

Main outputs:
- `05_steering/steering_benchmark.csv`
- `05_steering/steering_curves.csv`
- `05_steering/steering_benchmark_behavioral.csv`
- `05_steering/steering_curves_behavioral.csv`
- `05_steering/steering_benchmark_by_pair.csv`
- `05_steering/behavioral_appraisal_ablation.csv`

Why it exists:
This stage provides the strongest causal-style evidence in the pipeline.

## Stage 5B: Generation Behavior Benchmark

Main file:
- `pipeline/generation_behavior_benchmark.py`

Purpose:
Test whether appraisal and emotion interventions change the model's actual generated text behavior, not just the latent circuit readout.

Inputs:
- selected circuit sites from `02_circuit`
- emotion and appraisal probes from `01_probes`
- appraisal profiles from `03_appraisal_structure`
- benchmark prompts from `pipeline/input_data/generation_behavior/behavior_benchmark.csv`

Core computation:
- run real text generation under baseline and steering conditions (by default: **`appraisal_elicitation_steer`** is included when `GENERATION_BEHAVIOR_INCLUDE_ELICITATION_STEER` is `True` and appraisal probes exist — see `docs/GENERATION_BEHAVIOR_BENCHMARK.md`)
- optional **`ADAPTIVE_APPRAISAL_TARGET_ENABLED`** (`config.py`): pick steering **target emotion** by max z-profile contrast vs resolved source (see `docs/GENERATION_BEHAVIOR_BENCHMARK.md`); default **on** in `config.py` (set `False` for fixed CSV targets)
- optional **`RUNTIME_READOUT_EMOTION_MODE`**: default **`circuit_sigmoid_mean`** (per-site σ then mean — matches `circuit_evidence`); also `single_site`, **`circuit_linear_mean`**; see **`docs/RUNTIME_READOUT.md`**
- save raw generations for every prompt and intervention
- compute latent circuit readouts for the same prompts
- compute lightweight behavior proxies over the generated text

Main outputs:
- `05_generation_behavior/generation_behavior_outputs.csv`
- `05_generation_behavior/generation_behavior_latent_readouts.csv`
- `05_generation_behavior/generation_behavior_scores.csv`
- `05_generation_behavior/generation_behavior_summary_by_condition.csv`
- `05_generation_behavior/generation_behavior_summary_by_family.csv`
- `05_generation_behavior/summary.md`

Downstream use:
- synthesis
- future safety and persona-control evaluation
- actual-generation validation of appraisal/emotion interventions

Why it exists:
This stage is the bridge between latent-state steering and user-visible assistant behavior.

## Optional: Baseline probe steering study (`05_baseline_probe_steering`)

Main file:
- `pipeline/baseline_probe_steering_study.py`

**Documentation:** `docs/BASELINE_PROBE_STEERING_STUDY.md`

Purpose:
Log **prompted vs unprompted** probe readouts on the same benchmark prompts, rank emotions with **top-k + margin** on linear OvA logits at a data-selected readout site, steer using **top-m appraisal dimensions per emotion** frozen from `appraisal_zscore_by_emotion.csv` and ridge directions at circuit sites, and include **wrong-emotion / random / shuffle** controls. Complements Stage 5B when you want data-chosen targets instead of taxonomy `source_emotion` / `target_emotion` alone.

Inputs:
- same as generation behavior benchmark (CSV, probes, circuit selection, appraisal z-scores)

Main outputs:
- `05_baseline_probe_steering/baseline_probe_readouts.csv`
- `05_baseline_probe_steering/baseline_probe_steering_runs.csv`
- `05_baseline_probe_steering/dim_selection_policy.json`
- `05_baseline_probe_steering/summary.md`

Why it exists:
Separates **which appraisal axes** (from structure) from **how you push** (ridge geometry + alpha), with explicit pre-registered primary text/outcome flags in `config.py`.

## Stage 5C: Mental Health Steering Benchmark

Main file:
- `pipeline/mental_health_steering_benchmark.py`

**Documentation:** `docs/MENTAL_HEALTH_STEERING_BENCHMARK.md` (steering math, normalization, prompt-only generation, how to read results).

Purpose:
Test whether appraisal-based steering changes how the model responds to real mental health forum posts, with condition-specific appraisal targets and dose-response curves. Steering vectors are **L2 unit–normalized** by default; generation uses **prefill-only** intervention unless configured otherwise (see config + doc).

Inputs:
- mental health holdout CSV (`pipeline/input_data/mental_health_dataset_holdout.csv`)
- emotion and appraisal probes from `01_probes`
- circuit sites from `02_circuit`
- appraisal z-scores from `03_appraisal_structure`

Core computation:
- steering methods: baseline, appraisal_targeted, appraisal_full, emotion_steer, combined, plus **`appraisal_elicitation`** (goal-directed profile from `ELICITATION_APPRAISAL_PROFILE`) for \(\alpha > 0\) when the profile is non-empty
- optional **`ADAPTIVE_APPRAISAL_TARGET_ENABLED`**: per post × framing, recompute target emotion (max z-contrast), circuit pairs, and all steering vectors (see `docs/MENTAL_HEALTH_STEERING_BENCHMARK.md`); default **on** in `config.py` (set `False` for fixed contrastive labels)
- optional **`RUNTIME_READOUT_EMOTION_MODE`**: same runtime rank-1 options as generation behavior (see **`docs/RUNTIME_READOUT.md`**)
- default prompt framings from **`MENTAL_HEALTH_FRAMINGS`**: counselor, introspective, **honest_reply** (CLI `--framings` can subset)
- dose-response alpha sweep (`MENTAL_HEALTH_ALPHAS` in `config.py`; default grid tuned for unit-normalized vectors)
- expanded scoring with empathy, hope, validation, solution, blame, hostility markers

Main outputs:
- `05_mental_health_steering/mental_health_steering_scores.csv`
- `05_mental_health_steering/dose_response_curves.csv`
- `05_mental_health_steering/dose_response_*.png`
- `05_mental_health_steering/condition_method_heatmap.png`
- `05_mental_health_steering/summary.md`

Why it exists:
The existing generation benchmark fights the assistant/safety axis. Mental health posts provide emotionally rich context with no instruction-following confound, enabling a cleaner test of whether appraisal steering causally changes model behavior.

## Stage 6: Synthesis

Main file:
- `pipeline/synthesis.py`

Purpose:
Collect the evidence from all prior stages into a single model-level and cross-model summary.

Inputs:
- outputs from stages 0 through 5

Core computation:
- aggregate metrics
- copy or organize key outputs
- produce final written summaries

Main outputs:
- `06_synthesis/synthesis_metrics.csv`
- `06_synthesis/SUMMARY.md`
- copied key figures and tables

Why it exists:
The earlier stages answer narrow questions. This stage tells the full experimental story.

## Shared Infrastructure Files

The stage files above depend heavily on several shared modules:

- `pipeline/config.py`: scientific defaults
- `pipeline/model_config.py`: model-specific runtime settings
- `pipeline/research_contracts.py`: methodological rules and provenance helpers
- `experiments/utils/training_utils.py`: model loading, extraction, and probe fitting engine
- `utils.py`: hidden-state extraction and intervention backend
- `pipeline/theory_analysis.py`: shared plotting helpers for theory-facing outputs

## How To Read Dependencies Across Stages

The main flow is:

1. Robustness chooses the training representation.
2. Emotion and appraisal probes create the main readout objects.
3. Circuit evidence chooses the site set.
4. Structure, phase 2, and steering interpret and perturb those sites.
5. Synthesis combines everything into the final report.

This dependency structure is intentional. Each later stage reuses earlier outputs rather than recomputing the scientific assumptions from scratch.

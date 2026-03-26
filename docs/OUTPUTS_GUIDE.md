# Outputs Guide

This guide explains what the main pipeline outputs mean and how to read them safely.

See also:
- `docs/PIPELINE_MAP.md` for which stage produces each output
- `docs/EXPERIMENTAL_SETUP.md` for the split and selection rules behind those outputs
- `docs/RUNBOOK.md` for a practical checklist after each stage
- `docs/GLOSSARY.md` for definitions of the metrics reported in these files
- `docs/GENERATION_BEHAVIOR_BENCHMARK.md` for the new actual-generation benchmark stage

## Output Root

Per-model outputs are written under:

- `pipeline/outputs/<model_id>/`

The main subfolders are:

- `01_probe_robustness`
- `01_probes` (including `appraisal_binary_ova_probes/`)
- `02_circuit`
- `03_appraisal_structure`
- `04_appraisal_in_circuit`
- `04_appraisal_theory`
- `05_steering`
- `05_generation_behavior`
- `05_baseline_probe_steering` (optional; see `docs/BASELINE_PROBE_STEERING_STUDY.md`)
- `06_synthesis`

## Stage 0 Outputs

### `best_variant_selection.json`

What it is:
The selected text representation from the robustness stage.

How to read it:
This file tells you which dataset variant the main emotion probe stage should use by default.

Why it matters:
The pipeline treats text representation as a chosen condition rather than an invisible preprocessing detail.

### `supported_emotions.json`

What it is:
The main supported emotion set after applying the rigorous split protocol and minimum-support thresholds.

How to read it:
If an emotion is absent here, it is not considered stable enough for the main benchmark under the current thresholds.

### `supported_emotion_counts.csv`

What it is:
Per-emotion counts across train, selection, and test after canonical splitting.

How to read it:
Use this file to understand whether poor probe performance might reflect weak support rather than weak signal.

### `probe_robustness_comparison.csv`

What it is:
The quick comparison of training text variants.

How to read it:
Higher selection metrics indicate a stronger candidate representation for the main probe training stage.

Common mistake:
Do not treat this file as the final probe benchmark. It is a model-selection artifact.

## Stage 1 Outputs

### `01_probes/binary_ova_probes/probe_summary.csv`

What it is:
The main site-by-site emotion probe summary.

How to read it:
Each row describes one emotion probe at one `(layer, loc, token)` site and includes held-out metrics.

Why it matters:
This is the main ranking source for later circuit selection.

Common mistake:
Do not average across everything immediately. The pipeline often needs per-emotion site quality, not only a global mean.

### `01_probes/binary_ova_probes/binary_ova_probes_*.pt`

What it is:
The serialized emotion probe bundle.

How to read it:
This contains the actual per-emotion classifier objects, scalers, and cached weight information used downstream.

Why it matters:
Later stages use these weights for logits, raw-space direction comparisons, and interventions.

### `01_probes/binary_ova_probes/probe_manifest.json`

What it is:
The provenance manifest for the emotion probe bundle.

How to read it:
Use it to confirm the model ID and configuration match the outputs you think you are reading.

### `01_probes/appraisal_regression_probes.pt`

What it is:
The serialized appraisal regression bundle.

How to read it:
This contains a separate scaler and Ridge regressor for each appraisal dimension at each layer and loc.

Why it matters:
Phase 2 and steering analyses use these regressors to build appraisal directions and comparisons.

### `01_probes/appraisal_probe_validation_detail.csv`

What it is:
The held-out selection metrics for each appraisal dimension and each site.

How to read it:
Lower `selection_rmse` is better. Higher `selection_r2` and `selection_corr` are better.

Common mistake:
Do not interpret a good appraisal score at one site as evidence that the same site is automatically part of the best emotion circuit.

## Stage 2 Outputs

### `02_circuit/circuit_evidence_classification.csv`

What it is:
The main comparison of single-site and multi-site circuit classifiers.

How to read it:
Look for how `single_best`, `topk_fusion`, and `topk_fusion_global` compare on the held-out test split.

Aggregation rule:
- `single_best` uses one site's emotion score vector directly.
- `topk_fusion` averages each emotion's own probe score across that emotion's selected circuit sites, then predicts the emotion with the highest averaged score.
- `topk_fusion_global` averages the full emotion score vectors across one shared global top-k site list, then predicts the highest-scoring emotion.

Why it matters:
This is the central evidence file for the claim that a circuit can outperform a single best site.

### `02_circuit/circuit_evidence_classification.png`

What it is:
The plot version of the circuit evidence comparison.

How to read it:
Use it as a quick visual summary, but prefer the CSV for exact interpretation.

Common mistake:
If every method looks identical, inspect the CSV and the selected-site JSON before trusting the plot.

### `02_circuit/circuit_top_k_selection.json`

What it is:
The record of the top-k sweep and the chosen circuit size.

How to read it:
This file should tell you:

- which `k` was chosen
- which metric drove the choice
- which `(layer, loc)` pairs were selected

Why it matters:
This is the provenance record that proves `k` was selected empirically rather than hard-coded.

### `02_circuit/circuits.json`

What it is:
A simplified reusable circuit definition.

How to read it:
Use this when you need a compact summary of the circuit without all of the sweep details.

### `02_circuit/circuit_sites.json`

What it is:
The exact selected `(layer, loc)` sites used for the circuit.

How to read it:
This is the most precise downstream definition of the selected circuit.

### `02_circuit/selection_hidden_states.pt` and `02_circuit/test_hidden_states.pt`

What they are:
Cached hidden-state tensors for the selection and test splits.

How to read them:
These are computational artifacts used by later stages to avoid redundant extraction.

Why they matter:
They support reproducibility and reduce unnecessary recomputation.

## Stage 3 Outputs

### `03_appraisal_structure/baseline_metrics.csv`

What it is:
Held-out baseline emotion classification metrics for the appraisal structure stage.

How to read it:
Use it as a descriptive benchmark for the emotional landscape, not as the main circuit-selection result.

### `03_appraisal_structure/cluster_emotion_mapping.csv`

What it is:
A summary of how emotions cluster under the stage's descriptive procedure.

How to read it:
This helps interpret relationships among emotions, especially similar and contrastive pairs.

### `03_appraisal_structure/appraisal_zscore_by_emotion.csv`

What it is:
Per-emotion appraisal profiles expressed as z-scores.

How to read it:
Positive and negative values indicate how an emotion differs from the training reference for each appraisal dimension.

### `03_appraisal_structure/appraisal_zscore_heatmap.png`

What it is:
A heatmap visualization of emotion-by-appraisal z-scores.

How to read it:
Use it to compare how emotions differ along appraisal dimensions, not as a direct causal analysis.

## Stage 3B Outputs

### `03_appraisal_structure/pca_eda/pca_explained_variance.csv`

What it is:
The explained-variance ratios for PCA components at each site.

How to read it:
Higher first-component variance means more of the site's total variance is captured by the first axis, not necessarily that the axis is the "emotion axis."

### `03_appraisal_structure/pca_eda/pca_site_summary.csv`

What it is:
A compact summary of PCA results by site.

How to read it:
Use it to compare where structure appears more concentrated or more diffuse across layers and locs.

### `03_appraisal_structure/pca_eda/pca/layer_<L>_loc_<LOC>.html`

What it is:
Interactive PCA plots with hover metadata.

How to read it:
Hover information can include scenario text, gold label, and appraisal values, which helps relate geometry back to the underlying examples.

Common mistake:
PCA separation is suggestive and useful for exploration, but it is not by itself proof of a causal mechanism.

### `03_appraisal_structure/pca_eda/eda/*.csv`

What they are:
EDA tables summarizing dataset support, source counts, lengths, and appraisal means.

How to read them:
Use them to sanity-check the benchmark before over-interpreting model-level results.

## Stage 4 Outputs

### `04_appraisal_in_circuit/geometry_circuit_layers.csv`

What it is:
A geometry summary comparing emotion and appraisal directions at selected circuit sites.

How to read it:
This file asks whether the directions associated with emotion and appraisal are aligned, opposed, or largely orthogonal at those sites.

### `04_appraisal_in_circuit/correlation_circuit_vs_default.csv`

What it is:
A comparison between circuit-site appraisal estimates and the default appraisal readout.

How to read it:
High correspondence suggests that the selected circuit preserves similar appraisal structure to the broader default appraisal representation.

### `04_appraisal_in_circuit/appraisal_ablation_summary.csv`

What it is:
The result of removing full appraisal signatures or individual appraisal dimensions from the selected circuit sites.

How to read it:
Larger changes indicate that the removed appraisal information mattered more for the circuit readout.

Common mistake:
Ablation effects should be read alongside the baseline circuit quality. A large change in a weak circuit is less informative than a large change in a strong one.

### `04_appraisal_in_circuit/SUMMARY.md`

What it is:
The human-readable phase-2 synthesis.

How to read it:
Use it as a narrative summary after checking the underlying CSVs and figures.

## Stage 4B Outputs: Appraisal Theory

### `04_appraisal_theory/onset_comparison.csv`

What it is:
Layer onset comparison between appraisal dimensions and emotions. Shows the first layer at which each probe exceeds a performance threshold.

How to read it:
If appraisal onsets are systematically earlier than emotion onsets, that supports the feedforward model from appraisal theory. Columns: `name`, `type` (emotion/appraisal), `onset_layer`, `peak_layer`, `peak_metric`.

### `04_appraisal_theory/reconstruction_by_layer_loc.csv`

What it is:
Appraisal-to-emotion reconstruction accuracy at each (layer, loc). Tests whether the 14-dimensional appraisal profile can predict the correct emotion label.

How to read it:
Compare `reconstruction_accuracy` against `direct_emotion_accuracy`. If reconstruction approaches direct readout, appraisal is sufficient for emotion. If reconstruction onset is earlier (above chance at a lower layer), appraisal is computed before emotion.

### `04_appraisal_theory/cross_layer_prediction.csv`

What it is:
Cross-layer prediction asymmetry. Tests whether appraisal at layer L predicts emotion at L+1 better than the reverse.

How to read it:
Compare `r2_test` for `appraisal_to_emotion` vs `emotion_to_appraisal` directions. A consistent asymmetry favoring appraisal-to-emotion supports the feedforward model.

### `04_appraisal_theory/appraisal_space_3d.html`

What it is:
Interactive 3D PCA biplot of emotions in appraisal space. Open in a browser to rotate, zoom, and inspect hover text.

How to read it:
Points are emotions, colored by valence group. Arrows show appraisal dimension loadings. Emotions that cluster together share similar appraisal profiles. PC1-3 explain ~83% of variance.

### `04_appraisal_theory/SUMMARY.md`

What it is:
Narrative summary of all appraisal theory analyses with embedded figure references.

## Stage 5 Outputs

### `05_steering/steering_benchmark.csv`

What it is:
Cache-based steering results.

How to read it:
This file summarizes whether adding internal emotion or appraisal directions shifts the readout in the intended direction.

### `05_steering/steering_curves.csv`

What it is:
Per-strength steering traces for the cache-based setting.

How to read it:
Use it to inspect dose-response style behavior rather than only final aggregate values.

### `05_steering/steering_benchmark_behavioral.csv`

What it is:
Behavioral forward-pass steering results.

How to read it:
This is a stronger form of evidence than cache-only steering because it affects live model computation.

### `05_steering/steering_benchmark_by_pair.csv`

What it is:
Steering results broken down by emotion pair.

How to read it:
This helps identify which pair transitions are easy, hard, contrastive, or similar.

### `05_steering/behavioral_appraisal_ablation.csv`

What it is:
Behavioral results when appraisal signatures are erased during live forward passes.

How to read it:
This file asks whether removing appraisal information changes the model's downstream emotional behavior.

## Stage 5B Outputs

### `05_generation_behavior/generation_behavior_outputs.csv`

What it is:
The raw prompt and generated response for each benchmark row under each intervention condition.

How to read it:
This is the main user-visible behavior artifact. Read it first if you want to know whether steering actually changed the assistant's text.

**Runtime readout (when enabled):** may include `runtime_rank1_emotion`, `appraisal_source_mode` (`csv` / `runtime` / `fallback`), `runtime_readout_layer`, `runtime_readout_loc`, `runtime_readout_emotion_mode` (e.g. `circuit_sigmoid_mean`, `single_site`, `circuit_linear_mean`), `runtime_readout_union_n_sites`, `ranked_top_k_json`, optional **`runtime_linear_circuit_rank1_emotion`** / **`runtime_linear_circuit_ranked_top_k_json`** (linear-mean aux when `RUNTIME_READOUT_LOG_LINEAR_CIRCUIT_AUX`), `runtime_skip_reason`. See **`docs/RUNTIME_READOUT.md`**, `docs/GENERATION_BEHAVIOR_BENCHMARK.md`, and `RUNTIME_READOUT_*` in `config.py`.

When **`RUNTIME_READOUT_LOG_RANK_JSON`** is True, `ranked_top_k_json` is a JSON **object** with keys such as `readout_mode`, **`score_kind`**, `ranked_top_k` (list of `{emotion, logit}` — scores in the active space; key name is legacy), optional `union_n_sites`, and optional **`all_emotions_scores_json`** (truncated) if **`RUNTIME_READOUT_LOG_FULL_EMOTION_SPECTRUM`** is True.

**Post-generation readout** (when **`GENERATION_BEHAVIOR_POSTGEN_READOUT`** is True in `config.py`): `postgen_runtime_rank1_emotion`, `postgen_ranked_top_k_json` (includes `"readout_phase": "post_generation"`), `postgen_readout_union_n_sites`, `postgen_latent_predicted_emotion`, **`postgen_latent_target_primary_score`** — from an **unsteered** forward on **`full_text`** at the last extracted token. See `docs/GENERATION_BEHAVIOR_BENCHMARK.md`.

### `05_generation_behavior/generation_behavior_latent_readouts.csv`

What it is:
The latent circuit readout paired with each generated response.

How to read it:
Use it to connect a generated behavior change back to the model's internal emotion/appraisal control state.

Carries the same runtime-readout provenance columns as `generation_behavior_outputs.csv` when present.

### `05_generation_behavior/generation_behavior_scores.csv`

What it is:
Lightweight rule-based behavior proxies computed over the generated text.

How to read it:
These are first-pass scaffolds for hostility, empathy, refusal, blame, de-escalation, and unsafe-compliance proxies. They are useful for comparison, but they are not a final judge-model evaluation.

### `05_generation_behavior/generation_behavior_judge_scores.csv`

What it is:
An LLM-judge scoring table over the generated responses.

How to read it:
This is the stronger first-pass behavior evaluation artifact. It scores assistant-likeness, hostility, blame, empathy, de-escalation, unsafe compliance, refusal quality, and persona stability.

### `05_generation_behavior/generation_behavior_judge_summary_by_condition.csv`

What it is:
Aggregate judge scores grouped by intervention type.

How to read it:
Use this to compare whether baseline, appraisal steering, emotion steering, or combined steering produces better generated behavior under the benchmark.

### `05_generation_behavior/generation_behavior_summary_by_condition.csv`

What it is:
Aggregate behavior scores by intervention type.

How to read it:
Use this file to compare baseline vs appraisal steering vs emotion steering vs combined steering at a glance.

### `05_generation_behavior/generation_behavior_summary_by_family.csv`

What it is:
Aggregate behavior scores broken down by benchmark family.

How to read it:
This shows whether an intervention helps in one kind of interaction, such as provocation, but hurts in another, such as harmful-request refusal.

### `05_mental_health_steering/mental_health_steering_scores.csv`

What it is:
Per (post, framing, method, alpha) rows with generated reply text, heuristic scores, and latent readouts.

How to read it:
When **`RUNTIME_READOUT_USE_FOR_MENTAL_HEALTH`** is on, includes the same runtime-readout columns as generation behavior (including optional linear-circuit aux columns), plus `appraisal_full_source`, `mh_skip_reason`, and adaptive-target columns when enabled. With default **`RUNTIME_READOUT_EMOTION_MODE = circuit_sigmoid_mean`**, **`runtime_rank1_emotion`** and **`latent_predicted_emotion`** both use **mean-σ circuit** fusion and should usually agree; differences can appear if modes diverge or token alignment differs.

## Optional: `05_baseline_probe_steering`

Produced only if you run `python -m pipeline.baseline_probe_steering_study`. See `docs/BASELINE_PROBE_STEERING_STUDY.md` for columns and interpretation (`baseline_probe_readouts.csv`, `baseline_probe_steering_runs.csv`, `dim_selection_policy.json`).

## Stage 6 Outputs

### `06_synthesis/synthesis_metrics.csv`

What it is:
The main aggregated metrics summary for the model.

How to read it:
This file is useful for comparing models or high-level settings after the lower-level outputs have been validated.

**Mental health columns (when `05_mental_health_steering/dose_response_curves.csv` exists):**

- `mh_synthesis_reference_alpha` — α used for reference-tone columns: nearest value in the curves to `MENTAL_HEALTH_REPORT_ALPHA` in `config.py` (same rule as the MH benchmark heatmap).
- `mh_<method>_mean_tone_ref` — mean `mean_emotional_tone` from the dose table at `mh_synthesis_reference_alpha` (methods include `appraisal_targeted`, `appraisal_full`, `emotion_steer`, `combined`, `appraisal_elicitation`).
- `mh_<method>_best_alpha_mean_tone` / `mh_<method>_mean_tone_best` — α among **α > 0** that maximizes the aggregated mean emotional-tone proxy (for `appraisal_elicitation`, **minimizes** it — stronger negative-affect proxy on this heuristic).
- `mh_<method>_tone_alpha5` — **legacy alias** for `mh_<method>_mean_tone_ref` (name kept for older notebooks; it is **not** fixed at α = 5).

### `06_synthesis/SUMMARY.md`

What it is:
A written rollup of the evidence from the whole pipeline.

How to read it:
Use it as the final interpretation layer, not as a replacement for the stage-specific evidence files.

## Quick Interpretation Rules

1. Use `selection` outputs for model selection decisions and `test` outputs for final claims.
2. Prefer CSV files over plots when numbers and provenance matter.
3. Read descriptive outputs, geometry outputs, and intervention outputs together.
4. Treat any unusually flat or identical result pattern as a debugging cue rather than a scientific finding.
5. Always check the manifest and model ID when comparing outputs across runs.

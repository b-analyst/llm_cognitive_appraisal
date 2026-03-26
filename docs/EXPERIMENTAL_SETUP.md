# Experimental Setup

This document explains how the pipeline is configured and why the current setup was chosen.

See also:
- `docs/RESEARCH_QUESTIONS.md` for the scientific questions
- `docs/PIPELINE_MAP.md` for stage-by-stage flow
- `docs/OUTPUTS_GUIDE.md` for output interpretation
- `docs/GLOSSARY.md` for metric and method definitions
- `docs/RUNBOOK.md` for execution order and validation checkpoints
- `docs/GENERATION_BEHAVIOR_BENCHMARK.md` for the actual-generation extension of the steering work

## Data Source And Canonical Dataset

The default combined dataset is read from `pipeline/config.py` via `COMBINED_CSV`. The canonicalization logic lives in `pipeline/research_contracts.py`.

The canonical dataset contract is:

- `emotion` exists and is normalized to lowercase.
- `situation` is the base scenario text.
- `hidden_emo_text` may exist as a prompted version of the scenario text.
- `scenario_id` is added so prompt variants from the same underlying situation can be grouped together during splitting.

Why this matters:
Different prompted versions of the same scenario should not be allowed to leak across train, selection, and test.

## Text Representation Policy

The pipeline supports different text variants for training and evaluation. The core policy is configured in `pipeline/config.py`:

- `PROMPT_TEXT_POLICY = "generate_if_missing"`
- prompt indices for robustness variants come from `PROMPT_INDICES_FOR_VARIANTS`

Stage `01_probe_robustness` compares multiple text variants and saves the best one in `best_variant_selection.json`. The main emotion probe stage then uses that winning variant by default.

Why this matters:
Text formulation can materially change probe quality. The pipeline therefore treats text representation as a selectable experimental condition rather than a hidden assumption.

## Split Protocol

The rigorous split logic lives in `split_combined_dataset()` inside `pipeline/research_contracts.py`.

The current defaults in `pipeline/config.py` are:

- `SELECTION_SPLIT = 0.15`
- `FINAL_TEST_SPLIT = 0.15`
- `RANDOM_STATE = 42`

The split is done at the scenario-group level, not row-by-row. That means prompt variants derived from the same scenario stay in the same partition.

The partitions have distinct jobs:

- Train: fit probes and regressors.
- Selection: choose hyperparameters, choose the best text variant, and choose circuit size `k`.
- Test: report final held-out performance after those choices are fixed.

Why this matters:
The pipeline tries to separate model fitting, model selection, and final reporting.

## Supported Emotion Set

The supported emotion logic lives in `supported_emotion_stats()` in `pipeline/research_contracts.py`.

The current thresholds in `pipeline/config.py` are:

- `MIN_SUPPORTED_EMOTION_TRAIN_COUNT = 50`
- `MIN_SUPPORTED_EMOTION_SELECTION_COUNT = 20`

An emotion is included in the main supported set only if it has enough examples in both train and selection after the rigorous split.

Why this matters:
Sparse emotions can make ROC-AUC unstable, make one-vs-all datasets extremely noisy, and turn circuit selection into a comparison driven by data scarcity rather than signal quality.

## Emotion Probe Training

Emotion probes are trained in `pipeline/train_probes.py` using `OvaLogisticRegressionTrainer` from `experiments/utils/training_utils.py`.

The current design is:

- binary one-vs-all logistic regression
- trained for each supported emotion
- trained across all configured extraction layers
- trained across all configured extraction locs
- trained across all configured extraction tokens
- regularization strength `C` chosen from `PROBE_C_GRID`

Important defaults in `pipeline/config.py`:

- `PROBE_C_GRID = [0.01, 0.1, 1.0, 10.0]`
- `EXTRACTION_TOKENS = [-1]`

Important model-specific settings come from `pipeline/model_config.py`:

- `get_extraction_layers(model_id)`
- `get_extraction_locs(model_id)`
- `get_extraction_batch_size(model_id)`
- `get_default_probe_n_jobs()`

What the loc values mean:
All current models use locs 3, 6, and 7. These map to specific internal positions within each transformer block:
- loc 3: output of the self-attention block
- loc 6: output of the feedforward (MLP) block
- loc 7: full layer output after both blocks

For a plain-English walkthrough with diagrams, see `docs/LOC_REFERENCE.md`.

How fitting works:

1. Stage `01_probe_robustness` decides the default text representation.
2. `train_probes.py` loads the supported emotion set and the selected text variant.
3. Hidden states are extracted for all configured sites.
4. For each emotion and each site, a binary logistic regression probe is fitted.
5. `C` is selected on the selection split only.
6. The resulting metrics are written to `probe_summary.csv`.

Why this design was chosen:

- Binary OVA preserves per-emotion interpretability and extensibility.
- Logistic regression is simple, auditable, and easy to compare across many sites.
- Selection-only hyperparameter choice helps reduce optimistic reporting.

## Appraisal Probe Training

Appraisal probes are trained in `pipeline/train_appraisal_probes.py`.

The current design is:

- one separate Ridge regressor per appraisal dimension
- one separate regressor per layer
- one separate regressor per loc
- uses the first configured extraction token key
- fits on the train split and evaluates on the selection split

The current default regularization is:

- `RIDGE_ALPHA = 5.0`

Additional training behavior:

- the training partition can be emotion-balanced before regression fitting
- rows with missing targets for a given appraisal dimension are excluded for that dimension
- each site gets its own `StandardScaler` and its own Ridge coefficients

Why this design was chosen:

- Appraisal targets are continuous, so regression is a better fit than classification.
- Ridge is a stable linear regressor for high-dimensional hidden-state features.
- Site-specific regressors allow later analyses to ask where each appraisal dimension is most readable.

By default a single extraction token is used; `train_appraisal_probes.run_train_appraisal_probes(..., extraction_tokens_override=...)` can fit one Ridge model per **(dimension, layer, loc, token)** (used by the grid ablation runner). The validation CSV includes a `token` column (for the default run this is the configured integer token, e.g. `-1`).

## Probe grid ablation (scope A)

Optional **probe-only** experiment: train emotion and appraisal probes on a wider **loc × token** grid without changing `EXTRACTION_LOCS` / `EXTRACTION_TOKENS` for the main pipeline or writing into canonical `01_probes/`.

- **Config:** `PROBE_GRID_ABLATION_LOCS`, `PROBE_GRID_ABLATION_TOKENS`, `PROBE_GRID_ABLATION_OUTPUT_SUBDIR`, and `get_probe_grid_ablation_dir(model_id)` in `pipeline/config.py`.
- **Output root:** `outputs/<model_id>/01_probes_grid_ablation/` (sibling to `01_probes/`). Emotion artifacts go to `.../binary_ova_probes/`; appraisal `.pt` / CSV / manifest sit in the ablation root.
- **Runner:** from repo root, `python -m pipeline.run_probe_grid_ablation --model_id <id>`. Useful flags: `--datasets_dir`, `--locs 1,2,3`, `--tokens -1,0,mid`, `--skip_appraisal`, `--max_samples N` (appraisal dev cap), `--emotions anger,joy`.
- **Extractor token `mid`:** not a fixed index across examples. `utils.extract_hidden_states` gathers the hidden state at the masked middle `(first_real + last_real) // 2` from `attention_mask` (see `GRID_ABLATION_README.md` under the ablation folder after analysis).
- **Token `0` vs `-1`:** both are indices on the **padded** batch; with typical right-padding, `-1` is often the last column (commonly a pad token unless you mask). Interpret ablation metrics as **readout sensitivity**, not ground-truth localization.
- **Analysis:** `python -m pipeline.analyze_probe_grid_ablation --model_id <id>` builds `grid_ablation_summary.csv`, heatmaps (requires `matplotlib`), and `GRID_ABLATION_README.md`. Optional `--baseline_probe_summary` points at canonical `01_probes/binary_ova_probes/probe_summary.csv` for emotion delta columns.

**Explicit non-goals (scope A):** no `circuit_evidence` / `phase1_circuits` rerun and no steering changes until a grid choice is promoted into production config.

What the current setup does not do:

- It does not train a pooled cross-layer appraisal model.
- It does not learn one shared regressor across all sites.

## Circuit Selection Protocol

Circuit selection is implemented in `pipeline/circuit_evidence.py`.

The goal is to compare:

- `single_best`
- `topk_fusion`
- `topk_fusion_global`

The current protocol is:

1. Load the held-out selection and test splits.
2. Extract or load cached hidden states for those splits.
3. Rank candidate `(layer, loc)` pairs using probe results.
4. Sweep `k` from 1 up to `CIRCUIT_TOP_K_MAX`.
5. Choose the best `k` on the selection split.
6. Report the chosen method on the held-out test split.

How the final circuit decision is made:

- `single_best`: use one `(layer, loc)` site, compute all emotion probe scores at that site, and predict the highest-scoring emotion.
- `topk_fusion`: for each emotion separately, collect that emotion's probe score from each selected circuit site for that emotion, average those scores across sites, and then predict the highest-scoring emotion after all per-emotion averages are assembled.
- `topk_fusion_global`: use one shared global top-k site list for all emotions, average the full emotion-score vectors across those sites, and then predict the highest-scoring emotion.

Why this matters:

- The circuit is not voting by majority label.
- It is averaging probe outputs across selected sites and only then making a final argmax decision.
- Per-emotion top-k allows different emotions to rely on different site sets before the final multiclass decision is made.

Important default:

- `CIRCUIT_TOP_K_MAX = 16`

Why this design was chosen:

- The pipeline avoids hard-coding a single circuit size.
- Selection performance determines the chosen circuit size.
- Test performance is reserved for final reporting only.

## Optional adaptive contrastive appraisal target (generation + mental health)

`pipeline/config.py` defines **`ADAPTIVE_APPRAISAL_TARGET_ENABLED`** (default **`True`**). When enabled, **`pipeline/generation_behavior_benchmark.py`** and **`pipeline/mental_health_steering_benchmark.py`** can choose the **steering target emotion** as the row in **`appraisal_zscore_by_emotion.csv`** whose **z-vector** (over appraisal columns shared with **`COMMON_APPRAISAL`**) has **largest L2 distance** from the **source** row (runtime probe rank-1 when runtime readout is on; otherwise CSV/condition fallbacks). Candidates are restricted to emotions present in probe summaries and the z-score index; the static CSV/config target remains **fallback** if selection fails. This only changes **which table rows** feed the existing z-difference-weighted ridge composite in **`steering_benchmark._compute_appraisal_steering_vector`**; it does **not** use live ridge predictions on the prompt as the source appraisal signature (that would be a separate experiment). Emotion names here are **probe-defined research constructs**, not clinical diagnoses.

**Runtime emotion ranking (generation + mental health):** **`RUNTIME_READOUT_EMOTION_MODE`** (default **`circuit_sigmoid_mean`**) controls how rank-1 is computed when runtime readout is on — see **`docs/RUNTIME_READOUT.md`** for σ-mean vs linear-mean circuit fusion and optional **`RUNTIME_READOUT_LOG_LINEAR_CIRCUIT_AUX`**. Optional **`RUNTIME_READOUT_LOG_FULL_EMOTION_SPECTRUM`** adds a truncated JSON dump of all per-emotion scores when combined with **`RUNTIME_READOUT_LOG_RANK_JSON`** (`all_emotions_scores_json` on the primary payload).

## Hidden-State Extraction And Caching

Hidden-state extraction is performed through `utils.py` and `experiments/utils/training_utils.py`.

Important design points:

- the default backend uses stock Hugging Face `AutoModelForCausalLM`
- PyTorch hooks are used for extraction and steering
- extraction batch size is model-specific
- repeated text extraction is deduplicated in the training backend
- cached outputs are saved as CPU float32 for downstream sklearn compatibility

Why this matters:

- Hidden-state extraction is the main expensive step.
- Deduplication makes proper hyperparameter tuning feasible.
- Float32 normalization avoids downstream dtype incompatibilities.

## Provenance And Manifest Checks

The provenance helpers live in `pipeline/research_contracts.py`.

Important helpers:

- `dataset_fingerprint()`
- `make_split_manifest()`
- `manifest_matches()`
- `ensure_manifest_model_match()`

These are used to make sure cached activations and saved probe bundles match:

- the expected model
- the expected dataset fingerprint
- the expected split protocol
- the expected split name

Why this matters:
Without provenance checks, cached artifacts can silently mix incompatible runs.

## Pair Taxonomy

Theory-facing analyses distinguish:

- contrastive emotion pairs from `CONTRASTIVE_EMOTION_PAIRS`
- similar emotion pairs from `SIMILAR_EMOTION_PAIRS`

These categories are used in steering plots and phase-2 comparisons.

Why this matters:
The theory claim is not just that all emotions differ. It is also that similar and contrastive relationships should show meaningful structure.

## Metrics By Stage

Emotion probe and circuit stages commonly use:

- accuracy
- balanced accuracy
- ROC-AUC
- PR-AUC
- kappa

Appraisal regression stages commonly use:

- RMSE
- R-squared
- correlation

The exact meaning of each metric is explained in `docs/GLOSSARY.md`.

## What Is Chosen On Selection Versus Reported On Test

Selection is used for:

- best text variant in `01_probe_robustness`
- logistic regression `C` in `01_probes`
- circuit size `k` in `02_circuit`

Test is used for:

- final circuit evidence reporting
- downstream descriptive and intervention analyses that are intended to be held-out

This is a core principle of the current setup.

## Where To Look In Code

- `pipeline/config.py`
- `pipeline/model_config.py`
- `pipeline/research_contracts.py`
- `pipeline/probe_training_robustness.py`
- `pipeline/train_probes.py`
- `pipeline/train_appraisal_probes.py`
- `pipeline/circuit_evidence.py`
- `experiments/utils/training_utils.py`
- `utils.py`

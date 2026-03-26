# Appraisal Theory Analysis

## Scientific Motivation

Cognitive appraisal theory (Scherer, Lazarus, Smith & Ellsworth) holds that emotions
are not atomic primitives but emerge from a multi-dimensional appraisal process.
A person evaluates a situation along dimensions such as pleasantness, control,
certainty, goal relevance, and responsibility attribution. The resulting appraisal
profile converges into an emotion category (anger, fear, joy, etc.).

If a language model mirrors this structure internally, we would expect:

1. **Temporal precedence:** Appraisal information should appear in earlier layers
   (or earlier within-layer locations) than emotion-category information.
2. **Sufficiency:** The appraisal profile at a given layer should be enough to
   reconstruct the correct emotion label.
3. **Directional flow:** Appraisal representations at layer L should predict
   emotion representations at layer L+1 better than the reverse.
4. **Structural specificity:** Different emotions should occupy different circuits,
   and those circuits should reflect their appraisal composition.

This pipeline stage tests all four predictions.

## Evidence Chain

```
Layer onset (B) + Location ordering (C)
  → Do appraisals appear before emotions?

Reconstruction (D)
  → Can appraisals reconstruct emotions?

Cross-layer prediction (F)
  → Is there directional appraisal→emotion flow?

Circuit structure (A) + Direction comparison (E)
  → Do circuits and probe directions reflect appraisal structure?
```

## Pipeline Stage

- **Module:** `pipeline/appraisal_theory.py`
- **Entry point:** `run_appraisal_theory(model_id)`
- **Depends on:** `01_probes/`, `02_circuit/`, binary appraisal probes (optional for E)
- **Outputs:** `outputs/<model_id>/04_appraisal_theory/`
- **Pipeline position:** After `phase2_summary`, before `steering_benchmark`

**Related optional study:** Emotion–appraisal structure from this stage (and `03_appraisal_structure` z-scores) can feed a **frozen top-m dimension policy** for an optional follow-on run that probes baselines and steers in appraisal subspace — see `docs/BASELINE_PROBE_STEERING_STUDY.md` and `pipeline/baseline_probe_steering_study.py`.

## Analyses

### A: Circuit Structure Characterization

**Question:** How are emotion circuits structured spatially? Do different emotions
use different transformer components (attention vs MLP vs residual)?

**Method:** From `circuit_top_k_selection.json`, compute:
- Location distribution per emotion (fraction of sites at loc 3/6/7)
- Location entropy (higher = more spread across components)
- Jaccard overlap between all emotion circuit pairs

**Outputs:**
- `circuit_structure_summary.csv`
- `circuit_overlap_matrix.csv`
- `circuit_location_distribution.png` (stacked bar chart)
- `circuit_overlap_heatmap.png` (clustered similarity matrix)

**Interpretation:** Emotions with high location entropy (e.g. fear) use multiple
transformer components. Emotions with low entropy (e.g. anger at mostly loc 3) read
primarily from attention outputs. High inter-emotion Jaccard means circuits share
sites; low Jaccard means distinct processing pathways.

### B: Layer Onset Comparison

**Question:** Does appraisal information become decodable at earlier layers than
emotion information?

**Method:** For each emotion and appraisal dimension, find the earliest layer
where probe performance exceeds a threshold. Compare the distributions.

**Thresholds (configurable in `config.py`):**
- Emotion: `test_roc_auc >= 0.60`
- Appraisal: `selection_corr >= 0.25`

**Outputs:**
- `onset_comparison.csv`
- `onset_comparison.png` (performance curves + onset box plot)

**Interpretation:** If appraisal onset layers are systematically earlier than
emotion onset layers, that supports the feedforward model from appraisal theory.
If they appear simultaneously, appraisal and emotion may be computed in parallel.

### C: Within-Layer Location Ordering

**Question:** Within the same layer, does appraisal peak at attention output
(loc 3) while emotion peaks at residual (loc 7)?

**Method:** For each layer, find the best loc for mean appraisal corr and mean
emotion ROC-AUC.

**Outputs:**
- `location_ordering_by_layer.csv`
- `location_ordering_heatmaps.png`

**Interpretation:** Location 3 = after self-attention, 6 = after MLP, 7 = after
residual connection (full layer output). If appraisals consistently peak at earlier
locations than emotions, that implies within-layer sequential processing:
attention computes appraisals, then MLP/residual integrates them into emotion.

### D: Appraisal-to-Emotion Reconstruction

**Question:** Is the 14-dimensional appraisal profile at a given layer sufficient
to predict the correct emotion label?

**Method (split-safe):**
1. At each (layer, loc), run 14 Ridge appraisal probes on hidden states to get
   a 14-dim prediction vector per sample.
2. On the selection split, train a LogisticRegression: emotion ~ appraisal vector.
3. On the test split, measure reconstruction accuracy.
4. Compare against direct emotion probe accuracy at the same site.

**Outputs:**
- `reconstruction_by_layer_loc.csv`
- `reconstruction_curves.png` (the key figure)

**Interpretation:** If reconstruction accuracy approaches direct-readout accuracy,
appraisal information is sufficient for emotion classification. If reconstruction
onset is earlier (achieves above-chance at a lower layer), appraisal is computed
before emotion emerges. The gap between reconstruction and direct readout measures
how much emotion information goes beyond appraisal.

### E: Ridge vs Binary Probe Direction Comparison

**Question:** Do Ridge regression and binary OVA logistic probes agree on the
direction that encodes each appraisal dimension?

**Method:** For each dimension at each (layer, loc), compute cosine similarity
between the Ridge weight vector and the binary logistic weight vector (both
mapped to raw hidden-state space via inverse scaler transform).

**Outputs:**
- `ridge_vs_binary_cosine.csv`
- `ridge_vs_binary_comparison.png`

**Interpretation:** High cosine similarity (> 0.7) means both probe types find
the same direction, validating each other. Low similarity suggests the linear
separating boundary (binary) and the regression gradient (Ridge) capture different
aspects of the representation. For steering, binary directions may produce sharper
interventions since they represent a categorical boundary.

### F: Cross-Layer Appraisal-to-Emotion Prediction

**Question:** Does appraisal at layer L predict emotion at layer L+1 better than
emotion at L predicts appraisal at L+1?

**Method:**
1. At each layer L: compute 14-dim appraisal vector and emotion logit vector.
2. Train Ridge: emotion(L+1) ~ appraisal(L), measure R-squared on test.
3. Train Ridge: appraisal(L+1) ~ emotion(L), measure R-squared on test.
4. Compare asymmetry across layers.

**Outputs:**
- `cross_layer_prediction.csv`
- `cross_layer_asymmetry.png`

**Interpretation:** If appraisal-to-emotion R-squared consistently exceeds
emotion-to-appraisal R-squared across layers, that is evidence for a feedforward
processing model consistent with appraisal theory. The reverse would suggest
emotion representations drive appraisal computations (top-down model).

## Binary Appraisal Probes

**Module:** `pipeline/train_appraisal_binary_probes.py`

Each appraisal dimension is binarized at the train-set median. Logistic OVA
probes are trained per dimension/layer/loc. This produces:

- `01_probes/appraisal_binary_ova_probes/appraisal_binary_summary.csv`
- `01_probes/appraisal_binary_ova_probes/appraisal_binary_ova_probes_<model>.pt`

Binary probes serve two purposes:
1. Direction comparison (Analysis E) against Ridge probes
2. Potential sharper intervention directions for future steering experiments

## How to Run

Single model:
```
python -m pipeline.train_appraisal_binary_probes --model_id Llama3.2_1B
python -m pipeline.appraisal_theory --model_id Llama3.2_1B
```

Or via the full pipeline runner (stages are included automatically).

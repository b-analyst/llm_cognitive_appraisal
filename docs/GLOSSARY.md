# Glossary

This glossary defines the main technical concepts used in the pipeline and ties them to the design choices made in the codebase.

See also:
- `docs/EXPERIMENTAL_SETUP.md` for how these concepts are instantiated in the current pipeline
- `docs/RESEARCH_QUESTIONS.md` for why they matter scientifically
- `docs/OUTPUTS_GUIDE.md` for where the resulting metrics and artifacts appear

## Probing

What it is:
Probing is the practice of training a simple model to predict some property from a model's hidden states.

Why it matters here:
Probing asks whether emotion and appraisal information are linearly decodable from internal activations.

What this repository uses:
- binary logistic-regression probes for emotion
- Ridge-regression probes for appraisal

Why this choice was made:
Simple linear probes are easy to audit, easy to compare across layers and locs, and less likely to hide complicated modeling assumptions.

Important limitation:
Probing shows that information is readable. It does not by itself prove that the model causally uses that information.

## Linear Probe

What it is:
A linear probe predicts a target from hidden states using a weighted sum plus a bias term.

Why it matters here:
The repository wants site-by-site comparability across a large number of internal locations. Linear probes make that comparison straightforward.

Why this choice was made:
If a very flexible nonlinear model were used, it could become harder to tell whether differences came from the hidden states or from the probe's own complexity.

## Hidden State

What it is:
A hidden state is the internal activation vector produced by the model at a given layer, location, and token position.

Why it matters here:
Hidden states are the basic object used for probing, circuit construction, geometry analysis, and intervention.

What this repository uses:
Hidden states are extracted across a configured grid of layers and locs, with token positions specified by `EXTRACTION_TOKENS`.

## Layer

What it is:
A layer is one transformer block depth in the model.

Why it matters here:
Different layers can carry different kinds of information. Some layers may be better for emotion readout, while others may be better for appraisal readout.

What this repository uses:
The layer grid is model-specific through `get_extraction_layers(model_id)`.

Why this choice was made:
The repository does not assume one universally best layer across all models.

## Location (`loc`)

What it is:
`loc` identifies a specific extraction point within a transformer block. Each transformer layer runs several subcomputations in sequence, and `loc` says which step to intercept.

Why it matters here:
Information can differ depending on whether it is measured before, inside, or after different internal subcomputations.

What this repository uses:
The allowed locs are model-specific through `get_extraction_locs(model_id)`. All current models use locs 3, 6, and 7.

What those three locs mean:
- **loc 3**: after the self-attention block
- **loc 6**: after the feedforward (MLP) block
- **loc 7**: after the full layer (attention + MLP, residual stream output)

Why this choice was made:
Circuit and probe quality often depend on where inside the block the state is sampled. Using all three lets the analysis distinguish whether emotion or appraisal signal comes primarily from the attention step, the feedforward step, or the integrated layer output.

For a full plain-English explanation with diagrams, see `docs/LOC_REFERENCE.md`.

## Token Position

What it is:
The token position tells the extraction code which token's hidden state to read.

Why it matters here:
The emotional summary signal may be stronger at some token positions than others.

What this repository uses:
The default extraction token is `-1`, which usually means the last token.

Why this choice was made:
The final token is often the most useful summary position for autoregressive models in this kind of analysis.

## Probe

What it is:
A probe is the fitted predictive model used to read out information from hidden states.

Why it matters here:
The repository uses probes both as measurement tools and as sources of directions for later geometry and intervention analyses.

What this repository uses:
- emotion probe bundles from `train_probes.py`
- appraisal regression bundles from `train_appraisal_probes.py`

## Logistic Regression

What it is:
Logistic regression is a linear classifier that predicts class probability from input features.

Why it matters here:
Emotion labels are discrete categories, so classification is the natural form of readout.

What this repository uses:
Binary one-vs-all logistic regression for each emotion at each site.

Why this choice was made:
It is simple, interpretable, and stable enough to sweep across many site combinations.

## Ridge Regression

What it is:
Ridge regression is linear regression with L2 regularization.

Why it matters here:
Appraisal targets are continuous and hidden-state features are high-dimensional and often correlated.

What this repository uses:
One Ridge regressor per appraisal dimension, per layer, and per loc.

Why this choice was made:
Ridge shrinks unstable coefficients and is a strong baseline for high-dimensional continuous prediction.

Why this is a better fit than classification here:
Appraisal dimensions are scalar-valued targets such as certainty or urgency, not class labels.

## L2 Regularization

What it is:
L2 regularization penalizes large weight magnitudes.

Why it matters here:
It helps prevent probe weights from becoming unstable when features are numerous and correlated.

What this repository uses:
- logistic regression regularization controlled by `C`
- Ridge regularization controlled by `alpha`

## Binary One-Vs-All (OVA)

What it is:
OVA trains one binary classifier per emotion. Each classifier asks whether the example belongs to that emotion versus all other emotions.

Why it matters here:
It gives a per-emotion view of where signal lives and makes the setup easier to extend if more emotions are added later.

What this repository uses:
Emotion probes are trained in a binary OVA setup across the full configured site grid.

Why this choice was made:
- it supports per-emotion circuit analysis
- it keeps the benchmark extensible
- it better accommodates the idea that emotional structure may not always be cleanly represented as one rigid mutually exclusive softmax space

Design implication:
Even when a downstream stage picks one winning emotion label, the underlying measurement system is still built from per-emotion binary readouts.

## Multiclass Versus Multilabel Thinking

What it is:
Multiclass prediction assumes exactly one correct class. Multilabel thinking allows overlapping or partly co-occurring states.

Why it matters here:
The benchmark labels are usually single labels, but the conceptual modeling of emotional states is more flexible than a strict one-softmax worldview.

What this repository uses:
The pipeline trains binary OVA probes so it can remain extensible and analyze each emotion separately, even when some reporting stages convert scores into one winning label.

Why this choice was made:
It gives cleaner per-emotion diagnostics and is more future-proof if the benchmark grows toward richer emotion states.

## Hyperparameter

What it is:
A hyperparameter is a setting chosen by the experimenter rather than learned directly by gradient descent inside the probe.

Why it matters here:
Different hyperparameter choices can materially change probe stability and circuit selection.

What this repository uses:
- logistic-regression `C`
- Ridge `alpha`
- extraction layers
- extraction locs
- extraction tokens
- batch sizes
- support thresholds
- maximum circuit size for the top-k sweep

## Hyperparameter Sweep

What it is:
A sweep evaluates multiple candidate hyperparameter values and chooses among them using a designated selection criterion.

Why it matters here:
This is the key mechanism that keeps choices like probe strength and circuit size from being hidden hard-coded assumptions.

What this repository uses:
- a `C` sweep for emotion probes
- a `k` sweep for circuit size

Why this choice was made:
Selection-based sweeps are more rigorous than picking one value in advance and hoping it generalizes.

## `C` In Logistic Regression

What it is:
`C` is the inverse strength of regularization in logistic regression.

How to interpret it:
- smaller `C` means stronger regularization
- larger `C` means weaker regularization

What this repository uses:
`PROBE_C_GRID = [0.01, 0.1, 1.0, 10.0]`

Why this choice was made:
It gives a small but meaningful range from strongly regularized to lightly regularized linear probes.

## `alpha` In Ridge Regression

What it is:
`alpha` is the regularization strength in Ridge regression.

How to interpret it:
- larger `alpha` means stronger coefficient shrinkage
- smaller `alpha` means less shrinkage

What this repository uses:
`RIDGE_ALPHA = 5.0` in the appraisal training stage.

Why this choice was made:
It provides a stable default for high-dimensional regression without making the appraisal stage too brittle.

## Train Split

What it is:
The train split is the partition used to fit model parameters.

What this repository uses it for:
- fitting emotion probes
- fitting appraisal regressors

## Selection Split

What it is:
The selection split is the held-out partition used to choose among candidate settings.

What this repository uses it for:
- selecting the best text variant
- selecting logistic-regression `C`
- selecting the circuit size `k`
- validating appraisal regressors

Why it matters here:
This split keeps model-selection decisions separate from the final test report.

## Test Split

What it is:
The test split is the final held-out partition used for reporting.

What this repository uses it for:
- final circuit evidence
- downstream descriptive and intervention analyses intended to stay held out

Why it matters here:
The pipeline tries to preserve a cleaner final evaluation set by not using it to choose settings.

## Leakage

What it is:
Leakage happens when information from evaluation examples influences training or model selection in a way that inflates performance.

Why it matters here:
The combined dataset can contain multiple text realizations of the same underlying scenario.

What this repository uses:
`scenario_id` grouping and scenario-level splitting in `split_combined_dataset()`.

Why this choice was made:
Prompt variants of the same scenario should not land in different splits.

## Scenario-Level Split

What it is:
A split where all rows from the same underlying scenario are kept together.

Why it matters here:
It is the main defense against prompt leakage in this repository.

## Supported Emotion Set

What it is:
The subset of emotions that have enough support after rigorous splitting to be included in the main benchmark.

Why it matters here:
Rare emotions can produce unstable metrics and misleading comparisons.

What this repository uses:
Minimum counts in both train and selection.

Why this choice was made:
The pipeline prefers a smaller but stabler benchmark over an inflated benchmark with very weak support.

## Circuit

What it is:
In this repository, a circuit is a selected set of internal `(layer, loc)` sites whose combined signals support emotion readout.

Why it matters here:
The pipeline is trying to identify structured internal site sets rather than merely one best layer.

What this repository uses:
Selected sites from `circuit_evidence.py`, later saved in `circuits.json` and `circuit_sites.json`.

Why this choice was made:
Earlier experiments suggested that combining multiple sites often gives better performance than a single-site readout.

## Top-k Circuit Selection

What it is:
Choosing how many top-ranked sites to include in the circuit.

Why it matters here:
Too few sites may miss useful signal. Too many sites may dilute it.

What this repository uses:
A sweep of `k` values up to `CIRCUIT_TOP_K_MAX`, with selection performance choosing the winner.

Why this choice was made:
The pipeline avoids fixing one global circuit size without evidence.

## Global Top-k Versus Per-Emotion Top-k

What it is:
Global top-k uses one shared site list for all emotions. Per-emotion top-k allows different emotions to have different best site sets.

Why it matters here:
Different emotions may rely on different internal sites.

What this repository uses:
Both are evaluated in the circuit evidence stage.

Why this choice was made:
It lets the benchmark compare a simpler shared-circuit hypothesis against a more emotion-specific circuit hypothesis.

## Accuracy

What it is:
The fraction of predictions that are correct.

Why it matters here:
It is intuitive and easy to read.

Limitation:
Accuracy can be misleading when classes are imbalanced.

## Balanced Accuracy

What it is:
The average recall across classes.

Why it matters here:
It reduces the chance that common classes dominate the score.

Why it is useful here:
Emotion support can be uneven, so balanced accuracy is often more informative than raw accuracy alone.

## ROC-AUC

What it is:
The area under the receiver operating characteristic curve.

Why it matters here:
It measures how well a classifier ranks positives above negatives across thresholds.

Why it is useful here:
The repository uses it heavily for binary OVA evaluation and circuit selection because it is threshold-independent.

Limitation:
If only one class is present in the evaluation targets, ROC-AUC is undefined.

## PR-AUC

What it is:
The area under the precision-recall curve.

Why it matters here:
It can be especially informative when positives are relatively rare.

Why it is useful here:
One-vs-all emotion probes often create imbalance between positive and negative examples.

## Cohen's Kappa

What it is:
A metric of agreement corrected for chance.

Why it matters here:
It helps show whether a classifier is doing better than what would be expected from label frequencies alone.

Why it is useful here:
Kappa can give a more cautious view of performance than raw accuracy when class distributions are uneven.

## RMSE

What it is:
Root mean squared error.

Why it matters here:
It measures the typical size of regression error in the same units as the target.

Why it is useful here:
It is one of the clearest ways to read appraisal regression performance.

Interpretation:
Lower is better.

## R-squared

What it is:
The proportion of variance explained relative to predicting the mean.

Why it matters here:
It tells you whether the regressor is capturing meaningful structure beyond a simple baseline.

Important caution:
On held-out data, `R^2` can be negative if predictions are worse than predicting the mean.

## Correlation

What it is:
A measure of linear association between predictions and targets.

Why it matters here:
It complements RMSE and `R^2` by showing whether scores move in the right direction even when absolute calibration is imperfect.

## PCA

What it is:
Principal component analysis is a linear dimensionality-reduction method that finds directions capturing the most variance.

Why it matters here:
It helps visualize hidden-state geometry and compare how structure changes across layers and locs.

What this repository uses:
PCA plots and explained-variance summaries for every configured site.

Important limitation:
PCA is an exploratory visualization tool. It does not by itself prove mechanistic structure.

## Explained Variance

What it is:
The proportion of total variance captured by a PCA component.

Why it matters here:
It tells you how concentrated or diffuse the site's variance is across principal axes.

Important caution:
A high explained-variance component is not automatically the same thing as an emotion or appraisal axis.

## Geometry Analysis

What it is:
Comparing directions or vectors inside hidden-state space, often using cosine similarity or related summaries.

Why it matters here:
Phase 2 uses geometry to ask whether appraisal and emotion directions align inside the selected circuit.

Why this choice was made:
It connects probe-derived readout directions back to the shared raw hidden-state space.

## Raw Hidden-State Space

What it is:
The original hidden-state coordinate system before probe-specific scaling or normalization.

Why it matters here:
Directions from different trained probes are most meaningfully compared after being converted back into the same raw feature space.

What this repository uses:
`emotion_probe_direction_raw()` and `appraisal_probe_direction_raw()` in `research_contracts.py`.

## Steering

What it is:
Adding or moving along a direction in hidden-state space to see whether the model's output changes.

Why it matters here:
Steering tests whether the identified directions have functional influence, not just readout value.

What this repository uses:
- cache-based steering
- behavioral forward-pass steering

Why this choice was made:
It provides stronger evidence than passive probing alone.

## Ablation

What it is:
Removing or suppressing an internal signal to test whether it matters.

Why it matters here:
Ablation asks what breaks or changes when the targeted information is erased.

What this repository uses:
- cache-based appraisal erasure in phase 2
- behavioral appraisal erasure in the steering stage

Why this choice was made:
Erasure complements additive steering by testing necessity rather than only influence.

## Cache-Based Intervention

What it is:
An intervention applied to cached hidden states rather than to a full live forward pass.

Why it matters here:
It is cheaper and easier to run at scale.

Limitation:
It is not as strong as a live behavioral intervention because it does not fully re-run the downstream model computation.

## Behavioral Intervention

What it is:
An intervention applied during the actual forward pass of the model.

Why it matters here:
It is closer to a causal test of whether the internal signal matters for generated behavior.

Why this repository uses it:
Behavioral steering and behavioral appraisal ablation are the strongest intervention-style evidence in the current pipeline.

## Pair Taxonomy

What it is:
A grouping of emotion pairs into categories such as contrastive and similar.

Why it matters here:
The theory-facing analyses want to know not only whether emotions differ, but whether similar pairs and contrastive pairs behave differently in systematic ways.

What this repository uses:
- `CONTRASTIVE_EMOTION_PAIRS`
- `SIMILAR_EMOTION_PAIRS`

## Cognitive Appraisal Theory

What it is:
A family of theories that explain emotions partly in terms of how situations are appraised along dimensions such as control, certainty, responsibility, urgency, and pleasantness.

Why it matters here:
This is the main theoretical lens for interpreting the relationship between appraisal signals and emotion circuits.

What this repository is testing:
- whether appraisal variables are readable from hidden states
- whether those appraisal signals align with selected emotion circuits
- whether removing or adding those signals changes emotion behavior

Important caution:
Even strong alignment with appraisal theory inside the model does not mean the model is a psychological subject. It means the model's internal organization is consistent with that theoretical lens.

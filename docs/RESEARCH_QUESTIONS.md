# Research Questions

This file explains the scientific questions the pipeline is trying to answer, what evidence each stage contributes, and how to interpret those outputs conservatively.

See also:
- `docs/START_HERE.md` for the high-level overview
- `docs/EXPERIMENTAL_SETUP.md` for how the experiment is configured
- `docs/PIPELINE_MAP.md` for where each question is addressed in code
- `docs/OUTPUTS_GUIDE.md` for how to read the resulting artifacts

## Question A

Can emotion be read out from hidden states?

Why it matters:
If emotion labels are consistently decodable from internal states, then the model has linearly accessible emotion information at some internal sites.

Main stages:
- `pipeline/train_probes.py`
- `pipeline/circuit_evidence.py`

Main evidence:
- `probe_summary.csv`
- `circuit_evidence_classification.csv`

What would count as support:
- Some layers and locs show stable held-out emotion probe performance.
- Multi-site circuit readout performs as well as or better than a single best site.

What would not be enough:
- A single good training result without held-out evaluation.
- Visual clustering alone without probe evidence.

## Question B

Can appraisal dimensions be read out from hidden states?

Why it matters:
If appraisal variables such as certainty, control, urgency, and responsibility are readable from hidden states, then the model contains structured internal signals that may help organize emotion perception.

Main stages:
- `pipeline/train_appraisal_probes.py`
- `pipeline/appraisal_structure.py`

Main evidence:
- `appraisal_probe_validation_detail.csv`
- `appraisal_zscore_by_emotion.csv`

What would count as support:
- Some sites predict held-out appraisal targets above trivial baselines.
- Distinct emotions show distinct appraisal profiles on held-out data.

What would not be enough:
- Appraisal-like visual separation in PCA without direct regression results.
- In-sample regression metrics with no held-out validation.

## Question C

Are there specific internal sites that form an emotion circuit?

Why it matters:
The circuit claim is stronger than saying "emotion is somewhere in the network." It asks whether a selected subset of internal sites carries especially useful emotion signal when combined.

Main stages:
- `pipeline/circuit_evidence.py`
- `pipeline/phase1_circuits.py`

Main evidence:
- `circuit_top_k_selection.json`
- `circuit_evidence_classification.csv`
- `circuits.json`
- `circuit_sites.json`

What would count as support:
- The selected top-k site combination performs better than or comparably to single-site baselines on held-out data.
- Different emotions can rely on different high-value site combinations.

What would not be enough:
- Picking a fixed circuit size in advance without empirical selection.
- Reporting only a heuristic site ranking without a held-out top-k sweep.

## Question D

Does appraisal information live inside those selected emotion circuits?

Why it matters:
This is the main theory-facing bridge. It asks whether the same sites that matter for emotion readout also contain aligned appraisal structure.

Main stages:
- `pipeline/phase2_compute.py`
- `pipeline/phase2_summary.py`

Main evidence:
- `geometry_circuit_layers.csv`
- `correlation_circuit_vs_default.csv`
- `appraisal_ablation_summary.csv`

What would count as support:
- Appraisal and emotion directions show meaningful alignment at selected circuit sites.
- Circuit-site appraisal estimates resemble the default appraisal readout in structured ways.
- Removing appraisal information changes the circuit readout.

What would not be enough:
- Showing only that appraisal information exists somewhere else in the model.
- Showing only correlation without any ablation or intervention evidence.

## Question E

If we intervene on appraisal-related information, does the model's emotion perception change?

Why it matters:
Interventions are stronger than passive readout. They test whether changing the internal signal changes the model's downstream emotion behavior.

Main stages:
- `pipeline/steering_benchmark.py`
- `pipeline/phase2_compute.py`
- `pipeline/generation_behavior_benchmark.py`

Main evidence:
- `steering_benchmark.csv`
- `steering_benchmark_behavioral.csv`
- `behavioral_appraisal_ablation.csv`
- `appraisal_ablation_summary.csv`
- `generation_behavior_outputs.csv` (actual generated text under intervention)

What would count as support:
- Steering along an appraisal or emotion direction shifts the model's emotion readout in the intended direction.
- Erasing appraisal signatures reduces or changes the targeted emotion signal.

What would not be enough:
- Cache-based changes with no behavioral follow-up.
- Behavioral changes with no provenance about which direction was used.

## Question F

Are the overall findings consistent with cognitive appraisal theory?

Why it matters:
The project is not just asking whether emotions are decodable. It is asking whether the model's internal organization resembles an appraisal-based account of emotion.

Main stages:
- `pipeline/appraisal_structure.py`
- `pipeline/representation_analysis.py`
- `pipeline/phase2_compute.py`
- `pipeline/steering_benchmark.py`
- `pipeline/appraisal_theory.py`
- `pipeline/synthesis.py`

Main evidence:
- Appraisal heatmaps and cluster summaries
- PCA and explained-variance analyses
- Circuit geometry comparisons
- Steering and ablation results
- Layer onset comparison (do appraisals appear before emotions?)
- Appraisal-to-emotion reconstruction accuracy (are appraisals sufficient for emotion?)
- Cross-layer prediction asymmetry (does information flow appraisal→emotion?)
- Emotions in appraisal space PCA biplot (do emotions organize by appraisal profile?)
- Model-level synthesis summaries

What would count as support:
- Appraisal profiles distinguish emotions in interpretable ways.
- Similar and contrastive emotion pairs show systematic differences in geometry and interventions.
- Circuit and intervention findings point in the same direction as the descriptive structure analyses.
- Appraisal-to-emotion prediction is stronger than the reverse across layers.
- Appraisal information can reconstruct emotion labels above chance.

What would not be enough:
- A single plot with a plausible story.
- Treating internal similarity alone as proof of a psychological theory.

## How To Read The Full Study

The pipeline is intentionally cumulative:

1. **Readout:** Probes ask whether the information is there.
2. **Circuit:** Circuit selection asks where the strongest sites are and whether combining them helps.
3. **Structure:** Appraisal structure and PCA ask how the information is organized.
4. **Intervention:** Ablation and steering ask whether changing the signal changes the model's emotion behavior.
5. **Temporal/structural:** Appraisal theory analyses ask whether the model's processing order mirrors cognitive appraisal theory's feedforward model.

The most convincing result is not one large score. It is agreement across these layers of evidence under held-out evaluation.

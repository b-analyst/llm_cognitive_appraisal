# Start Here

This repository studies how large language models internally represent emotion and appraisal information.

The central hypothesis is that some internal model sites form emotion circuits, and that appraisal information lives inside those circuits in a way that helps determine the model's emotion perception. The pipeline is designed to test that idea with multiple evidence types rather than relying on a single analysis.

**Supported interface:** command-line only (`python -m pipeline.<module>`). See `README.md` and `docs/RUNBOOK.md` for setup.

**Where the code lives:** `pipeline/README.md` and `pipeline/STRUCTURE.md` map the tree; the master narrative is `pipeline/docs/PIPELINE.md`. Implementation lives under `core/`, `stage_01_probes/` … `stage_07_orchestration/`, and `stage_06_benchmarks/utils/`. Top-level `pipeline/*.py` files are mostly thin shims (see `pipeline/shims/redirect.py`) so `python -m pipeline.<name>` stays stable.

## Key Docs

- `pipeline/docs/PIPELINE.md`: master map of stages, outputs, and links into each `stage_*/README.md`
- `pipeline/STRUCTURE.md`: how root shims relate to `stage_*` folders (for code review)
- `docs/PIPELINE_TLDR_RESEARCH_PARTNER.md`: short high-level description of each pipeline stage (for collaborators)
- `docs/RESEARCH_QUESTIONS.md`: what the study is trying to answer
- `docs/EXPERIMENTAL_SETUP.md`: how the experiment is configured and why
- `docs/PIPELINE_MAP.md`: what each pipeline stage does
- `docs/OUTPUTS_GUIDE.md`: how to interpret the main outputs
- `docs/GLOSSARY.md`: what the main technical concepts mean and why they were chosen
- `docs/RUNBOOK.md`: how to run the pipeline and validate a run stage by stage
- `docs/LOC_REFERENCE.md`: plain-English explanation of what loc 3, 6, and 7 are inside a transformer and why they are used
- `docs/GENERATION_BEHAVIOR_BENCHMARK.md`: how the pipeline tests actual generated behavior under appraisal/emotion interventions
- `docs/BASELINE_PROBE_STEERING_STUDY.md`: optional prompted vs unprompted probe readouts, top-k + margin ranking, frozen appraisal-dim steering + null controls (`python -m pipeline.baseline_probe_steering_study`)
- `docs/MENTAL_HEALTH_STEERING_BENCHMARK.md`: mental-health forum posts, steering calibration (unit norm, prefill-only generation), interpreting dose curves
- `docs/BENCHMARK_DATASETS.md`: open-source dataset leads for persona, provocation, manipulation, and safety benchmarks
- `docs/PROMPT_TAXONOMY.md`: benchmark-family definitions and default emotion/appraisal intervention targets
- `docs/BENCHMARK_BUILD_PROCESS.md`: exactly how benchmark datasets are downloaded, cleaned, combined, and scored

## The One-Sentence Mental Model

Take a combined emotion-and-appraisal dataset, split it carefully at the scenario level, train emotion and appraisal probes across model sites, identify high-value circuit sites, test whether appraisal structure is present inside those sites, intervene on those directions, test actual generated behavior under those interventions, and then synthesize the results.

## What This Pipeline Is Trying To Answer

1. Can emotion labels be read out from hidden states?
2. Can appraisal dimensions be read out from hidden states?
3. Are there specific internal sites that jointly act like an emotion circuit?
4. Does appraisal information align with and matter inside those circuit sites?
5. Do causal-style interventions on those internal directions change model behavior?
6. Are the overall patterns consistent with cognitive appraisal theory?

## The Ladder Of Evidence

The documentation and pipeline are organized around a ladder of evidence:

1. Readout evidence: linear probes show whether information is decodable from hidden states.
2. Circuit evidence: multi-site combinations are compared against single best sites.
3. Representational evidence: appraisal maps, geometry analyses, and PCA show how information is organized.
4. Intervention evidence: steering and ablation test whether changing those internal directions affects the model's outputs.

No single stage is treated as definitive on its own. The goal is convergence across held-out evaluation, geometry, and interventions.

## What Makes The Setup Rigorous

The current documentation reflects several design choices intended to keep the experiment honest:

- Scenario-level train, selection, and test splitting to reduce leakage across prompt variants of the same underlying situation.
- Separate selection and test partitions so hyperparameters and circuit size are not chosen on the final evaluation set.
- Supported-emotion thresholds so the benchmark focuses on emotions with enough data for stable comparison.
- Model-aware extraction settings, manifests, and cache provenance checks so outputs can be traced back to their generating conditions.
- Separate interpretation of descriptive evidence, predictive evidence, and intervention evidence.

## Recommended Reading Order

1. `docs/RESEARCH_QUESTIONS.md`
2. `docs/EXPERIMENTAL_SETUP.md`
3. `docs/PIPELINE_MAP.md`
4. `docs/OUTPUTS_GUIDE.md`
5. `docs/GLOSSARY.md`
6. `docs/RUNBOOK.md`

## Core Files In The Codebase

- `pipeline/config.py`: global scientific defaults such as appraisal dimensions, split fractions, and probe tuning grids.
- `pipeline/model_config.py`: model-specific extraction layers, locs, batch sizes, and CPU parallel defaults.
- `pipeline/research_contracts.py`: methodological rules for normalization, splitting, manifests, pair taxonomy, and raw-space conversions.
- `pipeline/train_probes.py`: emotion probe training across the configured site grid.
- `pipeline/train_appraisal_probes.py`: appraisal probe training across the configured site grid.
- `pipeline/circuit_evidence.py`: top-k circuit-site selection and final held-out circuit comparison.
- `pipeline/phase2_compute.py`: appraisal-in-circuit geometry, correlations, and cache-based ablations.
- `pipeline/steering_benchmark.py`: cache-based and behavioral interventions.
- `pipeline/synthesis.py`: final model-level and cross-model rollup.

## What This Repository Does Not Claim

This pipeline does not claim that the model literally feels emotions, and it does not treat a linear probe or PCA plot as a complete causal explanation. The strongest claims come from consistent patterns across held-out probe performance, circuit-site selection, geometry analyses, and intervention results.

# Baseline probe readouts + top-k appraisal steering

Optional study stage: characterize **probe-defined** internal readouts under **prompted vs unprompted** text (same split as probe training vs raw benchmark string), rank emotions with a **multilabel-style** rule (top-k + consecutive margin on **linear OvA logits**), then steer along **frozen** appraisal subspaces derived from `appraisal_zscore_by_emotion.csv` and **ridge** probe directions at circuit sites.

This does **not** replace [GENERATION_BEHAVIOR_BENCHMARK.md](GENERATION_BEHAVIOR_BENCHMARK.md); it complements it with data-driven target selection and explicit null controls.

## Pre-specified choices (config)

Defined in [pipeline/config.py](../pipeline/config.py):

| Constant | Role |
|----------|------|
| `BASELINE_PROBE_STUDY_PRIMARY_TEXT` | `"unprompted"` or `"prompted"` — which text is used for **generation** and steering (both text types are always logged in readouts). |
| `BASELINE_PROBE_STUDY_PRIMARY_OUTCOME` | Pre-registered behavior proxy label for summaries (default: `assistantlike_proxy`). |
| `BASELINE_PROBE_STUDY_TOP_K` | Max length of ranked emotion list. |
| `BASELINE_PROBE_STUDY_TOP_M_APPRAISAL_DIMS` | How many appraisal dimensions per emotion enter the steer (by \|z\| in z-score table). |
| `BASELINE_PROBE_STUDY_RANK_MIN_MARGIN` | Minimum gap between consecutive ranked linear logits to include the next emotion. |
| `BASELINE_PROBE_STUDY_STEER_ALPHA` | Steering strength (with per-site division as in generation behavior). |
| `BASELINE_PROBE_STUDY_STEERING_UNIT_NORM` | L2 unit vectors per site before alpha. |
| `BASELINE_PROBE_STUDY_GEN_INTERVENTION_DURING_DECODE` | Match generation-behavior decode steering flag. |
| `BASELINE_PROBE_STUDY_MAX_ROWS` | Cap prompts (`None` = use `GENERATION_BEHAVIOR_MAX_ROWS` if set, else all). |

## Shared readout helpers

The **readout site** rule and **hidden-vector extraction** at that site live in [`pipeline/runtime_emotion_readout.py`](../pipeline/runtime_emotion_readout.py) (`select_probe_readout_site`, `hidden_vector_at_site`). The baseline study imports them so generation-behavior and mental-health benchmarks can use the **same** primitive when optional **runtime** appraisal-source mode is enabled (see `docs/GENERATION_BEHAVIOR_BENCHMARK.md` and `docs/MENTAL_HEALTH_STEERING_BENCHMARK.md`).

## Design chain

1. **Readout site** — single `(layer, loc)` maximizing mean `test_roc_auc` over emotions in `probe_summary.csv` (excludes `no-emotion`).
2. **Frozen dimension policy** — for each emotion, take top-m `COMMON_APPRAISAL` columns by absolute value in `outputs/<model>/03_appraisal_structure/appraisal_zscore_by_emotion.csv`. Saved as `dim_selection_policy.json` with a SHA-256 prefix of that CSV for reproducibility.
3. **Ridge geometry** — at each circuit site for the steer target emotion, build a vector  
   \(\sum_{d \in D} (z_{\text{target}}(d) - z_{\text{rank1}}(d)) \cdot \text{dir}_d\)  
   where `rank1` is the top-ranked emotion on the **primary** text condition (probe-linear logits).
4. **Controls** — `control_wrong_emotion_policy` (lowest-logit emotion with a circuit, that emotion’s dim set), `control_random_appraisal_mix` (random weights on random ridge directions), `control_shuffle_dim_weights` (permute weights across directions for the reference emotion).

## Contrastive vs goal-directed steering (elsewhere in the pipeline)

- **Contrastive (this stage and MH `appraisal_full` / emotion targets):** directions built from **differences** between profiles (e.g. target emotion vs top-ranked readout, or condition vs contrastive emotion in z-score space).
- **Goal-directed (fixed profile):** **`ELICITATION_APPRAISAL_PROFILE`** in `config.py` defines a **single hand-tuned** appraisal mix applied at circuit sites — used as **`appraisal_elicitation`** in `mental_health_steering_benchmark` and optionally **`appraisal_elicitation_steer`** in `generation_behavior_benchmark`. The baseline probe study does **not** add that condition by default; see the mental-health and generation-behavior docs for interpretation and ethics.

## Caveats (language and inference)

- OvA linear scores are **probe-defined**, not claims about phenomenal “emotion in the model.”
- Z-score tables indicate **which dimensions** to include; **sign and dose** are chosen here as contrast from rank-1 to target profile — not a claim of causal optimality at every prompt.
- Prompted framing can shift rankings vs unprompted; report both in `baseline_probe_readouts.csv`.

## Outputs (`outputs/<model_id>/05_baseline_probe_steering/`)

| File | Content |
|------|---------|
| `baseline_probe_readouts.csv` | Per prompt × `text_type`: fused argmax, ranked top-k + logits, ridge predictions JSON. |
| `baseline_probe_steering_runs.csv` | Per prompt × condition: generation, post-steer fused argmax, linear logits JSON, appraisal JSON, heuristic behavior scores, deltas vs baseline. |
| `baseline_probe_steering_summary_by_condition.csv` | Optional mean/std of `delta_<PRIMARY_OUTCOME>` by `intervention_type`. |
| `dim_selection_policy.json` | Frozen emotion → dimension list + z-score fingerprint. |
| `summary.md` | Run metadata. |

## How to run

Prerequisites: trained probes, `circuit_top_k_selection`, `appraisal_structure` (z-scores), generation behavior benchmark CSV.

```bash
# From repo root (with PYTHONPATH including repo root)
python -m pipeline.baseline_probe_steering_study --model_id Llama3.2_1B --max_rows 5
```

This module is **not** inserted into the default `run_pipeline_for_models` sequence (to avoid surprise GPU/time cost). Invoke explicitly when needed.

## Code map

- [pipeline/baseline_probe_steering_study.py](../pipeline/baseline_probe_steering_study.py) — main entrypoint.
- [pipeline/probe_latent_scoring.py](../pipeline/probe_latent_scoring.py) — linear OvA logits, ridge readouts, ranking helper, probe key resolution.
- Reuses: [pipeline/generation_behavior_benchmark.py](../pipeline/generation_behavior_benchmark.py) (CSV load, behavior proxies, steering spec), [pipeline/steering_benchmark.py](../pipeline/steering_benchmark.py) (`_circuit_logits`), [pipeline/research_contracts.py](../pipeline/research_contracts.py) (`build_prompted_texts`, `appraisal_probe_direction_raw`).

See also [APPRAISAL_THEORY.md](APPRAISAL_THEORY.md) for where z-scores and appraisal probes come from.

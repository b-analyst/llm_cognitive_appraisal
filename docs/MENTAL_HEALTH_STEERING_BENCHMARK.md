# Mental health steering benchmark

This page explains **what** the mental health steering benchmark does, **why** the steering math is set up the way it is (in plain language), and **how to interpret** the outputs. It pairs with the code in `pipeline/mental_health_steering_benchmark.py` and settings in `pipeline/config.py`.

**Related docs:** `docs/GENERATION_BEHAVIOR_BENCHMARK.md` (assistant/safety generation benchmark — uses the **same** unit-normalization and prefill-only generation steering defaults via `GENERATION_BEHAVIOR_*` config), `docs/APPRAISAL_THEORY.md`, `docs/PIPELINE_MAP.md` (stage 5C).

---

## What problem does this benchmark solve?

Earlier pipeline stages learn **directions in the model’s internal activations** that relate to:

- **Emotions** (from binary probes trained on hidden states), and  
- **Appraisal dimensions** (from regression probes, e.g. “how much control does this text imply?”).

The **circuit** stage picks a few **(layer, position)** sites where those signals are strong.

This benchmark asks a focused question:

> If we **nudge** those internal activations while the model reads a **real mental-health forum post** and writes a reply, does the **tone** of the reply change in ways we care about (more empathy, less blame, etc.)?

So it is not primarily about jailbreaks or “be a hacker” prompts—it’s about **emotionally heavy, supportive-reply** behavior under **controlled interventions**.

---

## Simple picture of the math (≈ 10th-grade level)

### 1. Vectors and hidden states

Inside the transformer, at each layer there is a big list of numbers (a **vector**) describing “what the model is thinking” at a certain token position. Think of it as coordinates in a huge space.

A **steering vector** is another list of numbers with the same length. We **add** a scaled copy of it to the model’s vector at chosen places:

\[
\text{new\_activation} = \text{old\_activation} + \alpha \cdot \hat{v}
\]

- **\(\hat{v}\)** is a **direction** we want to push the activation (in code: one vector per circuit site, after **L2 normalization**—see below).  
- **\(\alpha\)** (called **alpha** in the code and CSVs) is a **knob**: how hard we push in that direction.  
- **\(\alpha = 0\)** means “no push” → **baseline** behavior.

So steering is literally: **same prompt, same model, but we bias the internal state a little before the model decides what to say next.**

### 2. Why we **normalize** \(\hat{v}\) (unit length)

Probe weights are trained for **prediction** (e.g. “does this direction help predict ‘joy’?”). Their raw numeric size is **not** chosen so that “\(\alpha = 5\)” means something universal across models or layers.

If we multiply a **huge** raw vector by **\(\alpha = 5\)**, we can add something enormous to the hidden state. The next-token predictor then sees nonsense inputs and often collapses into **gibberish** (repeated characters, broken subwords)—even if a side metric (like a linear “emotion probe”) looks “better.”

**Fix:** before steering, we convert each site’s vector to **length 1** (L2 **unit norm**):

\[
\hat{v} = \frac{v}{\|v\|} \quad (\text{unless } \|v\| \approx 0)
\]

Then **\(\alpha\)** is easier to reason about: it’s “how many units of movement along that direction,” not “raw probe scale × mystery factor.”

**Config:** `MENTAL_HEALTH_STEERING_UNIT_NORM = True` in `config.py`. Turn off only for ablations (`--no_unit_norm` on the CLI).

### 3. Multiple circuit sites

We often steer at **\(k\)** different (layer, site) pairs at once. The code splits strength so each site gets **\(\alpha / k\)** times its unit vector. That way, doubling the number of sites doesn’t automatically double the **total** push unless you change \(\alpha\).

### 4. Prompt-only steering during **generation**

When the model **generates** the reply, it runs many forward passes: one big pass over the **prompt**, then one pass per **new token**.

**Problem:** Re-applying the **same** big push on **every** new token can **pile up** distortion and break fluent text.

**Mitigation (default for this benchmark):** during `generate()`, apply steering only when the forward pass looks like **prompt prefill** (sequence length greater than 1 in our hook), and **skip** the tiny single-token steps used with KV cache during decoding.

**Config:** `MENTAL_HEALTH_GEN_INTERVENTION_DURING_DECODE = False`.

- **`False` (default):** steer on prefill, **not** on each decode step → usually **much more fluent** text while still biasing **how generation starts** from the steered prompt state.  
- **`True`:** old behavior—steer on **every** step (stress test; often harsher on quality).

CLI: `--steer_during_decode` forces `True`.

**Note:** The separate **forward** pass used for **latent / probe readouts** still runs with the full steering spec (one pass over the prompt). That’s intentional: we want to see how probes respond under the same intervention.

### Runtime readout for `appraisal_full` only (optional)

**`RUNTIME_READOUT_USE_FOR_MENTAL_HEALTH`**: when `True`, the benchmark runs **one unsteered forward per (post × framing)** before the alpha loop, ranks emotions (see **`RUNTIME_READOUT_EMOTION_MODE`** below — same options as generation behavior), and builds **`appraisal_full`** vectors using **rank-1** as the z-contrast **source** when valid in `appraisal_zscore_by_emotion.csv`, else falls back to the condition name (or `sadness` if missing). **`appraisal_targeted`**, **`emotion_steer`**, **`combined`**, and **`appraisal_elicitation`** are unchanged.

**Cost:** +1 forward pass per post × framing (in addition to baseline + steered runs).

**Ranking modes** (`RUNTIME_READOUT_EMOTION_MODE`): same as generation behavior — default **`circuit_sigmoid_mean`** (per-site σ then mean over **`topk_per_emotion`**); **`single_site`** (σ at probe-summary-optimal site); **`circuit_linear_mean`** (mean **pre-sigmoid** linear scores). Canonical diagram: **`docs/RUNTIME_READOUT.md`**.

**Outputs:** `mental_health_steering_scores.csv` gains `runtime_rank1_emotion`, `appraisal_full_source` (`runtime` / `condition` / `fallback`), `runtime_readout_layer`, `runtime_readout_loc`, `runtime_readout_emotion_mode`, `runtime_readout_union_n_sites`, `ranked_top_k_json` (when `RUNTIME_READOUT_LOG_RANK_JSON`; includes `score_kind` + optional `all_emotions_scores_json`), optional **`runtime_linear_circuit_rank1_emotion`** / **`runtime_linear_circuit_ranked_top_k_json`** when **`RUNTIME_READOUT_LOG_LINEAR_CIRCUIT_AUX`**, `mh_skip_reason` (e.g. `appraisal_full_skipped_source_equals_contrastive` when rank-1 source equals the **effective** contrastive target and `appraisal_full` rows are omitted for that prompt). With adaptive target enabled, also `effective_target_emotion`, `adaptive_appraisal_target`, `adaptive_target_metric`, `adaptive_target_distance`, `adaptive_target_fallback_reason`, `adaptive_target_scores_json` (see **Adaptive contrastive target** above).

Ranking hyperparameters mirror **`RUNTIME_READOUT_TOP_K`**, **`RUNTIME_READOUT_MIN_MARGIN`**, **`RUNTIME_READOUT_MIN_TOP1_LOGIT`** in `config.py` (aligned with baseline probe study defaults). Thresholds apply to the **active score scale** (σ-mean vs linear-mean).

**Latent readout:** `latent_predicted_emotion` uses **`_circuit_logits`** (mean **sigmoid** per emotion over circuits). With default runtime mode **`circuit_sigmoid_mean`**, this matches the **primary** runtime readout definition; see **`docs/RUNTIME_READOUT.md`** when modes differ.

### Adaptive contrastive target (optional; `ADAPTIVE_APPRAISAL_TARGET_ENABLED`)

When **`True`**, the benchmark picks **`effective_target_emotion`** per **(post × framing)** by **maximum L2 z-profile contrast** to the resolved appraisal **source** (runtime rank-1 when `RUNTIME_READOUT_USE_FOR_MENTAL_HEALTH` is on, else the condition’s z-row fallback `condition` or `sadness`), over **`COMMON_APPRAISAL`** columns present in `appraisal_zscore_by_emotion.csv`. **`CONDITION_APPRAISAL_TARGETS[...]["contrastive_emotion"]`** becomes the **static fallback** only (same semantics as taxonomy `target_emotion` in generation behavior).

**Steering when adaptive is on:** For each post × framing, the code **rebuilds** `target_pairs`, **`appraisal_targeted`**, **`appraisal_full`**, **`emotion_steer`**, **`combined`**, and **`appraisal_elicitation`** vectors so **circuit sites and emotion probes** match **`effective_target_emotion`**, and **`appraisal_full`** uses z-contrast **`(source, effective_target)`** with **exact z-score index keys**. **`omit_appraisal_full`** (skip `appraisal_full` when source equals contrastive target) uses **`effective_target_emotion`** instead of the fixed config contrastive label.

**Cost:** Extra CPU to rebuild vectors per post × framing; forward-pass count is unchanged unless you also use runtime readout.

**Outputs:** Same diagnostic columns as generation behavior: `effective_target_emotion`, `adaptive_appraisal_target`, `adaptive_target_metric`, `adaptive_target_distance`, `adaptive_target_fallback_reason`, `adaptive_target_scores_json`. When adaptive is off, `adaptive_target_fallback_reason` is `disabled` and `effective_target_emotion` matches **`contrastive_emotion`**.

**Default:** `ADAPTIVE_APPRAISAL_TARGET_ENABLED = True` in `config.py` enables adaptive target selection; set `False` for fixed **`contrastive_emotion`** per condition only.

---

## What interventions are compared?

For each mental-health **condition** (e.g. depression, anxiety), each **post**, each **framing** (`counselor`, `introspective`, **`honest_reply`** — see below), and each **\(\alpha\)**:

| Method | Idea (plain language) |
|--------|------------------------|
| **baseline** | \(\alpha = 0\): no steering. |
| **appraisal_targeted** | Push along a **hand-designed mix** of appraisal probe directions (condition-specific “target profile”). |
| **appraisal_full** | Push along the **data-driven** appraisal difference vector (contrastive emotion vs condition in z-score space), same as elsewhere in the pipeline. |
| **emotion_steer** | Push along the **emotion probe** direction for a **contrastive emotion** (e.g. “joy” or “relief”) at the circuit sites chosen for that emotion. |
| **combined** | Add targeted appraisal direction and emotion direction **per site**, then **re-normalize** to unit length (when unit norm is on). |
| **appraisal_elicitation** | **Goal-directed** push toward a **fixed** appraisal profile (`ELICITATION_APPRAISAL_PROFILE` in `config.py`): tuned to evoke a **threat / frustration / negative affect toward the post text** readout (research construct for studying elicited tone; **not** a clinical diagnosis). Runs for \(\alpha > 0\) only when the profile is non-empty. **Contrasts with** contrastive methods above, which move toward relief/joy-style targets or data-driven differences. |

**Default framings** are listed in **`MENTAL_HEALTH_FRAMINGS`** (`counselor`, `introspective`, `honest_reply`). **`honest_reply`** asks for a direct, honest reaction without prescribing a counselor role, to reduce **assistant-axis** pressure relative to the other framings.

**Alphas** in `config.py` (`MENTAL_HEALTH_ALPHAS`) are chosen for **unit-normalized** vectors (e.g. `0, 0.5, 1, 2, 4`). Older runs used much larger \(\alpha\) on **raw** vectors; those are **not** comparable numbers.

### Ethics and scope (appraisal elicitation)

- **Research-only:** elicitation steering is for measuring how **generated language** shifts under a **fixed internal target** in a controlled offline benchmark. It is **not** a recipe for deployment, therapy, or harm.
- **Purpose:** isolate **tone and stance** effects when the model’s appraisal geometry is pushed toward **negative affect / annoyance toward the post** — analogous to stress-testing other steering directions, with explicit documentation.
- **Not clinical:** forum posts are real-ish text; outputs are **not** patient interactions and must not be framed as mental health advice.

### Console / logging

At **INFO** once per run:

`[mental_health_steering] model_id=... conditions=... posts~... framings=[...] alphas=... ~steps=... appraisal_elicitation=True|False unit_norm=... gen_intervention_during_decode=... runtime_readout=... adaptive_appraisal_target=True|False`

Long work is shown via tqdm (`Mental health steering`).

---

## What should you take from the results?

### Dose–response curves (`dose_response_*.png`, `dose_response_curves.csv`)

- **X-axis:** \(\alpha\) (steering strength on **unit** directions).  
- **Y-axis:** Aggregated **heuristic** scores (e.g. “emotional tone proxy” from simple word-pattern counts).

**If curves move smoothly** with \(\alpha\) and text stays fluent: steering **dose** and **direction** are in a reasonable regime; you can compare **methods** (which line goes up or down).

**If high \(\alpha\) still destroys text:** treat those points as **out-of-distribution** for the model—they show **breakdown**, not useful behavior change.

### Latent readouts (`latent_predicted_emotion`, `latent_target_logit`)

These read a **linear probe** on hidden states. They can move in the “expected” direction even when generated text is bad—**probes are not the same as human-quality text**. Use them as **mechanistic** side signals, not the only success metric.

### Heuristic text scores (empathy, blame, hostility, …)

These are **regex / pattern counts**, not clinical judgments. They’re good for **batch trends** and **comparing conditions**, not for claiming real-world therapeutic quality.

### LLM judge (optional, deferred)

Judge scores (`run_behavior_judges.py`) add a **richer** behavioral signal. They’re still automated and should be reported with uncertainty (parse failures, rubric limits).

---

## Design reasoning (why not “just copy a paper’s \(\alpha\)”?)

Papers differ on:

- **Which layer** and **how many** components they steer.  
- Whether vectors are **normalized** or scaled to activation **RMS**.  
- Whether intervention runs **once** or **throughout** decoding.

This pipeline ties steering to **your** probes and **your** circuit sites, so numeric \(\alpha\) from another setup **won’t transfer** without recalibration. The defaults here prioritize:

1. **Comparable \(\alpha\)** across methods via **unit-normalized** directions.  
2. **Fluent generations** via **prompt-only** steering during decode (by default).  
3. A **moderate** \(\alpha\) grid you can widen with `MENTAL_HEALTH_ALPHAS_FULL` once behavior looks stable.

---

## Config reference (`pipeline/config.py`)

| Constant | Role |
|----------|------|
| `MENTAL_HEALTH_ALPHAS` | Dose grid (with unit norm). |
| `STEERING_BENCHMARK_ALPHAS` | Same list as `MENTAL_HEALTH_ALPHAS` — default sweep for `run_steering_benchmark`. |
| `MENTAL_HEALTH_ALPHAS_FULL` | Wider grid for sweeps. |
| `MENTAL_HEALTH_STEERING_UNIT_NORM` | L2-normalize per-site vectors (recommended). |
| `MENTAL_HEALTH_GEN_INTERVENTION_DURING_DECODE` | Steer every decode step vs prefill-only. |
| `MENTAL_HEALTH_REPORT_ALPHA` | Preferred dose for MH `summary.md` / heatmap, steering-benchmark summary rows (nearest in sweep), generation-behavior default appraisal strength, baseline-probe steer α, and **`synthesis_metrics` MH rollups** (nearest α in `dose_response_curves.csv`). |
| `MENTAL_HEALTH_FRAMINGS` | Default framing list (stable order). |
| `ELICITATION_APPRAISAL_PROFILE` | Weights for **`appraisal_elicitation`** (and generation-behavior **`appraisal_elicitation_steer`** when enabled). |

---

## CLI overrides

```bash
python -m pipeline.mental_health_steering_benchmark --model_id Llama3.2_1B
# Fewer framings (faster; legacy two-framing comparison):
python -m pipeline.mental_health_steering_benchmark --model_id Llama3.2_1B --framings counselor introspective
# Ablations:
python -m pipeline.mental_health_steering_benchmark --no_unit_norm
python -m pipeline.mental_health_steering_benchmark --steer_during_decode
```

---

## Limitations (read before citing)

- **Not a clinical trial**—forum posts + pattern scores + (optional) LLM judge.  
- **Steering ≠ therapy**—we measure text statistics under interventions, not patient outcomes.  
- **Small models** may show **weak** or **noisy** effects even when steering is “working” mechanistically.  
- **Unit normalization** fixes **scale**; it does not guarantee that probe directions are **causally optimal** for supportive language—that’s an empirical question this benchmark helps probe.

---

## File outputs (per model)

Under `outputs/<model_id>/05_mental_health_steering/`:

| File | Contents |
|------|-----------|
| `mental_health_steering_scores.csv` | One row per (post, framing, method, \(\alpha\)); text + heuristics + latent readouts. |
| `dose_response_curves.csv` | Aggregated means by condition / framing / method / \(\alpha\). |
| `dose_response_*.png` | Plots of heuristics vs \(\alpha\). |
| `condition_method_heatmap.png` | Snapshot at `MENTAL_HEALTH_REPORT_ALPHA` (or nearest available). |
| `summary.md` | Auto-written overview; see `MENTAL_HEALTH_REPORT_ALPHA` for the comparison dose. |

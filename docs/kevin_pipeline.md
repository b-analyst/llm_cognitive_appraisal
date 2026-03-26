# Pipeline TLDR
---

## The questions (in order of strength)

| Layer | Question |
|--------|----------|
| **Readout** | Is emotion (and appraisal) information present in internal states in a stable, testable way? |
| **Localization** | Are there particular internal “sites” (layer + position) that matter most for emotion—and do several sites together act like a small **circuit**? |
| **Structure** | Do emotions line up with appraisal profiles the way theory might suggest (e.g. clusters, contrasts between similar vs opposite emotions)? |
| **Inside the circuit** | Does appraisal geometry **match** emotion geometry **at those circuit sites**, and do **ablations** of appraisal signal change emotion readout there? |
| **Theory-facing** | Do analyses of **order**, **reconstruction**, and **cross-layer flow** look more like “appraisal first, emotion later” or not? |
| **Behavior** | When we **steer** or **erase** directions derived from probes, does **generated text** or **downstream scores** move in interpretable ways? |


---

## Stage-by-stage 


### Stage 0 — Probe robustness (`01_probe_robustness`)

**Purpose:** Pick a sensible way to **format** scenarios for the model (prompt layout, etc.) and define which emotions have enough data so we do not train on empty categories.

**If it goes well:** Later probe scores are not artifacts of an arbitrary formatting choice.

**If it goes poorly:** We fix data or formatting before trusting any probe.

---

### Stage 1 — Probes (`01_probes`)

**Emotion probes:** Train linear regression classifiers that map internal states → “is this scenario about emotion X?” (one-vs-rest for each emotion), across many layers/locations.

**Appraisal probes:** Same idea but predict **continuous appraisal ratings** (regression) and, in a variant, **high vs low** appraisal (binary).

**Questions answered:** *Can we read emotion and appraisals from the inside, on held-out scenarios? Where (which layer/position) is it easiest?*

**Outcomes:** High held-out probe performance → evidence that **linearly accessible** emotion/appraisal information exists **somewhere** in the network. Modest performance → information may be **nonlinear**, **noisy**, or **not aligned** with our labels. This stage does **not** prove the model “uses” that signal for its normal output.

---

### Stage 2 — Circuit (`02_circuit`)

**Purpose:** From probe results, identify a **small set of internal sites** whose combined emotion readout works well—ideally better than any single site alone. Package that set as the **emotion circuit** for later steps.

**Questions answered:** *Is “emotion” concentrated in a handful of locations that work together?*

**Outcomes:** Clear gain from multi-site fusion → supports a **circuit** story. Little gain → emotion signal may be **diffuse** or already captured by one dominant site.

---

### Stage 3 — Appraisal structure (`03_appraisal_structure`)

**Purpose:** Describe how **emotions relate to appraisals** in the data and in the model’s readouts—e.g. appraisal **profiles by emotion**, heatmaps, contrasts. (Additional label–appraisal coupling summaries may appear here depending on configuration.)

**Questions answered:** *Do emotions separate along appraisal axes in an interpretable way on held-out data?*

**Outcomes:** Clean structure → nice **descriptive** support for appraisal theory in **this** benchmark. Messy structure → theory mapping is weak **for these labels and scenarios**, or the model’s internal organization does not mirror human ratings.

---

### Stage 4 — Appraisal in the circuit (`04_appraisal_in_circuit`)

**Purpose:** At the **same sites** chosen as the emotion circuit, compare **emotion directions** and **appraisal directions** in activation space, correlate circuit vs default readouts, and run **ablations** that remove appraisal-related components from representations.

**Questions answered:** *Does appraisal information **live in** the emotion circuit, and is it **functionally entangled** with emotion readout there?*

**Outcomes:** Aligned geometry + ablations that move emotion readout → appraisal is **plausibly part of** that subspace for emotion. Weak alignment or tiny ablation effects → appraisal may live **elsewhere**, be **nonlinear**, or be **decodable** without being **causally central** to that readout.

---

### Stage 4b — Appraisal theory analysis (`04_appraisal_theory`)

**Purpose:** Extra **theory-facing** checks: e.g. whether appraisal probes “turn on” earlier than emotion probes in depth, whether appraisals can **reconstruct** emotion labels, asymmetries in cross-layer prediction, and visualizations of emotions in appraisal space.

**Questions answered:** *Does the **depth profile** of the model look at all like a staged appraisal→emotion story?*

**Outcomes:** Consistent patterns → supportive **narrative** evidence (still not proof of human cognitive architecture). Null or mixed → appraisal and emotion may be **entangled** or **parallel** in ways this linear toolkit does not separate cleanly.

---

### Stage 5 — Interventions (`05_steering`, `05_generation_behavior`, `05_mental_health_steering`)

**Steering benchmark:** Push activations along directions learned from emotion/appraisal probes (or erase components) and measure changes in **probe-based emotion readout** and related metrics.

**Generation behavior:** Run **actual text generation** under some of those interventions on a small benchmark; save outputs and simple behavioral proxies (and optional LLM-as-judge scoring if you run that separately).

**Mental health steering:** Similar steering idea applied to **mental-health–style** prompts, to see if responses shift with appraisal-targeted pushes.

**Questions answered:** *If we **force** internal state along these directions, does the model’s **measured emotion behavior** or **wording** change?*

**Outcomes:** Reliable, dose-dependent shifts → those directions are **causally relevant** to **our** metrics (stronger than readout alone). Small or inconsistent shifts → directions may be **fragile**, **misaligned with generation**, or behavior may be driven by **other** mechanisms; **jailbreak / robustness** claims would need **different** evaluations.

---

## How to read “success” vs “failure”

- **Probes strong, steering weak:** Information is **there** but may not **drive** the behaviors we measure, or our interventions do not hit the right subspace.
- **Circuit helps for emotion, phase 2 flat:** Emotion may be **localized**, but appraisal may not **cohere** with that pocket the way we hoped.
- **Everything lines up:** Best case for an appraisal-in-circuit paper—but still bounded by **linear methods**, **chosen layers**, and **this dataset**.

---


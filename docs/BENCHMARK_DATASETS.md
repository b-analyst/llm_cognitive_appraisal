# Benchmark Datasets

This document lists the main open-source dataset leads for the new generation-behavior benchmark and explains how each source maps onto the research questions.

The benchmark should use a hybrid strategy:

- reuse open-source datasets wherever they are already strong
- use synthetic prompt generation to fill coverage gaps

This is especially important for persona drift, deception, and appraisal-targeted manipulations, where public datasets are incomplete.

---

## Current Implementation Status

The current builder already ingests these sources directly:

- `AdvBench`
- `Anthropic/hh-rlhf:red-team-attempts`
- `lmsys/toxic-chat`
- `JailbreakBench/JBB-Behaviors`
- `ESConv`
- `Counsel-Chat`
- synthetic prompt templates

The exact download and cleaning record for a given build is written to:

- `pipeline/input_data/generation_behavior/behavior_benchmark_manifest.json`

For the full build pipeline, see:

- `docs/BENCHMARK_BUILD_PROCESS.md`

---

## Recommended Benchmark Sources

## 1. Anthropic HH-RLHF Red-Team Attempts

Source:
- `Anthropic/hh-rlhf`, `data_dir="red-team-attempts"`

Why it matters:
- contains full red-team conversation transcripts
- includes task descriptions, red-team success ratings, and transcript-level harmfulness scores
- is one of the strongest open sources for adversarial multi-turn interaction

Best use:
- deception / manipulation benchmark
- jailbreak / exploitative prompting benchmark
- multi-turn attack trajectories

Limitations:
- not designed for dialogue-agent fine-tuning
- may need adaptation into a benchmark-row schema for controlled inference evaluation

---

## 2. HarmBench

Source:
- `centerforaisafety/HarmBench`

Why it matters:
- standardized open-source benchmark for harmful behaviors and red teaming
- broad and well-recognized safety benchmark

Best use:
- harmful compliance benchmark
- jailbreak resistance benchmark
- refusal quality evaluation

Limitations:
- focuses on harmful requests, not emotional manipulation specifically
- best combined with other prompt families
- may be gated depending on access route, so the first implementation uses AdvBench and JailbreakBench as the primary open harmful-behavior sources

---

## 3. AdvBench / JailbreakBench

Sources:
- `llm-attacks/llm-attacks` (`harmful_behaviors.csv`)
- JailbreakBench behavior datasets and tooling

Why it matters:
- strong open-source harmful-behavior/jailbreak prompt sources
- useful for consistent safety comparisons

Best use:
- harmful request prompts
- jailbreak prompts
- controlled attack-style evaluations

Limitations:
- not naturally rich in emotional or social manipulation context

---

## 4. ToxicChat

Source:
- `lmsys/toxic-chat`

Why it matters:
- real user prompts from a public chat demo
- includes toxicity and jailbreaking labels
- strong for realistic provocative and unsafe interaction inputs

Best use:
- frustration / provocation benchmark
- realistic toxic or adversarial user prompts
- conversational safety prompts

Limitations:
- access requires accepting dataset conditions
- not all prompts are emotionally structured in the way your theory work needs

---

## 5. ESConv

Source:
- `thu-coai/Emotional-Support-Conversation`

Why it matters:
- focused on emotional-support conversations
- includes support strategies and emotionally vulnerable contexts

Best use:
- emotional support benchmark
- de-escalation benchmark
- empathy and support-quality evaluation

Limitations:
- not adversarial by default
- better for the benign high-affect side of the benchmark

---

## 6. AnnoMI / CounselChat / Related Counseling Data

Sources:
- `AnnoMI`
- `counsel-chat`
- related counseling conversation datasets

Why it matters:
- emotionally vulnerable user scenarios
- supportive / therapeutic interaction styles
- can help test whether interventions preserve or improve prosocial behavior

Best use:
- therapy-adjacent interaction benchmark
- emotional support quality benchmark
- “do not mistake distress for manipulation” control set

Limitations:
- may have usage or license constraints depending on source
- often require preprocessing into a common prompt schema

---

## 7. Assistant Axis Work

Sources:
- Anthropic Assistant Axis paper
- associated open-source code / vector resources

Why it matters:
- closest mechanistic adjacent work on assistant persona drift
- provides a strong comparison frame for persona stabilization

Best use:
- research framing
- baseline comparison condition
- possible source of persona-style prompt families

Limitations:
- not a complete benchmark dataset by itself for your purposes
- should be used as a conceptual and comparative anchor, not the sole dataset source

---

## Where Synthetic Prompt Generation Is Still Needed

Open-source data alone is unlikely to cover all of these cleanly:

- false urgency
- subtle guilt induction
- manipulative praise / flattery
- social engineering framed as emotional vulnerability
- unfair blame that induces assistant frustration
- controlled persona-drift ladders

So the clean recommendation is:

- use open datasets for the backbone
- use synthetic prompt generation to fill specific appraisal and manipulation gaps

The current pipeline implements this with:

- `pipeline/build_generation_behavior_benchmark.py`
- `pipeline/input_data/generation_behavior/prompt_taxonomy.csv`
- `pipeline/input_data/generation_behavior/behavior_benchmark_manifest.json`

---

## Proposed Benchmark Families And Data Sources

| Benchmark family | Best sources | Role in pipeline |
|------------------|-------------|------------------|
| Persona drift | Assistant Axis-inspired prompts, synthetic prompts | test assistant identity stability |
| Frustration / provocation | ToxicChat, synthetic prompts | test de-escalation and reactivity |
| Deception / manipulation | HH red-team attempts, ToxicChat, synthetic prompts | test exploitative prompt detection and mitigation |
| Harmful compliance / jailbreak | HarmBench, AdvBench, JailbreakBench | test refusal quality and safety |
| Emotional support / therapy-adjacent | ESConv, AnnoMI, CounselChat | test supportive and non-harmful affective behavior |

---

## Recommended First Build

For the first implemented version of the benchmark, use:

1. **HH red-team attempts** for adversarial multi-turn interaction
2. **HarmBench / AdvBench** for harmful request prompts
3. **ToxicChat** for realistic provocative prompts
4. **ESConv** for benign high-affect support scenarios
5. **Synthetic prompt generation** for appraisal-specific edge cases

That mix gives you both:
- strong open-source grounding
- enough controlled coverage to actually test the hypothesis

---

## Caution

Do not treat all emotionally intense prompts as malicious.

A good benchmark must distinguish:

- genuine distress
- frustration
- deception
- coercion
- jailbreak intent
- benign role-play / persona prompts

That distinction is essential if the project is going to make claims about mental-health or psychotherapy-related applications.

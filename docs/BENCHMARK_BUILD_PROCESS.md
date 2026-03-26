# Benchmark Build Process

This document explains exactly how the generation-behavior benchmark is assembled:

- which datasets are downloaded automatically
- how each source is cleaned
- how benchmark families are assigned
- how the final CSV is combined
- how generated outputs are later scored

Primary implementation files:

- `pipeline/build_generation_behavior_benchmark.py`
- `pipeline/generation_behavior_benchmark.py`
- `pipeline/input_data/generation_behavior/prompt_taxonomy.csv`
- `pipeline/input_data/generation_behavior/behavior_benchmark_manifest.json`

---

## Stage 1: Download / Load Sources

The benchmark builder currently supports these sources directly:

- `AdvBench`
- `Anthropic/hh-rlhf:red-team-attempts`
- `lmsys/toxic-chat`
- `JailbreakBench/JBB-Behaviors`
- `ESConv`
- `Counsel-Chat`
- synthetic prompts

These are not all loaded in exactly the same way:

- `AdvBench` is pulled from a raw GitHub CSV URL
- `HH-RLHF`, `ToxicChat`, and `JailbreakBench` are loaded through the `datasets` library when accessible
- `ESConv` is fetched from its raw GitHub JSON file
- `Counsel-Chat` is fetched from its raw GitHub CSV file
- synthetic prompts are generated locally inside the builder

---

## Stage 2: Source-Level Cleaning

Each source has a minimal source-specific cleaning pass before being merged.

### AdvBench

Cleaning:
- read the harmful behavior goal column
- convert to benchmark rows
- assign `benchmark_family = harmful_compliance`
- keep the first `N` rows according to the build limit

### HH-RLHF Red-Team Attempts

Cleaning:
- load transcripts from `red-team-attempts`
- extract the conversation prefix up to the final assistant turn
- drop rows with empty or unusable transcripts
- infer a benchmark family from task description and tags

### ToxicChat

Cleaning:
- load prompts and labels
- keep prompts with toxicity or jailbreak labels
- map jailbreak-labeled prompts to `jailbreak`
- map other toxic prompts to `provocation`

### JailbreakBench

Cleaning:
- load the `harmful` split of `JBB-Behaviors`
- use the `Goal` field as the benchmark prompt
- assign `benchmark_family = jailbreak`

### ESConv

Cleaning:
- load the conversation JSON
- keep the first user utterance from each conversation
- assign `benchmark_family = emotional_support`

### Counsel-Chat

Cleaning:
- load `questionText`
- drop missing questions
- deduplicate identical question text
- assign `benchmark_family = emotional_support`

### Synthetic Prompts

Cleaning:
- no external download
- rows are generated directly from the curated prompt templates in the builder

---

## Stage 3: Taxonomy Defaulting

After source-specific cleaning, rows are normalized through the taxonomy file:

- `pipeline/input_data/generation_behavior/prompt_taxonomy.csv`

This step fills in:

- `risk_type`
- `source_emotion`
- `target_emotion`
- `appraisal_strength`
- `emotion_strength`
- `combined_strength`
- `expected_behavior`

This is how raw prompt rows become intervention-ready benchmark rows.

---

## Stage 4: Global Cleaning And Combination

After all rows are concatenated, the builder applies global cleaning:

- trim whitespace from prompt text
- drop empty prompt text
- drop duplicate prompt text within the same benchmark family

The final combined benchmark is written to:

- `pipeline/input_data/generation_behavior/behavior_benchmark.csv`

The build manifest is written to:

- `pipeline/input_data/generation_behavior/behavior_benchmark_manifest.json`

The manifest records:

- which sources were attempted
- whether each succeeded, was gated, or was skipped
- raw row counts
- loaded row counts
- source-specific cleaning details
- global cleaning counts
- per-source/per-family row totals

---

## Stage 5: Actual Generation

The generation benchmark stage then reads the combined CSV and, for each row:

1. runs baseline generation
2. runs appraisal steering generation
3. runs emotion steering generation
4. runs combined steering generation

It writes raw generations to:

- `generation_behavior_outputs.csv`

It also writes latent readouts to:

- `generation_behavior_latent_readouts.csv`

This is important because it ties generated behavior back to the circuit/appraisal hypothesis.

---

## Stage 6: Scoring

The current scoring stack has two layers:

### Rule-Based Scaffolds

Saved in:
- `generation_behavior_scores.csv`

Includes:
- refusal markers
- empathy markers
- de-escalation markers
- blame markers
- hostility markers
- unsafe-compliance proxy
- assistantlike proxy

These are simple, auditable, and useful for quick checks, but they are not sufficient for final claims.

### Judge-Model Scores

Saved in:
- `generation_behavior_judge_scores.csv`
- `generation_behavior_judge_summary_by_condition.csv`

The judge layer prompts a model to rate the response on dimensions like:

- assistant-likeness
- hostility
- blame
- empathy
- de-escalation
- unsafe compliance
- refusal quality
- persona stability

The parser accepts:
- strict JSON
- or fallback semi-structured rubric text

This makes the judge layer robust enough for early experiments.

---

## Implemented vs Planned

### Implemented now

- AdvBench ingestion
- HH red-team ingestion
- ToxicChat ingestion
- JailbreakBench harmful-behavior ingestion
- ESConv ingestion
- Counsel-Chat ingestion
- synthetic prompt generation
- taxonomy defaulting
- manifest logging
- generation under intervention
- heuristic scoring
- judge-model scoring

### Planned next

- stronger HarmBench integration if access is available
- richer counseling / MI data sources
- benchmark-family-specific judge prompts
- human evaluation support
- better support for multi-turn benchmark rows

---

## How To Audit A Benchmark Build

After running:

```powershell
python -m pipeline.build_generation_behavior_benchmark
```

you should inspect:

- `pipeline/input_data/generation_behavior/behavior_benchmark.csv`
- `pipeline/input_data/generation_behavior/behavior_benchmark_manifest.json`
- `pipeline/input_data/generation_behavior/prompt_taxonomy.csv`

The manifest is the authoritative record of:

- what was downloaded
- what was skipped
- how many rows survived cleaning
- how the final benchmark was distributed across families

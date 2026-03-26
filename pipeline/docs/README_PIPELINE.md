# Emotion circuit and appraisal steering pipeline

Config-driven pipeline from probe/circuit identification through steering, with a fixed output layout per model. Run stages in order; later stages depend on earlier outputs.

**Standalone:** To move this folder to another repo and run it there, see [STANDALONE_README.md](STANDALONE_README.md). Run `python pipeline/export_standalone.py` from the original repo root to create a full bundle (pipeline + data + code deps) you can copy.

Optional benchmark build step for actual-generation evaluation:
- `python -m pipeline.build_generation_behavior_benchmark`

## Order of runs

0. **Probe training robustness (optional)** — Compare dataset variants: prompted only, prompted+unprompted, 2 or 3 prompt variations + no prompt. Uses scenario-level train/selection/final-test splitting first, then **BinaryOvaDatasetProcessor** with **even_negative_distribution=True** inside the train/selection partitions only. Logs leakage-safe dataset splits and optionally trains probes at one probe-selected site per variant for a quick selection-set comparison.  
   `python -m pipeline.probe_training_robustness` (use `--no_quick_probes` to skip probe training and only build datasets + stats)
1. **Emotion probes** — Train binary OVA emotion probes on the train split and score them on the selection split. Outputs live under `outputs/<model_id>/01_probes/binary_ova_probes/` with a probe manifest and model-specific filenames.
2. **Appraisal probes** — Train appraisal regressors on the train split and validate them on the selection split.  
   `python -m pipeline.train_appraisal_probes`
3. **Circuit evidence** — Use the selection split to choose `k`, then report single-best vs top-k metrics on the held-out test split. Writes `selection_hidden_states.pt`, `test_hidden_states.pt`, and `circuit_top_k_selection.json`.  
   `python -m pipeline.circuit_evidence` (optionally `--skip_extract` if selection/test hidden states already cached)
4. **Circuit (Phase 1)** — Produce `circuits.json` from the saved circuit-site selection for compatibility with downstream legacy consumers.  
   `python -m pipeline.phase1_circuits`
5. **Appraisal structure** — Baseline classification on held-out test activations, clustering on held-out probe logits, and appraisal z-score heatmap from the train split labels.  
   `python -m pipeline.appraisal_structure` or run [`notebooks/appraisal_structure_analysis.ipynb`](../notebooks/appraisal_structure_analysis.ipynb)
6. **Appraisal in circuit (Phase 2)** — Geometry and agreement diagnostics are computed per model at runtime using the exact selected circuit sites; then summary:  
   `python -m pipeline.phase2_summary`  
   Optional isolate-directions: `python -m pipeline.phase2_isolate_directions`
7. **Steering** — Appraisal vs emotion steering benchmark on held-out test activations; behavioral prompted/unprompted evaluation uses a real prompted condition, generating prompts deterministically when the combined CSV only stores `situation`.  
   `python -m pipeline.steering_benchmark`
8. **Generation behavior benchmark** — Actual generated text under baseline, appraisal steering, emotion steering, and combined steering, with raw output CSVs, heuristic proxies, and judge-scored behavior summaries.  
   `python -m pipeline.generation_behavior_benchmark`
9. **Mental health steering benchmark** — Forum-style posts, counselor vs introspective framings, appraisal/emotion steering with **unit-normalized** vectors and (by default) **prefill-only** generation steering. See `docs/MENTAL_HEALTH_STEERING_BENCHMARK.md`.  
   `python -m pipeline.mental_health_steering_benchmark`
10. **Synthesis** — Read all outputs; produce summary figures and SUMMARY.md with cautious interpretation of held-out metrics vs internal diagnostics.  
   `python -m pipeline.synthesis` — also run automatically as the last step of `python -m pipeline.run_pipeline_for_models` per model, with an optional multi-model aggregate pass at the end of the runner.

## Config

- **config.py**: `DEFAULT_MODEL_ID`, paths to data and probes, `EXTRACTION_LAYERS`, `EXTRACTION_LOCS`, `COMMON_APPRAISAL`, `CIRCUIT_TOP_K_MAX` (upper bound for circuit pair sweep; actual *k* is chosen on val in `circuit_evidence`), and helpers `get_model_output_dir`, `get_circuit_dir`, etc.
- One config covers all stages; change `model_id` per run to scale to multiple models.

## Runtime backend

- The active experiment path now uses **stock Hugging Face `AutoModelForCausalLM` models plus PyTorch forward hooks** for extraction and steering at the supported locations.
- The vendored files under `LLMs/` remain in the repo for reference and optional legacy debugging, but the pipeline no longer depends on them by default.
- Use the pinned environment from `requirements_standalone.txt` / `pipeline/requirements_runtime_pinned.txt`, ideally inside a local `.venv`.

## Output layout (per model)

- `outputs/<model_id>/01_probe_robustness/` — Dataset variants (prompted_only, prompted_plus_unprompted, two_prompts_plus_unprompted, three_prompts_plus_unprompted), each with fair OVA train/val CSVs; dataset_stats_*_per_emotion_per_variant.csv; probe_robustness_comparison.csv and .png (if quick probes run); summary.md
- `outputs/<model_id>/02_circuit/` — `circuit_top_k_selection.json`, `circuits.json`, `circuit_evidence_classification.csv`, figures, `selection_hidden_states.pt`, `selection_labels.csv`, `test_hidden_states.pt`, `test_labels.csv` (plus legacy `val_*` aliases for compatibility)
- `outputs/<model_id>/03_appraisal_structure/` — baseline_metrics.csv, appraisal_zscore_by_emotion.csv, cluster_emotion_mapping.csv, appraisal_zscore_heatmap.pdf
- `outputs/<model_id>/04_appraisal_in_circuit/` — SUMMARY.md (from phase2_summary), isolate_directions_report.csv (optional)
- `outputs/<model_id>/05_steering/` — steering_benchmark.csv, steering_curves.csv, steering_benchmark.pdf
- `outputs/<model_id>/05_generation_behavior/` — raw generated responses, latent readout CSVs, lightweight behavior-score CSVs, summary.md
- `outputs/<model_id>/05_mental_health_steering/` — mental health steering scores, dose-response tables/plots, summary.md
- `outputs/<model_id>/06_synthesis/` — summary figures and tables (from synthesis script)

Phase 2 geometry and correlation are computed by `phase2_compute` (invoked automatically by phase2_summary when missing) and written per model to `04_appraisal_in_circuit/`.

## Scaling to other models

Use the same extraction_layers/locs and probe training interface (e.g. experiments.utils.training_utils). Set `model_id` when running each stage; synthesis can compare across models by reading multiple `outputs/<model_id>/` directories.

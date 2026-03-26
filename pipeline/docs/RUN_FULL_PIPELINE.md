# Full pipeline: preparation and command order

Run from the **repo root** (parent of `pipeline/`) so that `utils`, `prompt_manager`, `experiments/`, and `LLMs/` are on `PYTHONPATH`.

---

## 1. Preparation

### 1.1 Environment
```bash
# From repo root (emo_llm_pipeline_standalone or your project root)
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements_standalone.txt   # Windows
# source .venv/bin/activate && python -m pip install -r requirements_standalone.txt   # Linux/Mac
```

**Windows (PowerShell):**
```powershell
$env:PYTHONPATH = (Get-Location).Path
```

**Linux/Mac:**
```bash
export PYTHONPATH=.
```

### 1.2 Data (standalone setup)
- **`pipeline/input_data/emotion_appraisal_train_combined.csv`** must exist.  
  If you only have the pipeline folder, copy it from `experiments/datasets/` in the original repo (or use the export bundle).

### 1.3 Optional: pre-filled probes (skip full probe training)
If you already have probes from another run and want to skip **train_probes** (which is slow):
- Put **`probe_summary.csv`**, **`probe_manifest.json`**, and the matching **`binary_ova_probes_<model>_*.pt`** in **`pipeline/input_probes/binary_ova_probes/`**.

### 1.4 Sklearn version (if probe loading fails)
If you see `AttributeError: 'LogisticRegression' object has no attribute 'multi_class'`:
```bash
pip install --upgrade scikit-learn
```

---

## 2. Command order (full run)

Use the same `--model_id` in every command (default is `Llama3.2_1B`). Replace `Llama3.2_1B` if you use another model.

| Step | Command | What it does |
|------|---------|---------------|
| **0** | `.venv\\Scripts\\python -m pipeline.probe_training_robustness` | Build dataset variants + optional quick probes (1 layer/loc). Creates leakage-safe scenario-level train/selection splits used by **train_probes** if you don’t have them. Use `--no_quick_probes` to only build datasets. |
| **1** | `.venv\\Scripts\\python -m pipeline.train_probes` | **Full probe training** (emotion OVA) using the stock-HF hook backend. Writes to `outputs/<model_id>/01_probes/binary_ova_probes/`. Slow; needs GPU/RAM. |
| **1b** | `.venv\\Scripts\\python -m pipeline.train_appraisal_probes` | **Appraisal regression probes** (Ridge per layer/loc) trained on the train split and validated on the selection split. Writes `01_probes/appraisal_regression_probes.pt`. |
| **2** | `.venv\\Scripts\\python -m pipeline.circuit_evidence` | Circuit evidence with `k` selected on the selection split and metrics reported on the held-out test split. Creates `selection_hidden_states.pt` and `test_hidden_states.pt` in 02_circuit. Omit `--skip_extract` on first run. |
| **3** | `.venv\\Scripts\\python -m pipeline.phase1_circuits` | Build `circuits.json` from saved circuit-site selection. |
| **4** | `python -m pipeline.appraisal_structure` | Appraisal z-scores, clustering, heatmap, and **label_coupling/** (pairwise overlap dashboards; see `pipeline/stage_03_appraisal_structure/docs/APPRAISAL_LABEL_COUPLING.md`). Uses 02_circuit cache if available. |
| **5** | `python -m pipeline.phase2_summary` | Computes Phase 2 geometry/correlation per model (if missing) and writes SUMMARY.md into 04_appraisal_in_circuit. |
| **5a** | (optional) `python -m pipeline.phase2_isolate_directions --layer 10 --dimension pleasantness` | Isolate appraisal directions. |
| **6** | `python -m pipeline.steering_benchmark` | Steering benchmark (appraisal vs emotion). |
| **7** | `python -m pipeline.synthesis` | Collects outputs into 06_synthesis (figures, SUMMARY). |

---

## 3. One-shot copy-paste (default model)

From repo root, with `PYTHONPATH` set:

```bash
.venv\Scripts\python -m pipeline.probe_training_robustness
.venv\Scripts\python -m pipeline.train_probes
.venv\Scripts\python -m pipeline.train_appraisal_probes
.venv\Scripts\python -m pipeline.circuit_evidence
.venv\Scripts\python -m pipeline.phase1_circuits
.venv\Scripts\python -m pipeline.appraisal_structure
.venv\Scripts\python -m pipeline.phase2_summary
.venv\Scripts\python -m pipeline.steering_benchmark
.venv\Scripts\python -m pipeline.synthesis
```

Optional flags you may use:
- `.venv\Scripts\python -m pipeline.probe_training_robustness --no_quick_probes` — only build datasets, no quick probes.
- `.venv\Scripts\python -m pipeline.train_probes --datasets_dir path/to/ova/csvs` — use a specific OVA dataset dir.
- `.venv\Scripts\python -m pipeline.train_appraisal_probes --no_balance` — disable emotion-balanced sampling; `--max_samples N` — cap samples (default: no cap).
- `.venv\Scripts\python -m pipeline.circuit_evidence --skip_extract` — reuse existing selection/test hidden states.
- `.venv\Scripts\python -m pipeline.steering_benchmark --source anger --target joy` — change source/target emotions.

---

## 4. Output locations (per model)

- **01_probe_robustness/** — Dataset variants + optional quick probe comparison.
- **01_probes/binary_ova_probes/** — Full probes (after **train_probes**).
- **02_circuit/** — `circuit_top_k_selection.json`, `circuits.json`, circuit_evidence outputs, `selection_hidden_states.pt`, `test_hidden_states.pt`, and legacy `val_*` aliases.
- **03_appraisal_structure/** — Appraisal z-scores, heatmap, baseline metrics, and **label_coupling/** (2×2 figures + metrics for configured dimension pairs).
- **04_appraisal_in_circuit/** — Phase 2 SUMMARY, optional isolate_directions report.
- **05_steering/** — steering_benchmark.csv, steering_curves.csv, figures.
- **06_synthesis/** — Combined summary and figures.

---

## 5. Multi-model runs and VRAM filtering

To run the pipeline for **multiple models** and only those that fit your GPU:

- **By VRAM:** Set `PIPELINE_MAX_VRAM_GB` (e.g. `8`) or pass `--max_vram_gb 8`. The runner will run every model in the registry whose `estimated_vram_gb` is at most that value.
- **By list:** Pass `--model_ids Llama3.2_1B Gemma2_2B` to run only those models (no VRAM check).

From repo root:

```bash
# Run all models that fit in 8 GB VRAM
.venv\Scripts\python -m pipeline.run_pipeline_for_models --max_vram_gb 8

# Run specific models only
.venv\Scripts\python -m pipeline.run_pipeline_for_models --model_ids Llama3.2_1B Gemma2_2B
```

Optional flags: `--skip_probe_robustness`, `--skip_phase2_summary`, `--skip_appraisal_probes`, `--circuit_skip_extract`, `--no_aggregate_synthesis`. **Resume/skip:** by default, models that already have `06_synthesis/SUMMARY.md` are skipped, and incomplete models resume from the first missing step. Use **`--overwrite`** to run every model from step 0 (redo everything). Use `--no_skip_complete` or `--no_resume` for finer control. Extraction layers and locations are chosen per model from `pipeline/model_config.py` (see `MODEL_ARCH_REGISTRY`).

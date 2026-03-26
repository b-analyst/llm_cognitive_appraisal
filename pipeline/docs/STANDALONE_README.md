# Running the pipeline standalone (outside the main repo)

You can move the `pipeline` folder to a new repo and run it there **if** you also bring the data and Python dependencies.

## Option A: Full standalone bundle (recommended)

From the **original emo-llm repo root**, run:

```bash
python pipeline/export_standalone.py
```

This creates `emo_llm_pipeline_standalone/` with:

- This pipeline folder (with `input_data/` and `input_probes/` filled)
- `utils.py`, `prompt_manager.py`
- `experiments/utils/` (data_utils, training_utils, intervention_utils)
- `LLMs/`
- `requirements_standalone.txt` and `STANDALONE_README.md`

Then move or copy that whole folder to your new repo (or use it as the new repo). In that folder:

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements_standalone.txt   # Windows
# source .venv/bin/activate && python -m pip install -r requirements_standalone.txt   # Linux/Mac
set PYTHONPATH=.   # or export PYTHONPATH=. on Linux/Mac
.venv\Scripts\python -m pipeline.phase1_circuits
```

## Option B: Only copy the pipeline folder

If you only copy the `pipeline` folder (no export script run):

1. **Create and fill inputs inside the folder:**
   - `pipeline/input_data/emotion_appraisal_train_combined.csv` — copy from `experiments/datasets/`
   - `pipeline/input_probes/binary_ova_probes/` — copy from `experiments/outputs/combined_dataset_probe_training_v2/binary_ova_probes/` (probe_summary.csv and all .pt files)
   - `pipeline/input_probes/appraisal_regression_probes.pt` — copy from `experiments/outputs/combined_dataset_probe_training_v2/`
   - `pipeline/input_probes/appraisal_probe_validation_detail.csv` — same dir if you use it

2. **You still need the rest of the repo** (or the same layout) for imports:
   - Repo root must have: `utils.py`, `prompt_manager.py`, `experiments/utils/` (data_utils, training_utils, intervention_utils), `LLMs/`
   - Run with that repo root on `PYTHONPATH` and run the pipeline from there.

So “drag only the pipeline folder” works for **paths** (config will use `input_data` and `input_probes` when present), but **Python imports** still need the parent repo layout unless you use Option A.

## Config behavior

- If `pipeline/input_data/` exists → `COMBINED_CSV` and data paths point there.
- If `pipeline/input_probes/` exists → probe paths and Phase 2 paths point there.
- Otherwise, paths fall back to `REPO_ROOT/experiments/...` (when run inside the full repo).

## Dependency note

- The pipeline now defaults to **stock Hugging Face models plus PyTorch hooks** for the active experiment path.
- The vendored files under `LLMs/` are kept for historical reference and optional legacy use, but the default runtime no longer depends on them.
- Use the pinned `requirements_standalone.txt` inside a local `.venv`; do **not** rely on a machine-wide Python environment.

# Emotion circuit & appraisal pipeline (standalone)

This folder is a self-contained copy of the pipeline and its dependencies. You can use it as a new repo or move it anywhere.

## Setup

```bash
cd <this_folder>   # the folder containing STANDALONE_README.md
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements_standalone.txt
```

## Run

From this directory, set the Python path so imports find `utils`, `experiments`, `LLMs`:

**Windows (cmd):**
```cmd
set PYTHONPATH=%CD%
python -m pipeline.phase1_circuits
```

**Windows (PowerShell):**
```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m pipeline.phase1_circuits
```

**Linux/Mac:**
```bash
export PYTHONPATH=.
python -m pipeline.phase1_circuits
```

## Data and probes

- `pipeline/input_data/` — combined CSV (emotion + appraisal). Filled by export script.
- `pipeline/input_probes/` — probe_summary, binary_ova_probes, appraisal_regression_probes.pt. Filled by export script.

If you add new data or probes, put them in these folders so the pipeline uses them (see `pipeline/config.py`).

## CLI (supported interface)

Run stages with `python -m pipeline.<module>` (see `README.md`, `docs/RUNBOOK.md`). For a full multi-stage run:

```powershell
python -m pipeline.run_pipeline_for_models --model_ids YourModelId
```

## Pipeline stages

See `pipeline/docs/README_PIPELINE.md`. Typical order:

0. `python -m pipeline.probe_training_robustness` (optional)
1. `python -m pipeline.phase1_circuits`
2. `python -m pipeline.circuit_evidence`
3. `python -m pipeline.appraisal_structure`
4. Run pipeline; Phase 2 (appraisal in circuit) is computed per model at runtime when you run `phase2_summary`.
5. `python -m pipeline.steering_benchmark`
6. `python -m pipeline.synthesis`

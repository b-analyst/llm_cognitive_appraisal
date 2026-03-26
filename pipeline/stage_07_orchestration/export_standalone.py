"""
Export a standalone runnable bundle of the pipeline so you can move it to a new repo.

Run from the emo-llm repo root:
  python pipeline/export_standalone.py [output_dir]

Creates output_dir (default: emo_llm_pipeline_standalone) with:
  - pipeline/ (all scripts + config; input_data/ and input_probes/ populated)
  - utils.py, prompt_manager.py
  - experiments/utils/ (data_utils, training_utils, intervention_utils)
  - LLMs/
  - requirements_standalone.txt (pip install -r this)
  - STANDALONE_README.md

Then cd into output_dir, install deps, and run e.g.:
  set PYTHONPATH=.  (or export PYTHONPATH=.)
  python -m pipeline.phase1_circuits
"""
from pathlib import Path
import shutil
import sys

def main():
    pipeline_dir = Path(__file__).resolve().parent.parent  # .../pipeline (orchestration/ parent)
    repo_root = pipeline_dir.parent
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else repo_root / "emo_llm_pipeline_standalone"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Paths in source repo
    experiments_utils = repo_root / "experiments" / "utils"
    llms_dir = repo_root / "LLMs"
    combined_csv = repo_root / "experiments" / "datasets" / "emotion_appraisal_train_combined.csv"
    v2_dir = repo_root / "experiments" / "outputs" / "combined_dataset_probe_training_v2"
    v2_probes = v2_dir / "binary_ova_probes"
    phase2_dir = repo_root / "experiments" / "outputs" / "phase_2_circuit_appraisal"
    v3_dir = repo_root / "experiments" / "outputs" / "combined_dataset_probe_training_v3"

    def copy_file(src: Path, dst: Path):
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return True
        return False

    def copy_dir(src: Path, dst: Path, ignore=None):
        if src.exists():
            shutil.copytree(src, dst, ignore=ignore or (lambda d, names: [n for n in names if n in ("__pycache__", ".pyc", "outputs", ".git")]), dirs_exist_ok=True)
            return True
        return False

    print("Exporting standalone bundle to:", out_dir)

    # 1) Pipeline folder (excluding outputs and existing input_*)
    out_pipeline = out_dir / "pipeline"
    out_pipeline.mkdir(parents=True, exist_ok=True)
    for item in pipeline_dir.iterdir():
        if item.name in ("outputs", "input_data", "input_probes", "__pycache__"):
            continue
        if item.is_file():
            shutil.copy2(item, out_pipeline / item.name)
        elif item.is_dir() and not item.name.startswith("."):
            shutil.copytree(item, out_pipeline / item.name, ignore=lambda d, names: [n for n in names if n in ("__pycache__", ".pyc", "outputs")], dirs_exist_ok=True)

    # 2) Root deps
    copy_file(repo_root / "utils.py", out_dir / "utils.py")
    copy_file(repo_root / "prompt_manager.py", out_dir / "prompt_manager.py")

    # 3) experiments/utils (only what pipeline needs)
    out_exp_utils = out_dir / "experiments" / "utils"
    out_exp_utils.mkdir(parents=True, exist_ok=True)
    for name in ("data_utils.py", "training_utils.py", "intervention_utils.py"):
        copy_file(experiments_utils / name, out_exp_utils / name)
    (out_dir / "experiments").mkdir(parents=True, exist_ok=True)
    init_exp = out_dir / "experiments" / "__init__.py"
    init_exp.write_text("", encoding="utf-8")
    init_utils = out_exp_utils / "__init__.py"
    init_utils.write_text("", encoding="utf-8")

    # 4) LLMs
    if llms_dir.exists():
        copy_dir(llms_dir, out_dir / "LLMs", ignore=lambda d, names: [n for n in names if n == "__pycache__"])

    # 5) Pipeline input_data
    in_data = out_pipeline / "input_data"
    in_data.mkdir(parents=True, exist_ok=True)
    if combined_csv.exists():
        shutil.copy2(combined_csv, in_data / "emotion_appraisal_train_combined.csv")
        print("  Copied combined CSV to pipeline/input_data/")
    else:
        print("  WARNING: combined CSV not found at", combined_csv)

    # 6) Pipeline input_probes
    in_probes = out_pipeline / "input_probes"
    in_probes.mkdir(parents=True, exist_ok=True)
    if v2_probes.exists():
        copy_dir(v2_probes, in_probes / "binary_ova_probes", ignore=lambda d, names: [n for n in names if n == "__pycache__"])
        print("  Copied binary_ova_probes to pipeline/input_probes/")
    if (v2_dir / "appraisal_regression_probes.pt").exists():
        copy_file(v2_dir / "appraisal_regression_probes.pt", in_probes / "appraisal_regression_probes.pt")
    if (v2_probes / "probe_summary.csv").exists():
        copy_file(v2_probes / "probe_summary.csv", in_probes / "binary_ova_probes" / "probe_summary.csv")
    if (v2_dir / "appraisal_probe_validation_detail.csv").exists():
        copy_file(v2_dir / "appraisal_probe_validation_detail.csv", in_probes / "appraisal_probe_validation_detail.csv")
    if v3_dir.exists():
        (in_probes / "v3").mkdir(parents=True, exist_ok=True)
        for f in ("layers_ranked_emotion.csv", "topk_performance_by_k.csv"):
            copy_file(v3_dir / f, in_probes / "v3" / f)
    if phase2_dir.exists():
        copy_dir(phase2_dir, in_probes / "phase_2_circuit_appraisal", ignore=lambda d, names: [n for n in names if n == "__pycache__"])
        print("  Copied phase_2 outputs to pipeline/input_probes/phase_2_circuit_appraisal/")

    # 7) requirements and README (use captured deps from venv if available)
    req = out_dir / "requirements_standalone.txt"
    pinned = pipeline_dir / "requirements_runtime_pinned.txt"
    captured = pipeline_dir / "requirements_captured.txt"
    if pinned.exists():
        shutil.copy2(pinned, req)
        print("  Used pipeline/requirements_runtime_pinned.txt for requirements_standalone.txt")
    elif captured.exists():
        shutil.copy2(captured, req)
        print("  Used pipeline/requirements_captured.txt for requirements_standalone.txt")
    else:
        req.write_text("""# Install in a venv then run pipeline with PYTHONPATH=.
# For pinned versions, run from your venv: python pipeline/capture_requirements.py
torch>=2.0
transformers
scikit-learn
pandas
numpy
tqdm
matplotlib
seaborn
""", encoding="utf-8")

    readme = out_dir / "STANDALONE_README.md"
    readme.write_text("""# Emotion circuit & appraisal pipeline (standalone)

This folder is a self-contained copy of the pipeline and its dependencies. You can use it as a new repo or move it anywhere.

## Setup

```bash
cd <this_folder>   # the folder containing STANDALONE_README.md
python -m venv .venv
.venv\\Scripts\\activate   # Windows
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

## Pipeline stages

See `pipeline/docs/README_PIPELINE.md`. Typical order:

0. `python -m pipeline.probe_training_robustness` (optional)
1. `python -m pipeline.phase1_circuits`
2. `python -m pipeline.circuit_evidence`
3. `python -m pipeline.appraisal_structure`
4. Run Phase 2 notebook in original repo if needed, then `python -m pipeline.phase2_summary`
5. `python -m pipeline.steering_benchmark`
6. `python -m pipeline.synthesis`
""", encoding="utf-8")

    print("Done. Standalone bundle at:", out_dir)
    print("Next: cd to that folder, install requirements_standalone.txt, set PYTHONPATH=., then run pipeline stages.")


if __name__ == "__main__":
    main()

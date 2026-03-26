# Emotion, appraisal, and circuit analysis pipeline

Research code to probe how LLMs represent **emotion** and **appraisal** dimensions internally, identify multi-site **circuits**, run **interventions** (steering / ablation), and synthesize results. The supported interface is the **CLI** (`python -m pipeline.…`).

## Quick start

```bash
python -m venv .venv
# Windows:
.venv\Scripts\python -m pip install -r requirements_standalone.txt
set PYTHONPATH=.
# Linux/macOS:
# source .venv/bin/activate && pip install -r requirements_standalone.txt
# export PYTHONPATH=.
```

**Run the full pipeline for one or more models:**

```bash
python -m pipeline.run_pipeline_for_models --model_ids Llama3.2_1B
```

Optional editable install (see `pyproject.toml`): `pip install -e .` — you can still use `PYTHONPATH=.` if you prefer. Dev deps: `pip install -e ".[dev]"` for pytest.

## Environment note

`requirements_standalone.txt` pins **NumPy < 2** so matplotlib/scipy wheels stay ABI-compatible. If you upgrade NumPy to 2.x, upgrade those packages together.

## Documentation

| Doc | Purpose |
|-----|---------|
| [docs/START_HERE.md](docs/START_HERE.md) | Orientation and reading order |
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | Commands, flags, checkpoints |
| [docs/PIPELINE_TLDR_RESEARCH_PARTNER.md](docs/PIPELINE_TLDR_RESEARCH_PARTNER.md) | High-level stage summary for collaborators |
| [pipeline/README.md](pipeline/README.md) | **Package layout** (`core/`, `shims/`, `stage_*`, `docs/`, `notebooks/`) |
| [pipeline/STRUCTURE.md](pipeline/STRUCTURE.md) | **Code review map** — root shims vs `stage_*` implementation |
| [pipeline/docs/PIPELINE.md](pipeline/docs/PIPELINE.md) | End-to-end pipeline narrative |
| [pipeline/docs/README_PIPELINE.md](pipeline/docs/README_PIPELINE.md) | Stage order and output layout |

## CPU threads

Optional: set `PIPELINE_CPU_THREADS` (e.g. `4`) before running to cap BLAS/OpenMP/torch intra-op threads on shared machines. See [docs/RUNBOOK.md](docs/RUNBOOK.md).

## GitHub / what to commit

`.gitignore` excludes **virtualenvs** (`.venv/`), **run outputs** (`pipeline/outputs/`), **PyTorch / probe weights** (`*.pt`, `*.pth`, etc.), **local probe dumps** (`pipeline/input_probes/*` except `.gitkeep`), **captured pip freezes**, caches, and `.cursor/`. **Tracked by default:** source, `pyproject.toml`, `requirements_standalone.txt`, and **`pipeline/input_data/`** CSVs/JSON (tens of MB total—fine for normal GitHub). If any single data file grows past ~50–100 MB, switch that file to [Git LFS](https://git-lfs.github.com/).

## Tests

```bash
set PYTHONPATH=.
.venv\Scripts\python -m pytest pipeline/tests
```

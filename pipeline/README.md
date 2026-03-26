# Pipeline package layout

Start here for **navigation**. For a reviewer-oriented tour (what each top-level `.py` file does, where real code lives), read **[`STRUCTURE.md`](STRUCTURE.md)** first.

## Where things live

| Path | Role |
|------|------|
| **[`STRUCTURE.md`](STRUCTURE.md)** | Map of folders, shims vs implementation, table of CLI → `stage_*` targets. |
| [`core/`](core/) | `config`, `model_config`, `research_contracts`, `logutil`, `runtime_env`, `logger`. |
| [`shims/`](shims/) | `redirect.bind_shim` — shared logic for root-level CLI modules. |
| [`stage_01_probes/`](stage_01_probes/README.md) … [`stage_07_orchestration/`](stage_07_orchestration/README.md) | **All substantive pipeline code**, one stage per folder. |
| [`stage_06_benchmarks/utils/`](stage_06_benchmarks/utils/) | Benchmark-only helpers (probe scoring, runtime readout, adaptive targets). |
| [`docs/`](docs/README.md) | In-package docs: master pipeline map, stage order, run checklist, standalone notes. |
| [`notebooks/`](notebooks/README.md) | Jupyter notebooks (run from repo root with `PYTHONPATH` set). |
| [`tests/`](tests/) | Pytest. |

## Root `*.py` files

The package root still contains many **small** files so `python -m pipeline.<module>` and `import pipeline.config` keep working:

- **Shims** delegate to `stage_*` via [`shims/redirect.py`](shims/redirect.py) (see each file’s module docstring for the target).
- **Aliases** (`config.py`, `model_config.py`, …) re-export [`core/`](core/).

## Imports for new code

```python
from pipeline.core.config import OUTPUTS_ROOT, DEFAULT_MODEL_ID
from pipeline.stage_02_circuit.circuit_evidence import load_circuit_top_k_selection
```

## Related docs

- [`docs/PIPELINE.md`](docs/PIPELINE.md) — master pipeline map and stage order.
- [`docs/README_PIPELINE.md`](docs/README_PIPELINE.md) — short stage order and CLI list.
- [`../docs/PIPELINE_MAP.md`](../docs/PIPELINE_MAP.md) — detailed narrative per stage.
- [`../docs/RUNBOOK.md`](../docs/RUNBOOK.md) — how to run.

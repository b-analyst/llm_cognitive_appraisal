"""CLI shim: ``python -m pipeline.analyze_probe_grid_ablation`` delegates to ``pipeline.stage_01_probes.analyze_probe_grid_ablation`` (see ``pipeline/shims/redirect.py``)."""
from __future__ import annotations

from pipeline.shims.redirect import bind_shim

_TARGET = "pipeline.stage_01_probes.analyze_probe_grid_ablation"
_run = bind_shim(_TARGET, globals())

if __name__ == "__main__":
    _run()

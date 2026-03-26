"""CLI shim: ``python -m pipeline.phase2_compute`` delegates to ``pipeline.stage_04_appraisal_in_circuit.phase2_compute`` (see ``pipeline/shims/redirect.py``)."""
from __future__ import annotations

from pipeline.shims.redirect import bind_shim

_TARGET = "pipeline.stage_04_appraisal_in_circuit.phase2_compute"
_run = bind_shim(_TARGET, globals())

if __name__ == "__main__":
    _run()

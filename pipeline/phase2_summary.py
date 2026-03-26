"""CLI shim: ``python -m pipeline.phase2_summary`` delegates to ``pipeline.stage_04_appraisal_in_circuit.phase2_summary`` (see ``pipeline/shims/redirect.py``)."""
from __future__ import annotations

from pipeline.shims.redirect import bind_shim

_TARGET = "pipeline.stage_04_appraisal_in_circuit.phase2_summary"
_run = bind_shim(_TARGET, globals())

if __name__ == "__main__":
    _run()

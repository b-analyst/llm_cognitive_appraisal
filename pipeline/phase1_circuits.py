"""CLI shim: ``python -m pipeline.phase1_circuits`` delegates to ``pipeline.stage_02_circuit.phase1_circuits`` (see ``pipeline/shims/redirect.py``)."""
from __future__ import annotations

from pipeline.shims.redirect import bind_shim

_TARGET = "pipeline.stage_02_circuit.phase1_circuits"
_run = bind_shim(_TARGET, globals())

if __name__ == "__main__":
    _run()

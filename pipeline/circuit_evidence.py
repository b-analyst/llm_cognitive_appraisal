"""CLI shim: ``python -m pipeline.circuit_evidence`` delegates to ``pipeline.stage_02_circuit.circuit_evidence`` (see ``pipeline/shims/redirect.py``)."""
from __future__ import annotations

from pipeline.shims.redirect import bind_shim

_TARGET = "pipeline.stage_02_circuit.circuit_evidence"
_run = bind_shim(_TARGET, globals())

if __name__ == "__main__":
    _run()

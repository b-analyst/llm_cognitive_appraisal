"""CLI shim: ``python -m pipeline.capture_requirements`` delegates to ``pipeline.stage_07_orchestration.capture_requirements`` (see ``pipeline/shims/redirect.py``)."""
from __future__ import annotations

from pipeline.shims.redirect import bind_shim

_TARGET = "pipeline.stage_07_orchestration.capture_requirements"
_run = bind_shim(_TARGET, globals())

if __name__ == "__main__":
    _run()

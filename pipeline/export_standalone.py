"""CLI shim: ``python -m pipeline.export_standalone`` delegates to ``pipeline.stage_07_orchestration.export_standalone`` (see ``pipeline/shims/redirect.py``)."""
from __future__ import annotations

from pipeline.shims.redirect import bind_shim

_TARGET = "pipeline.stage_07_orchestration.export_standalone"
_run = bind_shim(_TARGET, globals())

if __name__ == "__main__":
    _run()

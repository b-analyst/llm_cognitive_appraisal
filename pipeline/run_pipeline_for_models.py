"""CLI shim: ``python -m pipeline.run_pipeline_for_models`` delegates to ``pipeline.stage_07_orchestration.runner`` (see ``pipeline/shims/redirect.py``)."""
from __future__ import annotations

from pipeline.shims.redirect import bind_shim

_TARGET = "pipeline.stage_07_orchestration.runner"
_run = bind_shim(_TARGET, globals())

if __name__ == "__main__":
    _run()

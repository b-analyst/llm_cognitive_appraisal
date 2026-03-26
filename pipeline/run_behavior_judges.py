"""CLI shim: ``python -m pipeline.run_behavior_judges`` delegates to ``pipeline.stage_06_benchmarks.run_behavior_judges`` (see ``pipeline/shims/redirect.py``)."""
from __future__ import annotations

from pipeline.shims.redirect import bind_shim

_TARGET = "pipeline.stage_06_benchmarks.run_behavior_judges"
_run = bind_shim(_TARGET, globals())

if __name__ == "__main__":
    _run()

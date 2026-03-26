"""CLI shim: ``python -m pipeline.generation_behavior_benchmark`` delegates to ``pipeline.stage_06_benchmarks.generation_behavior_benchmark`` (see ``pipeline/shims/redirect.py``)."""
from __future__ import annotations

from pipeline.shims.redirect import bind_shim

_TARGET = "pipeline.stage_06_benchmarks.generation_behavior_benchmark"
_run = bind_shim(_TARGET, globals())

if __name__ == "__main__":
    _run()

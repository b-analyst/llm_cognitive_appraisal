"""CLI shim: ``python -m pipeline.baseline_probe_steering_study`` delegates to ``pipeline.stage_06_benchmarks.baseline_probe_steering_study`` (see ``pipeline/shims/redirect.py``)."""
from __future__ import annotations

from pipeline.shims.redirect import bind_shim

_TARGET = "pipeline.stage_06_benchmarks.baseline_probe_steering_study"
_run = bind_shim(_TARGET, globals())

if __name__ == "__main__":
    _run()

"""CLI shim: ``python -m pipeline.probe_training_robustness`` delegates to ``pipeline.stage_01_probes.probe_training_robustness`` (see ``pipeline/shims/redirect.py``)."""
from __future__ import annotations

from pipeline.shims.redirect import bind_shim

_TARGET = "pipeline.stage_01_probes.probe_training_robustness"
_run = bind_shim(_TARGET, globals())

if __name__ == "__main__":
    _run()

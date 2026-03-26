"""CLI shim: ``python -m pipeline.train_appraisal_probes`` delegates to ``pipeline.stage_01_probes.train_appraisal_probes`` (see ``pipeline/shims/redirect.py``)."""
from __future__ import annotations

from pipeline.shims.redirect import bind_shim

_TARGET = "pipeline.stage_01_probes.train_appraisal_probes"
_run = bind_shim(_TARGET, globals())

if __name__ == "__main__":
    _run()

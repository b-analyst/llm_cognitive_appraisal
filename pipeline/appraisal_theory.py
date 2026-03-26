"""CLI shim: ``python -m pipeline.appraisal_theory`` delegates to ``pipeline.stage_05_appraisal_theory.appraisal_theory`` (see ``pipeline/shims/redirect.py``)."""
from __future__ import annotations

from pipeline.shims.redirect import bind_shim

_TARGET = "pipeline.stage_05_appraisal_theory.appraisal_theory"
_run = bind_shim(_TARGET, globals())

if __name__ == "__main__":
    _run()

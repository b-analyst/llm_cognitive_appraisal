"""CLI shim: ``python -m pipeline.theory_analysis`` delegates to ``pipeline.stage_05_appraisal_theory.theory_analysis`` (see ``pipeline/shims/redirect.py``)."""
from __future__ import annotations

from pipeline.shims.redirect import bind_shim

_TARGET = "pipeline.stage_05_appraisal_theory.theory_analysis"
_run = bind_shim(_TARGET, globals())

if __name__ == "__main__":
    _run()

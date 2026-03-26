"""CLI shim: ``python -m pipeline.appraisal_structure`` delegates to ``pipeline.stage_03_appraisal_structure.appraisal_structure`` (see ``pipeline/shims/redirect.py``)."""
from __future__ import annotations

from pipeline.shims.redirect import bind_shim

_TARGET = "pipeline.stage_03_appraisal_structure.appraisal_structure"
_run = bind_shim(_TARGET, globals())

if __name__ == "__main__":
    _run()

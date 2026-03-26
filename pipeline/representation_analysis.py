"""CLI shim: ``python -m pipeline.representation_analysis`` delegates to ``pipeline.stage_03_appraisal_structure.representation_analysis`` (see ``pipeline/shims/redirect.py``)."""
from __future__ import annotations

from pipeline.shims.redirect import bind_shim

_TARGET = "pipeline.stage_03_appraisal_structure.representation_analysis"
_run = bind_shim(_TARGET, globals())

if __name__ == "__main__":
    _run()

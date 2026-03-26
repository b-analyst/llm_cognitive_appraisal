"""Smoke imports for repo root on PYTHONPATH."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def test_import_pipeline_config():
    import pipeline.config  # noqa: F401


def test_import_run_pipeline_for_models():
    import pipeline.run_pipeline_for_models  # noqa: F401

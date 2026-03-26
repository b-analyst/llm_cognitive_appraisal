"""Tests for synthesis aggregation helpers (no GPU)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture()
def patched_outputs_root(tmp_path, monkeypatch):
    monkeypatch.setattr("pipeline.core.config.OUTPUTS_ROOT", tmp_path)
    monkeypatch.setattr("pipeline.stage_07_orchestration.runner.OUTPUTS_ROOT", tmp_path)
    return tmp_path


def test_completed_model_ids_only_complete(patched_outputs_root):
    from pipeline.run_pipeline_for_models import (
        PIPELINE_STEPS,
        SYNTHESIS_STEP_INDEX,
        _aggregate_synthesis_model_ids,
        _completed_model_ids,
    )

    rel = PIPELINE_STEPS[SYNTHESIS_STEP_INDEX][3]
    (patched_outputs_root / "Done" / Path(rel).parent).mkdir(parents=True, exist_ok=True)
    (patched_outputs_root / "Done" / rel).write_text("# ok\n", encoding="utf-8")
    (patched_outputs_root / "Half" / "01_probes").mkdir(parents=True, exist_ok=True)

    assert _completed_model_ids() == ["Done"]
    assert _aggregate_synthesis_model_ids(False) == ["Done"]
    assert set(_aggregate_synthesis_model_ids(True)) == {"Done", "Half"}


def test_aggregate_all_includes_incomplete(patched_outputs_root):
    from pipeline.run_pipeline_for_models import (
        PIPELINE_STEPS,
        SYNTHESIS_STEP_INDEX,
        _aggregate_synthesis_model_ids,
    )

    rel = PIPELINE_STEPS[SYNTHESIS_STEP_INDEX][3]
    (patched_outputs_root / "A" / Path(rel).parent).mkdir(parents=True, exist_ok=True)
    (patched_outputs_root / "A" / rel).write_text("x", encoding="utf-8")
    (patched_outputs_root / "B" / "02_circuit").mkdir(parents=True, exist_ok=True)

    assert _aggregate_synthesis_model_ids(True) == ["A", "B"]
    assert _aggregate_synthesis_model_ids(False) == ["A"]

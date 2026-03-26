"""Unit tests for runtime emotion readout / z-source resolution (stdlib only)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.stage_06_benchmarks.utils.runtime_emotion_readout import (
    appraisal_full_source_label,
    generation_appraisal_source_mode,
    resolve_appraisal_source_emotion,
    zscore_row_key,
)


class _FakeIndex:
    """Minimal pandas.Index-like object for zscore_row_key tests."""

    def __init__(self, keys: list):
        self._keys = list(keys)

    def __contains__(self, item) -> bool:
        return item in self._keys

    def __iter__(self):
        return iter(self._keys)


class _FakeZ:
    def __init__(self, keys: list):
        self.index = _FakeIndex(keys)
        self.empty = False


class TestRuntimeEmotionReadout(unittest.TestCase):
    def test_zscore_row_key_case_insensitive(self):
        z = _FakeZ(["Joy", "anger"])
        self.assertEqual(zscore_row_key("joy", z), "Joy")
        self.assertEqual(zscore_row_key("Anger", z), "anger")

    def test_resolve_runtime_then_fallback(self):
        z = _FakeZ(["calm", "storm"])
        s, n = resolve_appraisal_source_emotion("storm", z, "calm", "calm")
        self.assertEqual((s, n), ("storm", "runtime"))
        s2, n2 = resolve_appraisal_source_emotion(None, z, "missing", "calm")
        self.assertEqual((s2, n2), ("calm", "fallback_secondary"))

    def test_generation_appraisal_source_mode(self):
        self.assertEqual(generation_appraisal_source_mode(False, "runtime"), "csv")
        self.assertEqual(generation_appraisal_source_mode(True, "runtime"), "runtime")
        self.assertEqual(generation_appraisal_source_mode(True, "fallback_condition"), "fallback")

    def test_appraisal_full_source_label(self):
        self.assertEqual(appraisal_full_source_label("runtime"), "runtime")
        self.assertEqual(appraisal_full_source_label("fallback_condition"), "condition")
        self.assertEqual(appraisal_full_source_label("fallback_secondary"), "fallback")


if __name__ == "__main__":
    unittest.main()

"""Tests for adaptive contrastive target selection."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.stage_06_benchmarks.utils.adaptive_appraisal_target import (  # noqa: E402
    resolve_emotion_list_key,
    select_contrastive_target_emotion,
)


class TestSelectContrastiveTargetEmotion(unittest.TestCase):
    def test_picks_max_l2_distance_tie_lexicographic(self):
        df = pd.DataFrame(
            {
                "urgency": [0.0, 2.0, 0.0],
                "control": [0.0, 0.0, 2.0],
            },
            index=["fear", "anger", "joy"],
        )
        emotions_list = ["fear", "anger", "joy"]
        t, d = select_contrastive_target_emotion(
            "fear", df, emotions_list, ["urgency", "control"], static_fallback_target="relief"
        )
        # L2 fear->anger = 2, fear->joy = 2 -> tie -> lexicographic min: anger < joy
        self.assertEqual(t, "anger")
        self.assertEqual(d["adaptive_target_fallback_reason"], "")
        self.assertAlmostEqual(d["adaptive_target_distance"], 2.0)

    def test_furthest_winner_clear(self):
        df = pd.DataFrame(
            {
                "a": [0.0, 1.0, 10.0],
            },
            index=["x", "y", "z"],
        )
        emotions_list = ["x", "y", "z"]
        t, d = select_contrastive_target_emotion("x", df, emotions_list, ["a"], "y")
        self.assertEqual(t, "z")
        self.assertAlmostEqual(d["adaptive_target_distance"], 10.0)

    def test_source_not_in_zscore_falls_back(self):
        df = pd.DataFrame({"a": [1.0]}, index=["only"])
        t, d = select_contrastive_target_emotion("missing", df, ["only"], ["a"], "relief")
        self.assertEqual(t, "relief")
        self.assertEqual(d["adaptive_target_fallback_reason"], "source_not_in_zscore")

    def test_min_distance_triggers_fallback(self):
        df = pd.DataFrame(
            {"a": [0.0, 0.01]},
            index=["s", "t"],
        )
        emotions_list = ["s", "t"]
        t, d = select_contrastive_target_emotion(
            "s",
            df,
            emotions_list,
            ["a"],
            static_fallback_target="t",
            min_distance=1.0,
        )
        self.assertEqual(t, "t")
        self.assertEqual(d["adaptive_target_fallback_reason"], "below_min_distance")

    def test_resolve_emotion_list_key(self):
        self.assertEqual(resolve_emotion_list_key("JOY", ["fear", "joy"]), "joy")


if __name__ == "__main__":
    unittest.main()

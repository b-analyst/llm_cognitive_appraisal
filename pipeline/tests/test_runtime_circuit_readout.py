"""Tests for circuit-linear-mean runtime emotion readout (no model)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.stage_06_benchmarks.utils.runtime_emotion_readout import (  # noqa: E402
    circuit_linear_mean_fused_scores,
    circuit_sigmoid_mean_fused_scores,
    readout_ranked_emotions_auto,
    _sigmoid_stable_np,
)
from pipeline.stage_06_benchmarks.utils.probe_latent_scoring import emotion_linear_logits_at_site  # noqa: E402


def _fake_probe_rec(weight: float, bias: float = 0.0, dim: int = 4) -> dict:
    w = np.zeros(dim, dtype=np.float64)
    w[0] = weight
    return {"weights": w, "bias": float(bias)}


class TestCircuitLinearMeanFusedScores(unittest.TestCase):
    def setUp(self):
        self.emotions_list = ["a", "b"]
        self.token_key = -1
        self.extraction_layers = [0]
        self.extraction_locs = [0]
        self.hidden_size = 4
        # batch, n_layer, n_loc, token, dim
        self.hs_np = np.zeros((1, 1, 1, 1, self.hidden_size), dtype=np.float64)
        self.hs_np[0, 0, 0, 0, :] = np.array([1.0, 0.0, 0.0, 0.0])

    def test_mean_over_two_sites_matches_manual(self):
        """Emotion 'a' has two sites; fused score = mean of linear score for 'a' at each site."""
        probes = {
            0: {
                0: {
                    self.token_key: {
                        "a": _fake_probe_rec(2.0, bias=0.0, dim=self.hidden_size),
                        "b": _fake_probe_rec(0.0, bias=0.0, dim=self.hidden_size),
                    }
                },
                1: {
                    self.token_key: {
                        "a": _fake_probe_rec(4.0, bias=0.0, dim=self.hidden_size),
                        "b": _fake_probe_rec(0.0, bias=0.0, dim=self.hidden_size),
                    }
                },
            }
        }
        extraction_locs = [0, 1]
        hs_np = np.zeros((1, 1, 2, 1, self.hidden_size), dtype=np.float64)
        hs_np[0, 0, :, 0, :] = np.array([1.0, 0.0, 0.0, 0.0])
        topk = {"a": [(0, 0), (0, 1)], "b": [(0, 0)]}
        scores, union_n = circuit_linear_mean_fused_scores(
            hs_np,
            probes,
            topk,
            self.extraction_layers,
            extraction_locs,
            0,
            self.emotions_list,
            self.token_key,
        )
        self.assertEqual(union_n, 2)
        # At (0,0): a=2, b=0; at (0,1): a=4, b=0
        h = hs_np[0, 0, 0, 0].copy()
        v0 = emotion_linear_logits_at_site(probes, h, 0, 0, self.token_key, self.emotions_list)
        h1 = hs_np[0, 0, 1, 0].copy()
        v1 = emotion_linear_logits_at_site(probes, h1, 0, 1, self.token_key, self.emotions_list)
        self.assertAlmostEqual(scores[0], (float(v0[0]) + float(v1[0])) / 2.0, places=5)
        self.assertAlmostEqual(scores[1], float(v0[1]), places=5)

    def test_empty_circuit_is_nan(self):
        probes = {
            0: {
                0: {
                    self.token_key: {
                        "a": _fake_probe_rec(1.0, dim=self.hidden_size),
                        "b": _fake_probe_rec(1.0, dim=self.hidden_size),
                    }
                }
            }
        }
        topk = {"a": [], "b": [(0, 0)]}
        scores, _ = circuit_linear_mean_fused_scores(
            self.hs_np,
            probes,
            topk,
            self.extraction_layers,
            self.extraction_locs,
            0,
            self.emotions_list,
            self.token_key,
        )
        self.assertTrue(np.isnan(scores[0]))
        self.assertFalse(np.isnan(scores[1]))

    def test_union_eval_once_two_emotions_share_site(self):
        """Two emotions share one site: only one linear eval."""
        probes = {
            0: {
                0: {
                    self.token_key: {
                        "a": _fake_probe_rec(10.0, dim=self.hidden_size),
                        "b": _fake_probe_rec(-5.0, dim=self.hidden_size),
                    }
                }
            }
        }
        topk = {"a": [(0, 0)], "b": [(0, 0)]}
        scores, union_n = circuit_linear_mean_fused_scores(
            self.hs_np,
            probes,
            topk,
            self.extraction_layers,
            self.extraction_locs,
            0,
            self.emotions_list,
            self.token_key,
        )
        self.assertEqual(union_n, 1)
        self.assertAlmostEqual(scores[0], 10.0, places=5)
        self.assertAlmostEqual(scores[1], -5.0, places=5)


class TestCircuitSigmoidMeanFusedScores(unittest.TestCase):
    def test_mean_sigmoid_over_two_sites(self):
        """Emotion 'a' at two sites: fused = mean(sigmoid(linear_a)) at each site."""
        emotions_list = ["a", "b"]
        token_key = -1
        hidden_size = 4
        hs_np = np.zeros((1, 1, 2, 1, hidden_size), dtype=np.float64)
        hs_np[0, 0, :, 0, :] = np.array([1.0, 0.0, 0.0, 0.0])
        probes = {
            0: {
                0: {
                    token_key: {
                        "a": _fake_probe_rec(0.0, bias=0.0, dim=hidden_size),
                        "b": _fake_probe_rec(0.0, bias=0.0, dim=hidden_size),
                    }
                },
                1: {
                    token_key: {
                        "a": _fake_probe_rec(0.0, bias=0.0, dim=hidden_size),
                        "b": _fake_probe_rec(0.0, bias=0.0, dim=hidden_size),
                    }
                },
            }
        }
        # Same hidden; different linear score for 'a' via weight on dim0
        probes[0][0][token_key]["a"] = _fake_probe_rec(2.0, bias=0.0, dim=hidden_size)
        probes[0][1][token_key]["a"] = _fake_probe_rec(4.0, bias=0.0, dim=hidden_size)
        extraction_layers = [0]
        extraction_locs = [0, 1]
        topk = {"a": [(0, 0), (0, 1)], "b": [(0, 0)]}
        scores, union_n = circuit_sigmoid_mean_fused_scores(
            hs_np,
            probes,
            topk,
            extraction_layers,
            extraction_locs,
            0,
            emotions_list,
            token_key,
        )
        self.assertEqual(union_n, 2)
        s0 = float(_sigmoid_stable_np(np.array([2.0]))[0])
        s1 = float(_sigmoid_stable_np(np.array([4.0]))[0])
        self.assertAlmostEqual(scores[0], (s0 + s1) / 2.0, places=5)
        self.assertAlmostEqual(scores[1], float(_sigmoid_stable_np(np.array([0.0]))[0]), places=5)


class TestReadoutRankedEmotionsAuto(unittest.TestCase):
    def test_auto_single_vs_circuit(self):
        emotions_list = ["a", "b"]
        token_key = -1
        dim = 3
        hs = np.zeros((1, 1, 1, 1, dim), dtype=np.float64)
        hs[0, 0, 0, 0, :] = [1.0, 0.0, 0.0]
        probes = {
            0: {
                0: {
                    token_key: {
                        "a": {"weights": np.array([3.0, 0.0, 0.0]), "bias": 0.0},
                        "b": {"weights": np.array([1.0, 0.0, 0.0]), "bias": 0.0},
                    }
                }
            }
        }
        ext_layers = [0]
        ext_locs = [0]
        ra, sa = readout_ranked_emotions_auto(
            "single_site",
            hs,
            probes,
            ext_layers,
            ext_locs,
            0,
            emotions_list,
            token_key,
            top_k=5,
            min_margin=0.0,
            min_top1_logit=None,
            read_layer=0,
            read_loc=0,
        )
        self.assertEqual(ra[0], "a")
        topk_pe = {"a": [(0, 0)], "b": [(0, 0)]}
        rb, sb = readout_ranked_emotions_auto(
            "circuit_linear_mean",
            hs,
            probes,
            ext_layers,
            ext_locs,
            0,
            emotions_list,
            token_key,
            top_k=5,
            min_margin=0.0,
            min_top1_logit=None,
            topk_per_emotion=topk_pe,
        )
        self.assertEqual(rb[0], "a")
        # single_site ranks on sigmoid(linear); circuit_linear_mean uses raw linear margins
        self.assertAlmostEqual(sb[0], 3.0, places=5)
        exp_sig = float(_sigmoid_stable_np(np.array([3.0], dtype=np.float64))[0])
        self.assertAlmostEqual(sa[0], exp_sig, places=5)

    def test_auto_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            readout_ranked_emotions_auto(
                "nope",
                np.zeros((1, 1, 1, 1, 2)),
                {},
                [0],
                [0],
                0,
                ["a"],
                -1,
                top_k=1,
                min_margin=0.0,
                read_layer=0,
                read_loc=0,
            )


if __name__ == "__main__":
    unittest.main()

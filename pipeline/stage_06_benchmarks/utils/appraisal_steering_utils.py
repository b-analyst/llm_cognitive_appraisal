"""
Build per-circuit-site steering vectors from a weighted appraisal profile (ridge probe directions).

Shared by mental_health_steering_benchmark and generation_behavior_benchmark.
"""
from __future__ import annotations

import numpy as np

from .probe_latent_scoring import resolve_probe_layer_loc
from pipeline.core.research_contracts import appraisal_probe_direction_raw
def build_appraisal_profile_vectors(
    target_profile: dict[str, float],
    target_pairs: list[tuple[int, int]],
    appraisal_probes: dict,
    hidden_size: int,
    token_key: int,
) -> list[np.ndarray]:
    """
    For each (layer, loc) in target_pairs, sum weight * ridge direction for each dim in target_profile.

    Empty or missing probes yield zero vectors for that site.
    """
    out: list[np.ndarray] = []
    if not target_profile:
        return [np.zeros(hidden_size, dtype=np.float32) for _ in target_pairs]
    for layer, loc in target_pairs:
        vec = np.zeros(hidden_size, dtype=np.float32)
        for dim, weight in target_profile.items():
            if dim not in appraisal_probes:
                continue
            L, lc = resolve_probe_layer_loc(appraisal_probes[dim], layer, loc)
            if L is None or lc is None:
                continue
            try:
                rec = appraisal_probes[dim][L][lc][token_key]
            except (KeyError, TypeError):
                continue
            direction = appraisal_probe_direction_raw(rec, hidden_size=hidden_size)
            if direction is not None:
                vec += float(weight) * direction.astype(np.float32)
        out.append(vec)
    return out

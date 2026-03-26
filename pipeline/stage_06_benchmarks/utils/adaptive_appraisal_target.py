"""
Select appraisal steering target emotion by maximum z-profile contrast to the source row.

Used when ADAPTIVE_APPRAISAL_TARGET_ENABLED: candidates are emotions that appear in both
probe summary order (emotions_list) and appraisal_zscore_by_emotion.csv; source row is
excluded. Falls back to static taxonomy/config target when selection is invalid.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .runtime_emotion_readout import zscore_row_key
def select_contrastive_target_emotion(
    source_label: str | None,
    zscore_df: pd.DataFrame | None,
    emotions_list: list,
    common_dims: list[str],
    static_fallback_target: str | None,
    metric: str = "l2",
    min_distance: float | None = None,
    max_scores_json_len: int = 4000,
) -> tuple[str | None, dict[str, Any]]:
    """
    Pick target emotion maximizing distance between z-score profiles vs source.

    Args:
        source_label: Resolved appraisal source (e.g. runtime rank-1 or CSV); mapped into zscore index.
        zscore_df: appraisal_zscore_by_emotion (index = emotion keys).
        emotions_list: Probe summary emotion order (defines candidates and probe keys).
        common_dims: Appraisal column names to align (e.g. COMMON_APPRAISAL subset present in df).
        static_fallback_target: CSV/taxonomy target when adaptive selection fails.
        metric: "l2" only in v1.
        min_distance: If set and best distance < this, use fallback.
        max_scores_json_len: Truncate adaptive_target_scores_json for CSV safety.

    Returns:
        (target_emotion, diagnostics) — target is a key from emotions_list when possible, else fallback.
    """
    diag: dict[str, Any] = {
        "adaptive_target_metric": metric,
        "adaptive_target_distance": None,
        "adaptive_target_fallback_reason": "",
        "adaptive_target_scores_json": "",
        "source_zscore_key": None,
    }
    if zscore_df is None or zscore_df.empty:
        diag["adaptive_target_fallback_reason"] = "missing_zscore_df"
        return static_fallback_target, diag

    source_key = zscore_row_key(source_label, zscore_df)
    diag["source_zscore_key"] = source_key
    if source_key is None:
        diag["adaptive_target_fallback_reason"] = "source_not_in_zscore"
        return static_fallback_target, diag

    dims = [d for d in common_dims if d in zscore_df.columns]
    if not dims:
        diag["adaptive_target_fallback_reason"] = "no_common_appraisal_columns"
        return static_fallback_target, diag

    try:
        s_vec = zscore_df.loc[source_key, dims].astype(np.float64).values.ravel()
    except (KeyError, TypeError):
        diag["adaptive_target_fallback_reason"] = "source_row_missing_columns"
        return static_fallback_target, diag

    if np.any(~np.isfinite(s_vec)):
        diag["adaptive_target_fallback_reason"] = "source_non_finite_z"
        return static_fallback_target, diag

    # Deterministic candidate order: sorted emotion labels from emotions_list
    candidates_sorted = sorted(
        {str(e).strip() for e in emotions_list if str(e).strip()},
        key=lambda x: x.lower(),
    )
    scores: dict[str, float] = {}
    for em in candidates_sorted:
        zk = zscore_row_key(em, zscore_df)
        if zk is None:
            continue
        if str(zk).strip().lower() == str(source_key).strip().lower():
            continue
        try:
            v = zscore_df.loc[zk, dims].astype(np.float64).values.ravel()
        except (KeyError, TypeError):
            continue
        if v.shape != s_vec.shape or np.any(~np.isfinite(v)):
            continue
        dist = float(np.linalg.norm(v - s_vec)) if metric == "l2" else float(np.linalg.norm(v - s_vec))
        scores[em] = dist

    if not scores:
        diag["adaptive_target_fallback_reason"] = "no_contrastive_candidates"
        return static_fallback_target, diag

    max_d = max(scores.values())
    tied = [e for e, d in scores.items() if np.isclose(d, max_d)]
    best_em = sorted(tied, key=str.lower)[0]
    best_dist = scores[best_em]

    diag["adaptive_target_distance"] = best_dist
    try:
        sj = json.dumps({k: float(scores[k]) for k in sorted(scores.keys(), key=str.lower)})
        if len(sj) > max_scores_json_len:
            sj = sj[: max_scores_json_len - 3] + "..."
        diag["adaptive_target_scores_json"] = sj
    except (TypeError, ValueError):
        diag["adaptive_target_scores_json"] = ""

    if min_distance is not None and best_dist < float(min_distance):
        diag["adaptive_target_fallback_reason"] = "below_min_distance"
        return static_fallback_target, diag

    diag["adaptive_target_fallback_reason"] = ""
    return best_em, diag


def resolve_emotion_list_key(chosen: str | None, emotions_list: list) -> str | None:
    """Map chosen label to the emotions_list entry (case-insensitive)."""
    if chosen is None:
        return None
    c = str(chosen).strip()
    if not c:
        return None
    cl = c.lower()
    for e in emotions_list:
        if str(e).strip().lower() == cl:
            return str(e)
    return chosen

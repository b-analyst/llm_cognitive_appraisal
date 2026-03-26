"""
Shared helpers for linear OvA emotion scores and ridge appraisal readouts at a single site.

Used by baseline probe steering study and kept minimal to avoid duplicating
steering_benchmark / appraisal_theory logic.
"""
from __future__ import annotations

import numpy as np

from pipeline.core.config import COMMON_APPRAISAL


def resolve_probe_layer_loc(nested: dict, layer: int, loc: int) -> tuple:
    """Support int or str keys in saved .pt probe bundles."""
    L = layer if layer in nested else (str(layer) if str(layer) in nested else None)
    if L is None:
        return None, None
    sub = nested[L]
    loc_sub = loc if loc in sub else (str(loc) if str(loc) in sub else None)
    return L, loc_sub


def _apply_emotion_scaler(rec: dict, X: np.ndarray) -> np.ndarray:
    """X: (n_samples, dim) — apply stored scaler like steering_benchmark._probe_logits_at."""
    scaler = rec.get("scaler")
    if scaler is not None and hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        mean = np.asarray(scaler.mean_, dtype=np.float64)
        scale = np.asarray(scaler.scale_, dtype=np.float64)
        denom = np.where(np.abs(scale) < 1e-12, 1.0, scale)
        return (X - mean) / denom
    if scaler is not None and hasattr(scaler, "transform"):
        return scaler.transform(X)
    return X


def emotion_linear_logits_at_site(
    probes: dict,
    hidden_vec: np.ndarray,
    layer: int,
    loc: int,
    token_key: int,
    emotions_list: list[str],
) -> np.ndarray:
    """
    Raw linear scores (pre-sigmoid) for each emotion OvA probe at one (layer, loc).

    hidden_vec: (hidden_dim,) in raw activation space (same convention as hooks).
    """
    h = np.asarray(hidden_vec, dtype=np.float64).reshape(1, -1)
    out = np.full(len(emotions_list), np.nan, dtype=np.float64)
    L, lc = resolve_probe_layer_loc(probes, layer, loc)
    if L is None or lc is None:
        return out
    try:
        bucket = probes[L][lc][token_key]
    except (KeyError, TypeError):
        return out
    for e_idx, em in enumerate(emotions_list):
        if em not in bucket:
            continue
        rec = bucket[em]
        if not isinstance(rec, dict) or "error" in rec:
            continue
        Xe = _apply_emotion_scaler(rec, h)
        w = rec.get("weights", None)
        b = rec.get("bias", None)
        if w is None or b is None:
            clf = rec.get("classifier", None)
            if clf is None or not hasattr(clf, "coef_"):
                continue
            w = clf.coef_.ravel()
            b = float(np.ravel(clf.intercept_)[0])
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        b = float(b)
        if w.shape[0] != Xe.shape[1]:
            continue
        raw = Xe @ w + b
        out[e_idx] = float(np.asarray(raw, dtype=np.float64).ravel()[0])
    return out


def appraisal_ridge_values_at_site(
    appraisal_probes: dict,
    hidden_vec: np.ndarray,
    layer: int,
    loc: int,
    token_key: int,
    dims: list[str] | None = None,
) -> dict[str, float]:
    """Predict appraisal dimension values (ridge) for selected dims; missing keys skipped."""
    if dims is None:
        dims = [d for d in COMMON_APPRAISAL if d in appraisal_probes]
    h = np.asarray(hidden_vec, dtype=np.float64).reshape(1, -1)
    pred: dict[str, float] = {}
    for dim in dims:
        if dim not in appraisal_probes:
            continue
        L, lc = resolve_probe_layer_loc(appraisal_probes[dim], layer, loc)
        if L is None or lc is None:
            continue
        try:
            rec = appraisal_probes[dim][L][lc][token_key]
        except (KeyError, TypeError):
            continue
        if not rec or not rec.get("ridge") or not rec.get("scaler"):
            continue
        ridge = rec["ridge"]
        sc = rec["scaler"]
        try:
            X = sc.transform(h)
            y = ridge.predict(X)
            pred[dim] = float(np.ravel(y)[0])
        except Exception:
            continue
    return pred


def top_m_appraisal_dims_for_emotion(
    emotion: str,
    zscore_df,
    m: int,
    common_dims: list[str] | None = None,
) -> list[str]:
    """
    Freeze-style subset: top-m dimensions by |z-score| for this emotion in appraisal_zscore_by_emotion.
    zscore_df: DataFrame indexed by emotion (lowercase labels).
    """
    common_dims = common_dims or list(COMMON_APPRAISAL)
    if zscore_df is None or emotion not in zscore_df.index:
        return []
    row = zscore_df.loc[emotion]
    scored = []
    for d in common_dims:
        if d not in row.index:
            continue
        try:
            v = float(row[d])
        except (TypeError, ValueError):
            continue
        if np.isnan(v):
            continue
        scored.append((d, abs(v)))
    scored.sort(key=lambda x: -x[1])
    return [d for d, _ in scored[: max(0, m)]]


def rank_emotions_top_k_margin(
    logits: np.ndarray,
    emotions_list: list[str],
    top_k: int,
    min_margin: float,
    min_top1_logit: float | None = None,
) -> tuple[list[str], list[float]]:
    """
    Multilabel-style ranking: include emotions in order until either k reached or
    gap to next falls below min_margin (plan: top-k + margin).
    """
    valid = [(emotions_list[i], float(logits[i])) for i in range(len(emotions_list)) if not np.isnan(logits[i])]
    if not valid:
        return [], []
    valid.sort(key=lambda x: -x[1])
    if min_top1_logit is not None and valid[0][1] < min_top1_logit:
        return [], []
    chosen = [valid[0]]
    for j in range(1, len(valid)):
        if len(chosen) >= top_k:
            break
        if valid[j - 1][1] - valid[j][1] < min_margin:
            break
        chosen.append(valid[j])
    return [x[0] for x in chosen], [x[1] for x in chosen]

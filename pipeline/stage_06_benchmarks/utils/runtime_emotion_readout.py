"""
Runtime OvA emotion readout (single-site or circuit-fused).

Shared by baseline probe steering study, generation-behavior benchmark, and
mental-health steering (adaptive z-contrast source for appraisal steering).

**Primary circuit convention** (aligned with ``circuit_evidence`` / ``_circuit_logits``):
per-site **sigmoid(linear decision)**, then **mean across** each emotion's circuit sites.

**Auxiliary** ``circuit_linear_mean``: mean **pre-sigmoid** linear scores (margin space).

See ``docs/RUNTIME_READOUT.md``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from .probe_latent_scoring import emotion_linear_logits_at_site, rank_emotions_top_k_margin
def _sigmoid_stable_np(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid (matches ``steering_benchmark._sigmoid_stable``)."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def normalize_runtime_readout_mode(mode: str | None) -> str:
    return (mode or "single_site").strip().lower().replace("-", "_")


def runtime_readout_mode_uses_topk_per_emotion(mode: str | None) -> bool:
    """True if this mode needs ``topk_per_emotion`` (any circuit fusion variant)."""
    m = normalize_runtime_readout_mode(mode)
    return m in (
        "circuit_sigmoid_mean",
        "circuit_evidence",
        "circuit_linear_mean",
        "circuit",
        "circuit_mean",
    )


def runtime_readout_mode_is_sigmoid_circuit(mode: str | None) -> bool:
    """True when primary fusion is mean(sigmoid) per emotion — same family as ``circuit_evidence``."""
    m = normalize_runtime_readout_mode(mode)
    return m in ("circuit_sigmoid_mean", "circuit_evidence", "circuit", "circuit_mean")


def linear_circuit_auxiliary_csv_columns(
    hs_np: np.ndarray,
    probes: dict,
    topk_per_emotion: dict[str, list],
    extraction_layers: list,
    extraction_locs: list,
    token_idx: int,
    emotions_list: list[str],
    token_key: int,
    *,
    primary_mode: str,
    log_aux: bool,
    log_rank_json: bool,
    log_full_spectrum: bool,
    top_k: int,
    min_margin: float,
    min_top1_logit: float | None,
    max_json_len: int,
) -> tuple[str, str]:
    """
    When primary readout is **sigmoid-mean circuit** fusion, optionally log the **linear-mean**
    circuit as separate CSV columns (margin space; see ``docs/RUNTIME_READOUT.md``).

    Returns ``(runtime_linear_circuit_rank1_emotion, runtime_linear_circuit_ranked_top_k_json)``.
    """
    if not log_aux or not runtime_readout_mode_is_sigmoid_circuit(primary_mode):
        return "", ""
    if not topk_per_emotion or not any(len(v or []) > 0 for v in topk_per_emotion.values()):
        return "", ""
    lr, lscores = readout_ranked_emotions_circuit_linear_mean(
        hs_np,
        probes,
        topk_per_emotion,
        extraction_layers,
        extraction_locs,
        token_idx,
        emotions_list,
        token_key,
        top_k=top_k,
        min_margin=min_margin,
        min_top1_logit=min_top1_logit,
    )
    r1 = lr[0] if lr else ""
    j = ""
    if log_rank_json:
        fused_lin, u_lin = circuit_linear_mean_fused_scores(
            hs_np,
            probes,
            topk_per_emotion,
            extraction_layers,
            extraction_locs,
            token_idx,
            emotions_list,
            token_key,
        )
        spec_inner = ""
        if log_full_spectrum:
            spec_inner = format_circuit_fused_scores_json(
                fused_lin, emotions_list, max_len=max_json_len
            )
        payload = {
            "readout_role": "auxiliary_linear_circuit",
            "readout_mode": "circuit_linear_mean",
            "score_kind": "linear_mean_fused",
            "ranked_top_k": [{"emotion": n, "logit": float(s)} for n, s in zip(lr, lscores)],
            "union_n_sites": int(u_lin),
        }
        if spec_inner:
            payload["all_emotions_linear_scores_json"] = spec_inner
        j = json.dumps(payload)
    return r1, j


def runtime_readout_score_kind_label(mode: str | None) -> str:
    """Label for JSON ``score_kind`` (values in ``ranked_top_k`` use key ``logit`` for CSV back-compat)."""
    m = normalize_runtime_readout_mode(mode)
    if m in ("single_site", "singlesite"):
        return "sigmoid_single_site"
    if runtime_readout_mode_is_sigmoid_circuit(mode):
        return "sigmoid_mean_fused"
    if m == "circuit_linear_mean":
        return "linear_mean_fused"
    return "unknown"


def select_probe_readout_site(summary_df: Any) -> tuple[int, int]:
    """Mean test ROC-AUC across emotions per (layer, loc); exclude no-emotion."""
    import pandas as pd

    df = summary_df.copy()
    df["emotion"] = df["emotion"].astype(str)
    df = df[df["emotion"].str.lower() != "no-emotion"]
    if df.empty:
        raise ValueError("probe_summary has no usable emotion rows.")
    g = df.groupby(["layer", "loc"], as_index=False)["test_roc_auc"].mean()
    best = g.loc[g["test_roc_auc"].idxmax()]
    return int(best["layer"]), int(best["loc"])


def hidden_vector_at_site(
    hs: np.ndarray,
    extraction_layers: list,
    extraction_locs: list,
    layer: int,
    loc: int,
    token_idx: int = 0,
) -> np.ndarray:
    """Single-token hidden vector at (layer, loc) from a forward hidden-state array."""
    li = extraction_layers.index(layer)
    ci = extraction_locs.index(loc)
    x = hs[0, li, ci, token_idx, :]
    if hasattr(x, "cpu") and callable(getattr(x, "cpu", None)):
        x = x.cpu().numpy()
    return np.asarray(x, dtype=np.float64).ravel()


def zscore_row_key(emotion: str | None, zscore_df: Any | None) -> str | None:
    """Map a label to a row key in appraisal z-score table (exact or case-insensitive)."""
    if emotion is None or zscore_df is None or zscore_df.empty:
        return None
    e = str(emotion).strip()
    if not e:
        return None
    idx = zscore_df.index
    if e in idx:
        return str(e)
    el = e.lower()
    for k in idx:
        if str(k).lower() == el:
            return str(k)
    return None


def resolve_appraisal_source_emotion(
    rank1: str | None,
    zscore_df: Any | None,
    fallback_primary: str,
    fallback_secondary: str | None = None,
) -> tuple[str, str]:
    """
    Choose source emotion key for z-diff steering.

    Returns (emotion_key_used, resolution_note):
      - runtime: rank1 maps into zscore index
      - fallback_condition: used fallback_primary
      - fallback_secondary: used fallback_secondary (e.g. sadness)
      - missing: no zscore row found; returns best-effort string (primary, then secondary, then 'sadness')
    """
    if zscore_df is None or zscore_df.empty:
        for label, note in (
            (fallback_primary, "fallback_condition"),
            (fallback_secondary, "fallback_secondary"),
        ):
            if label and str(label).strip():
                return str(label), note
        return "sadness", "missing"

    k = zscore_row_key(rank1, zscore_df)
    if k is not None:
        return k, "runtime"

    k = zscore_row_key(fallback_primary, zscore_df)
    if k is not None:
        return k, "fallback_condition"

    if fallback_secondary is not None:
        k = zscore_row_key(fallback_secondary, zscore_df)
        if k is not None:
            return k, "fallback_secondary"

    if fallback_primary and str(fallback_primary).strip():
        return str(fallback_primary), "missing"
    if fallback_secondary and str(fallback_secondary).strip():
        return str(fallback_secondary), "missing"
    return "sadness", "missing"


def emotion_sigmoid_scores_at_site(
    probes: dict,
    hidden_vec: np.ndarray,
    layer: int,
    loc: int,
    token_key: int,
    emotions_list: list[str],
) -> np.ndarray:
    """Per-emotion **sigmoid(linear OvA)** at one (layer, loc); same convention as ``circuit_evidence``."""
    lin = np.asarray(
        emotion_linear_logits_at_site(
            probes, hidden_vec, layer, loc, token_key, emotions_list
        ),
        dtype=np.float64,
    )
    return _sigmoid_stable_np(lin)


def readout_ranked_emotions(
    hs_np: np.ndarray,
    probes: dict,
    read_layer: int,
    read_loc: int,
    extraction_layers: list,
    extraction_locs: list,
    token_idx: int,
    emotions_list: list[str],
    token_key: int,
    *,
    top_k: int,
    min_margin: float,
    min_top1_logit: float | None = None,
) -> tuple[list[str], list[float]]:
    """
    Hidden at read site -> OvA **sigmoid probabilities** per emotion (matches ``circuit_evidence``
    single-site matrix) -> margin-ranked emotion list.
    """
    h = hidden_vector_at_site(
        hs_np, extraction_layers, extraction_locs, read_layer, read_loc, token_idx=token_idx
    )
    probs = emotion_sigmoid_scores_at_site(
        probes, h, read_layer, read_loc, token_key, emotions_list
    )
    return rank_emotions_top_k_margin(
        probs, emotions_list, top_k=top_k, min_margin=min_margin, min_top1_logit=min_top1_logit
    )


def circuit_linear_mean_fused_scores(
    hs_np: np.ndarray,
    probes: dict,
    topk_per_emotion: dict[str, list],
    extraction_layers: list,
    extraction_locs: list,
    token_idx: int,
    emotions_list: list[str],
    token_key: int,
) -> tuple[np.ndarray, int]:
    """
    Fuse OvA probe outputs across the **circuit for each emotion** (multilabel-friendly scores).

    For **each** emotion label ``e`` independently:
    - Take the **pre-sigmoid linear** decision score from the **e-vs-rest** probe at every
      ``(layer, loc)`` in ``topk_per_emotion[e]``.
    - **Average those scores** → one scalar per emotion.

    So you still have **one score per emotion** (a multilabel vector over emotions). You are **not**
    mixing different emotions' probes: the mean is only over **spatial** sites for the **same** OvA
    head. (Argmax / margin-ranked top-k on this vector is an optional **multiclass** collapse.)

    Evaluates each unique ``(layer, loc)`` in the union of all emotions' circuits once for speed.
    Emotions with no valid finite scores at any site get NaN.
    """
    seen: set[tuple[int, int]] = set()
    ordered_sites: list[tuple[int, int]] = []
    layer_set = set(extraction_layers)
    loc_set = set(extraction_locs)
    for pairs in topk_per_emotion.values():
        for layer, loc in pairs or []:
            layer_i, loc_i = int(layer), int(loc)
            if layer_i not in layer_set or loc_i not in loc_set:
                continue
            t = (layer_i, loc_i)
            if t not in seen:
                seen.add(t)
                ordered_sites.append(t)

    site_logits: dict[tuple[int, int], np.ndarray] = {}
    for layer, loc in ordered_sites:
        h = hidden_vector_at_site(
            hs_np, extraction_layers, extraction_locs, layer, loc, token_idx=token_idx
        )
        site_logits[(layer, loc)] = emotion_linear_logits_at_site(
            probes, h, layer, loc, token_key, emotions_list
        )

    n_e = len(emotions_list)
    out = np.full(n_e, np.nan, dtype=np.float64)
    for e_idx, em in enumerate(emotions_list):
        pairs = topk_per_emotion.get(em, []) or []
        vals: list[float] = []
        for layer, loc in pairs:
            key = (int(layer), int(loc))
            vec = site_logits.get(key)
            if vec is None:
                continue
            v = vec[e_idx]
            if np.isfinite(v):
                vals.append(float(v))
        if vals:
            out[e_idx] = float(np.mean(vals))
    return out, len(ordered_sites)


def circuit_sigmoid_mean_fused_scores(
    hs_np: np.ndarray,
    probes: dict,
    topk_per_emotion: dict[str, list],
    extraction_layers: list,
    extraction_locs: list,
    token_idx: int,
    emotions_list: list[str],
    token_key: int,
) -> tuple[np.ndarray, int]:
    """
    **Circuit-evidence / steering fusion:** for each emotion ``e``, average **sigmoid(linear OvA)**
    at each ``(layer, loc)`` in ``topk_per_emotion[e]``.

    Matches ``circuit_evidence._per_emotion_fusion_logits`` and ``steering_benchmark._circuit_logits``
    (per-sample). Same site union as ``circuit_linear_mean_fused_scores``.
    """
    seen: set[tuple[int, int]] = set()
    ordered_sites: list[tuple[int, int]] = []
    layer_set = set(extraction_layers)
    loc_set = set(extraction_locs)
    for pairs in topk_per_emotion.values():
        for layer, loc in pairs or []:
            layer_i, loc_i = int(layer), int(loc)
            if layer_i not in layer_set or loc_i not in loc_set:
                continue
            t = (layer_i, loc_i)
            if t not in seen:
                seen.add(t)
                ordered_sites.append(t)

    site_probs: dict[tuple[int, int], np.ndarray] = {}
    for layer, loc in ordered_sites:
        h = hidden_vector_at_site(
            hs_np, extraction_layers, extraction_locs, layer, loc, token_idx=token_idx
        )
        lin = np.asarray(
            emotion_linear_logits_at_site(
                probes, h, layer, loc, token_key, emotions_list
            ),
            dtype=np.float64,
        )
        site_probs[(layer, loc)] = _sigmoid_stable_np(lin)

    n_e = len(emotions_list)
    out = np.full(n_e, np.nan, dtype=np.float64)
    for e_idx, em in enumerate(emotions_list):
        pairs = topk_per_emotion.get(em, []) or []
        vals: list[float] = []
        for layer, loc in pairs:
            key = (int(layer), int(loc))
            vec = site_probs.get(key)
            if vec is None:
                continue
            v = vec[e_idx]
            if np.isfinite(v):
                vals.append(float(v))
        if vals:
            out[e_idx] = float(np.mean(vals))
    return out, len(ordered_sites)


def readout_ranked_emotions_circuit_sigmoid_mean(
    hs_np: np.ndarray,
    probes: dict,
    topk_per_emotion: dict[str, list],
    extraction_layers: list,
    extraction_locs: list,
    token_idx: int,
    emotions_list: list[str],
    token_key: int,
    *,
    top_k: int,
    min_margin: float,
    min_top1_logit: float | None = None,
) -> tuple[list[str], list[float]]:
    """Margin-ranked emotions using circuit **sigmoid-mean** fusion (circuit_evidence convention)."""
    fused, _ = circuit_sigmoid_mean_fused_scores(
        hs_np,
        probes,
        topk_per_emotion,
        extraction_layers,
        extraction_locs,
        token_idx,
        emotions_list,
        token_key,
    )
    return rank_emotions_top_k_margin(
        fused, emotions_list, top_k=top_k, min_margin=min_margin, min_top1_logit=min_top1_logit
    )


def readout_ranked_emotions_circuit_linear_mean(
    hs_np: np.ndarray,
    probes: dict,
    topk_per_emotion: dict[str, list],
    extraction_layers: list,
    extraction_locs: list,
    token_idx: int,
    emotions_list: list[str],
    token_key: int,
    *,
    top_k: int,
    min_margin: float,
    min_top1_logit: float | None = None,
) -> tuple[list[str], list[float]]:
    """Margin-ranked emotions using circuit-linear-mean scores (same linear scale as single-site)."""
    fused, _ = circuit_linear_mean_fused_scores(
        hs_np,
        probes,
        topk_per_emotion,
        extraction_layers,
        extraction_locs,
        token_idx,
        emotions_list,
        token_key,
    )
    return rank_emotions_top_k_margin(
        fused, emotions_list, top_k=top_k, min_margin=min_margin, min_top1_logit=min_top1_logit
    )


def readout_ranked_emotions_auto(
    mode: str,
    hs_np: np.ndarray,
    probes: dict,
    extraction_layers: list,
    extraction_locs: list,
    token_idx: int,
    emotions_list: list[str],
    token_key: int,
    *,
    top_k: int,
    min_margin: float,
    min_top1_logit: float | None = None,
    read_layer: int | None = None,
    read_loc: int | None = None,
    topk_per_emotion: dict[str, list] | None = None,
) -> tuple[list[str], list[float]]:
    """
    Dispatch runtime emotion ranking.

    mode:
      - ``single_site``: read_layer, read_loc; **sigmoid(linear)** per emotion at that site
        (same as one row of ``circuit_evidence`` probe matrix).
      - ``circuit_sigmoid_mean`` / ``circuit_evidence`` / ``circuit`` / ``circuit_mean``:
        mean **sigmoid** per emotion over ``topk_per_emotion`` (default **primary** circuit readout).
      - ``circuit_linear_mean``: mean **pre-sigmoid linear** scores (auxiliary margin-space readout).
    """
    m = normalize_runtime_readout_mode(mode)
    if m in ("single_site", "singlesite"):
        if read_layer is None or read_loc is None:
            raise ValueError("read_layer and read_loc required for single_site mode")
        return readout_ranked_emotions(
            hs_np,
            probes,
            int(read_layer),
            int(read_loc),
            extraction_layers,
            extraction_locs,
            token_idx,
            emotions_list,
            token_key,
            top_k=top_k,
            min_margin=min_margin,
            min_top1_logit=min_top1_logit,
        )
    if m in ("circuit_sigmoid_mean", "circuit_evidence", "circuit", "circuit_mean"):
        if topk_per_emotion is None:
            raise ValueError("topk_per_emotion required for circuit sigmoid-mean mode")
        return readout_ranked_emotions_circuit_sigmoid_mean(
            hs_np,
            probes,
            topk_per_emotion,
            extraction_layers,
            extraction_locs,
            token_idx,
            emotions_list,
            token_key,
            top_k=top_k,
            min_margin=min_margin,
            min_top1_logit=min_top1_logit,
        )
    if m == "circuit_linear_mean":
        if topk_per_emotion is None:
            raise ValueError("topk_per_emotion required for circuit_linear_mean mode")
        return readout_ranked_emotions_circuit_linear_mean(
            hs_np,
            probes,
            topk_per_emotion,
            extraction_layers,
            extraction_locs,
            token_idx,
            emotions_list,
            token_key,
            top_k=top_k,
            min_margin=min_margin,
            min_top1_logit=min_top1_logit,
        )
    raise ValueError(f"Unknown RUNTIME_READOUT_EMOTION_MODE: {mode!r}")


def runtime_readout_full_spectrum_json(
    hs_np: np.ndarray,
    probes: dict,
    readout_mode: str,
    emotions_list: list[str],
    token_key: int,
    extraction_layers: list,
    extraction_locs: list,
    token_idx: int,
    topk_per_emotion: dict[str, list] | None,
    read_layer: int | None,
    read_loc: int | None,
    max_len: int = 8000,
) -> tuple[str, int]:
    """
    Optional all-emotion scores for logging (truncated JSON), matching the active readout mode.

    Returns (json_string, union_n_sites). union_n_sites is 1 for single_site mode.
    """
    m = normalize_runtime_readout_mode(readout_mode)
    if m in ("single_site", "singlesite"):
        if read_layer is None or read_loc is None:
            return "", 0
        h = hidden_vector_at_site(
            hs_np, extraction_layers, extraction_locs, int(read_layer), int(read_loc), token_idx=token_idx
        )
        lin = np.asarray(
            emotion_linear_logits_at_site(
                probes, h, int(read_layer), int(read_loc), token_key, emotions_list
            ),
            dtype=np.float64,
        )
        probs = _sigmoid_stable_np(lin)
        return format_circuit_fused_scores_json(probs, emotions_list, max_len=max_len), 1
    if m in ("circuit_sigmoid_mean", "circuit_evidence", "circuit", "circuit_mean"):
        if topk_per_emotion is None:
            return "", 0
        fused, union_n = circuit_sigmoid_mean_fused_scores(
            hs_np,
            probes,
            topk_per_emotion,
            extraction_layers,
            extraction_locs,
            token_idx,
            emotions_list,
            token_key,
        )
        return format_circuit_fused_scores_json(fused, emotions_list, max_len=max_len), int(union_n)
    if m == "circuit_linear_mean":
        if topk_per_emotion is None:
            return "", 0
        fused, union_n = circuit_linear_mean_fused_scores(
            hs_np,
            probes,
            topk_per_emotion,
            extraction_layers,
            extraction_locs,
            token_idx,
            emotions_list,
            token_key,
        )
        return format_circuit_fused_scores_json(fused, emotions_list, max_len=max_len), int(union_n)
    return "", 0


def format_circuit_fused_scores_json(
    scores: np.ndarray,
    emotions_list: list[str],
    max_len: int = 8000,
) -> str:
    """
    Sorted list of {emotion, score} for all emotions with finite fused scores; truncate for CSV.
    """
    items: list[dict[str, Any]] = []
    for i, em in enumerate(emotions_list):
        if i >= len(scores):
            continue
        s = scores[i]
        if not np.isfinite(s):
            continue
        items.append({"emotion": str(em), "score": float(s)})
    items.sort(key=lambda x: -x["score"])
    try:
        out = json.dumps(items)
    except (TypeError, ValueError):
        return ""
    if len(out) > max_len:
        out = out[: max(0, max_len - 3)] + "..."
    return out


def appraisal_full_source_label(resolution_note: str) -> str:
    """CSV-friendly label for mental-health appraisal_full_source column."""
    if resolution_note == "runtime":
        return "runtime"
    if resolution_note == "fallback_condition":
        return "condition"
    if resolution_note in ("fallback_secondary", "missing"):
        return "fallback"
    return "fallback"


@dataclass
class RuntimeReadoutResult:
    """Per-prompt runtime readout metadata (generation behavior)."""

    rank1_emotion: str | None
    ranked_names: list[str]
    ranked_scores: list[float]
    source_eff: str
    appraisal_source_mode: str  # csv | runtime | fallback
    skip_appraisal_and_combined: bool
    skip_reason: str
    read_layer: int
    read_loc: int


def generation_appraisal_source_mode(
    used_runtime_readout: bool, resolution_note: str
) -> str:
    if not used_runtime_readout:
        return "csv"
    if resolution_note == "runtime":
        return "runtime"
    return "fallback"


if __name__ == "__main__":
    import pandas as pd

    # Self-check: z-score key resolution
    z = pd.DataFrame({"a": [1.0]}, index=["Joy"])
    assert zscore_row_key("joy", z) == "Joy"
    s, n = resolve_appraisal_source_emotion("joy", z, "Anger", "Sadness")
    assert s == "Joy" and n == "runtime"
    s2, n2 = resolve_appraisal_source_emotion(None, z, "Joy", "Sadness")
    assert s2 == "Joy" and n2 == "fallback_condition"
    print("runtime_emotion_readout self-check OK")

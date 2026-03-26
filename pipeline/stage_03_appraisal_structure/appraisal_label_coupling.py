"""
Appraisal dimension coupling: label-space overlap (conditional KDEs, bivariate density)
and optional probe-score coupling on the test split when circuit caches include row indices.

Writes to pipeline/outputs/<model_id>/03_appraisal_structure/label_coupling/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import gaussian_kde, ks_2samp, mannwhitneyu, spearmanr

from pipeline.core.config import (
    REPO_ROOT,
    COMBINED_CSV,
    COMMON_APPRAISAL,
    DEFAULT_MODEL_ID,
    SELECTION_SPLIT,
    FINAL_TEST_SPLIT,
    RANDOM_STATE,
    APPRAISAL_LABEL_COUPLING_PAIRS,
    APPRAISAL_LABEL_COUPLING_SPLIT_METHOD,
    APPRAISAL_PROBE_COUPLING_LAYER,
    APPRAISAL_PROBE_COUPLING_LOC,
    get_appraisal_structure_dir,
    get_circuit_dir,
    get_probe_paths,
    EXTRACTION_TOKENS,
)
from pipeline.core.model_config import get_extraction_layers, get_extraction_locs
from pipeline.core.research_contracts import (
    canonicalize_combined_dataset,
    split_combined_dataset,
    dataset_fingerprint,
    ensure_manifest_model_match,
    resolve_text_columns,
)
FIGURE_DPI = 175
COLOR_LOW = "#0173B2"  # teal-blue (colorblind-friendly)
COLOR_HIGH = "#DE8F05"  # orange
MIN_GROUP_N = 8


def _backfill_test_labels_row_indices(circuit_dir: Path) -> bool:
    """
    Legacy caches wrote test_labels.csv with only `emotion`. Recompute dataset_row_idx from the
    canonical combined CSV + split protocol (same as circuit_evidence) when row counts and
    emotions match — avoids a full hidden-state re-extract.
    """
    lbl_path = circuit_dir / "test_labels.csv"
    if not lbl_path.exists():
        return False
    lbl_df = pd.read_csv(lbl_path)
    if "dataset_row_idx" in lbl_df.columns:
        return True
    try:
        raw = pd.read_csv(COMBINED_CSV)
        df = canonicalize_combined_dataset(raw)
        base_text_col, prompted_col = resolve_text_columns(df)
        text_col = prompted_col or base_text_col
        df = df.dropna(subset=["emotion", text_col])
        bundle = split_combined_dataset(
            df,
            selection_frac=SELECTION_SPLIT,
            test_frac=FINAL_TEST_SPLIT,
            random_state=RANDOM_STATE,
        )
        test_df = bundle["test"]
    except Exception as e:
        print(f"[appraisal_label_coupling] Backfill failed (could not rebuild test split): {e}")
        return False
    exp_em = test_df["emotion"].astype(str).str.strip().str.lower().tolist()
    got_em = lbl_df["emotion"].astype(str).str.strip().str.lower().tolist()
    if len(exp_em) != len(got_em):
        print(
            f"[appraisal_label_coupling] Backfill skipped: canonical test n={len(exp_em)} "
            f"vs test_labels.csv n={len(got_em)} (dataset or cache changed?)."
        )
        return False
    if exp_em != got_em:
        print(
            "[appraisal_label_coupling] Backfill skipped: emotion sequence does not match "
            "canonical test split (re-run circuit_evidence extract to refresh)."
        )
        return False
    lbl_df["dataset_row_idx"] = [int(i) for i in test_df.index.tolist()]
    lbl_df.to_csv(lbl_path, index=False)
    val_path = circuit_dir / "val_labels.csv"
    if val_path.exists():
        try:
            vdf = pd.read_csv(val_path)
            if len(vdf) == len(lbl_df) and list(vdf["emotion"].astype(str)) == list(
                lbl_df["emotion"].astype(str)
            ):
                vdf["dataset_row_idx"] = lbl_df["dataset_row_idx"].values
                vdf.to_csv(val_path, index=False)
        except Exception:
            pass
    print(
        "[appraisal_label_coupling] Backfilled `dataset_row_idx` into 02_circuit test_labels.csv "
        "(matches current combined CSV + split protocol)."
    )
    return True


def _ensure_repo_path() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def _figure_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": FIGURE_DPI,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "font.family": "sans-serif",
            "axes.grid": True,
            # y-only grid: use axes.grid.axis (there is no rc key "grid.axis").
            "axes.grid.axis": "y",
            "grid.alpha": 0.25,
        }
    )


def _cohens_d_pooled(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
    n1, n2 = len(a), len(b)
    pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / max(n1 + n2 - 2, 1))
    if pooled < 1e-12:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def _cohen_interpretation(d: float) -> str:
    ad = abs(d)
    if np.isnan(ad):
        return "n/a"
    if ad < 0.2:
        return "negligible separation"
    if ad < 0.5:
        return "small separation"
    if ad < 0.8:
        return "medium separation"
    return "large separation"


def _partial_corr(a: np.ndarray, b: np.ndarray, z: np.ndarray) -> float:
    """Pearson partial correlation of a,b controlling for columns of z (n, k)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    n = len(a)
    if n < z.shape[1] + 3:
        return float("nan")
    ones = np.ones((n, 1))
    Z = np.hstack([ones, z])
    try:
        ra = a - Z @ np.linalg.lstsq(Z, a, rcond=None)[0]
        rb = b - Z @ np.linalg.lstsq(Z, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return float("nan")
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _split_binary(
    z: np.ndarray, method: str
) -> tuple[np.ndarray, np.ndarray, float | tuple[float, float]]:
    """Return (low_mask, high_mask, threshold_description)."""
    z = np.asarray(z, dtype=np.float64)
    valid = np.isfinite(z)
    if method == "median":
        med = float(np.nanmedian(z[valid]))
        low = z <= med
        high = z > med
        return low, high, med
    if method == "tertile":
        t1, t2 = np.nanpercentile(z[valid], [33.33, 66.67])
        low = z <= t1
        high = z >= t2
        return low, high, (float(t1), float(t2))
    raise ValueError(f"Unknown split method: {method}")


def _conditional_stats(
    y: np.ndarray, low_mask: np.ndarray, high_mask: np.ndarray
) -> dict:
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(y) & low_mask & np.isfinite(low_mask)
    h = np.isfinite(y) & high_mask & np.isfinite(high_mask)
    y_lo, y_hi = y[m], y[h]
    n_lo, n_hi = int(y_lo.size), int(y_hi.size)
    out = {
        "n_low": n_lo,
        "n_high": n_hi,
        "cohens_d": float("nan"),
        "mannwhitney_p": float("nan"),
        "ks_statistic": float("nan"),
        "ks_pvalue": float("nan"),
    }
    if n_lo < MIN_GROUP_N or n_hi < MIN_GROUP_N:
        return out
    out["cohens_d"] = _cohens_d_pooled(y_hi, y_lo)
    try:
        u = mannwhitneyu(y_hi, y_lo, alternative="two-sided")
        out["mannwhitney_p"] = float(u.pvalue)
    except ValueError:
        pass
    try:
        ks = ks_2samp(y_lo, y_hi)
        out["ks_statistic"] = float(ks.statistic)
        out["ks_pvalue"] = float(ks.pvalue)
    except ValueError:
        pass
    return out


def _safe_kde(x: np.ndarray, x_grid: np.ndarray) -> np.ndarray | None:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < MIN_GROUP_N:
        return None
    try:
        kde = gaussian_kde(x)
        return kde(x_grid)
    except np.linalg.LinAlgError:
        return None


def _plot_pair_dashboard(
    a: np.ndarray,
    b: np.ndarray,
    dim_a: str,
    dim_b: str,
    split_method: str,
    metrics_row: dict,
    suptitle: str,
    subtitle_layer_loc: str | None,
    out_png: Path,
    out_pdf: Path,
    bivariate_panel_title: str = "Bivariate: labels",
) -> None:
    import matplotlib.pyplot as plt

    _figure_style()
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]

    low_b, high_b, thr_b = _split_binary(b, split_method)
    low_a, high_a, thr_a = _split_binary(a, split_method)

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    st = fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)
    if subtitle_layer_loc:
        fig.text(0.5, 0.98, subtitle_layer_loc, ha="center", fontsize=10, style="italic")

    # --- Top left: A | B split ---
    ax = axes[0, 0]
    a_lo_b = a[low_b & np.isfinite(a)]
    a_hi_b = a[high_b & np.isfinite(a)]
    lo_label = f"{dim_b} ≤ median" if split_method == "median" else f"{dim_b} low tertile"
    hi_label = f"{dim_b} > median" if split_method == "median" else f"{dim_b} high tertile"
    if split_method == "median":
        med_b = float(thr_b) if isinstance(thr_b, (int, float)) else float("nan")
        thr_str = f"Median({dim_b}) = {med_b:.3g}"
    else:
        thr_str = f"Tertile cutpoints on {dim_b}"

    all_a = np.concatenate([a_lo_b, a_hi_b]) if len(a_lo_b) and len(a_hi_b) else a
    if all_a.size:
        x_min, x_max = np.nanmin(all_a), np.nanmax(all_a)
        pad = 0.05 * (x_max - x_min + 1e-6)
        grid = np.linspace(x_min - pad, x_max + pad, 200)
        d_lo = _safe_kde(a_lo_b, grid)
        d_hi = _safe_kde(a_hi_b, grid)
        if d_lo is not None:
            ax.plot(grid, d_lo, color=COLOR_LOW, lw=2, label=lo_label)
            ax.fill_between(grid, d_lo, alpha=0.35, color=COLOR_LOW)
        if d_hi is not None:
            ax.plot(grid, d_hi, color=COLOR_HIGH, lw=2, label=hi_label)
            ax.fill_between(grid, d_hi, alpha=0.35, color=COLOR_HIGH)
    ax.set_xlabel(dim_a.replace("_", " "))
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution of {dim_a.replace('_', ' ')} | split on {dim_b.replace('_', ' ')}")
    ax.legend(loc="upper right")
    d_ab = metrics_row.get("cohens_d_a_given_split_on_b", float("nan"))
    p_ab = metrics_row.get("mannwhitney_p_a_given_b", float("nan"))
    ax.text(
        0.02,
        0.98,
        f"{thr_str}\nn_low={metrics_row.get('n_a_low_b', '?')}, n_high={metrics_row.get('n_a_high_b', '?')}\n"
        f"Cohen's d (high vs low {dim_b}) = {d_ab:.3f}\n"
        f"Mann–Whitney p = {p_ab:.3g}\n({_cohen_interpretation(d_ab)})",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.35),
    )

    # --- Top right: B | A split ---
    ax = axes[0, 1]
    b_lo_a = b[low_a & np.isfinite(b)]
    b_hi_a = b[high_a & np.isfinite(b)]
    lo_l2 = f"{dim_a} ≤ median" if split_method == "median" else f"{dim_a} low tertile"
    hi_l2 = f"{dim_a} > median" if split_method == "median" else f"{dim_a} high tertile"
    all_b = np.concatenate([b_lo_a, b_hi_a]) if len(b_lo_a) and len(b_hi_a) else b
    if all_b.size:
        x_min, x_max = np.nanmin(all_b), np.nanmax(all_b)
        pad = 0.05 * (x_max - x_min + 1e-6)
        grid_b = np.linspace(x_min - pad, x_max + pad, 200)
        d_lo = _safe_kde(b_lo_a, grid_b)
        d_hi = _safe_kde(b_hi_a, grid_b)
        if d_lo is not None:
            ax.plot(grid_b, d_lo, color=COLOR_LOW, lw=2, label=lo_l2)
            ax.fill_between(grid_b, d_lo, alpha=0.35, color=COLOR_LOW)
        if d_hi is not None:
            ax.plot(grid_b, d_hi, color=COLOR_HIGH, lw=2, label=hi_l2)
            ax.fill_between(grid_b, d_hi, alpha=0.35, color=COLOR_HIGH)
    ax.set_xlabel(dim_b.replace("_", " "))
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution of {dim_b.replace('_', ' ')} | split on {dim_a.replace('_', ' ')}")
    ax.legend(loc="upper right")
    d_ba = metrics_row.get("cohens_d_b_given_split_on_a", float("nan"))
    p_ba = metrics_row.get("mannwhitney_p_b_given_a", float("nan"))
    med_a = float(thr_a) if split_method == "median" and isinstance(thr_a, (int, float)) else None
    thr_str2 = f"Median({dim_a}) = {med_a:.3g}" if med_a is not None else f"Tertile cutpoints on {dim_a}"
    ax.text(
        0.02,
        0.98,
        f"{thr_str2}\nn_low={metrics_row.get('n_b_low_a', '?')}, n_high={metrics_row.get('n_b_high_a', '?')}\n"
        f"Cohen's d = {d_ba:.3f}\nMW p = {p_ba:.3g}\n({_cohen_interpretation(d_ba)})",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.35),
    )

    # --- Bottom left: scatter + 2D KDE contours ---
    ax = axes[1, 0]
    ax.scatter(a, b, alpha=0.3, s=12, c="#444444", edgecolors="none")
    if len(a) >= 20:
        try:
            xy = np.vstack([a, b])
            kde2 = gaussian_kde(xy)
            xi, yi = np.mgrid[
                a.min() : a.max() : 40j,
                b.min() : b.max() : 40j,
            ]
            zi = kde2(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)
            ax.contour(xi, yi, zi, levels=5, colors="#CC79A7", linewidths=1.2, alpha=0.9)
        except (np.linalg.LinAlgError, ValueError):
            pass
    if len(a) >= 2:
        m, c = np.polyfit(a, b, 1)
        xs = np.linspace(np.nanmin(a), np.nanmax(a), 50)
        ax.plot(xs, m * xs + c, color="#009E73", lw=2, label="OLS fit")
        ax.legend(loc="upper right")
    ax.set_xlabel(dim_a.replace("_", " "))
    ax.set_ylabel(dim_b.replace("_", " "))
    ax.set_title(bivariate_panel_title)

    # --- Bottom right: summary text ---
    ax = axes[1, 1]
    ax.axis("off")
    pr = metrics_row.get("pearson_r", float("nan"))
    sp = metrics_row.get("spearman_r", float("nan"))
    pc = metrics_row.get("partial_corr_ab", float("nan"))
    ks = metrics_row.get("ks_stat_a_given_b", float("nan"))
    n = metrics_row.get("n_valid", 0)
    blurb = metrics_row.get("interpretation_blurb", "")
    split_ctx = (
        "test split (ridge predictions)"
        if metrics_row.get("space") == "probe_predictions_test"
        else "train split (dataset labels)"
    )
    txt = (
        f"Sample n (valid pairs) = {n}\n"
        f"Context = {split_ctx}; conditioning split = {split_method}\n\n"
        f"Pearson r (A, B) = {pr:.4f}\n"
        f"Spearman ρ (A, B) = {sp:.4f}\n"
        f"Partial corr (A,B | other appraisals) = {pc:.4f}\n\n"
        f"KS statistic (A | low vs high B) = {ks:.4f}\n\n"
        f"{blurb}"
    )
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, va="top", fontsize=11, family="monospace")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", bbox_extra_artists=[st] if st else None)
    fig.savefig(out_pdf, bbox_inches="tight", bbox_extra_artists=[st] if st else None)
    plt.close(fig)


def _compute_pair_metrics(
    df: pd.DataFrame,
    dim_a: str,
    dim_b: str,
    split_method: str,
) -> dict | None:
    cols = [dim_a, dim_b] + [c for c in COMMON_APPRAISAL if c in df.columns and c not in (dim_a, dim_b)]
    sub = df[cols].copy()
    sub = sub.dropna(subset=[dim_a, dim_b])
    if len(sub) < 2 * MIN_GROUP_N:
        return None
    a = sub[dim_a].to_numpy(dtype=np.float64)
    b = sub[dim_b].to_numpy(dtype=np.float64)

    pearson_r = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-12 and np.std(b) > 1e-12 else float("nan")
    try:
        sp_r, _ = spearmanr(a, b)
        spearman_r = float(sp_r)
    except Exception:
        spearman_r = float("nan")

    cov_cols = [c for c in COMMON_APPRAISAL if c in sub.columns and c not in (dim_a, dim_b)]
    if cov_cols:
        Z = sub[cov_cols].to_numpy(dtype=np.float64)
        row_ok = np.all(np.isfinite(Z), axis=1) & np.isfinite(a) & np.isfinite(b)
        partial_corr = _partial_corr(a[row_ok], b[row_ok], Z[row_ok]) if row_ok.sum() > len(cov_cols) + 2 else float("nan")
    else:
        partial_corr = float("nan")

    a_arr = sub[dim_a].to_numpy(dtype=np.float64)
    b_arr = sub[dim_b].to_numpy(dtype=np.float64)
    low_b, high_b = _split_binary(b_arr, split_method)[:2]
    low_a, high_a = _split_binary(a_arr, split_method)[:2]

    st_ab = _conditional_stats(a_arr, low_b, high_b)
    st_ba = _conditional_stats(b_arr, low_a, high_a)

    # Interpretation
    d_ab = st_ab["cohens_d"]
    if abs(d_ab) < 0.35 if not np.isnan(d_ab) else True:
        blurb = "Heavy overlap of A across B groups ⇒ limited independent variation of A given B (descriptive)."
    else:
        blurb = "Clearer separation of A across B groups ⇒ more independent variation along A given B (descriptive)."

    return {
        "dimension_a": dim_a,
        "dimension_b": dim_b,
        "n_valid": int(len(sub)),
        "split_method": split_method,
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "partial_corr_ab": partial_corr,
        "cohens_d_a_given_split_on_b": st_ab["cohens_d"],
        "cohens_d_b_given_split_on_a": st_ba["cohens_d"],
        "mannwhitney_p_a_given_b": st_ab["mannwhitney_p"],
        "mannwhitney_p_b_given_a": st_ba["mannwhitney_p"],
        "ks_stat_a_given_b": st_ab["ks_statistic"],
        "ks_p_a_given_b": st_ab["ks_pvalue"],
        "ks_stat_b_given_a": st_ba["ks_statistic"],
        "ks_p_b_given_a": st_ba["ks_pvalue"],
        "n_a_low_b": st_ab["n_low"],
        "n_a_high_b": st_ab["n_high"],
        "n_b_low_a": st_ba["n_low"],
        "n_b_high_a": st_ba["n_high"],
        "interpretation_blurb": blurb,
    }


def _sanitize_filename_part(s: str) -> str:
    return s.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _ridge_predict_batch(
    hidden_states: np.ndarray,
    layer: int,
    loc: int,
    dim: str,
    appraisal_probes: dict,
    extraction_layers: list,
    extraction_locs: list,
    token_key,
) -> np.ndarray | None:
    if dim not in appraisal_probes:
        return None
    if layer not in extraction_layers or loc not in extraction_locs:
        return None
    li = extraction_layers.index(layer)
    lci = extraction_locs.index(loc)
    X = hidden_states[:, li, lci, 0, :].astype(np.float64)
    try:
        r = appraisal_probes[dim][layer][loc][token_key]
    except (KeyError, TypeError):
        return None
    if not r or "ridge" not in r:
        return None
    ridge = r["ridge"]
    scaler = r.get("scaler")
    Xe = scaler.transform(X) if scaler is not None else X
    return ridge.predict(Xe).astype(np.float64)


def run_appraisal_label_coupling(
    model_id: str = DEFAULT_MODEL_ID,
    split_method: str | None = None,
    pairs: list[tuple[str, str]] | None = None,
    label_only: bool = False,
) -> dict:
    """
    Run label-space coupling on the train split; optionally probe-space on test when
    circuit caches include dataset_row_idx and appraisal ridge probes exist.
    """
    split_method = split_method or APPRAISAL_LABEL_COUPLING_SPLIT_METHOD
    pairs = pairs or list(APPRAISAL_LABEL_COUPLING_PAIRS)
    out_root = get_appraisal_structure_dir(model_id) / "label_coupling"
    fig_dir = out_root / "figures"
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df_full = canonicalize_combined_dataset(pd.read_csv(COMBINED_CSV))
    base_text_col, prompted_col = resolve_text_columns(df_full)
    text_col = prompted_col or base_text_col
    df_full = df_full.dropna(subset=["emotion", text_col])
    split_bundle = split_combined_dataset(
        df_full,
        selection_frac=SELECTION_SPLIT,
        test_frac=FINAL_TEST_SPLIT,
        random_state=RANDOM_STATE,
    )
    train_df = split_bundle["train"].copy()

    label_rows = []
    for dim_a, dim_b in pairs:
        if dim_a not in train_df.columns or dim_b not in train_df.columns:
            print(f"[appraisal_label_coupling] Skip pair ({dim_a}, {dim_b}): missing column.")
            continue
        m = _compute_pair_metrics(train_df, dim_a, dim_b, split_method)
        if m is None:
            print(f"[appraisal_label_coupling] Skip pair ({dim_a}, {dim_b}): insufficient data.")
            continue
        label_rows.append(m)
        base = f"pair_{_sanitize_filename_part(dim_a)}__{_sanitize_filename_part(dim_b)}_label_train"
        _plot_pair_dashboard(
            train_df[dim_a].to_numpy(),
            train_df[dim_b].to_numpy(),
            dim_a,
            dim_b,
            split_method,
            m,
            f"Appraisal coupling (train split): {dim_a.replace('_', ' ')} vs {dim_b.replace('_', ' ')}",
            None,
            fig_dir / f"{base}.png",
            fig_dir / f"{base}.pdf",
            bivariate_panel_title="Bivariate: dataset labels",
        )

    label_csv = out_root / "label_coupling_pairwise_metrics.csv"
    if label_rows:
        pd.DataFrame(label_rows).to_csv(label_csv, index=False)
    else:
        pd.DataFrame(
            columns=[
                "dimension_a",
                "dimension_b",
                "n_valid",
                "split_method",
                "pearson_r",
                "spearman_r",
                "partial_corr_ab",
            ]
        ).to_csv(label_csv, index=False)

    manifest = {
        "version": 1,
        "model_id": model_id,
        "dataset": dataset_fingerprint(COMBINED_CSV),
        "split_protocol": {
            "selection_frac": SELECTION_SPLIT,
            "test_frac": FINAL_TEST_SPLIT,
            "random_state": RANDOM_STATE,
        },
        "label_split": "train",
        "n_train_rows": int(len(train_df)),
        "pairs": [{"a": a, "b": b} for a, b in pairs],
        "split_method": split_method,
        "matplotlib_dpi": FIGURE_DPI,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    readme = _build_readme()
    (out_root / "README.md").write_text(readme, encoding="utf-8")

    probe_rows: list[dict] = []
    if not label_only:
        probe_rows = _run_probe_coupling_if_available(
            model_id, df_full, pairs, split_method, fig_dir
        )
    pd.DataFrame(probe_rows).to_csv(out_root / "probe_coupling_pairwise_metrics.csv", index=False)

    print(f"[appraisal_label_coupling] Wrote {out_root}")
    return {"out_dir": out_root, "n_label_pairs": len(label_rows), "n_probe_pairs": len(probe_rows)}


def _run_probe_coupling_if_available(
    model_id: str,
    df_full: pd.DataFrame,
    pairs: list[tuple[str, str]],
    split_method: str,
    fig_dir: Path,
) -> list[dict]:
    circuit_dir = get_circuit_dir(model_id)
    hs_path = circuit_dir / "test_hidden_states.pt"
    lbl_path = circuit_dir / "test_labels.csv"
    if not hs_path.exists() or not lbl_path.exists():
        print("[appraisal_label_coupling] Probe coupling skipped: missing test hidden states or labels.")
        return []
    lbl_df = pd.read_csv(lbl_path)
    if "dataset_row_idx" not in lbl_df.columns:
        if not _backfill_test_labels_row_indices(circuit_dir):
            print(
                "[appraisal_label_coupling] Probe coupling skipped: could not add dataset_row_idx "
                "to test_labels.csv (see messages above)."
            )
            return []
        lbl_df = pd.read_csv(lbl_path)

    paths = get_probe_paths(model_id)
    if not paths.appraisal_probes_path.exists():
        print("[appraisal_label_coupling] Probe coupling skipped: no appraisal_regression_probes.pt.")
        return []

    ensure_manifest_model_match(paths.appraisal_manifest_path, model_id, "appraisal probe")
    appraisal_probes = torch.load(paths.appraisal_probes_path, weights_only=False)
    hidden_states = torch.load(hs_path, weights_only=False)
    if isinstance(hidden_states, torch.Tensor):
        hs = hidden_states.numpy()
    else:
        hs = np.asarray(hidden_states)

    extraction_layers = get_extraction_layers(model_id)
    extraction_locs = get_extraction_locs(model_id)
    token_key = EXTRACTION_TOKENS[0]
    layer = APPRAISAL_PROBE_COUPLING_LAYER if APPRAISAL_PROBE_COUPLING_LAYER is not None else extraction_layers[-1]
    loc = APPRAISAL_PROBE_COUPLING_LOC if APPRAISAL_PROBE_COUPLING_LOC is not None else extraction_locs[0]
    subtitle = f"Ridge readout at layer={layer}, loc={loc} (shared across dimensions)"

    row_indices = lbl_df["dataset_row_idx"].astype(int).tolist()
    # align df_full index: canonical df may have default RangeIndex 0..N-1
    pred_cache: dict[str, np.ndarray] = {}

    def preds_for(dim: str) -> np.ndarray | None:
        if dim in pred_cache:
            return pred_cache[dim]
        p = _ridge_predict_batch(hs, layer, loc, dim, appraisal_probes, extraction_layers, extraction_locs, token_key)
        if p is not None:
            pred_cache[dim] = p
        return p

    rows_out: list[dict] = []
    for dim_a, dim_b in pairs:
        pa, pb = preds_for(dim_a), preds_for(dim_b)
        if pa is None or pb is None:
            print(f"[appraisal_label_coupling] Probe skip ({dim_a},{dim_b}): missing ridge at layer={layer} loc={loc}.")
            continue
        if len(pa) != len(row_indices) or len(pb) != len(row_indices):
            print("[appraisal_label_coupling] Probe coupling skipped: length mismatch hidden states vs row indices.")
            return rows_out

        recs = []
        for i, ridx in enumerate(row_indices):
            if ridx not in df_full.index:
                continue
            if not np.isfinite(pa[i]) or not np.isfinite(pb[i]):
                continue
            recs.append({"pred_a": float(pa[i]), "pred_b": float(pb[i])})
        if len(recs) < 2 * MIN_GROUP_N:
            continue
        pdf = pd.DataFrame(recs)
        met_df = pdf.rename(columns={"pred_a": dim_a, "pred_b": dim_b})
        m = _compute_pair_metrics(met_df, dim_a, dim_b, split_method)
        if m is None:
            continue
        m["space"] = "probe_predictions_test"
        m["ridge_layer"] = layer
        m["ridge_loc"] = loc
        rows_out.append(m)
        base = f"pair_{_sanitize_filename_part(dim_a)}__{_sanitize_filename_part(dim_b)}_probe_test"
        _plot_pair_dashboard(
            pdf["pred_a"].to_numpy(),
            pdf["pred_b"].to_numpy(),
            dim_a,
            dim_b,
            split_method,
            m,
            f"Probe predictions (test split): {dim_a.replace('_', ' ')} vs {dim_b.replace('_', ' ')}",
            subtitle,
            fig_dir / f"{base}.png",
            fig_dir / f"{base}.pdf",
            bivariate_panel_title="Bivariate: ridge probe predictions",
        )

    return rows_out


def _build_readme() -> str:
    return """# Label coupling (03_appraisal_structure/label_coupling)

## Files

- `label_coupling_pairwise_metrics.csv` — One row per configured dimension pair on the **train** split (label space).
- `probe_coupling_pairwise_metrics.csv` — Same metrics on **ridge probe predictions** for the **test** split (only if `02_circuit/test_labels.csv` includes `dataset_row_idx` and appraisal probes exist).
- `figures/` — For each pair, a **2×2 dashboard** (PNG + PDF):
  1. **Top left:** distribution of dimension A split on high/low B (KDE overlap).
  2. **Top right:** distribution of B split on high/low A (check asymmetry).
  3. **Bottom left:** scatter of the two variables with 2D density contours and an OLS line.
  4. **Bottom right:** numeric summary (correlations, partial correlation, KS, sample sizes) and a short plain-language line.

## How to read the dashboard

- **Overlapping KDEs** in the top row ⇒ the conditioning dimension does little to separate the other (strong coupling / little independent variation).
- **Separated KDEs** ⇒ more independent variation; probes can more easily disentangle the two constructs.
- **Scatter + contours:** a tight diagonal or anti-diagonal suggests a **single shared axis**; a filled 2D blob suggests **two degrees of freedom** in the data (descriptive).
- **Probe panels** compare the same geometry on **model readouts**; if labels look 2D but probe scores collapse to 1D, the model may be compressing both constructs.

## Caveats

These analyses are **descriptive** and concern **identifiability** in the dataset and in linear readouts — **not** causal claims about the network.
"""


def main() -> None:
    p = argparse.ArgumentParser(description="Appraisal label coupling analysis.")
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--label_only", action="store_true", help="Skip probe-score coupling on test.")
    p.add_argument("--split_method", default=None, help="median or tertile (default: config).")
    args = p.parse_args()
    run_appraisal_label_coupling(
        model_id=args.model_id,
        split_method=args.split_method,
        label_only=args.label_only,
    )


if __name__ == "__main__":
    main()

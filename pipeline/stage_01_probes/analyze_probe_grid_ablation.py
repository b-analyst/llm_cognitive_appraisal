"""
Summarize probe grid ablation outputs: best (layer, loc, token) tables, heatmaps, README.

Reads:
  - <ablation_dir>/binary_ova_probes/probe_summary.csv
  - <ablation_dir>/appraisal_probe_validation_detail.csv

Writes:
  - grid_ablation_summary.csv
  - heatmap_*.png (if matplotlib is available)
  - GRID_ABLATION_README.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PIPELINE_ROOT.parent


def _ensure_paths():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def _norm_token(x) -> str:
    if isinstance(x, str):
        return x
    return str(int(x))


def _best_emotion_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if df.empty or "emotion" not in df.columns:
        return pd.DataFrame()
    metric = "test_roc_auc"
    for emotion, g in df.groupby("emotion"):
        g2 = g.copy()
        if metric not in g2.columns:
            continue
        g2 = g2.dropna(subset=[metric])
        if g2.empty:
            continue
        i = g2[metric].idxmax()
        r = g2.loc[i].to_dict()
        r["probe_type"] = "emotion"
        r["best_metric_name"] = metric
        r["best_metric_value"] = float(r[metric])
        rows.append(r)
    return pd.DataFrame(rows)


def _best_appraisal_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if df.empty or "dimension" not in df.columns:
        return pd.DataFrame()
    metric = "selection_corr"
    for dim, g in df.groupby("dimension"):
        g2 = g.copy()
        if metric not in g2.columns:
            continue
        g2 = g2.dropna(subset=[metric])
        if g2.empty:
            continue
        i = g2[metric].idxmax()
        r = g2.loc[i].to_dict()
        r["probe_type"] = "appraisal"
        r["best_metric_name"] = metric
        r["best_metric_value"] = float(r[metric])
        rows.append(r)
    return pd.DataFrame(rows)


def _merge_baseline_emotion(best: pd.DataFrame, baseline_path: Path | None) -> pd.DataFrame:
    if baseline_path is None or not baseline_path.exists() or best.empty:
        return best
    base = pd.read_csv(baseline_path)
    if base.empty or "emotion" not in base.columns:
        return best
    m = "test_roc_auc"
    if m not in base.columns:
        return best
    bbest = _best_emotion_rows(base)
    if bbest.empty:
        return best
    out = best.copy()
    out["baseline_test_roc_auc"] = np.nan
    out["delta_test_roc_auc_vs_baseline"] = np.nan
    bmap = {str(r["emotion"]).lower(): float(r[m]) for _, r in bbest.iterrows() if m in r and pd.notna(r[m])}
    for idx, row in out.iterrows():
        key = str(row.get("emotion", "")).lower()
        if key in bmap:
            out.at[idx, "baseline_test_roc_auc"] = bmap[key]
            out.at[idx, "delta_test_roc_auc_vs_baseline"] = float(row[m]) - bmap[key]
    return out


def _plot_heatmaps_emotion(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return paths
    if df.empty or "token" not in df.columns:
        return paths
    metric = "test_roc_auc"
    if metric not in df.columns:
        return paths
    df = df.copy()
    df["token_str"] = df["token"].map(_norm_token)
    for tok, g in df.groupby("token_str"):
        pivot = g.groupby(["layer", "loc"], as_index=False)[metric].max()
        if pivot.empty:
            continue
        mat = pivot.pivot(index="layer", columns="loc", values=metric)
        fig, ax = plt.subplots(figsize=(max(4, mat.shape[1] * 0.9), max(3, mat.shape[0] * 0.5)))
        im = ax.imshow(mat.values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(mat.shape[1]))
        ax.set_xticklabels([str(c) for c in mat.columns])
        ax.set_yticks(range(mat.shape[0]))
        ax.set_yticklabels([str(i) for i in mat.index])
        ax.set_xlabel("loc")
        ax.set_ylabel("layer")
        ax.set_title(f"Emotion probes: max {metric} over emotions (token={tok})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fp = out_dir / f"heatmap_emotion_{metric}_token_{tok}.png"
        fig.savefig(fp, dpi=150)
        plt.close(fig)
        paths.append(fp)
    return paths


def _plot_heatmaps_appraisal(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return paths
    if df.empty:
        return paths
    if "token" not in df.columns:
        return paths
    metric = "selection_corr"
    if metric not in df.columns:
        return paths
    df = df.copy()
    df["token_str"] = df["token"].map(_norm_token)
    for tok, g in df.groupby("token_str"):
        pivot = g.groupby(["layer", "loc"], as_index=False)[metric].max()
        if pivot.empty:
            continue
        mat = pivot.pivot(index="layer", columns="loc", values=metric)
        fig, ax = plt.subplots(figsize=(max(4, mat.shape[1] * 0.9), max(3, mat.shape[0] * 0.5)))
        im = ax.imshow(mat.values, aspect="auto", cmap="magma")
        ax.set_xticks(range(mat.shape[1]))
        ax.set_xticklabels([str(c) for c in mat.columns])
        ax.set_yticks(range(mat.shape[0]))
        ax.set_yticklabels([str(i) for i in mat.index])
        ax.set_xlabel("loc")
        ax.set_ylabel("layer")
        ax.set_title(f"Appraisal probes: max {metric} over dimensions (token={tok})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fp = out_dir / f"heatmap_appraisal_{metric}_token_{tok}.png"
        fig.savefig(fp, dpi=150)
        plt.close(fig)
        paths.append(fp)
    return paths


def _write_readme(path: Path, figs: list[str], had_mpl: bool):
    lines = [
        "# Probe grid ablation (scope A)",
        "",
        "This folder holds **probe-training-only** artifacts for a wider **loc × token** readout grid.",
        "It does **not** re-run `circuit_evidence`, `phase1_circuits`, or steering.",
        "",
        "## Token semantics",
        "",
        "- **`0` / `-1`:** Fixed indices on the **padded** batch tensor (same as the main extractor).",
        "  With right-padding, **`-1`** is often the last column (frequently a **pad** token unless masked).",
        "  **`0`** is often BOS / prompt start depending on the chat template.",
        "- **`mid`:** Per-sequence **masked middle**: from `attention_mask`,",
        "  `first_real` = first index with mask 1, `last_real` = sum(mask)-1,",
        "  `mid = (first_real + last_real) // 2`, then gather hidden states at that index.",
        "",
        "Interpret metrics as **sensitivity to readout policy**, not as ground-truth localization of emotion.",
        "",
        "## Outputs from `analyze_probe_grid_ablation.py`",
        "",
        "- `grid_ablation_summary.csv` — best (layer, loc, token) per emotion and per appraisal dimension.",
    ]
    if had_mpl and figs:
        lines.append("- Heatmaps: " + ", ".join(f"`{f}`" for f in figs))
    else:
        lines.append("- Heatmaps: skipped (install `matplotlib` and re-run the analyzer).")
    lines.extend(["", "See `docs/EXPERIMENTAL_SETUP.md` for how to run the ablation.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_probe_grid_ablation(
    ablation_dir: Path,
    baseline_probe_summary: Path | None = None,
) -> Path:
    _ensure_paths()
    ablation_dir = Path(ablation_dir).resolve()
    emotion_csv = ablation_dir / "binary_ova_probes" / "probe_summary.csv"
    appraisal_csv = ablation_dir / "appraisal_probe_validation_detail.csv"

    parts = []
    if emotion_csv.exists():
        edf = pd.read_csv(emotion_csv)
        best_e = _best_emotion_rows(edf)
        best_e = _merge_baseline_emotion(best_e, baseline_probe_summary)
        parts.append(best_e)
    if appraisal_csv.exists():
        adf = pd.read_csv(appraisal_csv)
        parts.append(_best_appraisal_rows(adf))

    summary = pd.concat(parts, axis=0, ignore_index=True) if parts else pd.DataFrame()
    out_csv = ablation_dir / "grid_ablation_summary.csv"
    summary.to_csv(out_csv, index=False)

    fig_paths = []
    if emotion_csv.exists():
        fig_paths.extend(_plot_heatmaps_emotion(pd.read_csv(emotion_csv), ablation_dir))
    if appraisal_csv.exists():
        fig_paths.extend(_plot_heatmaps_appraisal(pd.read_csv(appraisal_csv), ablation_dir))

    _write_readme(
        ablation_dir / "GRID_ABLATION_README.md",
        [p.name for p in fig_paths],
        bool(fig_paths),
    )
    return out_csv


def main():
    _ensure_paths()
    from pipeline.config import DEFAULT_MODEL_ID, get_probe_grid_ablation_dir

    p = argparse.ArgumentParser(description="Summarize probe grid ablation CSVs + heatmaps")
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID, help="Use outputs/<id>/01_probes_grid_ablation/")
    p.add_argument("--ablation_dir", type=Path, default=None, help="Override ablation directory")
    p.add_argument(
        "--baseline_probe_summary",
        type=Path,
        default=None,
        help="Optional canonical probe_summary.csv for delta columns (emotion only)",
    )
    args = p.parse_args()
    root = args.ablation_dir if args.ablation_dir is not None else get_probe_grid_ablation_dir(args.model_id)

    analyze_probe_grid_ablation(root, baseline_probe_summary=args.baseline_probe_summary)
    print(f"Wrote summary under {root}")


if __name__ == "__main__":
    main()

"""
Diagnostic: Where does appraisal live vs. the emotion circuit?

For each model with existing outputs, this script:
1. Ranks all (layer, loc) sites by mean appraisal probe selection_corr
2. Identifies the emotion circuit sites (union of per_emotion_pairs)
3. Computes overlap between top appraisal sites and circuit sites
4. Produces per-dimension breakdowns and summary figures

Run:  python -m pipeline.appraisal_overlap_diagnostic
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from pipeline.core.config import OUTPUTS_ROOT, COMMON_APPRAISAL

MODELS = ["Llama3.2_1B", "Llama3.1_8B"]


def _load_appraisal_detail(model_id: str) -> pd.DataFrame | None:
    path = OUTPUTS_ROOT / model_id / "01_probes" / "appraisal_probe_validation_detail.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _load_circuit_selection(model_id: str) -> dict | None:
    path = OUTPUTS_ROOT / model_id / "02_circuit" / "circuit_top_k_selection.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _circuit_sites_union(sel: dict) -> set[tuple[int, int]]:
    pairs = sel.get("per_emotion_pairs", {})
    sites: set[tuple[int, int]] = set()
    for emotion, pair_list in pairs.items():
        for layer, loc in pair_list:
            sites.add((int(layer), int(loc)))
    return sites


def _circuit_sites_per_emotion(sel: dict) -> dict[str, set[tuple[int, int]]]:
    pairs = sel.get("per_emotion_pairs", {})
    out: dict[str, set[tuple[int, int]]] = {}
    for emotion, pair_list in pairs.items():
        out[emotion] = {(int(l), int(c)) for l, c in pair_list}
    return out


def _total_sites(model_id: str) -> int:
    if "8B" in model_id:
        return 32 * 3
    return 16 * 3


def run_diagnostic(model_id: str) -> dict:
    appraisal_df = _load_appraisal_detail(model_id)
    circuit_sel = _load_circuit_selection(model_id)
    if appraisal_df is None or circuit_sel is None:
        print(f"  [skip] {model_id}: missing appraisal detail or circuit selection")
        return {}

    total = _total_sites(model_id)
    circuit_union = _circuit_sites_union(circuit_sel)
    circuit_per_em = _circuit_sites_per_emotion(circuit_sel)

    # --- Per-dimension best sites ---
    dim_best: dict[str, list[tuple[float, int, int]]] = {}
    for dim in COMMON_APPRAISAL:
        sub = appraisal_df[appraisal_df["dimension"] == dim].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("selection_corr", ascending=False)
        ranked = [(float(row["selection_corr"]), int(row["layer"]), int(row["loc"]))
                  for _, row in sub.iterrows()]
        dim_best[dim] = ranked

    # --- Aggregate: mean selection_corr across all dimensions per (layer, loc) ---
    site_agg = (
        appraisal_df.groupby(["layer", "loc"])["selection_corr"]
        .mean()
        .reset_index()
        .sort_values("selection_corr", ascending=False)
        .reset_index(drop=True)
    )
    site_agg["rank"] = range(1, len(site_agg) + 1)
    site_agg["in_circuit"] = site_agg.apply(
        lambda r: (int(r["layer"]), int(r["loc"])) in circuit_union, axis=1
    )

    # --- Build summary ---
    print(f"\n{'='*70}")
    print(f"  MODEL: {model_id}")
    print(f"  Total (layer, loc) sites: {total}")
    print(f"  Emotion circuit sites (union across emotions): {len(circuit_union)}")
    print(f"  Circuit coverage of grid: {len(circuit_union)/total:.1%}")
    print(f"{'='*70}")

    print(f"\n  TOP 15 APPRAISAL SITES (by mean selection_corr across all dimensions):")
    print(f"  {'Rank':>4}  {'Layer':>5}  {'Loc':>3}  {'Mean r':>8}  {'In Circuit?':>12}")
    print(f"  {'-'*4}  {'-'*5}  {'-'*3}  {'-'*8}  {'-'*12}")
    for _, row in site_agg.head(15).iterrows():
        marker = "YES" if row.in_circuit else "---"
        print(f"  {int(row['rank']):>4}  {int(row['layer']):>5}  {int(row['loc']):>3}  {row['selection_corr']:>8.4f}  {marker:>12}")

    # --- Overlap at various top-K thresholds ---
    print(f"\n  OVERLAP: top-K appraisal sites vs emotion circuit")
    print(f"  {'K':>4}  {'Overlap':>8}  {'In-circuit':>10}  {'Outside':>8}  {'Pct overlap':>12}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*12}")
    overlap_rows = []
    for k in [3, 5, 7, 10, 15, 20, len(circuit_union)]:
        if k > len(site_agg):
            continue
        top_k_sites = {(int(r["layer"]), int(r["loc"])) for _, r in site_agg.head(k).iterrows()}
        overlap = top_k_sites & circuit_union
        outside = top_k_sites - circuit_union
        pct = len(overlap) / k if k > 0 else 0
        print(f"  {k:>4}  {len(overlap):>8}  {len(overlap):>10}  {len(outside):>8}  {pct:>11.1%}")
        overlap_rows.append({"k": k, "overlap": len(overlap), "outside": len(outside), "pct_overlap": pct})

    # --- Per-dimension: is the best site in the circuit? ---
    print(f"\n  PER-DIMENSION BEST SITE (is it inside any emotion circuit?)")
    print(f"  {'Dimension':<22}  {'Best (layer,loc)':>16}  {'r':>7}  {'In Circuit?':>12}")
    print(f"  {'-'*22}  {'-'*16}  {'-'*7}  {'-'*12}")
    dim_in_circuit_count = 0
    for dim in COMMON_APPRAISAL:
        if dim not in dim_best:
            continue
        best_r, best_l, best_c = dim_best[dim][0]
        in_c = (best_l, best_c) in circuit_union
        if in_c:
            dim_in_circuit_count += 1
        marker = "YES" if in_c else "---"
        print(f"  {dim:<22}  {f'({best_l}, {best_c})':>16}  {best_r:>7.4f}  {marker:>12}")
    print(f"\n  {dim_in_circuit_count}/{len(dim_best)} dimension-best sites are inside the emotion circuit")

    # --- Per-dimension: top-3 sites and circuit overlap ---
    print(f"\n  PER-DIMENSION TOP-3 SITES:")
    for dim in COMMON_APPRAISAL:
        if dim not in dim_best:
            continue
        top3 = dim_best[dim][:3]
        labels = []
        for r, l, c in top3:
            in_c = (l, c) in circuit_union
            labels.append(f"({l},{c}) r={r:.3f}{'*' if in_c else ''}")
        print(f"  {dim:<22}  {' | '.join(labels)}")
    print(f"  (* = in emotion circuit)")

    # --- Key question: among circuit sites, what's the mean appraisal corr?
    #     vs among non-circuit sites? ---
    circuit_corrs = site_agg[site_agg["in_circuit"]]["selection_corr"].values
    non_circuit_corrs = site_agg[~site_agg["in_circuit"]]["selection_corr"].values
    print(f"\n  APPRAISAL STRENGTH: circuit vs non-circuit sites")
    print(f"    Circuit sites    (n={len(circuit_corrs):>2}): mean r = {np.mean(circuit_corrs):.4f}, "
          f"median = {np.median(circuit_corrs):.4f}, max = {np.max(circuit_corrs):.4f}")
    if len(non_circuit_corrs) > 0:
        print(f"    Non-circuit sites (n={len(non_circuit_corrs):>2}): mean r = {np.mean(non_circuit_corrs):.4f}, "
              f"median = {np.median(non_circuit_corrs):.4f}, max = {np.max(non_circuit_corrs):.4f}")
    else:
        print(f"    Non-circuit sites: NONE (circuit covers 100% of grid)")

    # --- Create figures ---
    out_dir = OUTPUTS_ROOT / model_id / "04_appraisal_in_circuit"
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_site_ranking(site_agg, circuit_union, model_id, out_dir)
    _plot_heatmap(site_agg, circuit_union, model_id, out_dir)
    _plot_per_dimension_heatmap(appraisal_df, circuit_union, model_id, out_dir)

    csv_path = out_dir / "appraisal_site_ranking_vs_circuit.csv"
    site_agg.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")
    print(f"  Figures saved to: {out_dir}")

    return {
        "model_id": model_id,
        "total_sites": total,
        "circuit_sites": len(circuit_union),
        "circuit_coverage": len(circuit_union) / total,
        "overlap_rows": overlap_rows,
        "circuit_mean_appraisal_r": float(np.mean(circuit_corrs)),
        "non_circuit_mean_appraisal_r": float(np.mean(non_circuit_corrs)) if len(non_circuit_corrs) > 0 else None,
    }


def _plot_site_ranking(site_agg: pd.DataFrame, circuit_union: set, model_id: str, out_dir: Path):
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#2196F3" if (int(r["layer"]), int(r["loc"])) in circuit_union else "#BDBDBD"
              for _, r in site_agg.iterrows()]
    ax.bar(range(len(site_agg)), site_agg["selection_corr"].values, color=colors, edgecolor="none")
    ax.set_xlabel("Site rank (by mean appraisal selection_corr)")
    ax.set_ylabel("Mean selection_corr across appraisal dims")
    ax.set_title(f"{model_id}: All sites ranked by appraisal probe strength\n"
                 f"(blue = in emotion circuit, gray = not)")

    ax.set_xticks(range(0, len(site_agg), max(1, len(site_agg) // 15)))
    labels = []
    for i in range(0, len(site_agg), max(1, len(site_agg) // 15)):
        row = site_agg.iloc[i]
        labels.append(f"L{int(row['layer'])}.{int(row['loc'])}")
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)

    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor="#2196F3", label="In emotion circuit"),
                    Patch(facecolor="#BDBDBD", label="Not in circuit")]
    ax.legend(handles=legend_elems, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "appraisal_site_ranking_vs_circuit.png", dpi=150)
    plt.close(fig)


def _plot_heatmap(site_agg: pd.DataFrame, circuit_union: set, model_id: str, out_dir: Path):
    layers = sorted(site_agg["layer"].unique())
    locs = sorted(site_agg["loc"].unique())
    n_layers = len(layers)
    n_locs = len(locs)

    layer_to_idx = {l: i for i, l in enumerate(layers)}
    loc_to_idx = {l: i for i, l in enumerate(locs)}

    grid = np.full((n_locs, n_layers), np.nan)
    for _, row in site_agg.iterrows():
        li = layer_to_idx[int(row["layer"])]
        ci = loc_to_idx[int(row["loc"])]
        grid[ci, li] = row["selection_corr"]

    fig, ax = plt.subplots(figsize=(max(10, n_layers * 0.5), 3))
    im = ax.imshow(grid, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([str(l) for l in layers], fontsize=7)
    ax.set_yticks(range(n_locs))
    loc_labels = {3: "attn_out (3)", 6: "mlp_out (6)", 7: "layer_out (7)"}
    ax.set_yticklabels([loc_labels.get(l, str(l)) for l in locs], fontsize=8)
    ax.set_xlabel("Layer")
    ax.set_title(f"{model_id}: Mean appraisal probe corr by site\n(boxes = emotion circuit sites)")

    for _, row in site_agg.iterrows():
        li = layer_to_idx[int(row["layer"])]
        ci = loc_to_idx[int(row["loc"])]
        if (int(row["layer"]), int(row["loc"])) in circuit_union:
            rect = plt.Rectangle((li - 0.5, ci - 0.5), 1, 1,
                                 linewidth=2, edgecolor="blue", facecolor="none")
            ax.add_patch(rect)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean selection_corr")
    fig.tight_layout()
    fig.savefig(out_dir / "appraisal_heatmap_with_circuit.png", dpi=150)
    plt.close(fig)


def _plot_per_dimension_heatmap(appraisal_df: pd.DataFrame, circuit_union: set,
                                model_id: str, out_dir: Path):
    dims_to_show = [d for d in COMMON_APPRAISAL if d in appraisal_df["dimension"].unique()]
    if not dims_to_show:
        return

    layers = sorted(appraisal_df["layer"].unique())
    locs = sorted(appraisal_df["loc"].unique())
    n_layers = len(layers)
    n_locs = len(locs)
    layer_to_idx = {l: i for i, l in enumerate(layers)}
    loc_to_idx = {l: i for i, l in enumerate(locs)}

    n_dims = len(dims_to_show)
    n_cols = 3
    n_rows = (n_dims + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(max(12, n_layers * 0.4 * n_cols), 2.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    loc_labels = {3: "attn(3)", 6: "mlp(6)", 7: "out(7)"}

    for idx, dim in enumerate(dims_to_show):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        sub = appraisal_df[appraisal_df["dimension"] == dim]
        grid = np.full((n_locs, n_layers), np.nan)
        for _, row in sub.iterrows():
            li = layer_to_idx[int(row["layer"])]
            ci = loc_to_idx[int(row["loc"])]
            grid[ci, li] = row["selection_corr"]

        im = ax.imshow(grid, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        ax.set_title(dim, fontsize=9)
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels([str(l) for l in layers], fontsize=5)
        ax.set_yticks(range(n_locs))
        ax.set_yticklabels([loc_labels.get(l, str(l)) for l in locs], fontsize=6)

        best_row = sub.sort_values("selection_corr", ascending=False).iloc[0]
        best_li = layer_to_idx[int(best_row["layer"])]
        best_ci = loc_to_idx[int(best_row["loc"])]
        in_c = (int(best_row["layer"]), int(best_row["loc"])) in circuit_union
        star_color = "blue" if in_c else "red"
        ax.plot(best_li, best_ci, marker="*", color=star_color, markersize=12)

    for idx in range(len(dims_to_show), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].set_visible(False)

    fig.suptitle(f"{model_id}: Per-dimension appraisal heatmaps\n"
                 f"(star = best site; blue star = in circuit, red star = outside circuit)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "appraisal_per_dimension_heatmaps.png", dpi=150)
    plt.close(fig)


def run_per_emotion_appraisal_diagnostic(model_id: str) -> dict | None:
    """For each emotion's circuit, measure appraisal probe strength per dimension.
    Produces an enrichment matrix: how much better is each appraisal dimension
    inside this emotion's specific circuit vs the global average?"""

    appraisal_df = _load_appraisal_detail(model_id)
    circuit_sel = _load_circuit_selection(model_id)
    if appraisal_df is None or circuit_sel is None:
        print(f"  [skip] {model_id}: missing data")
        return None

    circuit_per_em = _circuit_sites_per_emotion(circuit_sel)
    total = _total_sites(model_id)
    dims = [d for d in COMMON_APPRAISAL if d in appraisal_df["dimension"].unique()]

    # Build a lookup: (dim, layer, loc) -> selection_corr
    corr_lookup: dict[tuple[str, int, int], float] = {}
    for _, row in appraisal_df.iterrows():
        corr_lookup[(row["dimension"], int(row["layer"]), int(row["loc"]))] = float(row["selection_corr"])

    # Global mean corr per dimension (across all sites)
    global_mean = {}
    for dim in dims:
        vals = [v for (d, l, c), v in corr_lookup.items() if d == dim]
        global_mean[dim] = np.mean(vals) if vals else 0.0

    # Per-emotion, per-dimension: mean corr at that emotion's circuit sites
    emotions = sorted(circuit_per_em.keys())
    raw_matrix = np.zeros((len(emotions), len(dims)))
    enrichment_matrix = np.zeros((len(emotions), len(dims)))
    overlap_count_matrix = np.zeros((len(emotions), len(dims)), dtype=int)

    for ei, emotion in enumerate(emotions):
        em_sites = circuit_per_em[emotion]
        for di, dim in enumerate(dims):
            # Mean appraisal corr at this emotion's circuit sites for this dimension
            vals = [corr_lookup.get((dim, l, c), np.nan) for l, c in em_sites]
            vals = [v for v in vals if not np.isnan(v)]
            em_mean = np.mean(vals) if vals else 0.0
            raw_matrix[ei, di] = em_mean
            enrichment_matrix[ei, di] = em_mean - global_mean[dim]

            # How many of the top-5 sites for this dim are in this emotion's circuit?
            dim_sub = appraisal_df[appraisal_df["dimension"] == dim].sort_values(
                "selection_corr", ascending=False
            )
            top5 = {(int(r["layer"]), int(r["loc"])) for _, r in dim_sub.head(5).iterrows()}
            overlap_count_matrix[ei, di] = len(top5 & em_sites)

    out_dir = OUTPUTS_ROOT / model_id / "04_appraisal_in_circuit"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save raw data
    rows_out = []
    for ei, emotion in enumerate(emotions):
        for di, dim in enumerate(dims):
            rows_out.append({
                "emotion": emotion,
                "dimension": dim,
                "circuit_mean_corr": raw_matrix[ei, di],
                "global_mean_corr": global_mean[dim],
                "enrichment": enrichment_matrix[ei, di],
                "top5_overlap_count": int(overlap_count_matrix[ei, di]),
                "circuit_size": len(circuit_per_em[emotion]),
            })
    pd.DataFrame(rows_out).to_csv(
        out_dir / "emotion_appraisal_circuit_enrichment.csv", index=False
    )

    # --- Print summary ---
    print(f"\n{'='*70}")
    print(f"  PER-EMOTION APPRAISAL ENRICHMENT: {model_id}")
    print(f"{'='*70}")

    # Show the most enriched (emotion, dimension) pairs
    flat = [(emotions[ei], dims[di], enrichment_matrix[ei, di], raw_matrix[ei, di])
            for ei in range(len(emotions)) for di in range(len(dims))]
    flat.sort(key=lambda x: x[2], reverse=True)

    print(f"\n  TOP 15 ENRICHED (emotion, appraisal) pairs:")
    print(f"  (enrichment = circuit_mean_r - global_mean_r for that dimension)")
    print(f"  {'Emotion':<14}  {'Dimension':<22}  {'Circuit r':>10}  {'Global r':>10}  {'Enrichment':>10}")
    print(f"  {'-'*14}  {'-'*22}  {'-'*10}  {'-'*10}  {'-'*10}")
    for emotion, dim, enrich, raw in flat[:15]:
        print(f"  {emotion:<14}  {dim:<22}  {raw:>10.4f}  {global_mean[dim]:>10.4f}  {enrich:>+10.4f}")

    print(f"\n  BOTTOM 10 (most depleted):")
    for emotion, dim, enrich, raw in flat[-10:]:
        print(f"  {emotion:<14}  {dim:<22}  {raw:>10.4f}  {global_mean[dim]:>10.4f}  {enrich:>+10.4f}")

    # --- Figure 1: Enrichment heatmap (emotion x dimension) ---
    _plot_enrichment_heatmap(enrichment_matrix, emotions, dims, model_id, out_dir)

    # --- Figure 2: Raw appraisal strength heatmap ---
    _plot_raw_appraisal_in_circuit_heatmap(raw_matrix, emotions, dims, model_id, out_dir)

    # --- Figure 3: Top-5 overlap count matrix ---
    _plot_overlap_count_heatmap(overlap_count_matrix, emotions, dims, model_id, out_dir)

    # --- Figure 4: Per-emotion circuit site visualization ---
    _plot_emotion_circuit_sites(circuit_per_em, model_id, out_dir)

    print(f"\n  Per-emotion figures saved to: {out_dir}")
    return {"emotions": emotions, "dims": dims, "enrichment": enrichment_matrix,
            "raw": raw_matrix, "overlap_counts": overlap_count_matrix}


def _plot_enrichment_heatmap(enrichment: np.ndarray, emotions: list[str], dims: list[str],
                             model_id: str, out_dir: Path):
    fig, ax = plt.subplots(figsize=(max(12, len(dims) * 0.9), max(5, len(emotions) * 0.45)))
    vmax = max(abs(enrichment.min()), abs(enrichment.max()))
    im = ax.imshow(enrichment, aspect="auto", cmap="RdBu_r", interpolation="nearest",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels(dims, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(emotions)))
    ax.set_yticklabels(emotions, fontsize=9)
    ax.set_title(f"{model_id}: Appraisal enrichment in each emotion's circuit\n"
                 f"(red = appraisal stronger in this circuit than global avg, blue = weaker)",
                 fontsize=11)

    for ei in range(len(emotions)):
        for di in range(len(dims)):
            val = enrichment[ei, di]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(di, ei, f"{val:+.3f}", ha="center", va="center", fontsize=6, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Enrichment (circuit mean r - global mean r)")
    fig.tight_layout()
    fig.savefig(out_dir / "emotion_appraisal_enrichment_heatmap.png", dpi=150)
    plt.close(fig)


def _plot_raw_appraisal_in_circuit_heatmap(raw: np.ndarray, emotions: list[str], dims: list[str],
                                            model_id: str, out_dir: Path):
    fig, ax = plt.subplots(figsize=(max(12, len(dims) * 0.9), max(5, len(emotions) * 0.45)))
    im = ax.imshow(raw, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels(dims, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(emotions)))
    ax.set_yticklabels(emotions, fontsize=9)
    ax.set_title(f"{model_id}: Mean appraisal probe strength at each emotion's circuit sites",
                 fontsize=11)

    for ei in range(len(emotions)):
        for di in range(len(dims)):
            val = raw[ei, di]
            color = "white" if val > 0.4 else "black"
            ax.text(di, ei, f"{val:.3f}", ha="center", va="center", fontsize=6, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean selection_corr")
    fig.tight_layout()
    fig.savefig(out_dir / "emotion_appraisal_raw_in_circuit.png", dpi=150)
    plt.close(fig)


def _plot_overlap_count_heatmap(counts: np.ndarray, emotions: list[str], dims: list[str],
                                model_id: str, out_dir: Path):
    fig, ax = plt.subplots(figsize=(max(12, len(dims) * 0.9), max(5, len(emotions) * 0.45)))
    im = ax.imshow(counts, aspect="auto", cmap="Blues", interpolation="nearest",
                   vmin=0, vmax=5)
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels(dims, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(emotions)))
    ax.set_yticklabels(emotions, fontsize=9)
    ax.set_title(f"{model_id}: How many top-5 appraisal sites fall in each emotion's circuit?\n"
                 f"(max = 5; higher = that appraisal is concentrated inside that circuit)",
                 fontsize=10)

    for ei in range(len(emotions)):
        for di in range(len(dims)):
            val = int(counts[ei, di])
            color = "white" if val >= 3 else "black"
            ax.text(di, ei, str(val), ha="center", va="center", fontsize=8, color=color,
                    fontweight="bold" if val >= 3 else "normal")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Count of top-5 appraisal sites in circuit")
    fig.tight_layout()
    fig.savefig(out_dir / "emotion_appraisal_top5_overlap.png", dpi=150)
    plt.close(fig)


def _plot_emotion_circuit_sites(circuit_per_em: dict[str, set[tuple[int, int]]],
                                model_id: str, out_dir: Path):
    """Visualize each emotion's circuit sites on the (layer, loc) grid."""
    emotions = sorted(circuit_per_em.keys())
    all_layers = set()
    all_locs = set()
    for sites in circuit_per_em.values():
        for l, c in sites:
            all_layers.add(l)
            all_locs.add(c)
    layers = sorted(all_layers)
    locs = sorted(all_locs)
    layer_to_idx = {l: i for i, l in enumerate(layers)}
    loc_to_idx = {l: i for i, l in enumerate(locs)}

    grid = np.zeros((len(emotions), len(layers) * len(locs)))
    col_labels = []
    for li, layer in enumerate(layers):
        for ci, loc in enumerate(locs):
            col_labels.append(f"L{layer}.{loc}")
            for ei, emotion in enumerate(emotions):
                if (layer, loc) in circuit_per_em[emotion]:
                    grid[ei, li * len(locs) + ci] = 1.0

    fig, ax = plt.subplots(figsize=(max(14, len(col_labels) * 0.25), max(4, len(emotions) * 0.4)))
    im = ax.imshow(grid, aspect="auto", cmap="Blues", interpolation="nearest", vmin=0, vmax=1)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=5)
    ax.set_yticks(range(len(emotions)))
    ax.set_yticklabels(emotions, fontsize=8)
    ax.set_title(f"{model_id}: Which (layer, loc) sites are in each emotion's circuit?",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "emotion_circuit_site_membership.png", dpi=150)
    plt.close(fig)


def main():
    print("=" * 70)
    print("  APPRAISAL OVERLAP DIAGNOSTIC")
    print("  Where does appraisal information live vs. the emotion circuit?")
    print("=" * 70)

    results = {}
    for model_id in MODELS:
        r = run_diagnostic(model_id)
        if r:
            results[model_id] = r

    if len(results) > 1:
        print(f"\n{'='*70}")
        print("  CROSS-MODEL SUMMARY")
        print(f"{'='*70}")
        for mid, r in results.items():
            nc = r.get("non_circuit_mean_appraisal_r")
            nc_str = f"{nc:.4f}" if nc is not None else "N/A"
            print(f"  {mid}: circuit covers {r['circuit_coverage']:.0%} of grid, "
                  f"circuit appraisal r={r['circuit_mean_appraisal_r']:.4f}, "
                  f"non-circuit r={nc_str}")

    # Per-emotion breakdowns
    print(f"\n{'='*70}")
    print("  PER-EMOTION x PER-DIMENSION ANALYSIS")
    print(f"{'='*70}")
    for model_id in MODELS:
        run_per_emotion_appraisal_diagnostic(model_id)

    print("\nDone.")


if __name__ == "__main__":
    main()

"""
Appraisal theory analysis: tests whether appraisal information builds emotion
representations in LLMs, consistent with cognitive appraisal theory.

Six analyses:
  A. Circuit structure characterization (location entropy, overlap)
  B. Layer onset comparison (appraisal vs emotion)
  C. Within-layer location ordering
  D. Appraisal-to-emotion reconstruction
  E. Ridge vs binary probe direction comparison
  F. Cross-layer appraisal-to-emotion prediction

Run:  python -m pipeline.appraisal_theory [--model_id ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from pipeline.core.config import (
    REPO_ROOT, COMMON_APPRAISAL, DEFAULT_MODEL_ID,
    EXTRACTION_TOKENS, RANDOM_STATE,
    ONSET_THRESHOLD_EMOTION_AUC, ONSET_THRESHOLD_APPRAISAL_CORR,
    get_appraisal_theory_dir, get_appraisal_structure_dir,
    get_circuit_dir, get_probes_dir,
    get_appraisal_binary_probes_dir,
)
from pipeline.core.model_config import get_extraction_layers, get_extraction_locs
from pipeline.core.research_contracts import (
    appraisal_probe_direction_raw, emotion_probe_direction_raw,
)
def _ensure_repo():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Analysis A: Circuit Structure Characterization
# ---------------------------------------------------------------------------

def analyze_circuit_structure(model_id: str, out_dir: Path, logger) -> dict:
    circuit_path = get_circuit_dir(model_id) / "circuit_top_k_selection.json"
    if not circuit_path.exists():
        logger.warning("circuit_top_k_selection.json not found, skipping A.")
        return {}
    sel = json.loads(circuit_path.read_text(encoding="utf-8"))
    per_em = sel.get("per_emotion_pairs", {})
    if not per_em:
        return {}

    emotions = sorted(per_em.keys())
    all_locs_set: set[int] = set()
    for pairs in per_em.values():
        for _, loc in pairs:
            all_locs_set.add(int(loc))
    all_locs = sorted(all_locs_set)

    rows = []
    sites_by_em: dict[str, set[tuple[int, int]]] = {}
    for em in emotions:
        pairs = [(int(l), int(c)) for l, c in per_em[em]]
        sites_by_em[em] = set(pairs)
        loc_counts = {loc: 0 for loc in all_locs}
        layers_list = []
        for l, c in pairs:
            loc_counts[c] = loc_counts.get(c, 0) + 1
            layers_list.append(l)
        n = len(pairs)
        loc_fracs = {f"loc{loc}_frac": loc_counts[loc] / n if n else 0 for loc in all_locs}
        probs = [v for v in loc_fracs.values() if v > 0]
        loc_entropy = -sum(p * np.log2(p) for p in probs) if probs else 0.0
        rows.append({
            "emotion": em, "n_sites": n,
            **loc_fracs,
            "layer_min": min(layers_list) if layers_list else 0,
            "layer_max": max(layers_list) if layers_list else 0,
            "layer_mean": float(np.mean(layers_list)) if layers_list else 0,
            "location_entropy": loc_entropy,
        })
    struct_df = pd.DataFrame(rows)
    struct_df.to_csv(out_dir / "circuit_structure_summary.csv", index=False)

    overlap_rows = []
    for i, em_a in enumerate(emotions):
        for j, em_b in enumerate(emotions):
            sa, sb = sites_by_em[em_a], sites_by_em[em_b]
            inter = len(sa & sb)
            union = len(sa | sb)
            jaccard = inter / union if union else 0.0
            overlap_rows.append({
                "emotion_a": em_a, "emotion_b": em_b,
                "jaccard": jaccard, "shared_sites": inter, "total_union": union,
            })
    overlap_df = pd.DataFrame(overlap_rows)
    overlap_df.to_csv(out_dir / "circuit_overlap_matrix.csv", index=False)

    _plot_circuit_structure(struct_df, overlap_df, emotions, all_locs, out_dir)
    logger.info(f"  A: circuit structure -> {out_dir / 'circuit_structure_summary.csv'}")
    return {"circuit_structure": struct_df.to_dict(orient="records")}


def _plot_circuit_structure(struct_df, overlap_df, emotions, all_locs, out_dir):
    loc_labels = {3: "attn_out(3)", 6: "mlp_out(6)", 7: "layer_out(7)"}
    struct_sorted = struct_df.sort_values("location_entropy", ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(4, len(emotions) * 0.35)))
    bottom = np.zeros(len(struct_sorted))
    colors = ["#1976D2", "#FF9800", "#4CAF50", "#9C27B0"]
    loc_cols = [c for c in struct_sorted.columns if c.startswith("loc") and c.endswith("_frac")]
    for ci, col in enumerate(loc_cols):
        loc_val = int(col.replace("loc", "").replace("_frac", ""))
        vals = struct_sorted[col].values
        ax.barh(range(len(struct_sorted)), vals, left=bottom,
                label=loc_labels.get(loc_val, f"loc {loc_val}"),
                color=colors[ci % len(colors)])
        bottom += vals
    ax.set_yticks(range(len(struct_sorted)))
    ax.set_yticklabels(struct_sorted["emotion"].values, fontsize=9)
    ax.set_xlabel("Fraction of circuit sites")
    ax.set_title("Circuit location distribution (sorted by location entropy)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "circuit_location_distribution.png", dpi=150)
    plt.close(fig)

    n = len(emotions)
    jac_matrix = np.zeros((n, n))
    for _, row in overlap_df.iterrows():
        i = emotions.index(row["emotion_a"])
        j = emotions.index(row["emotion_b"])
        jac_matrix[i, j] = row["jaccard"]
    fig, ax = plt.subplots(figsize=(max(7, n * 0.55), max(6, n * 0.5)))
    dist = 1 - jac_matrix
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    order = dendrogram(Z, no_plot=True)["leaves"]
    jac_ordered = jac_matrix[np.ix_(order, order)]
    labels_ordered = [emotions[i] for i in order]
    im = ax.imshow(jac_ordered, cmap="YlOrRd", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels_ordered, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels_ordered, fontsize=8)
    for ei in range(n):
        for ej in range(n):
            ax.text(ej, ei, f"{jac_ordered[ei, ej]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if jac_ordered[ei, ej] > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Jaccard similarity")
    ax.set_title("Inter-emotion circuit overlap (clustered)")
    fig.tight_layout()
    fig.savefig(out_dir / "circuit_overlap_heatmap.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Analysis B: Layer Onset Comparison
# ---------------------------------------------------------------------------

def analyze_layer_onset(model_id: str, out_dir: Path, logger) -> dict:
    probes_dir = get_probes_dir(model_id)
    emotion_csv = probes_dir / "binary_ova_probes" / "probe_summary.csv"
    appraisal_csv = probes_dir / "appraisal_probe_validation_detail.csv"
    if not emotion_csv.exists() or not appraisal_csv.exists():
        logger.warning("Probe summaries not found, skipping B.")
        return {}

    em_df = pd.read_csv(emotion_csv)
    ap_df = pd.read_csv(appraisal_csv)
    extraction_locs = get_extraction_locs(model_id)

    onset_rows = []
    em_agg = em_df.groupby(["emotion", "layer"])["test_roc_auc"].max().reset_index()
    for emotion in em_agg["emotion"].unique():
        sub = em_agg[em_agg["emotion"] == emotion].sort_values("layer")
        peak_row = sub.loc[sub["test_roc_auc"].idxmax()]
        above = sub[sub["test_roc_auc"] >= ONSET_THRESHOLD_EMOTION_AUC]
        onset_layer = int(above["layer"].min()) if not above.empty else -1
        onset_rows.append({
            "name": emotion, "type": "emotion",
            "onset_layer": onset_layer,
            "peak_layer": int(peak_row["layer"]),
            "peak_metric": float(peak_row["test_roc_auc"]),
        })

    ap_agg = ap_df.groupby(["dimension", "layer"])["selection_corr"].max().reset_index()
    for dim in ap_agg["dimension"].unique():
        sub = ap_agg[ap_agg["dimension"] == dim].sort_values("layer")
        peak_row = sub.loc[sub["selection_corr"].idxmax()]
        above = sub[sub["selection_corr"] >= ONSET_THRESHOLD_APPRAISAL_CORR]
        onset_layer = int(above["layer"].min()) if not above.empty else -1
        onset_rows.append({
            "name": dim, "type": "appraisal",
            "onset_layer": onset_layer,
            "peak_layer": int(peak_row["layer"]),
            "peak_metric": float(peak_row["selection_corr"]),
        })

    onset_df = pd.DataFrame(onset_rows)
    onset_df.to_csv(out_dir / "onset_comparison.csv", index=False)
    _plot_onset(em_agg, ap_agg, onset_df, out_dir)
    logger.info(f"  B: layer onset -> {out_dir / 'onset_comparison.csv'}")
    return {"onset": onset_df.to_dict(orient="records")}


def _plot_onset(em_agg, ap_agg, onset_df, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    cmap_warm = plt.cm.Oranges
    cmap_cool = plt.cm.Blues
    ap_dims = sorted(ap_agg["dimension"].unique())
    em_names = sorted(em_agg["emotion"].unique())
    for i, dim in enumerate(ap_dims):
        sub = ap_agg[ap_agg["dimension"] == dim].sort_values("layer")
        color = cmap_warm(0.3 + 0.6 * i / max(1, len(ap_dims) - 1))
        ax1.plot(sub["layer"], sub["selection_corr"], alpha=0.5, color=color, linewidth=1)
    for i, em in enumerate(em_names):
        sub = em_agg[em_agg["emotion"] == em].sort_values("layer")
        color = cmap_cool(0.3 + 0.6 * i / max(1, len(em_names) - 1))
        ax1.plot(sub["layer"], sub["test_roc_auc"], alpha=0.5, color=color, linewidth=1)
    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(color=cmap_warm(0.6), label="Appraisal dims"),
                        Patch(color=cmap_cool(0.6), label="Emotions")], fontsize=8)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Probe performance")
    ax1.set_title("Performance by layer")

    valid_onset = onset_df[onset_df["onset_layer"] >= 0]
    ap_onsets = valid_onset[valid_onset["type"] == "appraisal"]["onset_layer"].values
    em_onsets = valid_onset[valid_onset["type"] == "emotion"]["onset_layer"].values
    bp_data = []
    bp_labels = []
    if len(ap_onsets):
        bp_data.append(ap_onsets)
        bp_labels.append(f"Appraisal\n(n={len(ap_onsets)}, med={np.median(ap_onsets):.0f})")
    if len(em_onsets):
        bp_data.append(em_onsets)
        bp_labels.append(f"Emotion\n(n={len(em_onsets)}, med={np.median(em_onsets):.0f})")
    if bp_data:
        bp = ax2.boxplot(bp_data, labels=bp_labels, patch_artist=True)
        colors = ["#FF9800", "#1976D2"]
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.5)
    ax2.set_ylabel("Onset layer")
    ax2.set_title("Onset layer comparison")
    fig.suptitle("Analysis B: When does appraisal vs emotion information appear?", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "onset_comparison.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Analysis C: Within-Layer Location Ordering
# ---------------------------------------------------------------------------

def analyze_location_ordering(model_id: str, out_dir: Path, logger) -> dict:
    probes_dir = get_probes_dir(model_id)
    emotion_csv = probes_dir / "binary_ova_probes" / "probe_summary.csv"
    appraisal_csv = probes_dir / "appraisal_probe_validation_detail.csv"
    if not emotion_csv.exists() or not appraisal_csv.exists():
        logger.warning("Probe summaries not found, skipping C.")
        return {}

    em_df = pd.read_csv(emotion_csv)
    ap_df = pd.read_csv(appraisal_csv)
    extraction_layers = get_extraction_layers(model_id)
    extraction_locs = get_extraction_locs(model_id)

    em_mean = em_df.groupby(["layer", "loc"])["test_roc_auc"].mean().reset_index()
    ap_mean = ap_df.groupby(["layer", "loc"])["selection_corr"].mean().reset_index()

    rows = []
    for layer in extraction_layers:
        em_sub = em_mean[em_mean["layer"] == layer]
        ap_sub = ap_mean[ap_mean["layer"] == layer]
        em_best_loc = int(em_sub.loc[em_sub["test_roc_auc"].idxmax(), "loc"]) if not em_sub.empty else -1
        ap_best_loc = int(ap_sub.loc[ap_sub["selection_corr"].idxmax(), "loc"]) if not ap_sub.empty else -1
        row = {"layer": layer, "emotion_best_loc": em_best_loc, "appraisal_best_loc": ap_best_loc}
        for loc in extraction_locs:
            em_val = em_sub.loc[em_sub["loc"] == loc, "test_roc_auc"]
            ap_val = ap_sub.loc[ap_sub["loc"] == loc, "selection_corr"]
            row[f"emotion_loc{loc}"] = float(em_val.values[0]) if len(em_val) else float("nan")
            row[f"appraisal_loc{loc}"] = float(ap_val.values[0]) if len(ap_val) else float("nan")
        rows.append(row)

    loc_df = pd.DataFrame(rows)
    loc_df.to_csv(out_dir / "location_ordering_by_layer.csv", index=False)
    _plot_location_ordering(loc_df, extraction_layers, extraction_locs, out_dir)
    logger.info(f"  C: location ordering -> {out_dir / 'location_ordering_by_layer.csv'}")
    return {"location_ordering": loc_df.to_dict(orient="records")}


def _plot_location_ordering(loc_df, layers, locs, out_dir):
    loc_labels = {3: "attn(3)", 6: "mlp(6)", 7: "out(7)"}
    n_layers = len(layers)
    n_locs = len(locs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, n_layers * 0.5), 4))
    for ax, prefix, title, cmap in [
        (ax1, "appraisal_loc", "Appraisal probe strength", "YlOrRd"),
        (ax2, "emotion_loc", "Emotion probe strength", "YlGnBu"),
    ]:
        grid = np.full((n_locs, n_layers), np.nan)
        for li, layer in enumerate(layers):
            row = loc_df[loc_df["layer"] == layer]
            if row.empty:
                continue
            for ci, loc in enumerate(locs):
                col = f"{prefix}{loc}"
                if col in row.columns:
                    grid[ci, li] = float(row[col].values[0])
        im = ax.imshow(grid, aspect="auto", cmap=cmap, interpolation="nearest")
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels([str(l) for l in layers], fontsize=6)
        ax.set_yticks(range(n_locs))
        ax.set_yticklabels([loc_labels.get(l, str(l)) for l in locs], fontsize=8)
        ax.set_xlabel("Layer")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Analysis C: Appraisal vs emotion performance by (layer, loc)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "location_ordering_heatmaps.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Analysis D: Appraisal-to-Emotion Reconstruction
# ---------------------------------------------------------------------------

def analyze_reconstruction(model_id: str, out_dir: Path, logger) -> dict:
    import torch

    circuit_dir = get_circuit_dir(model_id)
    probes_dir = get_probes_dir(model_id)
    extraction_layers = get_extraction_layers(model_id)
    extraction_locs = get_extraction_locs(model_id)
    token_key = EXTRACTION_TOKENS[0]

    sel_hs_path = circuit_dir / "selection_hidden_states.pt"
    test_hs_path = circuit_dir / "test_hidden_states.pt"
    if not test_hs_path.exists():
        test_hs_path = circuit_dir / "val_hidden_states.pt"
    if not sel_hs_path.exists() or not test_hs_path.exists():
        logger.warning("Cached hidden states not found, skipping D.")
        return {}

    sel_labels_path = circuit_dir / "selection_labels.csv"
    test_labels_path = circuit_dir / "test_labels.csv"
    if not test_labels_path.exists():
        test_labels_path = circuit_dir / "val_labels.csv"
    if not sel_labels_path.exists() or not test_labels_path.exists():
        logger.warning("Label CSVs not found, skipping D.")
        return {}

    appraisal_probes_path = probes_dir / "appraisal_regression_probes.pt"
    if not appraisal_probes_path.exists():
        logger.warning("Appraisal regression probes not found, skipping D.")
        return {}

    logger.info("  D: loading cached hidden states and probes for reconstruction...")
    sel_hs = torch.load(sel_hs_path, map_location="cpu", weights_only=False)
    test_hs = torch.load(test_hs_path, map_location="cpu", weights_only=False)
    if isinstance(sel_hs, torch.Tensor):
        sel_hs = sel_hs.numpy()
    if isinstance(test_hs, torch.Tensor):
        test_hs = test_hs.numpy()

    sel_labels_df = pd.read_csv(sel_labels_path)
    test_labels_df = pd.read_csv(test_labels_path)
    sel_emotions = sel_labels_df["emotion"].values
    test_emotions = test_labels_df["emotion"].values

    all_emotions = sorted(set(sel_emotions) | set(test_emotions))
    em_to_idx = {e: i for i, e in enumerate(all_emotions)}
    sel_y = np.array([em_to_idx[e] for e in sel_emotions])
    test_y = np.array([em_to_idx[e] for e in test_emotions])

    appraisal_probes = torch.load(appraisal_probes_path, map_location="cpu", weights_only=False)
    dims = [d for d in COMMON_APPRAISAL if d in appraisal_probes]

    emotion_probes_path = list((probes_dir / "binary_ova_probes").glob("binary_ova_probes_*.pt"))
    emotion_probes = None
    if emotion_probes_path:
        emotion_probes = torch.load(emotion_probes_path[0], map_location="cpu", weights_only=False)

    recon_rows = []
    best_recon_acc = -1
    best_recon_layer_loc = (0, 0)
    best_cm = None

    for layer_idx, layer in enumerate(extraction_layers):
        for loc_idx, loc in enumerate(extraction_locs):
            sel_appraisal_vecs = np.zeros((len(sel_emotions), len(dims)))
            test_appraisal_vecs = np.zeros((len(test_emotions), len(dims)))
            ok = True
            for di, dim in enumerate(dims):
                rec = appraisal_probes.get(dim, {}).get(layer, {}).get(loc, {}).get(token_key)
                if rec is None:
                    ok = False
                    break
                scaler = rec.get("scaler")
                ridge = rec.get("ridge")
                if ridge is None or scaler is None:
                    ok = False
                    break
                X_sel = sel_hs[:, layer_idx, loc_idx, 0, :].astype(np.float64)
                X_test = test_hs[:, layer_idx, loc_idx, 0, :].astype(np.float64)
                sel_appraisal_vecs[:, di] = ridge.predict(scaler.transform(X_sel))
                test_appraisal_vecs[:, di] = ridge.predict(scaler.transform(X_test))
            if not ok:
                continue

            scaler_r = StandardScaler()
            X_sel_z = scaler_r.fit_transform(sel_appraisal_vecs)
            X_test_z = scaler_r.transform(test_appraisal_vecs)
            clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=RANDOM_STATE)
            clf.fit(X_sel_z, sel_y)
            test_pred = clf.predict(X_test_z)
            recon_acc = float(accuracy_score(test_y, test_pred))
            recon_f1 = float(f1_score(test_y, test_pred, average="macro", zero_division=0))

            direct_acc = float("nan")
            direct_auc = float("nan")
            if emotion_probes is not None:
                scores = np.zeros((len(test_emotions), len(all_emotions)))
                for ei, em in enumerate(all_emotions):
                    p_layer = emotion_probes.get(layer, emotion_probes.get(str(layer), {}))
                    p_loc = p_layer.get(loc, p_layer.get(str(loc), {}))
                    p_tok = p_loc.get(token_key, p_loc.get(str(token_key), {}))
                    rec = p_tok.get(em, {})
                    w = rec.get("weights")
                    b = rec.get("bias", 0.0)
                    sc = rec.get("scaler")
                    if w is not None:
                        X_t = test_hs[:, layer_idx, loc_idx, 0, :].astype(np.float64)
                        if sc is not None and hasattr(sc, "transform"):
                            X_t = sc.transform(X_t)
                        scores[:, ei] = X_t @ np.asarray(w).ravel() + float(b)
                em_pred = np.argmax(scores, axis=1)
                direct_acc = float(accuracy_score(test_y, em_pred))
                try:
                    direct_auc = float(roc_auc_score(
                        test_y, scores, multi_class="ovr", average="macro"
                    ))
                except Exception:
                    direct_auc = float("nan")

            recon_rows.append({
                "layer": int(layer), "loc": int(loc),
                "reconstruction_accuracy": recon_acc,
                "reconstruction_macro_f1": recon_f1,
                "direct_emotion_accuracy": direct_acc,
                "direct_emotion_roc_auc": direct_auc,
                "n_test": len(test_y),
                "n_classes": len(all_emotions),
            })
            if recon_acc > best_recon_acc:
                best_recon_acc = recon_acc
                best_recon_layer_loc = (layer, loc)
                best_cm = confusion_matrix(test_y, test_pred)

    recon_df = pd.DataFrame(recon_rows)
    recon_df.to_csv(out_dir / "reconstruction_by_layer_loc.csv", index=False)
    _plot_reconstruction(recon_df, best_cm, all_emotions, best_recon_layer_loc, out_dir)
    logger.info(f"  D: reconstruction -> {out_dir / 'reconstruction_by_layer_loc.csv'}")
    return {"reconstruction": recon_df.to_dict(orient="records")}


def _plot_reconstruction(recon_df, best_cm, all_emotions, best_ll, out_dir):
    if recon_df.empty:
        return
    best_loc = recon_df.groupby("layer", group_keys=False).apply(
        lambda g: g.loc[g["reconstruction_accuracy"].idxmax()]
    ).reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(best_loc["layer"], best_loc["reconstruction_accuracy"], "o-",
             color="#FF5722", label="Appraisal reconstruction", linewidth=2)
    if "direct_emotion_accuracy" in best_loc.columns:
        valid = best_loc["direct_emotion_accuracy"].notna()
        ax1.plot(best_loc.loc[valid, "layer"], best_loc.loc[valid, "direct_emotion_accuracy"],
                 "s-", color="#1976D2", label="Direct emotion readout", linewidth=2)
    n_classes = best_loc["n_classes"].iloc[0] if "n_classes" in best_loc.columns else 13
    ax1.axhline(1.0 / n_classes, color="gray", linestyle="--", alpha=0.5, label=f"Chance (1/{n_classes})")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Appraisal reconstruction vs direct emotion readout")
    ax1.legend(fontsize=9)

    if best_cm is not None and len(all_emotions) > 0:
        cm_norm = best_cm.astype(float) / best_cm.sum(axis=1, keepdims=True).clip(1)
        im = ax2.imshow(cm_norm, cmap="Blues", interpolation="nearest", vmin=0, vmax=1)
        ax2.set_xticks(range(len(all_emotions)))
        ax2.set_xticklabels(all_emotions, rotation=45, ha="right", fontsize=6)
        ax2.set_yticks(range(len(all_emotions)))
        ax2.set_yticklabels(all_emotions, fontsize=6)
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("True")
        ax2.set_title(f"Reconstruction confusion (L{best_ll[0]}, loc{best_ll[1]})")
        fig.colorbar(im, ax=ax2, shrink=0.8)
    fig.suptitle("Analysis D: Can appraisal information reconstruct emotion?", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "reconstruction_curves.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Analysis E: Ridge vs Binary Probe Direction Comparison
# ---------------------------------------------------------------------------

def analyze_direction_comparison(model_id: str, out_dir: Path, logger) -> dict:
    import torch

    probes_dir = get_probes_dir(model_id)
    ridge_path = probes_dir / "appraisal_regression_probes.pt"
    binary_dir = get_appraisal_binary_probes_dir(model_id)
    binary_path = list(binary_dir.glob("appraisal_binary_ova_probes_*.pt")) if binary_dir.exists() else []
    if not ridge_path.exists() or not binary_path:
        logger.warning("Ridge or binary probes not found, skipping E.")
        return {}

    ridge_probes = torch.load(ridge_path, map_location="cpu", weights_only=False)
    binary_probes = torch.load(binary_path[0], map_location="cpu", weights_only=False)
    token_key = EXTRACTION_TOKENS[0]

    ridge_csv = probes_dir / "appraisal_probe_validation_detail.csv"
    binary_csv = binary_dir / "appraisal_binary_summary.csv"
    ridge_metrics = pd.read_csv(ridge_csv) if ridge_csv.exists() else None
    binary_metrics = pd.read_csv(binary_csv) if binary_csv.exists() else None

    rows = []
    for dim in COMMON_APPRAISAL:
        if dim not in ridge_probes or dim not in binary_probes:
            continue
        for layer in ridge_probes[dim]:
            if layer not in binary_probes[dim]:
                continue
            for loc in ridge_probes[dim][layer]:
                if loc not in binary_probes[dim][layer]:
                    continue
                ridge_rec = ridge_probes[dim][layer][loc].get(token_key)
                binary_rec = binary_probes[dim][layer][loc].get(token_key)
                if ridge_rec is None or binary_rec is None:
                    continue
                ridge_dir = appraisal_probe_direction_raw(ridge_rec)
                binary_dir_vec = appraisal_probe_direction_raw(binary_rec)
                if ridge_dir is None or binary_dir_vec is None:
                    continue
                cos = _cos_sim(ridge_dir, binary_dir_vec)

                ridge_corr = float("nan")
                binary_auc = float("nan")
                if ridge_metrics is not None:
                    m = ridge_metrics[(ridge_metrics["dimension"] == dim) &
                                     (ridge_metrics["layer"] == int(layer)) &
                                     (ridge_metrics["loc"] == int(loc))]
                    if not m.empty:
                        ridge_corr = float(m["selection_corr"].values[0])
                if binary_metrics is not None:
                    m = binary_metrics[(binary_metrics["dimension"] == dim) &
                                      (binary_metrics["layer"] == int(layer)) &
                                      (binary_metrics["loc"] == int(loc))]
                    if not m.empty:
                        binary_auc = float(m["selection_roc_auc"].values[0])

                rows.append({
                    "dimension": dim, "layer": int(layer), "loc": int(loc),
                    "cosine_sim": cos, "ridge_corr": ridge_corr, "binary_auc": binary_auc,
                })

    cos_df = pd.DataFrame(rows)
    cos_df.to_csv(out_dir / "ridge_vs_binary_cosine.csv", index=False)
    _plot_direction_comparison(cos_df, out_dir)
    logger.info(f"  E: direction comparison -> {out_dir / 'ridge_vs_binary_cosine.csv'}")
    return {"direction_comparison": {"mean_cosine": float(cos_df["cosine_sim"].mean()) if len(cos_df) else 0}}


def _plot_direction_comparison(cos_df, out_dir):
    if cos_df.empty:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.hist(cos_df["cosine_sim"], bins=40, color="#4CAF50", alpha=0.7, edgecolor="white")
    ax1.axvline(cos_df["cosine_sim"].mean(), color="red", linestyle="--",
                label=f"Mean = {cos_df['cosine_sim'].mean():.3f}")
    ax1.set_xlabel("Cosine similarity")
    ax1.set_ylabel("Count")
    ax1.set_title("Ridge vs Binary probe direction alignment")
    ax1.legend()

    valid = cos_df.dropna(subset=["ridge_corr", "binary_auc"])
    if not valid.empty:
        sc = ax2.scatter(valid["ridge_corr"], valid["binary_auc"],
                         c=valid["cosine_sim"], cmap="RdYlGn", s=15, alpha=0.7, vmin=0, vmax=1)
        fig.colorbar(sc, ax=ax2, shrink=0.8, label="Cosine sim")
        ax2.set_xlabel("Ridge selection_corr")
        ax2.set_ylabel("Binary selection_roc_auc")
        ax2.set_title("Ridge performance vs Binary performance")
    fig.suptitle("Analysis E: Do Ridge and Binary probes agree on direction?", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "ridge_vs_binary_comparison.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Analysis F: Cross-Layer Appraisal-to-Emotion Prediction
# ---------------------------------------------------------------------------

def analyze_cross_layer(model_id: str, out_dir: Path, logger) -> dict:
    import torch

    circuit_dir = get_circuit_dir(model_id)
    probes_dir = get_probes_dir(model_id)
    extraction_layers = get_extraction_layers(model_id)
    extraction_locs = get_extraction_locs(model_id)
    token_key = EXTRACTION_TOKENS[0]

    sel_hs_path = circuit_dir / "selection_hidden_states.pt"
    test_hs_path = circuit_dir / "test_hidden_states.pt"
    if not test_hs_path.exists():
        test_hs_path = circuit_dir / "val_hidden_states.pt"
    if not sel_hs_path.exists() or not test_hs_path.exists():
        logger.warning("Cached hidden states not found, skipping F.")
        return {}

    appraisal_probes_path = probes_dir / "appraisal_regression_probes.pt"
    emotion_probes_paths = list((probes_dir / "binary_ova_probes").glob("binary_ova_probes_*.pt"))
    if not appraisal_probes_path.exists() or not emotion_probes_paths:
        logger.warning("Probes not found, skipping F.")
        return {}

    sel_labels_path = circuit_dir / "selection_labels.csv"
    test_labels_path = circuit_dir / "test_labels.csv"
    if not test_labels_path.exists():
        test_labels_path = circuit_dir / "val_labels.csv"
    if not sel_labels_path.exists() or not test_labels_path.exists():
        logger.warning("Label CSVs not found, skipping F.")
        return {}

    logger.info("  F: loading data for cross-layer prediction...")
    sel_hs = torch.load(sel_hs_path, map_location="cpu", weights_only=False)
    test_hs = torch.load(test_hs_path, map_location="cpu", weights_only=False)
    if isinstance(sel_hs, torch.Tensor):
        sel_hs = sel_hs.numpy()
    if isinstance(test_hs, torch.Tensor):
        test_hs = test_hs.numpy()
    appraisal_probes = torch.load(appraisal_probes_path, map_location="cpu", weights_only=False)
    emotion_probes = torch.load(emotion_probes_paths[0], map_location="cpu", weights_only=False)

    sel_labels = pd.read_csv(sel_labels_path)
    test_labels = pd.read_csv(test_labels_path)
    all_emotions = sorted(set(sel_labels["emotion"]) | set(test_labels["emotion"]))

    dims = [d for d in COMMON_APPRAISAL if d in appraisal_probes]

    def _get_appraisal_vec(hs, layer_idx, loc_idx):
        n = hs.shape[0]
        vecs = np.zeros((n, len(dims)))
        for di, dim in enumerate(dims):
            rec = appraisal_probes.get(dim, {}).get(extraction_layers[layer_idx], {}).get(
                extraction_locs[loc_idx], {}
            ).get(token_key)
            if rec and rec.get("ridge") and rec.get("scaler"):
                X = hs[:, layer_idx, loc_idx, 0, :].astype(np.float64)
                vecs[:, di] = rec["ridge"].predict(rec["scaler"].transform(X))
        return vecs

    def _get_emotion_logits(hs, layer_idx, loc_idx):
        n = hs.shape[0]
        logits = np.zeros((n, len(all_emotions)))
        layer = extraction_layers[layer_idx]
        loc = extraction_locs[loc_idx]
        for ei, em in enumerate(all_emotions):
            p_layer = emotion_probes.get(layer, emotion_probes.get(str(layer), {}))
            p_loc = p_layer.get(loc, p_layer.get(str(loc), {}))
            p_tok = p_loc.get(token_key, p_loc.get(str(token_key), {}))
            p = p_tok.get(em, {})
            w = p.get("weights")
            b = p.get("bias", 0.0)
            sc = p.get("scaler")
            if w is not None:
                X = hs[:, layer_idx, loc_idx, 0, :].astype(np.float64)
                if sc and hasattr(sc, "transform"):
                    X = sc.transform(X)
                logits[:, ei] = X @ np.asarray(w).ravel() + float(b)
        return logits

    rows = []
    for layer_idx in range(len(extraction_layers) - 1):
        for loc_idx in range(len(extraction_locs)):
            sel_ap = _get_appraisal_vec(sel_hs, layer_idx, loc_idx)
            sel_em = _get_emotion_logits(sel_hs, layer_idx, loc_idx)
            test_ap = _get_appraisal_vec(test_hs, layer_idx, loc_idx)
            test_em = _get_emotion_logits(test_hs, layer_idx, loc_idx)

            next_idx = layer_idx + 1
            sel_ap_next = _get_appraisal_vec(sel_hs, next_idx, loc_idx)
            sel_em_next = _get_emotion_logits(sel_hs, next_idx, loc_idx)
            test_ap_next = _get_appraisal_vec(test_hs, next_idx, loc_idx)
            test_em_next = _get_emotion_logits(test_hs, next_idx, loc_idx)

            for direction, X_train, y_train, X_test, y_test in [
                ("appraisal_to_emotion", sel_ap, sel_em_next, test_ap, test_em_next),
                ("emotion_to_appraisal", sel_em, sel_ap_next, test_em, test_ap_next),
            ]:
                if X_train.shape[1] == 0 or y_train.shape[1] == 0:
                    continue
                sc_x = StandardScaler()
                sc_y = StandardScaler()
                X_tr = sc_x.fit_transform(X_train)
                y_tr = sc_y.fit_transform(y_train)
                X_te = sc_x.transform(X_test)
                y_te = sc_y.transform(y_test)
                ridge = Ridge(alpha=5.0)
                ridge.fit(X_tr, y_tr)
                pred = ridge.predict(X_te)
                ss_res = np.sum((y_te - pred) ** 2)
                ss_tot = np.sum((y_te - y_te.mean(axis=0)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
                rows.append({
                    "layer_from": int(extraction_layers[layer_idx]),
                    "layer_to": int(extraction_layers[next_idx]),
                    "loc": int(extraction_locs[loc_idx]),
                    "direction": direction,
                    "r2_test": float(r2),
                })

    cross_df = pd.DataFrame(rows)
    cross_df.to_csv(out_dir / "cross_layer_prediction.csv", index=False)
    _plot_cross_layer(cross_df, out_dir)
    logger.info(f"  F: cross-layer prediction -> {out_dir / 'cross_layer_prediction.csv'}")
    return {"cross_layer": cross_df.to_dict(orient="records") if len(cross_df) < 200 else {}}


def _plot_cross_layer(cross_df, out_dir):
    if cross_df.empty:
        return
    best_loc = cross_df.groupby(["layer_from", "direction"])["r2_test"].max().reset_index()
    a2e = best_loc[best_loc["direction"] == "appraisal_to_emotion"].sort_values("layer_from")
    e2a = best_loc[best_loc["direction"] == "emotion_to_appraisal"].sort_values("layer_from")

    fig, ax = plt.subplots(figsize=(10, 5))
    if not a2e.empty:
        ax.plot(a2e["layer_from"], a2e["r2_test"], "o-", color="#FF5722",
                label="Appraisal(L) → Emotion(L+1)", linewidth=2)
    if not e2a.empty:
        ax.plot(e2a["layer_from"], e2a["r2_test"], "s-", color="#1976D2",
                label="Emotion(L) → Appraisal(L+1)", linewidth=2)
    if not a2e.empty and not e2a.empty:
        merged = a2e.merge(e2a, on="layer_from", suffixes=("_a2e", "_e2a"))
        dom_a2e = merged[merged["r2_test_a2e"] > merged["r2_test_e2a"]]
        if not dom_a2e.empty:
            ax.fill_between(dom_a2e["layer_from"],
                            dom_a2e["r2_test_e2a"], dom_a2e["r2_test_a2e"],
                            alpha=0.15, color="#FF5722", label="Appraisal dominates")
    ax.set_xlabel("Layer (source)")
    ax.set_ylabel("R² (test)")
    ax.set_title("Analysis F: Cross-layer prediction asymmetry")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "cross_layer_asymmetry.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Analysis G: Emotions in Appraisal Space (PCA biplot)
# ---------------------------------------------------------------------------

def analyze_appraisal_space(model_id: str, out_dir: Path, logger) -> dict:
    """PCA of emotions in 14-dim appraisal z-score space, with dimension loading vectors."""
    from sklearn.decomposition import PCA

    zscore_path = get_appraisal_structure_dir(model_id) / "appraisal_zscore_by_emotion.csv"
    if not zscore_path.exists():
        logger.warning("appraisal_zscore_by_emotion.csv not found, skipping G.")
        return {}

    zdf = pd.read_csv(zscore_path, index_col=0)
    dims = [c for c in COMMON_APPRAISAL if c in zdf.columns]
    if len(dims) < 3:
        return {}
    X = zdf[dims].values
    emotions = list(zdf.index)

    n_components = min(3, len(dims), len(emotions))
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(X)
    loadings = pca.components_.T  # (n_dims, n_components)
    ev = pca.explained_variance_ratio_

    coord_df = pd.DataFrame({"emotion": emotions})
    for i in range(n_components):
        coord_df[f"PC{i+1}"] = coords[:, i]
    loading_df = pd.DataFrame({"dimension": dims})
    for i in range(n_components):
        loading_df[f"PC{i+1}_loading"] = loadings[:, i]

    coord_df.to_csv(out_dir / "appraisal_space_coordinates.csv", index=False)
    loading_df.to_csv(out_dir / "appraisal_space_loadings.csv", index=False)

    _plot_appraisal_space(coord_df, loading_df, ev, out_dir)
    if n_components >= 3:
        _plot_appraisal_space_3d(coord_df, loading_df, ev, zdf, dims, out_dir)
    logger.info(f"  G: appraisal space PCA -> {out_dir / 'appraisal_space_coordinates.csv'}")
    return {"explained_variance": [float(v) for v in ev]}


def _plot_appraisal_space(coord_df, loading_df, ev, out_dir):
    fig, ax = plt.subplots(figsize=(11, 9))

    valence_colors = {
        "anger": "#D32F2F", "disgust": "#795548", "fear": "#F57C00",
        "sadness": "#1565C0", "shame": "#7B1FA2", "guilt": "#6A1B9A",
        "boredom": "#9E9E9E", "disappointment": "#78909C",
        "joy": "#FFD600", "pride": "#43A047", "relief": "#66BB6A",
        "trust": "#29B6F6", "surprise": "#FF9800", "hope": "#81C784",
        "no-emotion": "#BDBDBD", "other": "#E0E0E0",
    }

    for _, row in coord_df.iterrows():
        em = row["emotion"]
        color = valence_colors.get(em, "#616161")
        ax.scatter(row["PC1"], row["PC2"], s=120, c=color, edgecolors="black",
                   linewidth=0.8, zorder=5)
        ax.annotate(em, (row["PC1"], row["PC2"]), fontsize=9, fontweight="bold",
                    xytext=(6, 6), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, lw=0))

    scale = max(abs(coord_df["PC1"]).max(), abs(coord_df["PC2"]).max()) * 0.8
    max_loading = max(abs(loading_df["PC1_loading"]).max(), abs(loading_df["PC2_loading"]).max())
    if max_loading > 1e-6:
        arrow_scale = scale / max_loading
    else:
        arrow_scale = 1.0

    for _, row in loading_df.iterrows():
        dx = row["PC1_loading"] * arrow_scale
        dy = row["PC2_loading"] * arrow_scale
        ax.annotate(
            "", xy=(dx, dy), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#B71C1C", lw=1.5, alpha=0.6),
        )
        ax.text(dx * 1.08, dy * 1.08, row["dimension"], fontsize=7, color="#B71C1C",
                ha="center", va="center", alpha=0.85,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.5, lw=0))

    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.3)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.3)
    ax.set_xlabel(f"PC1 ({ev[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1%} variance)" if len(ev) > 1 else "PC2")
    ax.set_title("Emotions in appraisal space (PCA biplot)\n"
                 "Points = emotions, arrows = appraisal dimension loadings")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(out_dir / "appraisal_space_biplot.png", dpi=150)
    plt.close(fig)


def _wrap_hover(text: str, width: int = 45) -> str:
    """Wrap long text with <br> for plotly hover."""
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        if cur and len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        lines.append(cur)
    return "<br>".join(lines)


_VALENCE_GROUPS = {
    "negative-high": {"anger", "fear", "disgust"},
    "negative-low": {"sadness", "boredom", "disappointment"},
    "self-conscious": {"guilt", "shame"},
    "positive": {"joy", "pride", "hope", "relief"},
    "social": {"trust", "surprise"},
    "neutral": {"no-emotion", "other"},
}
_GROUP_COLORS = {
    "negative-high": "#D32F2F",
    "negative-low": "#1565C0",
    "self-conscious": "#7B1FA2",
    "positive": "#43A047",
    "social": "#FF9800",
    "neutral": "#9E9E9E",
}


def _emotion_group(em: str) -> str:
    for grp, members in _VALENCE_GROUPS.items():
        if em in members:
            return grp
    return "neutral"


def _plot_appraisal_space_3d(coord_df, loading_df, ev, zdf, dims, out_dir):
    """Interactive 3D PCA scatter with loading arrows and rich hover text."""
    import plotly.graph_objects as go

    coord_df = coord_df.copy()
    coord_df["group"] = coord_df["emotion"].apply(_emotion_group)
    coord_df["color"] = coord_df["group"].map(_GROUP_COLORS).fillna("#616161")

    hover_texts = []
    for _, row in coord_df.iterrows():
        em = row["emotion"]
        grp = row["group"]
        header = f"<b>{em}</b>  ({grp})"
        coords_line = (f"PC1: {row['PC1']:+.2f}  |  "
                       f"PC2: {row['PC2']:+.2f}  |  "
                       f"PC3: {row['PC3']:+.2f}")
        appraisal_lines = []
        if em in zdf.index:
            zrow = zdf.loc[em]
            sorted_dims = sorted(dims, key=lambda d: abs(zrow.get(d, 0)), reverse=True)
            for d in sorted_dims:
                val = zrow.get(d, 0)
                if abs(val) < 0.01:
                    continue
                bar = "+" * int(min(abs(val), 3)) if val > 0 else "-" * int(min(abs(val), 3))
                appraisal_lines.append(f"  {d}: {val:+.2f}  {bar}")
        appraisal_block = "<br>".join(appraisal_lines[:10]) if appraisal_lines else "(no strong appraisals)"
        full = f"{header}<br>{coords_line}<br><br><b>Appraisal profile (z-score):</b><br>{appraisal_block}"
        hover_texts.append(full)

    fig = go.Figure()

    for grp in sorted(_GROUP_COLORS.keys()):
        mask = coord_df["group"] == grp
        sub = coord_df[mask]
        if sub.empty:
            continue
        idx = sub.index.tolist()
        fig.add_trace(go.Scatter3d(
            x=sub["PC1"], y=sub["PC2"], z=sub["PC3"],
            mode="markers+text",
            marker=dict(size=10, color=_GROUP_COLORS[grp], opacity=0.9,
                        line=dict(width=1, color="black")),
            text=sub["emotion"],
            textposition="top center",
            textfont=dict(size=10, color="black"),
            hovertext=[hover_texts[i] for i in idx],
            hoverinfo="text",
            hoverlabel=dict(
                bgcolor="white", bordercolor=_GROUP_COLORS[grp],
                font=dict(size=11, family="Consolas, monospace"),
            ),
            name=grp,
            legendgroup=grp,
            showlegend=True,
        ))

    scale = max(
        abs(coord_df["PC1"]).max(),
        abs(coord_df["PC2"]).max(),
        abs(coord_df["PC3"]).max(),
    ) * 0.75
    max_load = max(
        abs(loading_df["PC1_loading"]).max(),
        abs(loading_df["PC2_loading"]).max(),
        abs(loading_df["PC3_loading"]).max(),
    )
    arrow_s = scale / max_load if max_load > 1e-6 else 1.0

    for _, row in loading_df.iterrows():
        dx = row["PC1_loading"] * arrow_s
        dy = row["PC2_loading"] * arrow_s
        dz = row["PC3_loading"] * arrow_s
        fig.add_trace(go.Scatter3d(
            x=[0, dx], y=[0, dy], z=[0, dz],
            mode="lines+text",
            line=dict(color="rgba(183,28,28,0.5)", width=3),
            text=["", row["dimension"]],
            textposition="top center",
            textfont=dict(size=8, color="#B71C1C"),
            hoverinfo="skip",
            showlegend=False,
        ))
        fig.add_trace(go.Cone(
            x=[dx], y=[dy], z=[dz],
            u=[dx * 0.15], v=[dy * 0.15], w=[dz * 0.15],
            colorscale=[[0, "rgba(183,28,28,0.6)"], [1, "rgba(183,28,28,0.6)"]],
            showscale=False, hoverinfo="skip", showlegend=False,
            sizemode="absolute", sizeref=0.3,
        ))

    total_var = sum(ev[:3])
    fig.update_layout(
        title=dict(
            text=(f"Emotions in Appraisal Space (3D PCA)<br>"
                  f"<span style='font-size:12px;color:gray'>"
                  f"PC1-3 explain {total_var:.1%} of variance  |  "
                  f"Arrows = appraisal dimension loadings</span>"),
            x=0.5,
        ),
        scene=dict(
            xaxis=dict(title=f"PC1 ({ev[0]:.1%})", gridcolor="#eee"),
            yaxis=dict(title=f"PC2 ({ev[1]:.1%})", gridcolor="#eee"),
            zaxis=dict(title=f"PC3 ({ev[2]:.1%})", gridcolor="#eee"),
            bgcolor="white",
        ),
        legend=dict(
            title="Emotion group",
            itemsizing="constant",
            font=dict(size=11),
        ),
        margin=dict(l=10, r=10, t=80, b=10),
        width=950, height=750,
    )

    fig.write_html(str(out_dir / "appraisal_space_3d.html"),
                   include_plotlyjs="cdn", full_html=True)


# ---------------------------------------------------------------------------
# Summary Generation
# ---------------------------------------------------------------------------

def _write_summary(model_id: str, out_dir: Path, results: dict, logger):
    lines = [
        f"# Appraisal Theory Analysis: {model_id}\n",
        "## Overview\n",
        "This analysis tests whether appraisal information precedes and builds emotion ",
        "representations in this LLM, consistent with cognitive appraisal theory.\n",
        "## Evidence Chain\n",
        "```",
        "Layer onset (B) + Location ordering (C) → Do appraisals appear before emotions?",
        "Reconstruction (D) → Can appraisals reconstruct emotions?",
        "Cross-layer prediction (F) → Is there directional appraisal→emotion flow?",
        "Circuit structure (A) → Do circuits reflect appraisal structure?",
        "Direction comparison (E) → Do Ridge and binary probes agree?",
        "```\n",
    ]

    onset_csv = out_dir / "onset_comparison.csv"
    if onset_csv.exists():
        odf = pd.read_csv(onset_csv)
        valid = odf[odf["onset_layer"] >= 0]
        ap_med = valid[valid["type"] == "appraisal"]["onset_layer"].median()
        em_med = valid[valid["type"] == "emotion"]["onset_layer"].median()
        lines.append("## B: Layer Onset\n")
        lines.append(f"- Appraisal median onset layer: **{ap_med:.0f}**\n")
        lines.append(f"- Emotion median onset layer: **{em_med:.0f}**\n")
        if ap_med < em_med:
            lines.append("- **Appraisals appear earlier** than emotions.\n")
        elif ap_med > em_med:
            lines.append("- Emotions appear earlier than appraisals.\n")
        else:
            lines.append("- Appraisals and emotions appear at the same layer.\n")
        lines.append("![Onset comparison](onset_comparison.png)\n")

    recon_csv = out_dir / "reconstruction_by_layer_loc.csv"
    if recon_csv.exists():
        rdf = pd.read_csv(recon_csv)
        if not rdf.empty:
            best = rdf.loc[rdf["reconstruction_accuracy"].idxmax()]
            lines.append("## D: Appraisal-to-Emotion Reconstruction\n")
            lines.append(f"- Best reconstruction accuracy: **{best['reconstruction_accuracy']:.1%}** ")
            lines.append(f"at layer {int(best['layer'])}, loc {int(best['loc'])}\n")
            if "direct_emotion_accuracy" in best.index and not np.isnan(best["direct_emotion_accuracy"]):
                lines.append(f"- Direct emotion probe accuracy at same site: **{best['direct_emotion_accuracy']:.1%}**\n")
            lines.append("![Reconstruction](reconstruction_curves.png)\n")

    cos_csv = out_dir / "ridge_vs_binary_cosine.csv"
    if cos_csv.exists():
        cdf = pd.read_csv(cos_csv)
        if not cdf.empty:
            lines.append("## E: Ridge vs Binary Direction Comparison\n")
            lines.append(f"- Mean cosine similarity: **{cdf['cosine_sim'].mean():.3f}**\n")
            lines.append(f"- Median: {cdf['cosine_sim'].median():.3f}, "
                         f"Min: {cdf['cosine_sim'].min():.3f}, Max: {cdf['cosine_sim'].max():.3f}\n")
            lines.append("![Direction comparison](ridge_vs_binary_comparison.png)\n")

    cross_csv = out_dir / "cross_layer_prediction.csv"
    if cross_csv.exists():
        xdf = pd.read_csv(cross_csv)
        if not xdf.empty:
            a2e_mean = xdf[xdf["direction"] == "appraisal_to_emotion"]["r2_test"].mean()
            e2a_mean = xdf[xdf["direction"] == "emotion_to_appraisal"]["r2_test"].mean()
            lines.append("## F: Cross-Layer Prediction Asymmetry\n")
            lines.append(f"- Appraisal→Emotion mean R²: **{a2e_mean:.4f}**\n")
            lines.append(f"- Emotion→Appraisal mean R²: **{e2a_mean:.4f}**\n")
            if a2e_mean > e2a_mean:
                lines.append("- **Appraisal-to-emotion prediction is stronger**, supporting feedforward model.\n")
            else:
                lines.append("- Emotion-to-appraisal prediction is equal or stronger.\n")
            lines.append("![Cross-layer](cross_layer_asymmetry.png)\n")

    struct_csv = out_dir / "circuit_structure_summary.csv"
    if struct_csv.exists():
        lines.append("## A: Circuit Structure\n")
        lines.append("![Location distribution](circuit_location_distribution.png)\n")
        lines.append("![Overlap heatmap](circuit_overlap_heatmap.png)\n")

    coord_csv = out_dir / "appraisal_space_coordinates.csv"
    if coord_csv.exists():
        lines.append("## G: Emotions in Appraisal Space\n")
        lines.append("PCA of emotions projected into the 14-dimensional appraisal z-score space.\n")
        lines.append("Arrows show appraisal dimension loading vectors.\n")
        lines.append("![Appraisal space biplot (2D)](appraisal_space_biplot.png)\n")
        if (out_dir / "appraisal_space_3d.html").exists():
            lines.append("Interactive 3D version: [appraisal_space_3d.html](appraisal_space_3d.html)\n")

    lines.append("\n## Figures\n")
    for png in sorted(out_dir.glob("*.png")):
        lines.append(f"- [{png.name}]({png.name})\n")

    (out_dir / "SUMMARY.md").write_text("".join(lines), encoding="utf-8")
    logger.info(f"  Summary -> {out_dir / 'SUMMARY.md'}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_appraisal_theory(model_id: str = DEFAULT_MODEL_ID, logger=None, **kwargs) -> dict:
    _ensure_repo()
    from utils import Log
    if logger is None:
        logger = Log("appraisal_theory").logger

    out_dir = get_appraisal_theory_dir(model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Running appraisal theory analysis for {model_id} -> {out_dir}")

    results = {}
    results["A"] = analyze_circuit_structure(model_id, out_dir, logger)
    results["B"] = analyze_layer_onset(model_id, out_dir, logger)
    results["C"] = analyze_location_ordering(model_id, out_dir, logger)
    results["D"] = analyze_reconstruction(model_id, out_dir, logger)
    results["E"] = analyze_direction_comparison(model_id, out_dir, logger)
    results["F"] = analyze_cross_layer(model_id, out_dir, logger)
    results["G"] = analyze_appraisal_space(model_id, out_dir, logger)

    _write_summary(model_id, out_dir, results, logger)
    logger.info(f"Appraisal theory analysis complete for {model_id}.")
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Appraisal theory analysis")
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    args = p.parse_args()
    run_appraisal_theory(model_id=args.model_id)

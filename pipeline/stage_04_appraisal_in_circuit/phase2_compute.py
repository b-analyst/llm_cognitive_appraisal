"""
Phase 2 computation: per-model geometry (appraisal vs emotion directions) and
correlation (circuit-layer vs default-layer appraisal predictions).

Runs at pipeline runtime for each model; no external notebook. Writes to
outputs/<model_id>/04_appraisal_in_circuit/.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

from pipeline.core.config import (
    get_circuit_dir,
    get_appraisal_in_circuit_dir,
    get_appraisal_structure_dir,
    get_probe_paths,
    EXTRACTION_TOKENS,
    COMMON_APPRAISAL,
    DEFAULT_MODEL_ID,
    CONTRASTIVE_EMOTION_PAIRS,
    SIMILAR_EMOTION_PAIRS,
)
from pipeline.core.model_config import get_extraction_layers, get_extraction_locs, probe_filename_suffix
from pipeline.core.research_contracts import (
    emotion_probe_direction_raw,
    appraisal_probe_direction_raw,
    read_json_if_exists,
    ensure_manifest_model_match,
)
from pipeline.stage_05_appraisal_theory.theory_analysis import save_phase2_theory_outputs
def _progress(message: str) -> None:
    print(f"[phase2_compute] {message}", flush=True)


def _probe_logits_at_layer_loc(probes, hidden_states, layer, loc, token_key, emotions_list, layer_idx, loc_idx, token_idx=0):
    X = hidden_states[:, layer_idx, loc_idx, token_idx, :]
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    X = X.reshape(X.shape[0], -1)
    logits = np.zeros((X.shape[0], len(emotions_list)), dtype=np.float64)

    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    for e_idx, em in enumerate(emotions_list):
        if layer not in probes or loc not in probes[layer] or token_key not in probes[layer][loc] or em not in probes[layer][loc][token_key]:
            continue
        rec = probes[layer][loc][token_key][em]
        if "error" in rec:
            continue
        scaler = rec.get("scaler")
        if scaler is not None and hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
            mean = np.asarray(scaler.mean_, dtype=np.float64)
            scale = np.asarray(scaler.scale_, dtype=np.float64)
            Xe = (X - mean) / np.where(np.abs(scale) < 1e-12, 1.0, scale)
        elif scaler is not None and hasattr(scaler, "transform"):
            Xe = scaler.transform(X)
        else:
            Xe = X
        w = rec.get("weights")
        b = rec.get("bias")
        if w is None or b is None:
            clf = rec.get("classifier")
            if clf is None or not hasattr(clf, "coef_") or not hasattr(clf, "intercept_"):
                continue
            w = clf.coef_.ravel()
            b = float(np.ravel(clf.intercept_)[0])
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        if w.shape[0] != Xe.shape[1]:
            continue
        logits[:, e_idx] = _sigmoid(Xe @ w + float(b))
    return logits


def _circuit_logits(hidden_states, probes, selected_pairs, extraction_layers, extraction_locs, emotions_list, token_key, token_idx=0):
    layer_to_idx = {l: i for i, l in enumerate(extraction_layers)}
    loc_to_idx = {l: i for i, l in enumerate(extraction_locs)}
    n = hidden_states.shape[0]
    logits = np.zeros((n, len(emotions_list)), dtype=np.float64)
    for e_idx, em in enumerate(emotions_list):
        cols = []
        for layer, loc in selected_pairs.get(em, []):
            li = layer_to_idx.get(layer)
            lci = loc_to_idx.get(loc)
            if li is None or lci is None:
                continue
            L = _probe_logits_at_layer_loc(probes, hidden_states, layer, loc, token_key, emotions_list, li, lci, token_idx)
            cols.append(L[:, e_idx])
        if cols:
            logits[:, e_idx] = np.mean(cols, axis=0)
    return logits


def _emotion_appraisal_signature_vectors(emotion: str, zscore_df: pd.DataFrame, appraisal_probes, selected_pairs: list[tuple[int, int]], token_key: int, hidden_size: int) -> dict[tuple[int, int], np.ndarray]:
    """Build one full appraisal signature vector per selected site for an emotion."""
    out = {}
    if zscore_df is None or emotion not in zscore_df.index:
        return out
    weights = zscore_df.loc[emotion]
    for layer, loc in selected_pairs:
        vec = np.zeros(hidden_size, dtype=np.float64)
        used = 0
        for dim in COMMON_APPRAISAL:
            if dim not in appraisal_probes or dim not in weights.index:
                continue
            try:
                rec = appraisal_probes[dim][layer][loc][token_key]
            except (KeyError, TypeError):
                continue
            direction = appraisal_probe_direction_raw(rec, hidden_size=hidden_size)
            if direction is None:
                continue
            vec += float(weights[dim]) * direction
            used += 1
        if used > 0:
            out[(layer, loc)] = vec
    return out


def _erase_vectors_at_pairs(hidden_states, vectors_by_pair, extraction_layers, extraction_locs, token_idx=0):
    out = hidden_states.clone() if isinstance(hidden_states, torch.Tensor) else np.array(hidden_states, copy=True)
    layer_to_idx = {l: i for i, l in enumerate(extraction_layers)}
    loc_to_idx = {l: i for i, l in enumerate(extraction_locs)}
    for (layer, loc), vec in vectors_by_pair.items():
        li = layer_to_idx.get(layer)
        lci = loc_to_idx.get(loc)
        if li is None or lci is None:
            continue
        v = np.asarray(vec, dtype=np.float64).reshape(-1)
        if isinstance(out, torch.Tensor):
            x = out[:, li, lci, token_idx, :].float()
            d = torch.tensor(v, dtype=x.dtype, device=x.device)
            d_sq = (d @ d).clamp(min=1e-8)
            coef = (x @ d) / d_sq
            x = x - coef.unsqueeze(1) * d.unsqueeze(0)
            out[:, li, lci, token_idx, :] = x.to(out.dtype)
        else:
            x = out[:, li, lci, token_idx, :].astype(np.float64)
            d_sq = max(float(np.dot(v, v)), 1e-8)
            coef = (x @ v) / d_sq
            x = x - coef[:, None] * v[None, :]
            out[:, li, lci, token_idx, :] = x.astype(out.dtype, copy=False)
    return out


def _find_probes_pt(probes_dir: Path, model_id: str) -> Path:
    """Same logic as circuit_evidence / steering_benchmark."""
    ensure_manifest_model_match(probes_dir / "probe_manifest.json", model_id, "probe")
    suffix = probe_filename_suffix(model_id)
    name = f"binary_ova_probes_{suffix}.pt"
    path = probes_dir / name
    dir_is_model_specific = str(model_id) in str(probes_dir)
    if path.exists():
        return path
    legacy = [f for f in probes_dir.glob("binary_ova_probes_layers_*.pt") if "locs_" in f.name and "tokens_" in f.name]
    if len(legacy) == 1 and (dir_is_model_specific or str(model_id) in legacy[0].name):
        return legacy[0]
    generic = list(probes_dir.glob("binary_ova_probes_*.pt"))
    if len(generic) == 1 and (dir_is_model_specific or str(model_id) in generic[0].name):
        return generic[0]
    if len(legacy) > 1 or len(generic) > 1:
        raise FileNotFoundError(f"Ambiguous probe bundle under {probes_dir}; expected a unique file for {model_id}.")
    raise FileNotFoundError(f"No combined probes .pt in {probes_dir}")


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n = (np.linalg.norm(a) * np.linalg.norm(b))
    if n < 1e-12:
        return 0.0
    return float(np.dot(a, b) / n)


def _selected_pairs_by_emotion(circuit_dir: Path, extraction_locs: list[int]) -> dict[str, list[tuple[int, int]]]:
    """Use exact selected (layer, loc) sites when available; otherwise fall back to all locs for legacy circuits.json."""
    sel = read_json_if_exists(circuit_dir / "circuit_top_k_selection.json")
    if sel and sel.get("per_emotion_pairs"):
        return {
            str(em): [(int(p[0]), int(p[1])) for p in pairs]
            for em, pairs in sel["per_emotion_pairs"].items()
        }
    with open(circuit_dir / "circuits.json", encoding="utf-8") as f:
        circuits = json.load(f)
    return {
        str(em): [(int(layer), int(loc)) for layer in layers for loc in extraction_locs]
        for em, layers in circuits.items()
    }


def compute_geometry_circuit_layers(model_id: str = DEFAULT_MODEL_ID) -> Path:
    """
    Compute cosine similarity between appraisal and emotion probe directions at circuit layers.
    Writes geometry_circuit_layers.csv to 04_appraisal_in_circuit/.
    """
    circuit_dir = get_circuit_dir(model_id)
    out_dir = get_appraisal_in_circuit_dir(model_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = get_probe_paths(model_id)
    token_key = EXTRACTION_TOKENS[0]
    extraction_layers = get_extraction_layers(model_id)
    extraction_locs = get_extraction_locs(model_id)
    selected_pairs = _selected_pairs_by_emotion(circuit_dir, extraction_locs)

    emotion_probes = torch.load(_find_probes_pt(paths.probes_dir_v2, model_id), weights_only=False)
    appraisal_path = paths.appraisal_probes_path
    if not appraisal_path.exists():
        # Write empty placeholder so summary doesn't fail
        pd.DataFrame(columns=["emotion", "dimension", "layer", "loc", "cos_sim"]).to_csv(
            out_dir / "geometry_circuit_layers.csv", index=False
        )
        return out_dir / "geometry_circuit_layers.csv"
    ensure_manifest_model_match(paths.appraisal_manifest_path, model_id, "appraisal probe")
    appraisal_probes = torch.load(appraisal_path, weights_only=False)

    rows = []
    for emotion, pair_list in selected_pairs.items():
        for layer, loc in pair_list:
            if layer not in extraction_layers or loc not in extraction_locs:
                continue
            if layer not in emotion_probes or loc not in emotion_probes[layer]:
                continue
            tok = emotion_probes[layer][loc].get(token_key)
            if not tok or emotion not in tok or "error" in tok.get(emotion, {}):
                continue
            emotion_coef = emotion_probe_direction_raw(tok[emotion])
            if emotion_coef is None:
                continue
            for dim in COMMON_APPRAISAL:
                if dim not in appraisal_probes:
                    continue
                try:
                    r = appraisal_probes[dim][layer][loc][token_key]
                except (KeyError, TypeError):
                    continue
                appraisal_coef = appraisal_probe_direction_raw(r, hidden_size=len(emotion_coef))
                if appraisal_coef is None:
                    continue
                cos = _cos_sim(emotion_coef, appraisal_coef)
                rows.append({"emotion": emotion, "dimension": dim, "layer": layer, "loc": loc, "cos_sim": cos})

    df = pd.DataFrame(rows)
    out_path = out_dir / "geometry_circuit_layers.csv"
    df.to_csv(out_path, index=False)
    return out_path


def compute_correlation_circuit_vs_default(model_id: str = DEFAULT_MODEL_ID) -> Path | None:
    """
    Correlate appraisal predictions at circuit layers with appraisal at default (last) layer.
    Writes correlation_circuit_vs_default.csv to 04_appraisal_in_circuit/.
    Returns None if val hidden states or appraisal probes missing.
    """
    circuit_dir = get_circuit_dir(model_id)
    out_dir = get_appraisal_in_circuit_dir(model_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    hs_path = circuit_dir / "test_hidden_states.pt"
    labels_path = circuit_dir / "test_labels.csv"
    if not hs_path.exists():
        hs_path = circuit_dir / "val_hidden_states.pt"
        labels_path = circuit_dir / "val_labels.csv"
    if not hs_path.exists():
        return None
    hidden_states = torch.load(hs_path, weights_only=False)
    if isinstance(hidden_states, torch.Tensor):
        hs = hidden_states.numpy()
    else:
        hs = np.asarray(hidden_states)

    paths = get_probe_paths(model_id)
    if not paths.appraisal_probes_path.exists():
        return None
    ensure_manifest_model_match(paths.appraisal_manifest_path, model_id, "appraisal probe")
    appraisal_probes = torch.load(paths.appraisal_probes_path, weights_only=False)
    token_key = EXTRACTION_TOKENS[0]
    extraction_layers = get_extraction_layers(model_id)
    extraction_locs = get_extraction_locs(model_id)

    selected_pairs = _selected_pairs_by_emotion(circuit_dir, extraction_locs)
    circuit_pairs = sorted(set((layer, loc) for pairs in selected_pairs.values() for layer, loc in pairs if layer in extraction_layers and loc in extraction_locs))
    default_layer = extraction_layers[-1] if extraction_layers else None
    if default_layer is None:
        return None

    def pred_at_layer_loc(layer_idx: int, loc_idx: int) -> dict[str, np.ndarray]:
        X = hs[:, layer_idx, loc_idx, 0, :].astype(np.float64)
        out = {}
        for dim in COMMON_APPRAISAL:
            if dim not in appraisal_probes:
                continue
            layer = extraction_layers[layer_idx]
            loc = extraction_locs[loc_idx]
            try:
                r = appraisal_probes[dim][layer][loc][token_key]
            except (KeyError, TypeError):
                continue
            if not r or "ridge" not in r:
                continue
            ridge = r["ridge"]
            scaler = r.get("scaler")
            Xe = scaler.transform(X) if scaler is not None else X
            pred = ridge.predict(Xe)
            out[dim] = pred
        return out

    default_preds = {}
    for loc_idx in range(len(extraction_locs)):
        p = pred_at_layer_loc(extraction_layers.index(default_layer), loc_idx)
        for dim, arr in p.items():
            default_preds[(dim, loc_idx)] = arr

    rows = []
    for layer, loc in circuit_pairs:
        layer_idx = extraction_layers.index(layer)
        loc_idx = extraction_locs.index(loc)
        circuit_preds = pred_at_layer_loc(layer_idx, loc_idx)
        for dim, c_pred in circuit_preds.items():
            key = (dim, loc_idx)
            if key not in default_preds:
                continue
            d_pred = default_preds[key]
            if len(c_pred) < 2:
                continue
            corr = float(np.corrcoef(c_pred, d_pred)[0, 1]) if np.std(c_pred) > 1e-12 and np.std(d_pred) > 1e-12 else 0.0
            rows.append({
                "dimension": dim,
                "circuit_layer": layer,
                "loc": loc,
                "evaluation_split": "test" if labels_path.name.startswith("test_") else "legacy_val",
                "mean_corr_with_default": corr,
            })
    if not rows:
        return None
    df = pd.DataFrame(rows)
    out_path = out_dir / "correlation_circuit_vs_default.csv"
    df.to_csv(out_path, index=False)
    return out_path


def compute_appraisal_ablation_effects(model_id: str = DEFAULT_MODEL_ID) -> Path | None:
    """
    Cache-based appraisal ablations on held-out activations.

    For each emotion, erase either:
    - the full appraisal signature vector at its selected circuit sites
    - one appraisal dimension at a time

    Measure how much the circuit readout for that emotion changes on examples labeled
    with that same emotion.
    """
    circuit_dir = get_circuit_dir(model_id)
    out_dir = get_appraisal_in_circuit_dir(model_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    hs_path = circuit_dir / "test_hidden_states.pt"
    labels_path = circuit_dir / "test_labels.csv"
    if not hs_path.exists() or not labels_path.exists():
        return None
    hidden_states = torch.load(hs_path, weights_only=False)
    test_labels = list(pd.read_csv(labels_path)["emotion"].astype(str))

    paths = get_probe_paths(model_id)
    if not paths.appraisal_probes_path.exists():
        return None
    ensure_manifest_model_match(paths.appraisal_manifest_path, model_id, "appraisal probe")
    appraisal_probes = torch.load(paths.appraisal_probes_path, weights_only=False)
    ensure_manifest_model_match(paths.probe_manifest_path, model_id, "probe")
    emotion_probes = torch.load(_find_probes_pt(paths.probes_dir_v2, model_id), weights_only=False)
    summary_df = pd.read_csv(paths.probe_summary_csv)
    emotions_list = summary_df["emotion"].unique().tolist()
    if hidden_states.shape[0] != len(test_labels):
        return None

    zscore_path = get_appraisal_structure_dir(model_id) / "appraisal_zscore_by_emotion.csv"
    zscore_df = pd.read_csv(zscore_path, index_col=0) if zscore_path.exists() else None

    extraction_layers = get_extraction_layers(model_id)
    extraction_locs = get_extraction_locs(model_id)
    token_key = EXTRACTION_TOKENS[0]
    selected_pairs = _selected_pairs_by_emotion(circuit_dir, extraction_locs)
    label_to_idx = {e: i for i, e in enumerate(emotions_list)}
    hidden_size = int(hidden_states.shape[-1])

    rows = []
    for emotion in emotions_list:
        mask = np.array([label == emotion for label in test_labels], dtype=bool)
        if not mask.any():
            continue
        pairs = selected_pairs.get(emotion, [])
        if not pairs:
            continue
        hs_em = hidden_states[mask]
        baseline_logits = _circuit_logits(
            hs_em, emotion_probes, selected_pairs, extraction_layers, extraction_locs, emotions_list, token_key
        )
        e_idx = label_to_idx[emotion]
        baseline_self_logit = float(baseline_logits[:, e_idx].mean())
        baseline_self_success = float((np.argmax(baseline_logits, axis=1) == e_idx).mean())

        full_vectors = _emotion_appraisal_signature_vectors(
            emotion, zscore_df, appraisal_probes, pairs, token_key, hidden_size
        )
        if full_vectors:
            hs_full = _erase_vectors_at_pairs(hs_em, full_vectors, extraction_layers, extraction_locs)
            logits_full = _circuit_logits(
                hs_full, emotion_probes, selected_pairs, extraction_layers, extraction_locs, emotions_list, token_key
            )
            rows.append(
                {
                    "emotion": emotion,
                    "ablation_type": "full_appraisal_signature",
                    "dimension": "all",
                    "n_samples": int(mask.sum()),
                    "n_sites": int(len(full_vectors)),
                    "baseline_self_logit": baseline_self_logit,
                    "ablated_self_logit": float(logits_full[:, e_idx].mean()),
                    "delta_self_logit": float(logits_full[:, e_idx].mean() - baseline_self_logit),
                    "baseline_self_success": baseline_self_success,
                    "ablated_self_success": float((np.argmax(logits_full, axis=1) == e_idx).mean()),
                    "delta_self_success": float((np.argmax(logits_full, axis=1) == e_idx).mean() - baseline_self_success),
                }
            )

        for dim in COMMON_APPRAISAL:
            dim_vectors = {}
            for layer, loc in pairs:
                try:
                    rec = appraisal_probes[dim][layer][loc][token_key]
                except (KeyError, TypeError):
                    continue
                direction = appraisal_probe_direction_raw(rec, hidden_size=hidden_size)
                if direction is not None:
                    dim_vectors[(layer, loc)] = direction
            if not dim_vectors:
                continue
            hs_dim = _erase_vectors_at_pairs(hs_em, dim_vectors, extraction_layers, extraction_locs)
            logits_dim = _circuit_logits(
                hs_dim, emotion_probes, selected_pairs, extraction_layers, extraction_locs, emotions_list, token_key
            )
            rows.append(
                {
                    "emotion": emotion,
                    "ablation_type": "single_dimension",
                    "dimension": dim,
                    "n_samples": int(mask.sum()),
                    "n_sites": int(len(dim_vectors)),
                    "baseline_self_logit": baseline_self_logit,
                    "ablated_self_logit": float(logits_dim[:, e_idx].mean()),
                    "delta_self_logit": float(logits_dim[:, e_idx].mean() - baseline_self_logit),
                    "baseline_self_success": baseline_self_success,
                    "ablated_self_success": float((np.argmax(logits_dim, axis=1) == e_idx).mean()),
                    "delta_self_success": float((np.argmax(logits_dim, axis=1) == e_idx).mean() - baseline_self_success),
                }
            )

    if not rows:
        return None
    df = pd.DataFrame(rows)
    out_path = out_dir / "appraisal_ablation_summary.csv"
    df.to_csv(out_path, index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        full_df = df[df["ablation_type"] == "full_appraisal_signature"].copy()
        if not full_df.empty:
            fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(full_df) + 2)))
            sns.barplot(data=full_df, y="emotion", x="delta_self_logit", ax=ax, color="tab:red")
            ax.axvline(0.0, color="black", linewidth=1)
            ax.set_title("Effect of erasing full appraisal signature")
            _save_path = out_dir / "appraisal_ablation_full_signature"
            fig.tight_layout()
            fig.savefig(_save_path.with_suffix(".png"), bbox_inches="tight")
            fig.savefig(_save_path.with_suffix(".pdf"), bbox_inches="tight")
            plt.close(fig)

        dim_df = df[df["ablation_type"] == "single_dimension"].copy()
        if not dim_df.empty:
            pivot = dim_df.pivot_table(index="emotion", columns="dimension", values="delta_self_logit", aggfunc="mean").fillna(0.0)
            fig, ax = plt.subplots(figsize=(12, max(5, 0.35 * len(pivot.index) + 2)))
            sns.heatmap(pivot, cmap="RdBu_r", center=0, ax=ax)
            ax.set_title("Effect of erasing individual appraisal dimensions")
            _save_path = out_dir / "appraisal_ablation_by_dimension"
            fig.tight_layout()
            fig.savefig(_save_path.with_suffix(".png"), bbox_inches="tight")
            fig.savefig(_save_path.with_suffix(".pdf"), bbox_inches="tight")
            plt.close(fig)
    except Exception as e:
        print(f"Could not save appraisal ablation plots: {e}")
    return out_path


def run_phase2_compute(model_id: str = DEFAULT_MODEL_ID) -> dict:
    """
    Run both geometry and correlation computation for this model; write to 04_appraisal_in_circuit/.
    Returns paths to written files (or None for correlation if skipped).
    """
    _progress(f"Starting Phase 2 compute for model `{model_id}`.")
    _progress("Computing geometry at selected circuit sites.")
    geo_path = compute_geometry_circuit_layers(model_id)
    _progress("Computing circuit-vs-default appraisal correlations.")
    corr_path = compute_correlation_circuit_vs_default(model_id)
    _progress("Computing appraisal ablation effects.")
    ablation_path = compute_appraisal_ablation_effects(model_id)
    geo_df = pd.read_csv(geo_path) if geo_path is not None and Path(geo_path).exists() else pd.DataFrame()
    corr_df = pd.read_csv(corr_path) if corr_path is not None and Path(corr_path).exists() else pd.DataFrame()
    zscore_path = get_appraisal_structure_dir(model_id) / "appraisal_zscore_by_emotion.csv"
    zscore_df = pd.read_csv(zscore_path, index_col=0) if zscore_path.exists() else None
    _progress("Writing theory-facing Phase 2 plots.")
    save_phase2_theory_outputs(
        geo_df=geo_df,
        corr_df=corr_df if corr_path is not None else None,
        appraisal_zscore_df=zscore_df,
        out_dir=get_appraisal_in_circuit_dir(model_id),
        contrastive_pairs=CONTRASTIVE_EMOTION_PAIRS,
        similar_pairs=SIMILAR_EMOTION_PAIRS,
    )
    _progress("Phase 2 compute complete.")
    return {
        "geometry_circuit_layers.csv": geo_path,
        "correlation_circuit_vs_default.csv": corr_path,
        "appraisal_ablation_summary.csv": ablation_path,
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    args = p.parse_args()
    run_phase2_compute(model_id=args.model_id)

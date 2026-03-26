"""
Native appraisal structure: baseline emotion classification, clustering on probe-derived
features, and appraisal z-score heatmap from dataset labels (opposing/comorbid framing).

Saves to pipeline/outputs/<model_id>/03_appraisal_structure/.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

from pipeline.core.config import (
    REPO_ROOT,
    COMBINED_CSV,
    get_probe_paths,
    EXTRACTION_TOKENS,
    COMMON_APPRAISAL,
    get_model_output_dir,
    get_appraisal_structure_dir,
    DEFAULT_MODEL_ID,
    SELECTION_SPLIT,
    FINAL_TEST_SPLIT,
    RANDOM_STATE,
)
from pipeline.core.model_config import get_extraction_layers, get_extraction_locs, probe_filename_suffix, get_extraction_batch_size
from pipeline.core.research_contracts import canonicalize_combined_dataset, split_combined_dataset, ensure_manifest_model_match, macro_ovr_roc_auc_from_scores
def _ensure_repo_path():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def _load_probes_and_summary(model_id: str):
    """Load probe summary and combined probes .pt (from get_probe_paths)."""
    paths = get_probe_paths(model_id)
    ensure_manifest_model_match(paths.probe_manifest_path, model_id, "probe")
    summary_df = pd.read_csv(paths.probe_summary_csv)
    suffix = probe_filename_suffix(model_id)
    name = f"binary_ova_probes_{suffix}.pt"
    path = paths.probes_dir_v2 / name
    dir_is_model_specific = str(model_id) in str(paths.probes_dir_v2)
    if not path.exists():
        # Backwards-compatible fallbacks: old long-form filenames and any binary_ova_probes_*.pt
        legacy = [f for f in paths.probes_dir_v2.glob("binary_ova_probes_layers_*.pt") if "locs_" in f.name and "tokens_" in f.name]
        if len(legacy) == 1 and (dir_is_model_specific or str(model_id) in legacy[0].name):
            path = legacy[0]
        elif len(legacy) > 1:
            raise FileNotFoundError(f"Ambiguous legacy probe bundles under {paths.probes_dir_v2}")
        if not path.exists():
            generic = list(paths.probes_dir_v2.glob("binary_ova_probes_*.pt"))
            if len(generic) == 1 and (dir_is_model_specific or str(model_id) in generic[0].name):
                path = generic[0]
            elif len(generic) > 1:
                raise FileNotFoundError(f"Ambiguous generic probe bundles under {paths.probes_dir_v2}")
    if not path.exists():
        raise FileNotFoundError(f"Probes not found under {paths.probes_dir_v2}")
    probes = torch.load(path, weights_only=False)
    return summary_df, probes


def _best_layer_loc(summary_df):
    """Single best (layer, loc) by mean test_roc_auc."""
    by = summary_df.groupby(["layer", "loc"])["test_roc_auc"].mean().reset_index()
    by = by.sort_values("test_roc_auc", ascending=False)
    r = by.iloc[0]
    return int(r["layer"]), int(r["loc"])


def _probe_logits_at(probes, hidden_states, layer, loc, token_key, emotions_list, layer_idx, loc_idx, token_idx=0):
    """(n_samples, n_emotions) logits."""
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


def _appraisal_zscore_by_emotion(df, appraisal_cols=None):
    """
    For each emotion, mean appraisal over samples; then z-score each column across emotions.
    Returns (emotions, appraisal_cols) matrix and emotion order.
    """
    appraisal_cols = appraisal_cols or [c for c in COMMON_APPRAISAL if c in df.columns]
    df = df.dropna(subset=appraisal_cols, how="all")
    by_emotion = df.groupby("emotion")[appraisal_cols].mean()
    emotions = by_emotion.index.tolist()
    mat = by_emotion.values.astype(np.float64)
    # Fill remaining nans with column mean
    for j in range(mat.shape[1]):
        col = mat[:, j]
        if np.isnan(col).any():
            col[np.isnan(col)] = np.nanmean(col)
    # Z-score per column (across emotions)
    mat = (mat - mat.mean(axis=0)) / (mat.std(axis=0) + 1e-8)
    return mat, emotions, appraisal_cols


def run_appraisal_structure(
    model_id: str = DEFAULT_MODEL_ID,
    max_per_emotion: int = 200,
    val_frac: float = 0.2,
    random_state: int = 42,
    use_cached_hidden_states: bool = True,
) -> dict:
    """
    Run baseline classification, clustering, and appraisal z-score heatmap.
    Uses a stratified sample; hidden states loaded from circuit cache if available (same val set),
    otherwise expects val_hidden_states.pt in 02_circuit (run circuit_evidence first).
    """
    out_dir = get_appraisal_structure_dir(model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    circuit_dir = get_model_output_dir(model_id) / "02_circuit"
    extraction_layers = get_extraction_layers(model_id)
    extraction_locs = get_extraction_locs(model_id)
    extraction_batch_size = get_extraction_batch_size(model_id)

    # Load data and preserve the same strict train/selection/test protocol.
    df = canonicalize_combined_dataset(pd.read_csv(COMBINED_CSV))
    text_col = "hidden_emo_text" if "hidden_emo_text" in df.columns else "situation"
    df = df.dropna(subset=["emotion", text_col])
    df["emotion"] = df["emotion"].astype(str).str.strip().str.lower()
    split_bundle = split_combined_dataset(
        df,
        selection_frac=SELECTION_SPLIT,
        test_frac=FINAL_TEST_SPLIT,
        random_state=RANDOM_STATE,
    )
    train_df = split_bundle["train"].copy()
    test_df = split_bundle["test"].copy()

    summary_df, probes = _load_probes_and_summary(model_id)
    emotions_list = summary_df["emotion"].unique().tolist()
    label_to_idx = {e: i for i, e in enumerate(emotions_list)}

    # Hidden states: prefer the held-out test cache created by circuit_evidence.
    cache_hs = circuit_dir / "test_hidden_states.pt"
    cache_labels = circuit_dir / "test_labels.csv"
    hidden_states = None
    y_true = None

    if cache_hs.exists() and cache_labels.exists():
        hidden_states = torch.load(cache_hs, weights_only=False)
        lbl_df = pd.read_csv(cache_labels)
        val_labels = list(lbl_df["emotion"].astype(str))
        y_true = np.array([label_to_idx.get(l, -1) for l in val_labels])
    elif use_cached_hidden_states and (circuit_dir / "val_hidden_states.pt").exists() and (circuit_dir / "val_labels.csv").exists():
        hidden_states = torch.load(circuit_dir / "val_hidden_states.pt", weights_only=False)
        val_labels = list(pd.read_csv(circuit_dir / "val_labels.csv")["emotion"].astype(str))
        y_true = np.array([label_to_idx.get(l, -1) for l in val_labels])

    if hidden_states is None or y_true is None:
        _ensure_repo_path()
        from utils import Log
        from experiments.utils.training_utils import OvaLogisticRegressionTrainer
        trainer = OvaLogisticRegressionTrainer(model_id, logger=Log("appraisal_structure").logger)
        trainer.load_model_and_tokenizer()
        rng = np.random.default_rng(random_state)
        sample_idx = []
        for em in test_df["emotion"].unique():
            sub = test_df.index[test_df["emotion"] == em].tolist()
            if len(sub) > max_per_emotion:
                sub = rng.choice(sub, size=max_per_emotion, replace=False).tolist()
            sample_idx.extend(sub)
        sample_df = test_df.loc[sample_idx].reset_index(drop=True)
        texts = sample_df[text_col].astype(str).tolist()
        hidden_states = trainer.extract_hidden_states(
            texts=texts,
            extraction_layers=extraction_layers,
            extraction_locs=extraction_locs,
            extraction_tokens=EXTRACTION_TOKENS,
            batch_size=extraction_batch_size,
        )
        torch.save(hidden_states, out_dir / "sample_hidden_states.pt")
        pd.DataFrame({"emotion": list(sample_df["emotion"].astype(str))}).to_csv(out_dir / "sample_labels.csv", index=False)
        y_true = np.array([label_to_idx.get(e, -1) for e in sample_df["emotion"].astype(str).tolist()])

    valid = y_true >= 0
    if not np.all(valid):
        if isinstance(hidden_states, torch.Tensor):
            hidden_states = hidden_states[torch.as_tensor(valid)]
        else:
            hidden_states = hidden_states[valid]
        y_true = y_true[valid]

    best_layer, best_loc = _best_layer_loc(summary_df)
    layer_idx = extraction_layers.index(best_layer) if best_layer in extraction_layers else 0
    loc_idx = extraction_locs.index(best_loc) if best_loc in extraction_locs else 0
    token_key = EXTRACTION_TOKENS[0]

    logits = _probe_logits_at(
        probes, hidden_states, best_layer, best_loc, token_key,
        emotions_list, layer_idx, loc_idx,
    )
    pred = np.argmax(logits, axis=1)
    baseline_acc = float(accuracy_score(y_true, pred))
    try:
        baseline_roc = float(macro_ovr_roc_auc_from_scores(y_true, logits))
    except Exception:
        baseline_roc = float("nan")

    baseline_metrics = pd.DataFrame([
        {"metric": "accuracy", "value": baseline_acc},
        {"metric": "macro_roc_auc", "value": baseline_roc},
    ])
    baseline_metrics.to_csv(out_dir / "baseline_metrics.csv", index=False)

    # Clustering on logits
    n_clusters = min(len(emotions_list), max(2, len(np.unique(y_true))))
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_ids = kmeans.fit_predict(logits)
    cluster_emotion_map = []
    for c in range(n_clusters):
        mask = cluster_ids == c
        if mask.sum() == 0:
            continue
        counts = pd.Series(y_true[mask]).value_counts()
        majority_emotion = emotions_list[counts.index[0]] if len(counts) else "unknown"
        cluster_emotion_map.append({"cluster": c, "majority_emotion": majority_emotion, "size": int(mask.sum())})
    cluster_df = pd.DataFrame(cluster_emotion_map)
    cluster_df.to_csv(out_dir / "cluster_emotion_mapping.csv", index=False)

    # Appraisal z-scores from the train split only; do not use held-out evaluation rows.
    mat, emotion_order, appraisal_cols = _appraisal_zscore_by_emotion(train_df)
    zscore_df = pd.DataFrame(mat, index=emotion_order, columns=appraisal_cols)
    zscore_df.to_csv(out_dir / "appraisal_zscore_by_emotion.csv")

    # Heatmap
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(10, max(6, len(emotion_order) * 0.35)))
        sns.heatmap(zscore_df, ax=ax, cmap="RdBu_r", center=0, vmin=-2, vmax=2)
        ax.set_title("Appraisal z-scores by emotion (dataset labels)")
        fig.tight_layout()
        fig.savefig(out_dir / "appraisal_zscore_heatmap.pdf", bbox_inches="tight")
        fig.savefig(out_dir / "appraisal_zscore_heatmap.png", bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"Heatmap failed: {e}")

    try:
        from .appraisal_label_coupling import run_appraisal_label_coupling
        run_appraisal_label_coupling(model_id=model_id)
    except Exception as e:
        print(f"Appraisal label coupling failed: {e}")

    summary_lines = [
        "# 03_appraisal_structure",
        "",
        "Baseline emotion classification on the held-out test split, clustering on held-out probe logits, and appraisal z-scores by emotion computed from the train split only.",
        "",
        f"- Baseline accuracy: `{baseline_acc:.3f}`.",
        f"- Baseline macro OVR ROC-AUC: `{baseline_roc:.3f}`.",
        f"- Number of clusters fit on held-out logits: `{n_clusters}`.",
        f"- Z-score table covers `{len(emotion_order)}` emotions and `{len(appraisal_cols)}` appraisal dimensions.",
        "",
        "Use the heatmap and `appraisal_zscore_by_emotion.csv` as descriptive structure outputs, not as direct causal evidence.",
        "",
        "## Label coupling (appraisal dimensions)",
        "",
        "- Outputs under `label_coupling/`: pairwise metrics CSVs, 2×2 dashboard figures (PNG+PDF), `manifest.json`, and `README.md`.",
        "- Label-space metrics use the **train split only**; optional probe-space metrics use **test** ridge predictions when `02_circuit/test_labels.csv` includes `dataset_row_idx` (re-run `circuit_evidence` extract if missing).",
        "- See `pipeline/stage_03_appraisal_structure/docs/APPRAISAL_LABEL_COUPLING.md` for definitions and interpretation.",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Appraisal structure outputs in {out_dir}")
    return {
        "baseline_accuracy": baseline_acc,
        "baseline_roc_auc": baseline_roc,
        "out_dir": out_dir,
        "zscore_df": zscore_df,
        "cluster_emotion_mapping": cluster_df,
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--max_per_emotion", type=int, default=200)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--no_cached_hs", action="store_true", help="Ignore circuit val cache")
    args = p.parse_args()
    run_appraisal_structure(
        model_id=args.model_id,
        max_per_emotion=args.max_per_emotion,
        val_frac=args.val_frac,
        use_cached_hidden_states=not args.no_cached_hs,
    )

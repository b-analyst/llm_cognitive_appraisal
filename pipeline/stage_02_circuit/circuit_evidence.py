"""
Circuit evidence: compare single-best (layer, loc) vs top-k fusion for emotion classification.

Loads validation hidden states (or extracts and caches), probe summary, and trained probes;
reports accuracy and macro ROC-AUC for single-best and top-k fusion; saves CSV and figure.
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score

from pipeline.core.config import (
    REPO_ROOT,
    COMBINED_CSV,
    get_probe_paths,
    EXTRACTION_TOKENS,
    get_circuit_dir,
    DEFAULT_MODEL_ID,
    CIRCUIT_TOP_K_MAX,
    SELECTION_SPLIT,
    FINAL_TEST_SPLIT,
    RANDOM_STATE,
)
from pipeline.core.model_config import get_extraction_layers, get_extraction_locs, probe_filename_suffix, get_extraction_batch_size
from pipeline.core.research_contracts import (
    canonicalize_combined_dataset,
    split_combined_dataset,
    resolve_text_columns,
    write_json,
    read_json_if_exists,
    manifest_matches,
    make_split_manifest,
    ensure_manifest_model_match,
    macro_ovr_roc_auc_from_scores,
)
# Backward compatibility for imports expecting this name
CIRCUIT_EVIDENCE_TOP_K_PAIRS = CIRCUIT_TOP_K_MAX

CIRCUIT_TOP_K_SELECTION_FILENAME = "circuit_top_k_selection.json"


def _progress(message: str) -> None:
    print(f"[circuit_evidence] {message}", flush=True)


def _ensure_repo_path():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def _load_combined_eval_splits(
    selection_frac: float = SELECTION_SPLIT,
    test_frac: float = FINAL_TEST_SPLIT,
    random_state: int = RANDOM_STATE,
):
    """
    Load the canonical combined dataset and return grouped selection/test splits.

    Selection is used for choosing k; final metrics are reported on the held-out test split.
    """
    df = canonicalize_combined_dataset(pd.read_csv(COMBINED_CSV))
    base_text_col, prompted_col = resolve_text_columns(df)
    text_col = prompted_col or base_text_col
    df = df.dropna(subset=["emotion", text_col])
    split_bundle = split_combined_dataset(
        df,
        selection_frac=selection_frac,
        test_frac=test_frac,
        random_state=random_state,
    )
    out = {}
    for split_name in ("selection", "test"):
        split_df = split_bundle[split_name].copy()
        out[split_name] = {
            "df": split_df,
            "texts": split_df[text_col].astype(str).tolist(),
            "labels": split_df["emotion"].astype(str).tolist(),
            "text_col": text_col,
            # Aligns each cached hidden-state row with combined CSV row index (for appraisal_label_coupling).
            "row_indices": [int(i) for i in split_df.index.tolist()],
        }
    return out


def _load_combined_val_split(val_frac=SELECTION_SPLIT, random_state=RANDOM_STATE):
    """
    Backward-compatible wrapper.

    Older callers expected a single validation split; we now return the strict selection split.
    """
    splits = _load_combined_eval_splits(
        selection_frac=val_frac,
        test_frac=FINAL_TEST_SPLIT,
        random_state=random_state,
    )
    return splits["selection"]["texts"], splits["selection"]["labels"]


def _split_cache_paths(circuit_dir: Path, split_name: str) -> tuple[Path, Path, Path]:
    return (
        circuit_dir / f"{split_name}_hidden_states.pt",
        circuit_dir / f"{split_name}_labels.csv",
        circuit_dir / f"{split_name}_manifest.json",
    )


def _get_or_extract_split_hidden_states(
    model_id,
    split_name,
    texts,
    labels,
    circuit_dir,
    extraction_layers,
    extraction_locs,
    manifest_payload,
    batch_size=4,
    dataset_row_idx: list[int] | None = None,
):
    """Load cached split hidden states when the manifest matches; otherwise re-extract and overwrite.

    Returns (hidden_states, labels, row_indices) where row_indices parallels labels when available.
    """
    cache_hs, cache_labels, cache_manifest = _split_cache_paths(circuit_dir, split_name)
    manifest = read_json_if_exists(cache_manifest)
    matches, _ = manifest_matches(
        manifest,
        {
            "model_id": manifest_payload["model_id"],
            "dataset": manifest_payload["dataset"],
            "split_protocol": manifest_payload["split_protocol"],
            "split_name": manifest_payload["split_name"],
        },
    )
    if matches and cache_hs.exists() and cache_labels.exists():
        _progress(f"Loading cached hidden states for `{split_name}` split.")
        hs = torch.load(cache_hs, weights_only=False)
        lbl_df = pd.read_csv(cache_labels)
        emotions = list(lbl_df["emotion"].astype(str))
        if "dataset_row_idx" in lbl_df.columns:
            row_idx = [int(x) for x in lbl_df["dataset_row_idx"].tolist()]
        elif dataset_row_idx is not None and len(dataset_row_idx) == len(emotions):
            # Legacy CSV (emotion only): backfill without re-extracting activations.
            row_idx = [int(x) for x in dataset_row_idx]
            lbl_df["dataset_row_idx"] = row_idx
            lbl_df.to_csv(cache_labels, index=False)
            if split_name == "test":
                lbl_df.to_csv(circuit_dir / "val_labels.csv", index=False)
            _progress(f"Backfilled `dataset_row_idx` onto cached `{split_name}` labels.")
        else:
            row_idx = None
        return hs, emotions, row_idx
    _progress(
        f"Extracting hidden states for `{split_name}` split "
        f"({len(texts)} texts, batch_size={batch_size})."
    )
    _ensure_repo_path()
    from utils import Log
    from experiments.utils.training_utils import OvaLogisticRegressionTrainer
    trainer = OvaLogisticRegressionTrainer(model_id, logger=Log("circuit_evidence").logger)
    trainer.load_model_and_tokenizer()
    hs = trainer.extract_hidden_states(
        texts=texts,
        extraction_layers=extraction_layers,
        extraction_locs=extraction_locs,
        extraction_tokens=EXTRACTION_TOKENS,
        batch_size=batch_size,
    )
    circuit_dir.mkdir(parents=True, exist_ok=True)
    torch.save(hs, cache_hs)
    if dataset_row_idx is not None and len(dataset_row_idx) == len(labels):
        pd.DataFrame({"emotion": labels, "dataset_row_idx": dataset_row_idx}).to_csv(cache_labels, index=False)
    else:
        pd.DataFrame({"emotion": labels}).to_csv(cache_labels, index=False)
    write_json(cache_manifest, manifest_payload)
    if split_name == "test":
        # Legacy compatibility for downstream code and older notebooks.
        torch.save(hs, circuit_dir / "val_hidden_states.pt")
        if dataset_row_idx is not None and len(dataset_row_idx) == len(labels):
            pd.DataFrame({"emotion": labels, "dataset_row_idx": dataset_row_idx}).to_csv(
                circuit_dir / "val_labels.csv", index=False
            )
        else:
            pd.DataFrame({"emotion": labels}).to_csv(circuit_dir / "val_labels.csv", index=False)
    row_out = dataset_row_idx if dataset_row_idx is not None and len(dataset_row_idx) == len(labels) else None
    return hs, labels, row_out


def _find_probes_pt(probes_dir: Path, model_id: str):
    """Return path to combined binary OVA probes .pt in probes_dir (from get_probe_paths)."""
    ensure_manifest_model_match(probes_dir / "probe_manifest.json", model_id, "probe")
    suffix = probe_filename_suffix(model_id)
    name = f"binary_ova_probes_{suffix}.pt"
    path = probes_dir / name
    dir_is_model_specific = str(model_id) in str(probes_dir)
    if path.exists():
        return path
    # Fallback for older naming schemes; refuse ambiguous matches.
    legacy = [f for f in probes_dir.glob("binary_ova_probes_layers_*.pt") if "locs_" in f.name and "tokens_" in f.name]
    if len(legacy) == 1 and (dir_is_model_specific or str(model_id) in legacy[0].name):
        return legacy[0]
    generic = list(probes_dir.glob("binary_ova_probes_*.pt"))
    if len(generic) == 1 and (dir_is_model_specific or str(model_id) in generic[0].name):
        return generic[0]
    if len(legacy) > 1 or len(generic) > 1:
        raise FileNotFoundError(f"Ambiguous probe bundle under {probes_dir}; expected a unique file for {model_id}.")
    raise FileNotFoundError(
        f"No combined probes .pt found under {probes_dir}. "
        "Ensure train_probes completed (it writes probe_summary.csv and binary_ova_probes_*.pt). "
        "If only probe_summary.csv exists, re-run the pipeline or run train_probes for this model."
    )


def _probe_logits_at_layer_loc(probes, hidden_states, layer, loc, token_key, emotions_list, layer_idx, loc_idx, token_idx):
    """Get (n_samples, n_emotions) logits from probes at one (layer, loc)."""
    # hidden_states: (n, n_layers, n_locs, n_tokens, dim)
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
            denom = np.where(np.abs(scale) < 1e-12, 1.0, scale)
            Xe = (X - mean) / denom
        elif scaler is not None and hasattr(scaler, "transform"):
            Xe = scaler.transform(X)
        else:
            Xe = X

        w = rec.get("weights", None)
        b = rec.get("bias", None)
        if w is None or b is None:
            clf = rec.get("classifier")
            coef = getattr(clf, "coef_", None) if clf is not None else None
            intercept = getattr(clf, "intercept_", None) if clf is not None else None
            if coef is None or intercept is None:
                continue
            w = coef.ravel()
            b = float(np.ravel(intercept)[0])
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        if w.shape[0] != Xe.shape[1]:
            continue
        logits[:, e_idx] = _sigmoid(Xe @ w + float(b))
    return logits


def _single_best_and_topk_pairs(probe_summary_df, top_k_pairs=CIRCUIT_EVIDENCE_TOP_K_PAIRS):
    """Compute (best_layer, best_loc) and list of top-k (layer, loc) by mean test_roc_auc."""
    by_layer_loc = probe_summary_df.groupby(["layer", "loc"])["test_roc_auc"].mean().reset_index()
    by_layer_loc = by_layer_loc.sort_values("test_roc_auc", ascending=False)
    best_row = by_layer_loc.iloc[0]
    best_layer = int(best_row["layer"])
    best_loc = int(best_row["loc"])
    topk_rows = by_layer_loc.head(top_k_pairs)
    topk_pairs = [(int(r["layer"]), int(r["loc"])) for _, r in topk_rows.iterrows()]
    return best_layer, best_loc, topk_pairs


def _topk_pairs_per_emotion(probe_summary_df, emotions_list, top_k):
    """For each emotion, return its top-k (layer, loc) pairs by test_roc_auc for that emotion."""
    result = {}
    for em in emotions_list:
        sub = probe_summary_df[probe_summary_df["emotion"] == em].sort_values(
            "test_roc_auc", ascending=False
        ).head(top_k)
        result[em] = [(int(r["layer"]), int(r["loc"])) for _, r in sub.iterrows()]
    return result


def _topk_global_pairs_list(probe_summary_df, k: int):
    """Top-k (layer, loc) by mean test_roc_auc across emotions (same list for all emotions)."""
    by_layer_loc = probe_summary_df.groupby(["layer", "loc"])["test_roc_auc"].mean().reset_index()
    by_layer_loc = by_layer_loc.sort_values("test_roc_auc", ascending=False)
    rows = by_layer_loc.head(max(1, k))
    return [(int(r["layer"]), int(r["loc"])) for _, r in rows.iterrows()]


def _score_multiclass(y_true, logits) -> tuple[float, float]:
    """Return (accuracy, macro one-vs-rest ROC-AUC over present classes, or nan)."""
    pred = np.argmax(logits, axis=1)
    acc = float(accuracy_score(y_true, pred))
    try:
        roc = float(macro_ovr_roc_auc_from_scores(y_true, logits))
    except Exception:
        roc = float("nan")
    return acc, roc


def _per_emotion_fusion_logits(
    summary_df,
    emotions_list,
    k_pairs: int,
    y_true,
    get_logits_single,
):
    """Mean logits per emotion over that emotion's top-k (layer,loc); shape (n, n_emotions)."""
    topk_per_emotion = _topk_pairs_per_emotion(summary_df, emotions_list, k_pairs)
    n_samples = len(y_true)
    logits_per_emotion = np.zeros((n_samples, len(emotions_list)), dtype=np.float64)
    for e_idx, em in enumerate(emotions_list):
        cols = []
        for layer, loc in topk_per_emotion.get(em, []):
            L = get_logits_single(layer, loc)
            if L is not None:
                cols.append(L[:, e_idx])
        if cols:
            logits_per_emotion[:, e_idx] = np.mean(cols, axis=0)
    return logits_per_emotion, topk_per_emotion


def _global_fusion_logits(summary_df, k_pairs: int, get_logits_single, emotions_list):
    pairs = _topk_global_pairs_list(summary_df, k_pairs)
    logits_list = []
    for layer, loc in pairs:
        L = get_logits_single(layer, loc)
        if L is not None:
            logits_list.append(L)
    if not logits_list:
        return None, pairs
    return np.mean(logits_list, axis=0), pairs


def _better_k_candidate(
    roc_new: float,
    acc_new: float,
    k_new: int,
    roc_best: float,
    acc_best: float,
    k_best: int,
    tie_prefer_smaller_k: bool,
) -> bool:
    """Prefer higher macro ROC-AUC, then accuracy, then smaller k if tied."""
    if not np.isnan(roc_new):
        if np.isnan(roc_best) or roc_new > roc_best:
            return True
        if roc_new < roc_best:
            return False
        # roc tie
        if acc_new > acc_best:
            return True
        if acc_new < acc_best:
            return False
        return tie_prefer_smaller_k and k_new < k_best
    # roc_new is nan
    if not np.isnan(roc_best):
        return False
    if acc_new > acc_best:
        return True
    if acc_new < acc_best:
        return False
    return tie_prefer_smaller_k and k_new < k_best


def _auto_select_k_pairs(
    summary_df,
    emotions_list,
    y_true,
    get_logits_single,
    top_k_max: int,
    tie_break_smaller_k: bool = True,
) -> tuple[int, list[dict], int, list[dict], dict, list]:
    """
    Sweep k=1..top_k_max (capped by available pairs) on val logits; maximize macro ROC-AUC,
    then accuracy, then smaller k. Returns (best_k_pe, sweep_pe, best_k_gl, sweep_gl, topk_per_emotion, global_pairs).
    """
    by_layer_loc = summary_df.groupby(["layer", "loc"])["test_roc_auc"].mean().reset_index()
    n_global = len(by_layer_loc)
    n_per_em = (
        min(len(summary_df[summary_df["emotion"] == em]) for em in emotions_list)
        if emotions_list
        else top_k_max
    )
    k_hi = min(top_k_max, n_global, n_per_em)
    k_hi = max(1, k_hi)

    best_pe, best_gl = 1, 1
    best_roc_pe = float("nan")
    best_acc_pe = -1.0
    best_roc_gl = float("nan")
    best_acc_gl = -1.0
    sweep_pe, sweep_gl = [], []
    final_topk_pe: dict = {}
    final_pairs_gl: list = []

    for k in range(1, k_hi + 1):
        _progress(f"Top-k sweep progress: k={k}/{k_hi}")
        logits_pe, tk_pe = _per_emotion_fusion_logits(
            summary_df, emotions_list, k, y_true, get_logits_single
        )
        acc_pe, roc_pe = _score_multiclass(y_true, logits_pe)
        sweep_pe.append({"k": k, "accuracy": acc_pe, "roc_auc": roc_pe})
        if _better_k_candidate(roc_pe, acc_pe, k, best_roc_pe, best_acc_pe, best_pe, tie_break_smaller_k):
            best_roc_pe, best_acc_pe, best_pe = roc_pe, acc_pe, k
            final_topk_pe = {e: list(v) for e, v in tk_pe.items()}

        logits_gl, pairs_gl = _global_fusion_logits(summary_df, k, get_logits_single, emotions_list)
        if logits_gl is None:
            sweep_gl.append({"k": k, "accuracy": float("nan"), "roc_auc": float("nan")})
            continue
        acc_gl, roc_gl = _score_multiclass(y_true, logits_gl)
        sweep_gl.append({"k": k, "accuracy": acc_gl, "roc_auc": roc_gl})
        if _better_k_candidate(roc_gl, acc_gl, k, best_roc_gl, best_acc_gl, best_gl, tie_break_smaller_k):
            best_roc_gl, best_acc_gl, best_gl = roc_gl, acc_gl, k
            final_pairs_gl = list(pairs_gl)

    if not final_topk_pe:
        final_topk_pe = _topk_pairs_per_emotion(summary_df, emotions_list, best_pe)
    if not final_pairs_gl:
        final_pairs_gl = _topk_global_pairs_list(summary_df, best_gl)

    return best_pe, sweep_pe, best_gl, sweep_gl, final_topk_pe, final_pairs_gl


def load_circuit_top_k_selection(circuit_dir: Path) -> dict | None:
    """Load circuit_top_k_selection.json if present."""
    path = circuit_dir / CIRCUIT_TOP_K_SELECTION_FILENAME
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def topk_per_emotion_from_selection(
    sel: dict,
    emotions_list: list,
    summary_df: pd.DataFrame | None = None,
    k_fallback: int | None = None,
) -> dict[str, list[tuple[int, int]]]:
    """Build topk_per_emotion dict from saved selection JSON (string keys -> list of [layer, loc])."""
    raw = sel.get("per_emotion_pairs") or {}
    k_fb = k_fallback
    if k_fb is None:
        pe = sel.get("per_emotion") or {}
        k_fb = int(pe.get("k_pairs") or CIRCUIT_TOP_K_MAX)
    out = {}
    for em in emotions_list:
        pairs = raw.get(em) or raw.get(str(em)) or []
        if pairs:
            out[em] = [(int(p[0]), int(p[1])) for p in pairs]
        elif summary_df is not None:
            sub = summary_df[summary_df["emotion"] == em].sort_values(
                "test_roc_auc", ascending=False
            ).head(k_fb)
            out[em] = [(int(r["layer"]), int(r["loc"])) for _, r in sub.iterrows()]
        else:
            out[em] = []
    return out


def run_classification_experiment(
    model_id: str = DEFAULT_MODEL_ID,
    val_frac: float = 0.15,
    top_k_pairs: int | None = None,
    top_k_max: int | None = None,
    auto_top_k: bool = True,
    skip_extract: bool = False,
) -> pd.DataFrame:
    """
    Run single-best vs top-k fusion classification; return metrics DataFrame and save outputs.

    When auto_top_k is True (default), sweeps k in 1..top_k_max on the validation hidden states
    and picks k that maximizes macro ROC-AUC (then accuracy, then smaller k). Writes
    circuit_top_k_selection.json for phase1_circuits and steering_benchmark.
    When auto_top_k is False, uses fixed top_k_pairs (default top_k_max).
    """
    if top_k_pairs is not None:
        auto_top_k = False

    _progress(f"Starting circuit evidence for model `{model_id}`.")
    paths = get_probe_paths(model_id)
    circuit_dir = get_circuit_dir(model_id)
    circuit_dir.mkdir(parents=True, exist_ok=True)
    k_bound = int(top_k_max if top_k_max is not None else CIRCUIT_TOP_K_MAX)

    extraction_layers = get_extraction_layers(model_id)
    extraction_locs = get_extraction_locs(model_id)
    extraction_batch_size = get_extraction_batch_size(model_id)
    _progress(
        f"Using {len(extraction_layers)} layers, {len(extraction_locs)} locs, "
        f"top_k_max={k_bound}, skip_extract={skip_extract}."
    )

    _progress("Loading canonical selection/test splits.")
    split_bundle = _load_combined_eval_splits(
        selection_frac=val_frac if val_frac is not None else SELECTION_SPLIT,
        test_frac=FINAL_TEST_SPLIT,
        random_state=RANDOM_STATE,
    )

    # Emotions: from probe summary so order matches probes (per-model path via get_probe_paths)
    _progress("Loading probe summary and serialized probe bundle.")
    summary_df = pd.read_csv(paths.probe_summary_csv)
    emotions_list = summary_df["emotion"].unique().tolist()
    label_to_idx = {e: i for i, e in enumerate(emotions_list)}

    layer_to_idx = {l: i for i, l in enumerate(extraction_layers)}
    loc_to_idx = {l: i for i, l in enumerate(extraction_locs)}
    token_key = EXTRACTION_TOKENS[0]
    token_idx = 0

    probes_path = _find_probes_pt(paths.probes_dir_v2, model_id)
    probes = torch.load(probes_path, weights_only=False)

    def _filter_split(hidden_states, labels, row_indices=None):
        y = np.array([label_to_idx.get(l, -1) for l in labels])
        valid = y >= 0
        hs_filtered = hidden_states[valid] if isinstance(hidden_states, np.ndarray) else hidden_states[valid]
        labels_filtered = [labels[i] for i in np.where(valid)[0]]
        ri_filtered = None
        if row_indices is not None and len(row_indices) == len(labels):
            ri_filtered = [row_indices[i] for i in np.where(valid)[0]]
        return hs_filtered, labels_filtered, y[valid], ri_filtered

    selection_manifest = make_split_manifest(
        model_id=model_id,
        dataset_path=COMBINED_CSV,
        selection_frac=val_frac if val_frac is not None else SELECTION_SPLIT,
        test_frac=FINAL_TEST_SPLIT,
        random_state=RANDOM_STATE,
        text_policy=split_bundle["selection"]["text_col"],
        split_name="selection",
        n_rows=len(split_bundle["selection"]["labels"]),
    )
    test_manifest = make_split_manifest(
        model_id=model_id,
        dataset_path=COMBINED_CSV,
        selection_frac=val_frac if val_frac is not None else SELECTION_SPLIT,
        test_frac=FINAL_TEST_SPLIT,
        random_state=RANDOM_STATE,
        text_policy=split_bundle["test"]["text_col"],
        split_name="test",
        n_rows=len(split_bundle["test"]["labels"]),
    )
    if not skip_extract:
        selection_hidden_states, selection_labels, selection_row_idx = _get_or_extract_split_hidden_states(
            model_id,
            "selection",
            split_bundle["selection"]["texts"],
            split_bundle["selection"]["labels"],
            circuit_dir,
            extraction_layers,
            extraction_locs,
            selection_manifest,
            batch_size=extraction_batch_size,
            dataset_row_idx=split_bundle["selection"]["row_indices"],
        )
        test_hidden_states, test_labels, test_row_idx = _get_or_extract_split_hidden_states(
            model_id,
            "test",
            split_bundle["test"]["texts"],
            split_bundle["test"]["labels"],
            circuit_dir,
            extraction_layers,
            extraction_locs,
            test_manifest,
            batch_size=extraction_batch_size,
            dataset_row_idx=split_bundle["test"]["row_indices"],
        )
    else:
        _progress("Using previously cached selection/test hidden states.")
        sel_hs_path, sel_labels_path, _ = _split_cache_paths(circuit_dir, "selection")
        test_hs_path, test_labels_path, _ = _split_cache_paths(circuit_dir, "test")
        if not (sel_hs_path.exists() and sel_labels_path.exists() and test_hs_path.exists() and test_labels_path.exists()):
            raise FileNotFoundError("Run without skip_extract first to create selection/test caches.")
        selection_hidden_states = torch.load(sel_hs_path, weights_only=False)
        sel_lbl_df = pd.read_csv(sel_labels_path)
        selection_labels = list(sel_lbl_df["emotion"].astype(str))
        selection_row_idx = (
            [int(x) for x in sel_lbl_df["dataset_row_idx"].tolist()]
            if "dataset_row_idx" in sel_lbl_df.columns
            else None
        )
        test_hidden_states = torch.load(test_hs_path, weights_only=False)
        test_lbl_df = pd.read_csv(test_labels_path)
        test_labels = list(test_lbl_df["emotion"].astype(str))
        test_row_idx = (
            [int(x) for x in test_lbl_df["dataset_row_idx"].tolist()]
            if "dataset_row_idx" in test_lbl_df.columns
            else None
        )

    selection_hidden_states, selection_labels, y_selection, _ = _filter_split(
        selection_hidden_states, selection_labels, selection_row_idx
    )
    test_hidden_states, test_labels, y_test, _ = _filter_split(
        test_hidden_states, test_labels, test_row_idx
    )
    _progress(
        f"Filtered splits to known probe emotions: selection={len(y_selection)} rows, "
        f"test={len(y_test)} rows, emotions={len(emotions_list)}."
    )

    def get_logits_single(layer, loc, hidden_states):
        li, lci = layer_to_idx.get(layer), loc_to_idx.get(loc)
        if li is None or lci is None:
            return None
        return _probe_logits_at_layer_loc(
            probes, hidden_states, layer, loc, token_key, emotions_list, li, lci, token_idx
        )

    best_layer, best_loc, _probe_topk_global = _single_best_and_topk_pairs(
        summary_df, top_k_pairs=k_bound
    )

    # Single-best (from probe-summary ranking; same as before)
    _progress(f"Evaluating single-best site at layer={best_layer}, loc={best_loc}.")
    logits_sb = get_logits_single(best_layer, best_loc, test_hidden_states)
    if logits_sb is None:
        raise RuntimeError(f"Single-best (layer={best_layer}, loc={best_loc}) not in extraction grid.")
    acc_sb, roc_sb = _score_multiclass(y_test, logits_sb)

    if auto_top_k and top_k_pairs is None:
        _progress("Running selection-split top-k sweep.")
        selection_get_logits = lambda layer, loc: get_logits_single(layer, loc, selection_hidden_states)
        k_pe, sweep_pe, k_gl, sweep_gl, topk_per_emotion, topk_pairs = _auto_select_k_pairs(
            summary_df, emotions_list, y_selection, selection_get_logits, k_bound
        )
        selection_payload = {
            "version": 1,
            "model_id": model_id,
            "auto_top_k": True,
            "top_k_max": k_bound,
            "selection_split": selection_manifest["split_protocol"],
            "evaluation_split": test_manifest["split_protocol"],
            "selection_rule": "maximize_macro_roc_auc_then_accuracy_then_smaller_k",
            "per_emotion": {"k_pairs": k_pe, "sweep": sweep_pe},
            "global": {"k_pairs": k_gl, "sweep": sweep_gl},
            "per_emotion_pairs": {em: [[a, b] for a, b in pairs] for em, pairs in topk_per_emotion.items()},
            "global_pairs": [[a, b] for a, b in topk_pairs],
        }
        sel_path = circuit_dir / CIRCUIT_TOP_K_SELECTION_FILENAME
        with open(sel_path, "w", encoding="utf-8") as f:
            json.dump(selection_payload, f, indent=2)
        print(f"Wrote {sel_path} (per-emotion k={k_pe}, global k={k_gl})")
    else:
        _progress(f"Using fixed top-k={int(top_k_pairs if top_k_pairs is not None else k_bound)}.")
        k_fix = int(top_k_pairs if top_k_pairs is not None else k_bound)
        topk_per_emotion = _topk_pairs_per_emotion(summary_df, emotions_list, k_fix)
        topk_pairs = _topk_global_pairs_list(summary_df, k_fix)
        k_pe = k_gl = k_fix
        sweep_pe = sweep_gl = []
        # Still persist pairs so phase1_circuits / steering use the same circuit without re-sweeping
        selection_payload = {
            "version": 1,
            "model_id": model_id,
            "auto_top_k": False,
            "top_k_max": k_bound,
            "selection_split": selection_manifest["split_protocol"],
            "evaluation_split": test_manifest["split_protocol"],
            "selection_rule": "fixed_k_from_config_or_cli",
            "per_emotion": {"k_pairs": k_fix, "sweep": []},
            "global": {"k_pairs": k_fix, "sweep": []},
            "per_emotion_pairs": {em: [[a, b] for a, b in pairs] for em, pairs in topk_per_emotion.items()},
            "global_pairs": [[a, b] for a, b in topk_pairs],
        }
        sel_path = circuit_dir / CIRCUIT_TOP_K_SELECTION_FILENAME
        with open(sel_path, "w", encoding="utf-8") as f:
            json.dump(selection_payload, f, indent=2)
        print(f"Wrote {sel_path} (fixed k={k_fix})")

    # Top-k fusion (per-emotion) at chosen k
    _progress(f"Evaluating final held-out test metrics for per-emotion top-k (k={k_pe}).")
    n_samples = len(y_test)
    logits_per_emotion = np.zeros((n_samples, len(emotions_list)), dtype=np.float64)
    for e_idx, em in enumerate(emotions_list):
        pairs = topk_per_emotion.get(em, [])
        cols = []
        for layer, loc in pairs:
            L = get_logits_single(layer, loc, test_hidden_states)
            if L is not None:
                cols.append(L[:, e_idx])
        if cols:
            logits_per_emotion[:, e_idx] = np.mean(cols, axis=0)
    acc_tk, roc_tk = _score_multiclass(y_test, logits_per_emotion)

    # Top-k fusion (global)
    _progress(f"Evaluating final held-out test metrics for global top-k (k={k_gl}).")
    logits_list = []
    for layer, loc in topk_pairs:
        L = get_logits_single(layer, loc, test_hidden_states)
        if L is not None:
            logits_list.append(L)
    if not logits_list:
        raise RuntimeError("No valid top-k (layer, loc) found.")
    logits_tk_global = np.mean(logits_list, axis=0)
    acc_tk_global, roc_tk_global = _score_multiclass(y_test, logits_tk_global)

    rows = [
        {"method": "single_best", "accuracy": acc_sb, "roc_auc": roc_sb, "layer": best_layer, "loc": best_loc, "evaluation_split": "test"},
        {"method": "topk_fusion", "accuracy": acc_tk, "roc_auc": roc_tk, "k_pairs": k_pe, "selection_split": "selection", "evaluation_split": "test"},
        {"method": "topk_fusion_global", "accuracy": acc_tk_global, "roc_auc": roc_tk_global, "k_pairs": k_gl, "selection_split": "selection", "evaluation_split": "test"},
    ]
    metrics_df = pd.DataFrame(rows)
    out_csv = circuit_dir / "circuit_evidence_classification.csv"
    metrics_df.to_csv(out_csv, index=False)
    _progress("Writing circuit evidence CSV, summary, and figure outputs.")
    summary_md = circuit_dir / "summary.md"
    summary_lines = [
        "# 02_circuit",
        "",
        "## Circuit evidence",
        "",
        "Single-best vs top-k fusion classification on the held-out test split.",
        "",
        f"- `single_best`: layer `{best_layer}`, loc `{best_loc}`, accuracy `{acc_sb:.3f}`, macro OVR ROC-AUC `{roc_sb:.3f}`.",
        f"- `topk_fusion`: per-emotion selected k `{k_pe}`, accuracy `{acc_tk:.3f}`, macro OVR ROC-AUC `{roc_tk:.3f}`.",
        f"- `topk_fusion_global`: global selected k `{k_gl}`, accuracy `{acc_tk_global:.3f}`, macro OVR ROC-AUC `{roc_tk_global:.3f}`.",
        "",
        "Selection details are stored in `circuit_top_k_selection.json`. The saved sweeps use the selection split to choose k, then report final metrics on the held-out test split.",
        "",
    ]
    summary_md.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Wrote {out_csv}")

    # Figure: bar chart (three methods)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        x = np.arange(len(metrics_df))
        width = 0.35
        ax.bar(x - width / 2, metrics_df["accuracy"], width, label="Accuracy")
        ax.bar(x + width / 2, metrics_df["roc_auc"], width, label="Macro ROC-AUC")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_df["method"].tolist(), rotation=15, ha="right")
        ax.set_ylabel("Score")
        ax.set_title("Circuit evidence: single-best vs top-k fusion (per-emotion vs global)")
        ax.legend()
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(circuit_dir / "circuit_evidence_classification.pdf", bbox_inches="tight")
        fig.savefig(circuit_dir / "circuit_evidence_classification.png", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved figure to {circuit_dir / 'circuit_evidence_classification.pdf'}")
    except Exception as e:
        print(f"Could not save figure: {e}")

    return metrics_df


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument(
        "--top_k_max",
        type=int,
        default=CIRCUIT_TOP_K_MAX,
        help="Max k to sweep when auto-selecting (default from config.CIRCUIT_TOP_K_MAX)",
    )
    p.add_argument(
        "--top_k_pairs",
        type=int,
        default=None,
        help="Fixed k for both fusion modes (disables validation-based auto sweep)",
    )
    p.add_argument(
        "--no_auto_top_k",
        action="store_true",
        help="Use fixed k = top_k_max for both modes (no sweep; does not write new selection logic beyond metrics)",
    )
    p.add_argument("--skip_extract", action="store_true", help="Use cached val hidden states only")
    args = p.parse_args()
    auto = not args.no_auto_top_k and args.top_k_pairs is None
    fixed_k = None if auto else int(args.top_k_pairs if args.top_k_pairs is not None else args.top_k_max)
    run_classification_experiment(
        model_id=args.model_id,
        val_frac=args.val_frac,
        top_k_pairs=fixed_k,
        top_k_max=args.top_k_max,
        auto_top_k=auto,
        skip_extract=args.skip_extract,
    )

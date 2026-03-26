"""
Steering benchmark: appraisal vs emotion steering for empathy control (e.g. anger -> joy).

Default alpha/beta sweeps match `STEERING_BENCHMARK_ALPHAS` (= `MENTAL_HEALTH_ALPHAS` in config).
Summary row picks use `MENTAL_HEALTH_REPORT_ALPHA` (nearest dose in the sweep).

Uses the same top-k circuit (per-emotion layer/loc pairs) as circuit_evidence. Steers at
each circuit site for the target emotion; evaluates with top-k fusion logits. Supports
all contrastive emotion pairs, prompted + unprompted robustness (behavioral), and progress bars.
Saves steering_benchmark.csv, steering_curves.csv, and figure to 05_steering/.
"""
from pathlib import Path
import math
import sys
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from pipeline.core.config import (
    REPO_ROOT,
    COMBINED_CSV,
    get_circuit_dir,
    get_steering_dir,
    get_appraisal_structure_dir,
    get_probe_paths,
    EXTRACTION_TOKENS,
    COMMON_APPRAISAL,
    DEFAULT_MODEL_ID,
    CONTRASTIVE_EMOTION_PAIRS,
    SIMILAR_EMOTION_PAIRS,
    VAL_SPLIT,
    RANDOM_STATE,
    CIRCUIT_TOP_K_MAX,
    SELECTION_SPLIT,
    FINAL_TEST_SPLIT,
    MENTAL_HEALTH_REPORT_ALPHA,
    STEERING_BENCHMARK_ALPHAS,
)
from pipeline.core.model_config import get_extraction_layers, get_extraction_locs, probe_filename_suffix, get_behavioral_batch_size
from pipeline.stage_02_circuit.circuit_evidence import (
    _topk_pairs_per_emotion,
    load_circuit_top_k_selection,
    topk_per_emotion_from_selection,
)
from pipeline.core.research_contracts import (
    canonicalize_combined_dataset,
    split_combined_dataset,
    build_prompted_texts,
    appraisal_probe_direction_raw,
    emotion_probe_direction_raw,
    ensure_manifest_model_match,
)
from pipeline.stage_05_appraisal_theory.theory_analysis import save_steering_pair_outputs
def _progress(message: str) -> None:
    print(f"[steering_benchmark] {message}", flush=True)


def _sigmoid_stable(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for large-magnitude probe decisions."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def _ensure_repo():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def _load_eval_split_both_text_types(selection_frac=None, test_frac=None, random_state=None):
    """
    Load the held-out test split with both prompted and unprompted text conditions.

    If `hidden_emo_text` is missing in the combined dataset, prompted text is generated
    deterministically from `situation` with the same prompt builder used during training.
    """
    selection_frac = selection_frac if selection_frac is not None else SELECTION_SPLIT
    test_frac = test_frac if test_frac is not None else FINAL_TEST_SPLIT
    random_state = random_state if random_state is not None else RANDOM_STATE
    df = canonicalize_combined_dataset(pd.read_csv(COMBINED_CSV))
    df = df.dropna(subset=["emotion", "situation"])
    split_bundle = split_combined_dataset(
        df,
        selection_frac=selection_frac,
        test_frac=test_frac,
        random_state=random_state,
    )
    test_df = split_bundle["test"].copy()
    emotions = test_df["emotion"].astype(str).tolist()
    has_prompted = "hidden_emo_text" in df.columns
    if has_prompted:
        prompted_texts = test_df["hidden_emo_text"].astype(str).tolist()
    else:
        prompted_texts = build_prompted_texts(
            test_df["situation"].astype(str).tolist(),
            emotions_list=sorted(df["emotion"].astype(str).unique().tolist()),
            prompt_index=1,
        )
    unprompted_texts = test_df["situation"].astype(str).tolist()
    return emotions, prompted_texts, unprompted_texts


def _find_probes_pt(probes_dir: Path, model_id: str):
    """Find combined binary OVA probes .pt in probes_dir (from get_probe_paths)."""
    ensure_manifest_model_match(probes_dir / "probe_manifest.json", model_id, "probe")
    suffix = probe_filename_suffix(model_id)
    name = f"binary_ova_probes_{suffix}.pt"
    path = probes_dir / name
    dir_is_model_specific = str(model_id) in str(probes_dir)
    if path.exists():
        return path
    # Backwards-compatible fallbacks
    legacy = [f for f in probes_dir.glob("binary_ova_probes_layers_*.pt") if "locs_" in f.name and "tokens_" in f.name]
    if len(legacy) == 1 and (dir_is_model_specific or str(model_id) in legacy[0].name):
        return legacy[0]
    generic = list(probes_dir.glob("binary_ova_probes_*.pt"))
    if len(generic) == 1 and (dir_is_model_specific or str(model_id) in generic[0].name):
        return generic[0]
    if len(legacy) > 1 or len(generic) > 1:
        raise FileNotFoundError(f"Ambiguous probe bundle under {probes_dir}; expected a unique file for {model_id}.")
    raise FileNotFoundError(f"No combined probes .pt found under {probes_dir}")


def _compute_appraisal_steering_vector(
    source_emotion: str,
    target_emotion: str,
    best_layer: int,
    best_loc: int,
    hidden_size: int,
    model_id: str,
    appraisal_probes=None,
    zscore_df: pd.DataFrame | None = None,
) -> np.ndarray | None:
    """
    Build appraisal steering vector from appraisal regression probe directions,
    weighted by (target - source) appraisal z-scores. Uses the same appraisal
    directions found in the emotion circuit. Returns None if probes/zscore unavailable.
    If appraisal_probes and zscore_df are provided, skips loading from disk (faster when called repeatedly).
    """
    token_key = EXTRACTION_TOKENS[0]
    if appraisal_probes is None:
        appraisal_path = get_probe_paths(model_id).appraisal_probes_path
        if not appraisal_path.exists():
            return None
        ensure_manifest_model_match(get_probe_paths(model_id).appraisal_manifest_path, model_id, "appraisal probe")
        appraisal_probes = torch.load(appraisal_path, weights_only=False)
    if zscore_df is None:
        zscore_path = get_appraisal_structure_dir(model_id) / "appraisal_zscore_by_emotion.csv"
        if not zscore_path.exists():
            return None
        zscore_df = pd.read_csv(zscore_path, index_col=0)
    if source_emotion not in zscore_df.index or target_emotion not in zscore_df.index:
        return None
    weights = zscore_df.loc[target_emotion] - zscore_df.loc[source_emotion]

    vec = np.zeros(hidden_size, dtype=np.float32)
    n_used = 0
    for dim in COMMON_APPRAISAL:
        if dim not in weights.index or dim not in appraisal_probes:
            continue
        try:
            r = appraisal_probes[dim][best_layer][best_loc][token_key]
        except (KeyError, TypeError):
            continue
        direction = appraisal_probe_direction_raw(r, hidden_size=hidden_size)
        if direction is None:
            continue
        direction = direction.astype(np.float32)
        if direction.shape[0] != hidden_size:
            continue
        w = float(weights[dim])
        vec += w * direction
        n_used += 1
    if n_used == 0:
        return None
    return vec


def _emotion_appraisal_signature_vectors_for_emotion(
    emotion: str,
    target_pairs: list[tuple[int, int]],
    hidden_size: int,
    appraisal_probes,
    zscore_df: pd.DataFrame | None,
) -> dict[tuple[int, int], np.ndarray]:
    """Build one appraisal-signature vector per selected site for an emotion."""
    out = {}
    token_key = EXTRACTION_TOKENS[0]
    if zscore_df is None or appraisal_probes is None or emotion not in zscore_df.index:
        return out
    weights = zscore_df.loc[emotion]
    for layer, loc in target_pairs:
        vec = np.zeros(hidden_size, dtype=np.float32)
        used = 0
        for dim in COMMON_APPRAISAL:
            if dim not in weights.index or dim not in appraisal_probes:
                continue
            try:
                rec = appraisal_probes[dim][layer][loc][token_key]
            except (KeyError, TypeError):
                continue
            direction = appraisal_probe_direction_raw(rec, hidden_size=hidden_size)
            if direction is None:
                continue
            vec += float(weights[dim]) * direction.astype(np.float32)
            used += 1
        if used > 0:
            out[(layer, loc)] = vec
    return out


def _probe_logits_at(probes, hidden_states, layer, loc, token_key, emotions_list, layer_idx, loc_idx, token_idx=0):
    X = hidden_states[:, layer_idx, loc_idx, token_idx, :]
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    X = X.reshape(X.shape[0], -1)
    logits = np.zeros((X.shape[0], len(emotions_list)), dtype=np.float64)
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

        # Prefer stored numeric weights/bias for compatibility across sklearn versions.
        w = rec.get("weights", None)
        b = rec.get("bias", None)
        if w is None or b is None:
            clf = rec.get("classifier", None)
            coef = getattr(clf, "coef_", None) if clf is not None else None
            intercept = getattr(clf, "intercept_", None) if clf is not None else None
            if coef is None or intercept is None:
                continue
            w = coef.ravel()
            b = float(np.ravel(intercept)[0])
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        b = float(b)
        if w.shape[0] != Xe.shape[1]:
            continue
        decision = Xe @ w + b
        logits[:, e_idx] = _sigmoid_stable(decision)
    return logits


def _circuit_logits(
    hidden_states,
    probes,
    topk_per_emotion,
    extraction_layers,
    extraction_locs,
    emotions_list,
    token_key,
    token_idx=0,
):
    """Top-k fusion logits: per emotion, average probe logits at that emotion's circuit (layer, loc) pairs."""
    n_samples = hidden_states.shape[0]
    layer_to_idx = {l: i for i, l in enumerate(extraction_layers)}
    loc_to_idx = {l: i for i, l in enumerate(extraction_locs)}
    logits = np.zeros((n_samples, len(emotions_list)), dtype=np.float64)
    for e_idx, em in enumerate(emotions_list):
        pairs = topk_per_emotion.get(em, [])
        cols = []
        for layer, loc in pairs:
            li, lci = layer_to_idx.get(layer), loc_to_idx.get(loc)
            if li is None or lci is None:
                continue
            L = _probe_logits_at(
                probes, hidden_states, layer, loc, token_key, emotions_list, li, lci, token_idx
            )
            cols.append(L[:, e_idx])
        if cols:
            logits[:, e_idx] = np.mean(cols, axis=0)
    return logits


def _nearest_sweep_strength(candidates: list | np.ndarray, preferred: float) -> float:
    """Pick a dose from the sweep closest to `preferred` (e.g. MENTAL_HEALTH_REPORT_ALPHA)."""
    uniq = sorted({float(x) for x in candidates})
    if not uniq:
        return float(preferred)
    return float(min(uniq, key=lambda a: abs(a - float(preferred))))


def _run_behavioral_steering(
    model_id: str,
    source_emotion: str,
    target_emotion: str,
    paths,
    circuit_dir: Path,
    out_dir: Path,
    summary_df: pd.DataFrame,
    emotions_list: list,
    target_idx: int,
    extraction_layers: list,
    extraction_locs: list,
    topk_per_emotion: dict,
    target_circuit_pairs: list,
    appraisal_vecs: list,
    emotion_vecs: list,
    k: int,
    alphas: list,
    betas: list,
    token_key: int,
    token_idx: int,
    source_texts: list | None = None,
    text_type: str | None = None,
    model=None,
    tokenizer=None,
    logger=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run model with in-pass steering hooks (real forward pass); compute circuit logits
    on the resulting hidden states. If source_texts is provided, use those (e.g. prompted
    or unprompted); otherwise load val and filter to source_emotion. If text_type is set,
    add it to returned rows for robustness reporting. Returns (benchmark_df, curves_df, samples_df)
    with samples_df having scenario, label, prediction, intervention_type for inspection.
    """
    _ensure_repo()
    from torch.utils.data import DataLoader
    from utils import run_forward_with_steering, TextDataset
    from experiments.utils.training_utils import OvaLogisticRegressionTrainer

    if logger is None:
        from utils import Log
        logger = Log("steering_behavioral").logger

    if source_texts is not None:
        anger_texts = list(source_texts)
    else:
        try:
            from pipeline.stage_02_circuit.circuit_evidence import _load_combined_val_split
        except ImportError:
            from circuit_evidence import _load_combined_val_split
        val_texts, val_labels = _load_combined_val_split(val_frac=VAL_SPLIT, random_state=RANDOM_STATE)
        anger_indices = [i for i, l in enumerate(val_labels) if l == source_emotion]
        if not anger_indices:
            raise ValueError(f"No '{source_emotion}' in val split for behavioral steering.")
        anger_texts = [val_texts[i] for i in anger_indices]
    n_anger = len(anger_texts)

    if model is None or tokenizer is None:
        trainer = OvaLogisticRegressionTrainer(model_id, logger=logger)
        trainer.load_model_and_tokenizer()
        model, tokenizer = trainer.model, trainer.tokenizer
    if model is None or tokenizer is None:
        raise RuntimeError("Failed to load model/tokenizer for behavioral steering.")

    dataset = TextDataset(texts=anger_texts, labels=[0] * n_anger)
    batch_size = get_behavioral_batch_size(model_id)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    probes = torch.load(_find_probes_pt(paths.probes_dir_v2, model_id), weights_only=False)
    extraction_tokens = list(EXTRACTION_TOKENS) if isinstance(EXTRACTION_TOKENS, (list, tuple)) else [EXTRACTION_TOKENS[0]]

    def _spec_from_vectors(vectors: list, strength: float):
        strength_per_site = strength / k if k else strength
        return {(layer, loc): (torch.tensor(vec, dtype=torch.float32), strength_per_site) for (layer, loc), vec in zip(target_circuit_pairs, vectors)}

    curve_rows = []
    ref_alpha = _nearest_sweep_strength(alphas, MENTAL_HEALTH_REPORT_ALPHA)
    ref_beta = _nearest_sweep_strength(betas, MENTAL_HEALTH_REPORT_ALPHA)

    # Baseline (no steering)
    if text_type:
        logger.info(f"Behavioral: baseline forward pass ({text_type})...")
    else:
        logger.info("Behavioral: baseline forward pass (no steering)...")
    hs_baseline, _ = run_forward_with_steering(
        dataloader, tokenizer, model, extraction_layers, extraction_locs,
        extraction_tokens, steering_spec={}, logger=logger,
    )
    logits_baseline = _circuit_logits(
        hs_baseline, probes, topk_per_emotion, extraction_layers, extraction_locs,
        emotions_list, token_key, token_idx,
    )
    joy_baseline = float(logits_baseline[:, target_idx].mean())
    pred_baseline = np.argmax(logits_baseline, axis=1)
    pred_appraisal_1 = None
    pred_emotion_1 = None

    for alpha in tqdm(alphas, desc="Behavioral appraisal", leave=False):
        spec = _spec_from_vectors(appraisal_vecs, alpha)
        logger.info(f"Behavioral: appraisal steering strength={alpha}...")
        hs_steered, _ = run_forward_with_steering(
            dataloader, tokenizer, model, extraction_layers, extraction_locs,
            extraction_tokens, steering_spec=spec, logger=logger,
        )
        logits = _circuit_logits(
            hs_steered, probes, topk_per_emotion, extraction_layers, extraction_locs,
            emotions_list, token_key, token_idx,
        )
        pred = np.argmax(logits, axis=1)
        if math.isclose(float(alpha), ref_alpha, rel_tol=0.0, abs_tol=1e-9):
            pred_appraisal_1 = pred
        success = float((pred == target_idx).mean())
        mean_joy = float(logits[:, target_idx].mean())
        row = {"method": "appraisal_steer", "strength": alpha, "success_rate": success, "mean_target_logit": mean_joy, "mean_delta_target": mean_joy - joy_baseline}
        if text_type is not None:
            row["text_type"] = text_type
        curve_rows.append(row)
    if pred_appraisal_1 is None:
        pred_appraisal_1 = pred_baseline

    for beta in tqdm(betas, desc="Behavioral emotion", leave=False):
        spec = _spec_from_vectors(emotion_vecs, beta)
        logger.info(f"Behavioral: emotion steering strength={beta}...")
        hs_steered, _ = run_forward_with_steering(
            dataloader, tokenizer, model, extraction_layers, extraction_locs,
            extraction_tokens, steering_spec=spec, logger=logger,
        )
        logits = _circuit_logits(
            hs_steered, probes, topk_per_emotion, extraction_layers, extraction_locs,
            emotions_list, token_key, token_idx,
        )
        pred = np.argmax(logits, axis=1)
        if beta == 1.0:
            pred_emotion_1 = pred
        success = float((pred == target_idx).mean())
        mean_joy = float(logits[:, target_idx].mean())
        row = {"method": "emotion_steer", "strength": beta, "success_rate": success, "mean_target_logit": mean_joy, "mean_delta_target": mean_joy - joy_baseline}
        if text_type is not None:
            row["text_type"] = text_type
        curve_rows.append(row)
    if pred_emotion_1 is None:
        pred_emotion_1 = pred_baseline

    curves_df = pd.DataFrame(curve_rows)
    app_row = curves_df[(curves_df["method"] == "appraisal_steer") & (curves_df["strength"] == ref_alpha)]
    emo_row = curves_df[(curves_df["method"] == "emotion_steer") & (curves_df["strength"] == ref_beta)]
    if app_row.empty:
        app_row = curves_df[curves_df["method"] == "appraisal_steer"].iloc[-1:]
    if emo_row.empty:
        emo_row = curves_df[curves_df["method"] == "emotion_steer"].iloc[-1:]
    benchmark_rows = [
        {"method": "appraisal_steer", "success_rate": app_row["success_rate"].values[0], "mean_delta_target_logit": app_row["mean_delta_target"].values[0]},
        {"method": "emotion_steer", "success_rate": emo_row["success_rate"].values[0], "mean_delta_target_logit": emo_row["mean_delta_target"].values[0]},
    ]
    if text_type is not None:
        for r in benchmark_rows:
            r["text_type"] = text_type
    benchmark_df = pd.DataFrame(benchmark_rows)

    # Per-sample rows for inspection: scenario, label, prediction, intervention_type
    tt = text_type if text_type is not None else ""
    sample_rows = []
    for i in range(n_anger):
        sample_rows.append({
            "scenario": anger_texts[i], "label": source_emotion, "source_emotion": source_emotion, "target_emotion": target_emotion, "text_type": tt,
            "prediction": emotions_list[pred_baseline[i]], "intervention_type": "baseline",
        })
        sample_rows.append({
            "scenario": anger_texts[i], "label": source_emotion, "source_emotion": source_emotion, "target_emotion": target_emotion, "text_type": tt,
            "prediction": emotions_list[pred_appraisal_1[i]], "intervention_type": f"appraisal_steer_{ref_alpha:g}",
        })
        sample_rows.append({
            "scenario": anger_texts[i], "label": source_emotion, "source_emotion": source_emotion, "target_emotion": target_emotion, "text_type": tt,
            "prediction": emotions_list[pred_emotion_1[i]], "intervention_type": f"emotion_steer_{ref_beta:g}",
        })
    samples_df = pd.DataFrame(sample_rows)
    return benchmark_df, curves_df, samples_df


def run_steering_benchmark(
    model_id: str = DEFAULT_MODEL_ID,
    source_emotion: str | None = None,
    target_emotion: str | None = None,
    emotion_pairs: list | None = None,
    alphas: list | None = None,
    betas: list | None = None,
    top_k_pairs: int | None = None,
    run_behavioral: bool = False,
) -> pd.DataFrame:
    """
    Circuit-level steering over contrastive emotion pairs. Uses cached hidden states
    (ablation) and optionally real forward pass (behavioral) with prompted + unprompted
    robustness. Saves steering_benchmark.csv (and _behavioral.csv) with source_emotion,
    target_emotion, method; behavioral adds text_type (prompted/unprompted). No sample cap.
    """
    _progress(f"Starting steering benchmark for model `{model_id}`.")
    _ensure_repo()
    from experiments.utils.intervention_utils import steer_hidden_states, compute_steering_vector

    alphas = alphas or [0.0, 0.5, 1.0, 2.0, 5.0]
    betas = betas or [0.0, 0.5, 1.0, 2.0]

    paths = get_probe_paths(model_id)
    circuit_dir = get_circuit_dir(model_id)
    out_dir = get_steering_dir(model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    _progress(f"Behavioral steering enabled: {run_behavioral}.")
    # Ensure appraisal probes exist; otherwise steering quality can silently degrade.
    if not paths.appraisal_probes_path.exists():
        print(f"Missing appraisal probes for {model_id}: {paths.appraisal_probes_path}")
        print("Training appraisal probes now (one-time prerequisite for robust appraisal steering)...")
        try:
            try:
                from pipeline.train_appraisal_probes import run_train_appraisal_probes
            except ImportError:
                from train_appraisal_probes import run_train_appraisal_probes
            run_train_appraisal_probes(model_id=model_id)
            # Re-resolve paths after training
            paths = get_probe_paths(model_id)
        except Exception as e:
            print(
                "Could not auto-train appraisal probes. "
                "Proceeding with fallback appraisal vectors (may reduce quality). "
                f"Reason: {e}"
            )

    hs_path = circuit_dir / "test_hidden_states.pt"
    labels_path = circuit_dir / "test_labels.csv"
    evaluation_split = "test"
    if not hs_path.exists() or not labels_path.exists():
        hs_path = circuit_dir / "val_hidden_states.pt"
        labels_path = circuit_dir / "val_labels.csv"
        evaluation_split = "legacy_val"
    if not hs_path.exists() or not labels_path.exists():
        raise FileNotFoundError("Run circuit_evidence first to create held-out test caches.")

    _progress(f"Loading cached hidden states from `{hs_path.name}` and probe artifacts.")
    hidden_states = torch.load(hs_path, weights_only=False)
    val_labels = list(pd.read_csv(labels_path)["emotion"].astype(str))
    summary_df = pd.read_csv(paths.probe_summary_csv)
    emotions_list = summary_df["emotion"].unique().tolist()

    sel = load_circuit_top_k_selection(circuit_dir)
    if top_k_pairs is not None:
        topk_per_emotion = _topk_pairs_per_emotion(summary_df, emotions_list, top_k_pairs)
    elif sel is not None:
        topk_per_emotion = topk_per_emotion_from_selection(
            sel, emotions_list, summary_df=summary_df
        )
    else:
        print(
            "Warning: circuit_top_k_selection.json missing; using probe-summary top-"
            f"{CIRCUIT_TOP_K_MAX} (layer,loc) per emotion. Run circuit_evidence for val-optimal k."
        )
        topk_per_emotion = _topk_pairs_per_emotion(
            summary_df, emotions_list, CIRCUIT_TOP_K_MAX
        )

    if emotion_pairs is not None:
        pairs = [(s, t) for s, t in emotion_pairs if s in emotions_list and t in emotions_list]
    elif source_emotion or target_emotion:
        if not (source_emotion and target_emotion):
            raise ValueError("Provide both source_emotion and target_emotion, or neither to run all default contrastive pairs.")
        pairs = [(source_emotion, target_emotion)] if source_emotion in emotions_list and target_emotion in emotions_list else []
    else:
        pairs = [(s, t) for s, t in CONTRASTIVE_EMOTION_PAIRS if s in emotions_list and t in emotions_list]
    if not pairs:
        raise ValueError("No valid emotion pairs (check CONTRASTIVE_EMOTION_PAIRS or emotions_list).")
    if source_emotion and target_emotion:
        _progress(f"Running a single requested pair: {source_emotion}->{target_emotion}.")
    else:
        _progress("Running the default full contrastive-pair benchmark.")
    _progress(f"Evaluating {len(pairs)} emotion pair(s) on split `{evaluation_split}`.")

    extraction_layers = get_extraction_layers(model_id)
    extraction_locs = get_extraction_locs(model_id)
    layer_to_idx = {l: i for i, l in enumerate(extraction_layers)}
    loc_to_idx = {l: i for i, l in enumerate(extraction_locs)}
    token_key = EXTRACTION_TOKENS[0]
    token_idx = 0
    probes = torch.load(_find_probes_pt(paths.probes_dir_v2, model_id), weights_only=False)
    hidden_size = hidden_states.shape[-1]

    # Load appraisal data once to avoid repeated disk I/O per pair/site
    appraisal_probes_cache = None
    zscore_df_cache = None
    appraisal_path = paths.appraisal_probes_path
    zscore_path = get_appraisal_structure_dir(model_id) / "appraisal_zscore_by_emotion.csv"
    if appraisal_path.exists():
        ensure_manifest_model_match(paths.appraisal_manifest_path, model_id, "appraisal probe")
        appraisal_probes_cache = torch.load(appraisal_path, weights_only=False)
    if zscore_path.exists():
        zscore_df_cache = pd.read_csv(zscore_path, index_col=0)

    def _emotion_vec_from_probe_record(rec) -> np.ndarray:
        """Convert a binary emotion probe direction into raw hidden-state space."""
        vec = emotion_probe_direction_raw(rec, hidden_size=hidden_size)
        if vec is None:
            return np.zeros(hidden_size, dtype=np.float32)
        return vec.astype(np.float32)

    all_bench_rows = []
    all_curve_rows = []
    ref_alpha_global = _nearest_sweep_strength(alphas, MENTAL_HEALTH_REPORT_ALPHA)
    ref_beta_global = _nearest_sweep_strength(betas, MENTAL_HEALTH_REPORT_ALPHA)

    for source_emotion, target_emotion in tqdm(pairs, desc="Emotion pairs (cache)", unit="pair"):
        source_idx = emotions_list.index(source_emotion)
        target_idx = emotions_list.index(target_emotion)
        source_mask = np.array([l == source_emotion for l in val_labels])
        n_source = source_mask.sum()
        if n_source == 0:
            continue
        hs_source = hidden_states[source_mask]
        joy_mask = np.array([l == target_emotion for l in val_labels])
        hs_target = hidden_states[joy_mask] if joy_mask.any() else None

        target_circuit_pairs = topk_per_emotion.get(target_emotion, [])
        if not target_circuit_pairs:
            by_layer_loc = summary_df.groupby(["layer", "loc"])["test_roc_auc"].mean().reset_index()
            by_layer_loc = by_layer_loc.sort_values("test_roc_auc", ascending=False)
            target_circuit_pairs = [(int(by_layer_loc.iloc[0]["layer"]), int(by_layer_loc.iloc[0]["loc"]))]
        k = len(target_circuit_pairs)

        appraisal_vecs = []
        for layer, loc in target_circuit_pairs:
            vec = _compute_appraisal_steering_vector(
                source_emotion, target_emotion, layer, loc, hidden_size, model_id,
                appraisal_probes=appraisal_probes_cache, zscore_df=zscore_df_cache,
            )
            if vec is None and hs_target is not None and len(hs_target) > 0:
                li, lci = layer_to_idx.get(layer), loc_to_idx.get(loc)
                if li is not None and lci is not None:
                    v = compute_steering_vector(hs_target, hs_source, li, loc_idx=lci, token_idx=token_idx)
                    vec = v.cpu().numpy().astype(np.float32)
            appraisal_vecs.append(vec if vec is not None else np.zeros(hidden_size, dtype=np.float32))

        emotion_vecs = []
        for layer, loc in target_circuit_pairs:
            if layer in probes and loc in probes[layer] and token_key in probes[layer][loc] and target_emotion in probes[layer][loc][token_key]:
                rec = probes[layer][loc][token_key][target_emotion]
                emotion_vecs.append(_emotion_vec_from_probe_record(rec))
            else:
                emotion_vecs.append(np.zeros(hidden_size, dtype=np.float32))

        def steer_at_circuit(hs, vectors, strength_per_site):
            out = hs
            for (layer, loc), vec in zip(target_circuit_pairs, vectors):
                li, lci = layer_to_idx.get(layer), loc_to_idx.get(loc)
                if li is None or lci is None:
                    continue
                out = steer_hidden_states(out, vec, strength_per_site, li, loc_idx=lci, token_idx=token_idx)
            return out

        logits_baseline = _circuit_logits(hs_source, probes, topk_per_emotion, extraction_layers, extraction_locs, emotions_list, token_key, token_idx)
        joy_baseline = logits_baseline[:, target_idx].mean()

        for alpha in tqdm(alphas, desc=f"{source_emotion}->{target_emotion} appraisal", leave=False):
            strength_per_site = alpha / k if k else alpha
            hs_steered = steer_at_circuit(hs_source, appraisal_vecs, strength_per_site)
            logits = _circuit_logits(hs_steered, probes, topk_per_emotion, extraction_layers, extraction_locs, emotions_list, token_key, token_idx)
            pred = np.argmax(logits, axis=1)
            success = (pred == target_idx).mean()
            mean_joy = logits[:, target_idx].mean()
            all_curve_rows.append({"source_emotion": source_emotion, "target_emotion": target_emotion, "method": "appraisal_steer", "strength": alpha, "success_rate": float(success), "mean_target_logit": float(mean_joy), "mean_delta_target": float(mean_joy - joy_baseline)})
        for beta in tqdm(betas, desc=f"{source_emotion}->{target_emotion} emotion", leave=False):
            strength_per_site = beta / k if k else beta
            hs_steered = steer_at_circuit(hs_source, emotion_vecs, strength_per_site)
            logits = _circuit_logits(hs_steered, probes, topk_per_emotion, extraction_layers, extraction_locs, emotions_list, token_key, token_idx)
            pred = np.argmax(logits, axis=1)
            success = (pred == target_idx).mean()
            mean_joy = logits[:, target_idx].mean()
            all_curve_rows.append({"source_emotion": source_emotion, "target_emotion": target_emotion, "method": "emotion_steer", "strength": beta, "success_rate": float(success), "mean_target_logit": float(mean_joy), "mean_delta_target": float(mean_joy - joy_baseline)})

        curves_sub = pd.DataFrame([r for r in all_curve_rows if r["source_emotion"] == source_emotion and r["target_emotion"] == target_emotion])
        app_row = curves_sub[(curves_sub["method"] == "appraisal_steer") & (curves_sub["strength"] == ref_alpha_global)]
        emo_row = curves_sub[(curves_sub["method"] == "emotion_steer") & (curves_sub["strength"] == ref_beta_global)]
        if app_row.empty:
            app_row = curves_sub[curves_sub["method"] == "appraisal_steer"].iloc[-1:]
        if emo_row.empty:
            emo_row = curves_sub[curves_sub["method"] == "emotion_steer"].iloc[-1:]
        all_bench_rows.append({"source_emotion": source_emotion, "target_emotion": target_emotion, "method": "appraisal_steer", "success_rate": app_row["success_rate"].values[0], "mean_delta_target_logit": app_row["mean_delta_target"].values[0], "n_source_samples": int(n_source), "evaluation_split": evaluation_split})
        all_bench_rows.append({"source_emotion": source_emotion, "target_emotion": target_emotion, "method": "emotion_steer", "success_rate": emo_row["success_rate"].values[0], "mean_delta_target_logit": emo_row["mean_delta_target"].values[0], "n_source_samples": int(n_source), "evaluation_split": evaluation_split})

    benchmark_df = pd.DataFrame(all_bench_rows)
    curves_df = pd.DataFrame(all_curve_rows)
    benchmark_df.to_csv(out_dir / "steering_benchmark.csv", index=False)
    curves_df.to_csv(out_dir / "steering_curves.csv", index=False)
    _progress("Saved cache-based steering benchmark outputs.")
    save_steering_pair_outputs(
        benchmark_df=benchmark_df,
        curves_df=curves_df,
        out_dir=out_dir,
        contrastive_pairs=CONTRASTIVE_EMOTION_PAIRS,
        similar_pairs=SIMILAR_EMOTION_PAIRS,
        behavioral=False,
    )
    print(f"Wrote {out_dir / 'steering_benchmark.csv'} and {out_dir / 'steering_curves.csv'} ({len(pairs)} pairs)")

    bench_b_df = None
    if run_behavioral:
        _progress("Preparing behavioral steering run (model load + prompted/unprompted texts).")
        val_labels_both, val_prompted, val_unprompted = _load_eval_split_both_text_types()
        all_bench_b = []
        all_curves_b = []
        all_samples_b = []
        all_ablation_b = []
        behavioral_model = None
        behavioral_tokenizer = None
        try:
            from utils import Log
            from experiments.utils.training_utils import OvaLogisticRegressionTrainer
            behavioral_logger = Log("steering_behavioral").logger
            trainer = OvaLogisticRegressionTrainer(model_id, logger=behavioral_logger)
            trainer.load_model_and_tokenizer()
            behavioral_model, behavioral_tokenizer = trainer.model, trainer.tokenizer
            if behavioral_model is None or behavioral_tokenizer is None:
                raise RuntimeError("Failed to load model/tokenizer for behavioral steering.")
        except Exception as e:
            print(f"Behavioral setup failed while loading model/tokenizer: {e}")
            raise
        for source_emotion, target_emotion in tqdm(pairs, desc="Emotion pairs (behavioral)", unit="pair"):
            target_idx = emotions_list.index(target_emotion)
            source_indices = [i for i, l in enumerate(val_labels_both) if l == source_emotion]
            if not source_indices:
                continue
            prompted_texts = [val_prompted[i] for i in source_indices]
            unprompted_texts = [val_unprompted[i] for i in source_indices]
            target_circuit_pairs = topk_per_emotion.get(target_emotion, [])
            if not target_circuit_pairs:
                by_layer_loc = summary_df.groupby(["layer", "loc"])["test_roc_auc"].mean().reset_index()
                by_layer_loc = by_layer_loc.sort_values("test_roc_auc", ascending=False)
                target_circuit_pairs = [(int(by_layer_loc.iloc[0]["layer"]), int(by_layer_loc.iloc[0]["loc"]))]
            k = len(target_circuit_pairs)
            hs_source = hidden_states[np.array([l == source_emotion for l in val_labels])]
            hs_target = hidden_states[np.array([l == target_emotion for l in val_labels])] if any(l == target_emotion for l in val_labels) else None
            appraisal_vecs = []
            for layer, loc in target_circuit_pairs:
                vec = _compute_appraisal_steering_vector(
                    source_emotion, target_emotion, layer, loc, hidden_size, model_id,
                    appraisal_probes=appraisal_probes_cache, zscore_df=zscore_df_cache,
                )
                if vec is None and hs_target is not None and len(hs_target) > 0:
                    li, lci = layer_to_idx.get(layer), loc_to_idx.get(loc)
                    if li is not None and lci is not None:
                        v = compute_steering_vector(hs_target, hs_source, li, loc_idx=lci, token_idx=token_idx)
                        vec = v.cpu().numpy().astype(np.float32)
                appraisal_vecs.append(vec if vec is not None else np.zeros(hidden_size, dtype=np.float32))
            emotion_vecs = []
            for layer, loc in target_circuit_pairs:
                if layer in probes and loc in probes[layer] and token_key in probes[layer][loc] and target_emotion in probes[layer][loc][token_key]:
                    rec = probes[layer][loc][token_key][target_emotion]
                    emotion_vecs.append(_emotion_vec_from_probe_record(rec))
                else:
                    emotion_vecs.append(np.zeros(hidden_size, dtype=np.float32))

            for text_type, texts in [("prompted", prompted_texts), ("unprompted", unprompted_texts)]:
                try:
                    bench_b, curves_b, samples_b = _run_behavioral_steering(
                        model_id=model_id, source_emotion=source_emotion, target_emotion=target_emotion,
                        paths=paths, circuit_dir=circuit_dir, out_dir=out_dir, summary_df=summary_df,
                        emotions_list=emotions_list, target_idx=target_idx, extraction_layers=extraction_layers,
                        extraction_locs=extraction_locs, topk_per_emotion=topk_per_emotion,
                        target_circuit_pairs=target_circuit_pairs, appraisal_vecs=appraisal_vecs, emotion_vecs=emotion_vecs,
                        k=k, alphas=alphas, betas=betas, token_key=token_key, token_idx=token_idx,
                        source_texts=texts, text_type=text_type, model=behavioral_model, tokenizer=behavioral_tokenizer, logger=None,
                    )
                    for _, r in bench_b.iterrows():
                        all_bench_b.append({"source_emotion": source_emotion, "target_emotion": target_emotion, "text_type": text_type, "method": r["method"], "success_rate": r["success_rate"], "mean_delta_target_logit": r["mean_delta_target_logit"], "n_source_samples": int(len(texts)), "evaluation_split": evaluation_split})
                    for _, r in curves_b.iterrows():
                        all_curves_b.append({"source_emotion": source_emotion, "target_emotion": target_emotion, "text_type": text_type, **{k: v for k, v in r.items() if k != "text_type"}})
                    all_samples_b.append(samples_b)
                except Exception as e:
                    print(f"Behavioral {source_emotion}->{target_emotion} ({text_type}) failed: {e}")

        # Behavioral appraisal ablation: erase each emotion's own appraisal signature at its circuit sites.
        for emotion in tqdm(emotions_list, desc="Behavioral appraisal ablation", unit="emotion"):
            source_indices = [i for i, l in enumerate(val_labels_both) if l == emotion]
            if not source_indices:
                continue
            target_pairs = topk_per_emotion.get(emotion, [])
            if not target_pairs:
                continue
            ablation_vectors = _emotion_appraisal_signature_vectors_for_emotion(
                emotion=emotion,
                target_pairs=target_pairs,
                hidden_size=hidden_size,
                appraisal_probes=appraisal_probes_cache,
                zscore_df=zscore_df_cache,
            )
            if not ablation_vectors:
                continue
            for text_type, texts in [
                ("prompted", [val_prompted[i] for i in source_indices]),
                ("unprompted", [val_unprompted[i] for i in source_indices]),
            ]:
                try:
                    from torch.utils.data import DataLoader
                    from utils import run_forward_with_steering, TextDataset

                    dataset = TextDataset(texts=texts, labels=[0] * len(texts))
                    batch_size = get_behavioral_batch_size(model_id)
                    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

                    baseline_hs, _ = run_forward_with_steering(
                        dataloader, behavioral_tokenizer, behavioral_model,
                        extraction_layers, extraction_locs, list(EXTRACTION_TOKENS),
                        steering_spec={}, logger=None,
                    )
                    erase_spec = {
                        pair: {"op": "erase", "vector": vec, "strength": 1.0}
                        for pair, vec in ablation_vectors.items()
                    }
                    ablated_hs, _ = run_forward_with_steering(
                        dataloader, behavioral_tokenizer, behavioral_model,
                        extraction_layers, extraction_locs, list(EXTRACTION_TOKENS),
                        steering_spec=erase_spec, logger=None,
                    )
                    baseline_logits = _circuit_logits(
                        baseline_hs, probes, topk_per_emotion, extraction_layers, extraction_locs,
                        emotions_list, token_key, token_idx,
                    )
                    ablated_logits = _circuit_logits(
                        ablated_hs, probes, topk_per_emotion, extraction_layers, extraction_locs,
                        emotions_list, token_key, token_idx,
                    )
                    e_idx = emotions_list.index(emotion)
                    base_self_logit = float(baseline_logits[:, e_idx].mean())
                    ablated_self_logit = float(ablated_logits[:, e_idx].mean())
                    base_self_success = float((np.argmax(baseline_logits, axis=1) == e_idx).mean())
                    ablated_self_success = float((np.argmax(ablated_logits, axis=1) == e_idx).mean())
                    all_ablation_b.append(
                        {
                            "emotion": emotion,
                            "text_type": text_type,
                            "ablation_type": "full_appraisal_signature",
                            "n_source_samples": int(len(texts)),
                            "baseline_self_logit": base_self_logit,
                            "ablated_self_logit": ablated_self_logit,
                            "delta_self_logit": ablated_self_logit - base_self_logit,
                            "baseline_self_success": base_self_success,
                            "ablated_self_success": ablated_self_success,
                            "delta_self_success": ablated_self_success - base_self_success,
                        }
                    )
                except Exception as e:
                    print(f"Behavioral appraisal ablation failed for {emotion} ({text_type}): {e}")

        if all_bench_b:
            bench_b_df = pd.DataFrame(all_bench_b)
            bench_b_df.to_csv(out_dir / "steering_benchmark_behavioral.csv", index=False)
            curves_b_df = pd.DataFrame(all_curves_b)
            curves_b_df.to_csv(out_dir / "steering_curves_behavioral.csv", index=False)
            _progress("Saved behavioral steering benchmark outputs.")
            save_steering_pair_outputs(
                benchmark_df=bench_b_df,
                curves_df=curves_b_df,
                out_dir=out_dir,
                contrastive_pairs=CONTRASTIVE_EMOTION_PAIRS,
                similar_pairs=SIMILAR_EMOTION_PAIRS,
                behavioral=True,
            )
            if all_samples_b:
                samples_b_df = pd.concat(all_samples_b, ignore_index=True)
                samples_b_df.to_csv(out_dir / "steering_behavioral_samples.csv", index=False)
                print(f"Wrote {out_dir / 'steering_behavioral_samples.csv'} ({len(samples_b_df)} rows)")
            print(f"Wrote {out_dir / 'steering_benchmark_behavioral.csv'} and {out_dir / 'steering_curves_behavioral.csv'}")
            if all_ablation_b:
                abl_b_df = pd.DataFrame(all_ablation_b)
                abl_b_df.to_csv(out_dir / "behavioral_appraisal_ablation.csv", index=False)
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, 0.35 * abl_b_df["emotion"].nunique() + 2)))
                    sns.barplot(data=abl_b_df, y="emotion", x="delta_self_logit", hue="text_type", ax=axes[0])
                    axes[0].axvline(0.0, color="black", linewidth=1)
                    axes[0].set_title("Behavioral ablation: delta self logit")
                    sns.barplot(data=abl_b_df, y="emotion", x="delta_self_success", hue="text_type", ax=axes[1])
                    axes[1].axvline(0.0, color="black", linewidth=1)
                    axes[1].set_title("Behavioral ablation: delta self success")
                    fig.tight_layout()
                    fig.savefig(out_dir / "behavioral_appraisal_ablation.png", bbox_inches="tight")
                    fig.savefig(out_dir / "behavioral_appraisal_ablation.pdf", bbox_inches="tight")
                    plt.close(fig)
                except Exception as e:
                    print(f"Behavioral ablation figure save failed: {e}")
            # Behavioral bar chart (mean over pairs and text_type per method)
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                agg_b = bench_b_df.groupby("method").agg({"success_rate": "mean", "mean_delta_target_logit": "mean"}).reset_index()
                fig, ax = plt.subplots(1, 1, figsize=(6, 4))
                x = np.arange(len(agg_b))
                ax.bar(x - 0.2, agg_b["success_rate"], width=0.35, label="Success rate (mean over pairs)")
                ax.bar(x + 0.2, agg_b["mean_delta_target_logit"], width=0.35, label="Mean delta target logit")
                ax.set_xticks(x)
                ax.set_xticklabels(agg_b["method"].tolist())
                ax.set_ylabel("Score")
                ax.set_title("Steering benchmark (behavioral, mean over pairs & text_type)")
                ax.legend()
                fig.tight_layout()
                fig.savefig(out_dir / "steering_benchmark_behavioral.pdf", bbox_inches="tight")
                fig.savefig(out_dir / "steering_benchmark_behavioral.png", bbox_inches="tight")
                plt.close(fig)
                print(f"Wrote {out_dir / 'steering_benchmark_behavioral.png'} and {out_dir / 'steering_benchmark_behavioral.pdf'}")
            except Exception as e:
                print(f"Behavioral figure save failed: {e}")
        _progress("Behavioral steering run complete.")
        # Explicitly free model memory after behavioral run
        try:
            del behavioral_model
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        agg = benchmark_df.groupby("method").agg({"success_rate": "mean", "mean_delta_target_logit": "mean"}).reset_index()
        x = np.arange(len(agg))
        ax.bar(x - 0.2, agg["success_rate"], width=0.35, label="Success rate (mean over pairs)")
        ax.bar(x + 0.2, agg["mean_delta_target_logit"], width=0.35, label="Mean delta target logit")
        ax.set_xticks(x)
        ax.set_xticklabels(agg["method"].tolist())
        ax.set_ylabel("Score")
        ax.set_title("Steering benchmark (circuit top-k, mean over emotion pairs)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "steering_benchmark.pdf", bbox_inches="tight")
        fig.savefig(out_dir / "steering_benchmark.png", bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"Figure save failed: {e}")

    summary_lines = [
        "# 05_steering",
        "",
        "Circuit-level steering over contrastive emotion pairs using the selected per-emotion top-k circuit.",
        "",
    ]
    if not benchmark_df.empty:
        def _weighted_success(df: pd.DataFrame, method: str) -> float:
            sub = df[df["method"] == method].copy()
            if sub.empty:
                return float("nan")
            weights = sub["n_source_samples"].astype(float)
            return float(np.average(sub["success_rate"].astype(float), weights=weights))

        app_success = _weighted_success(benchmark_df, "appraisal_steer")
        emo_success = _weighted_success(benchmark_df, "emotion_steer")
        summary_lines.extend([
            f"- Cached steering weighted success: appraisal `{app_success:.3f}`, emotion `{emo_success:.3f}`.",
        ])
        pair_winners = (
            benchmark_df.pivot_table(
                index=["source_emotion", "target_emotion"],
                columns="method",
                values="success_rate",
                aggfunc="mean",
            )
            .reset_index()
        )
        if {"appraisal_steer", "emotion_steer"}.issubset(pair_winners.columns):
            pair_winners["gap_emotion_minus_appraisal"] = (
                pair_winners["emotion_steer"] - pair_winners["appraisal_steer"]
            )
            worst_gap = pair_winners.sort_values("gap_emotion_minus_appraisal", ascending=False).head(3)
            if not worst_gap.empty:
                formatted = ", ".join(
                    f"{r['source_emotion']}->{r['target_emotion']} ({r['gap_emotion_minus_appraisal']:.3f})"
                    for _, r in worst_gap.iterrows()
                )
                summary_lines.append(f"- Largest cached emotion-over-appraisal success gaps: {formatted}.")
        summary_lines.append("")

    if bench_b_df is not None and not bench_b_df.empty:
        def _weighted_behavioral(df: pd.DataFrame, method: str) -> float:
            sub = df[df["method"] == method].copy()
            if sub.empty:
                return float("nan")
            weights = sub["n_source_samples"].astype(float)
            return float(np.average(sub["success_rate"].astype(float), weights=weights))

        app_b = _weighted_behavioral(bench_b_df, "appraisal_steer")
        emo_b = _weighted_behavioral(bench_b_df, "emotion_steer")
        summary_lines.extend([
            f"- Behavioral steering weighted success: appraisal `{app_b:.3f}`, emotion `{emo_b:.3f}`.",
        ])
        behavior_pairs = (
            bench_b_df.pivot_table(
                index=["source_emotion", "target_emotion", "text_type"],
                columns="method",
                values="success_rate",
                aggfunc="mean",
            )
            .reset_index()
        )
        if {"appraisal_steer", "emotion_steer"}.issubset(behavior_pairs.columns):
            behavior_pairs["gap_emotion_minus_appraisal"] = (
                behavior_pairs["emotion_steer"] - behavior_pairs["appraisal_steer"]
            )
            worst_behavior = behavior_pairs.sort_values("gap_emotion_minus_appraisal", ascending=False).head(4)
            if not worst_behavior.empty:
                formatted = ", ".join(
                    f"{r['source_emotion']}->{r['target_emotion']} {r['text_type']} ({r['gap_emotion_minus_appraisal']:.3f})"
                    for _, r in worst_behavior.iterrows()
                )
                summary_lines.append(f"- Largest behavioral emotion-over-appraisal success gaps: {formatted}.")
        if (out_dir / "behavioral_appraisal_ablation.csv").exists():
            summary_lines.append("- Behavioral appraisal ablation results are saved separately in `behavioral_appraisal_ablation.csv` and the paired figure outputs.")
        summary_lines.append("")

    summary_lines.extend([
        "Use `steering_benchmark.csv` and `steering_benchmark_behavioral.csv` for aggregate comparisons, and the by-pair outputs to see whether effects are broad or concentrated in a few transitions.",
        "",
    ])
    (out_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    _progress("Steering benchmark complete.")

    return benchmark_df


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--source", default=None, help="Optional source emotion; must be paired with --target.")
    p.add_argument("--target", default=None, help="Optional target emotion; must be paired with --source.")
    p.add_argument(
        "--top_k_pairs",
        type=int,
        default=None,
        help="Override circuit_top_k_selection.json with fixed k (probe-summary ranking)",
    )
    p.add_argument("--behavioral", action="store_true", help="Also run real forward pass with in-pass steering (slower).")
    args = p.parse_args()
    run_steering_benchmark(
        model_id=args.model_id,
        source_emotion=args.source,
        target_emotion=args.target,
        top_k_pairs=args.top_k_pairs,
        run_behavioral=args.behavioral,
    )

"""
Train appraisal regression probes on the train split only and validate on the selection split.

This stage now follows the same scenario-level split protocol as the rest of the rigorous
pipeline so downstream phase-2 and steering analyses do not rely on in-sample appraisal fits.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from pipeline.core.config import (
    REPO_ROOT,
    COMBINED_CSV,
    EXTRACTION_TOKENS,
    COMMON_APPRAISAL,
    get_probes_dir,
    DEFAULT_MODEL_ID,
    SELECTION_SPLIT,
    FINAL_TEST_SPLIT,
    RANDOM_STATE,
)
from pipeline.core.model_config import get_extraction_layers, get_extraction_locs, get_extraction_batch_size
from pipeline.core.research_contracts import (
    canonicalize_combined_dataset,
    split_combined_dataset,
    write_json,
    resolve_text_columns,
)
RIDGE_ALPHA = 5.0
MIN_EMOTION_COUNT_FOR_BALANCE = 50
RANDOM_STATE_APPRAISAL = RANDOM_STATE


def _ensure_repo():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def _balance_by_emotion(
    df: pd.DataFrame,
    emotion_col: str = "emotion",
    random_state: int = RANDOM_STATE_APPRAISAL,
) -> pd.DataFrame:
    """Balance only the training partition to stabilize appraisal fits across emotions."""
    counts = df[emotion_col].value_counts()
    eligible = counts[counts >= MIN_EMOTION_COUNT_FOR_BALANCE]
    if eligible.empty:
        return df
    target_count = int(eligible.max())
    rng = np.random.default_rng(random_state)
    pieces = []
    for emotion in eligible.index:
        sub = df.loc[df[emotion_col] == emotion]
        replace = len(sub) < target_count
        indices = rng.choice(sub.index, size=target_count, replace=replace)
        pieces.append(df.loc[indices])
    return pd.concat(pieces, axis=0).reset_index(drop=True)


def _token_col_value(token_key) -> str | int:
    """CSV-serializable token label (matches emotion probe_summary token column style)."""
    if isinstance(token_key, str):
        return token_key
    return int(token_key)


def _metric_row(
    dim: str,
    layer: int,
    loc: int,
    token_key,
    n_valid: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_true) > 1e-12 and np.std(y_pred) > 1e-12 else 0.0
    return {
        "dimension": dim,
        "layer": int(layer),
        "loc": int(loc),
        "token": _token_col_value(token_key),
        "n_valid": int(n_valid),
        "selection_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "selection_r2": float(r2_score(y_true, y_pred)),
        "selection_corr": corr,
    }


def run_train_appraisal_probes(
    model_id: str = DEFAULT_MODEL_ID,
    max_samples: Optional[int] = None,
    ridge_alpha: float = RIDGE_ALPHA,
    balance_by_emotion: bool = True,
    logger=None,
    output_dir: Optional[Path] = None,
    extraction_locs_override: Optional[list] = None,
    extraction_tokens_override: Optional[list] = None,
) -> Path:
    """
    Fit Ridge appraisal probes on the global train split and validate on the selection split.
    Save probe weights, scalers, and selection metrics under outputs/<model_id>/01_probes/ by default.

    output_dir: Optional alternate directory (e.g. grid ablation under `01_probes_grid_ablation/`).
    extraction_locs_override / extraction_tokens_override: Optional grid overrides; token list may
        include the string ``"mid"`` (per-sequence masked middle; requires utils.extract_hidden_states support).
    """
    _ensure_repo()
    from utils import Log
    from experiments.utils.training_utils import OvaLogisticRegressionTrainer

    if logger is None:
        logger = Log("train_appraisal_probes").logger

    if not COMBINED_CSV.exists():
        raise FileNotFoundError(f"COMBINED_CSV not found: {COMBINED_CSV}. Required for appraisal probe training.")

    df = canonicalize_combined_dataset(pd.read_csv(COMBINED_CSV))
    base_text_col, prompted_col = resolve_text_columns(df)
    text_col = prompted_col or base_text_col
    df = df.dropna(subset=[text_col])
    appraisal_cols = [c for c in COMMON_APPRAISAL if c in df.columns]
    if not appraisal_cols:
        raise ValueError(f"None of COMMON_APPRAISAL columns found in {COMBINED_CSV}. Columns: {list(df.columns)}")

    split_bundle = split_combined_dataset(
        df,
        selection_frac=SELECTION_SPLIT,
        test_frac=FINAL_TEST_SPLIT,
        random_state=RANDOM_STATE_APPRAISAL,
    )
    train_df = split_bundle["train"].copy()
    selection_df = split_bundle["selection"].copy()
    if balance_by_emotion and "emotion" in train_df.columns:
        n_before = len(train_df)
        train_df = _balance_by_emotion(train_df, random_state=RANDOM_STATE_APPRAISAL)
        logger.info(f"Emotion-balanced appraisal training set: {n_before} -> {len(train_df)} rows.")
    if max_samples is not None and len(train_df) > max_samples:
        train_df = train_df.sample(n=max_samples, random_state=RANDOM_STATE_APPRAISAL).reset_index(drop=True)
        logger.info(f"Subsampled training set to {max_samples} rows (max_samples cap).")

    train_texts = train_df[text_col].astype(str).tolist()
    selection_texts = selection_df[text_col].astype(str).tolist()
    extraction_layers = get_extraction_layers(model_id)
    extraction_locs = extraction_locs_override if extraction_locs_override is not None else get_extraction_locs(model_id)
    extraction_tokens = list(extraction_tokens_override) if extraction_tokens_override is not None else list(EXTRACTION_TOKENS)
    extraction_batch_size = get_extraction_batch_size(model_id)

    logger.info(
        f"Extracting appraisal hidden states for {model_id}: train={len(train_texts)}, "
        f"selection={len(selection_texts)}, held_out_test={len(split_bundle['test'])}, "
        f"batch_size={extraction_batch_size}, locs={extraction_locs}, tokens={extraction_tokens}"
    )
    trainer = OvaLogisticRegressionTrainer(model_id, logger=logger)
    train_hidden_states = trainer.extract_hidden_states(
        texts=train_texts,
        extraction_layers=extraction_layers,
        extraction_locs=extraction_locs,
        extraction_tokens=extraction_tokens,
        batch_size=extraction_batch_size,
    )
    selection_hidden_states = trainer.extract_hidden_states(
        texts=selection_texts,
        extraction_layers=extraction_layers,
        extraction_locs=extraction_locs,
        extraction_tokens=extraction_tokens,
        batch_size=extraction_batch_size,
    )
    train_hs = train_hidden_states.cpu().numpy() if isinstance(train_hidden_states, torch.Tensor) else np.asarray(train_hidden_states)
    selection_hs = selection_hidden_states.cpu().numpy() if isinstance(selection_hidden_states, torch.Tensor) else np.asarray(selection_hidden_states)

    appraisal_probes: dict = {}
    metric_rows = []
    for dim in appraisal_cols:
        y_train_full = train_df[dim].values.astype(np.float64)
        y_sel_full = selection_df[dim].values.astype(np.float64)
        valid_train = ~np.isnan(y_train_full)
        valid_sel = ~np.isnan(y_sel_full)
        if valid_train.sum() < 10 or valid_sel.sum() < 10:
            logger.warning(f"Skipping dimension {dim}: too few valid train/selection targets.")
            continue
        appraisal_probes[dim] = {}
        for layer_idx, layer in enumerate(extraction_layers):
            appraisal_probes[dim][layer] = {}
            for loc_idx, loc in enumerate(extraction_locs):
                appraisal_probes[dim][layer][loc] = {}
                for tok_idx, token_key in enumerate(extraction_tokens):
                    X_train = train_hs[:, layer_idx, loc_idx, tok_idx, :].astype(np.float64)[valid_train]
                    X_sel = selection_hs[:, layer_idx, loc_idx, tok_idx, :].astype(np.float64)[valid_sel]
                    y_train = y_train_full[valid_train]
                    y_sel = y_sel_full[valid_sel]
                    if len(y_train) < 10 or len(y_sel) < 10:
                        continue
                    scaler = StandardScaler()
                    X_train_z = scaler.fit_transform(X_train)
                    X_sel_z = scaler.transform(X_sel)
                    ridge = Ridge(alpha=ridge_alpha, fit_intercept=True)
                    ridge.fit(X_train_z, y_train)
                    pred_sel = ridge.predict(X_sel_z)
                    metric_rows.append(_metric_row(dim, layer, loc, token_key, len(y_sel), y_sel, pred_sel))
                    appraisal_probes[dim][layer][loc][token_key] = {
                        "ridge": ridge,
                        "scaler": scaler,
                        "weights": ridge.coef_.ravel().astype(np.float64),
                        "bias": float(np.ravel(getattr(ridge, "intercept_", [0.0]))[0]),
                    }

    out_dir = Path(output_dir) if output_dir is not None else get_probes_dir(model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "appraisal_regression_probes.pt"
    torch.save(appraisal_probes, out_path)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = out_dir / "appraisal_probe_validation_detail.csv"
    metrics_df.to_csv(metrics_path, index=False)
    write_json(
        out_dir / "appraisal_regression_probes_manifest.json",
        {
            "version": 1,
            "model_id": model_id,
            "text_column": text_col,
            "selection_split": SELECTION_SPLIT,
            "final_test_split": FINAL_TEST_SPLIT,
            "random_state": RANDOM_STATE_APPRAISAL,
            "ridge_alpha": ridge_alpha,
            "train_rows": int(len(train_df)),
            "selection_rows": int(len(selection_df)),
            "held_out_test_rows": int(len(split_bundle["test"])),
            "dimensions": appraisal_cols,
            "extraction_locs": list(extraction_locs),
            "extraction_tokens": [t if isinstance(t, str) else int(t) for t in extraction_tokens],
            "validation_metrics_file": metrics_path.name,
        },
    )
    logger.info(f"Saved {out_path} ({len(appraisal_probes)} dimensions).")
    logger.info(f"Saved {metrics_path} ({len(metrics_df)} rows).")
    return out_path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Train appraisal regression probes per layer/loc")
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--max_samples", type=int, default=None, help="If set, cap training rows")
    p.add_argument("--ridge_alpha", type=float, default=RIDGE_ALPHA)
    p.add_argument("--no_balance", action="store_true", help="Disable emotion-balanced sampling for the train split")
    args = p.parse_args()
    run_train_appraisal_probes(
        model_id=args.model_id,
        max_samples=args.max_samples,
        ridge_alpha=args.ridge_alpha,
        balance_by_emotion=not args.no_balance,
    )

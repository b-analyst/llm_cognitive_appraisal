"""
Train binary OVA logistic probes for appraisal dimensions (high vs low).

Each appraisal dimension is binarized using the median of the train split.
Binary probes enable sharper intervention directions and support the
appraisal-to-emotion reconstruction experiment in appraisal_theory.py.

Run:  python -m pipeline.train_appraisal_binary_probes [--model_id ...]
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from pipeline.core.config import (
    REPO_ROOT, COMBINED_CSV, EXTRACTION_TOKENS, COMMON_APPRAISAL,
    get_probes_dir, get_appraisal_binary_probes_dir, DEFAULT_MODEL_ID,
    SELECTION_SPLIT, FINAL_TEST_SPLIT, RANDOM_STATE,
    APPRAISAL_BINARY_THRESHOLD,
)
from pipeline.core.model_config import get_extraction_layers, get_extraction_locs, get_extraction_batch_size
from pipeline.core.research_contracts import (
    canonicalize_combined_dataset, split_combined_dataset,
    write_json, resolve_text_columns,
)
RANDOM_STATE_BINARY = RANDOM_STATE


def _ensure_repo():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def _binarize(values: np.ndarray, threshold: float) -> np.ndarray:
    return (values >= threshold).astype(np.int64)


def run_train_appraisal_binary_probes(
    model_id: str = DEFAULT_MODEL_ID,
    max_samples: Optional[int] = None,
    threshold_method: str = APPRAISAL_BINARY_THRESHOLD,
    logger=None,
) -> Path:
    """
    Train binary OVA logistic probes for each appraisal dimension.

    Binarization uses the train-set median (or a fixed float threshold).
    Probes are evaluated on the selection split.
    """
    _ensure_repo()
    from utils import Log
    from experiments.utils.training_utils import OvaLogisticRegressionTrainer

    if logger is None:
        logger = Log("train_appraisal_binary_probes").logger

    if not COMBINED_CSV.exists():
        raise FileNotFoundError(f"COMBINED_CSV not found: {COMBINED_CSV}")

    df = canonicalize_combined_dataset(pd.read_csv(COMBINED_CSV))
    base_text_col, prompted_col = resolve_text_columns(df)
    text_col = prompted_col or base_text_col
    df = df.dropna(subset=[text_col])
    appraisal_cols = [c for c in COMMON_APPRAISAL if c in df.columns]
    if not appraisal_cols:
        raise ValueError(f"No COMMON_APPRAISAL columns found in {COMBINED_CSV}")

    split_bundle = split_combined_dataset(
        df,
        selection_frac=SELECTION_SPLIT,
        test_frac=FINAL_TEST_SPLIT,
        random_state=RANDOM_STATE_BINARY,
    )
    train_df = split_bundle["train"].copy()
    selection_df = split_bundle["selection"].copy()
    if max_samples is not None and len(train_df) > max_samples:
        train_df = train_df.sample(n=max_samples, random_state=RANDOM_STATE_BINARY).reset_index(drop=True)

    train_texts = train_df[text_col].astype(str).tolist()
    selection_texts = selection_df[text_col].astype(str).tolist()
    extraction_layers = get_extraction_layers(model_id)
    extraction_locs = get_extraction_locs(model_id)
    extraction_batch_size = get_extraction_batch_size(model_id)
    token_key = EXTRACTION_TOKENS[0]

    logger.info(
        f"Extracting hidden states for binary appraisal probes ({model_id}): "
        f"train={len(train_texts)}, selection={len(selection_texts)}"
    )
    trainer = OvaLogisticRegressionTrainer(model_id, logger=logger)
    train_hs_raw = trainer.extract_hidden_states(
        texts=train_texts,
        extraction_layers=extraction_layers,
        extraction_locs=extraction_locs,
        extraction_tokens=EXTRACTION_TOKENS,
        batch_size=extraction_batch_size,
    )
    sel_hs_raw = trainer.extract_hidden_states(
        texts=selection_texts,
        extraction_layers=extraction_layers,
        extraction_locs=extraction_locs,
        extraction_tokens=EXTRACTION_TOKENS,
        batch_size=extraction_batch_size,
    )
    train_hs = train_hs_raw.cpu().numpy() if isinstance(train_hs_raw, torch.Tensor) else np.asarray(train_hs_raw)
    sel_hs = sel_hs_raw.cpu().numpy() if isinstance(sel_hs_raw, torch.Tensor) else np.asarray(sel_hs_raw)

    binary_probes: dict = {}
    metric_rows: list[dict] = []
    thresholds_used: dict[str, float] = {}

    for dim in appraisal_cols:
        y_train_raw = train_df[dim].values.astype(np.float64)
        y_sel_raw = selection_df[dim].values.astype(np.float64)
        valid_train = ~np.isnan(y_train_raw)
        valid_sel = ~np.isnan(y_sel_raw)
        if valid_train.sum() < 20 or valid_sel.sum() < 10:
            logger.warning(f"Skipping {dim}: too few valid samples.")
            continue

        if threshold_method == "median":
            thresh = float(np.median(y_train_raw[valid_train]))
        else:
            thresh = float(threshold_method)
        thresholds_used[dim] = thresh

        y_train_bin = _binarize(y_train_raw[valid_train], thresh)
        y_sel_bin = _binarize(y_sel_raw[valid_sel], thresh)
        if len(np.unique(y_train_bin)) < 2 or len(np.unique(y_sel_bin)) < 2:
            logger.warning(f"Skipping {dim}: binarization produced single class (threshold={thresh:.2f}).")
            continue

        binary_probes[dim] = {}
        n_high_train = int(y_train_bin.sum())
        n_low_train = int(len(y_train_bin) - n_high_train)

        for layer_idx, layer in enumerate(extraction_layers):
            binary_probes[dim][layer] = {}
            for loc_idx, loc in enumerate(extraction_locs):
                X_train = train_hs[:, layer_idx, loc_idx, 0, :].astype(np.float64)[valid_train]
                X_sel = sel_hs[:, layer_idx, loc_idx, 0, :].astype(np.float64)[valid_sel]
                if len(X_train) < 20:
                    continue

                scaler = StandardScaler()
                X_train_z = scaler.fit_transform(X_train)
                X_sel_z = scaler.transform(X_sel)
                clf = LogisticRegression(
                    C=1.0, max_iter=2000, solver="lbfgs",
                    random_state=RANDOM_STATE_BINARY,
                )
                clf.fit(X_train_z, y_train_bin)
                pred_proba = clf.predict_proba(X_sel_z)[:, 1]
                pred_label = clf.predict(X_sel_z)

                try:
                    auc = float(roc_auc_score(y_sel_bin, pred_proba))
                except ValueError:
                    auc = float("nan")
                acc = float(accuracy_score(y_sel_bin, pred_label))
                f1 = float(f1_score(y_sel_bin, pred_label, zero_division=0.0))

                binary_probes[dim][layer][loc] = {
                    token_key: {
                        "classifier": clf,
                        "scaler": scaler,
                        "weights": clf.coef_.ravel().astype(np.float64),
                        "bias": float(clf.intercept_[0]),
                    }
                }
                metric_rows.append({
                    "dimension": dim,
                    "layer": int(layer),
                    "loc": int(loc),
                    "threshold": thresh,
                    "n_high_train": n_high_train,
                    "n_low_train": n_low_train,
                    "selection_roc_auc": auc,
                    "selection_accuracy": acc,
                    "selection_f1": f1,
                })

    out_dir = get_appraisal_binary_probes_dir(model_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    probes_path = out_dir / f"appraisal_binary_ova_probes_{model_id}.pt"
    torch.save(binary_probes, probes_path)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = out_dir / "appraisal_binary_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)

    write_json(
        out_dir / "appraisal_binary_manifest.json",
        {
            "version": 1,
            "model_id": model_id,
            "threshold_method": threshold_method,
            "thresholds": thresholds_used,
            "selection_split": SELECTION_SPLIT,
            "final_test_split": FINAL_TEST_SPLIT,
            "random_state": RANDOM_STATE_BINARY,
            "train_rows": int(len(train_df)),
            "selection_rows": int(len(selection_df)),
            "dimensions": list(binary_probes.keys()),
        },
    )
    logger.info(f"Saved {probes_path} ({len(binary_probes)} dimensions).")
    logger.info(f"Saved {metrics_path} ({len(metrics_df)} rows).")
    return probes_path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Train binary OVA appraisal probes")
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--max_samples", type=int, default=None)
    args = p.parse_args()
    run_train_appraisal_binary_probes(model_id=args.model_id, max_samples=args.max_samples)

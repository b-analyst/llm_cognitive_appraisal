from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


DEFAULT_PROMPT_SHOTS = ("fear", "anger", "sadness")


def normalize_emotion_values(series: pd.Series) -> pd.Series:
    """Normalize emotion labels to the lowercase canonical form used by the pipeline."""
    return series.astype(str).str.strip().str.lower()


def canonicalize_combined_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the combined dataset schema and attach stable scenario IDs for split grouping.

    The contract is:
    - `emotion` exists and is lowercase
    - `situation` exists when raw text is available
    - `scenario_id` groups prompt variants from the same underlying situation
    """
    out = df.copy()
    if "emotion" not in out.columns:
        raise ValueError("Combined dataset must contain an 'emotion' column.")
    out["emotion"] = normalize_emotion_values(out["emotion"])
    if "situation" in out.columns:
        base_source = out["source"].astype(str) if "source" in out.columns else pd.Series([""] * len(out))
        base_text = out["situation"].astype(str)
        keys = (base_source.fillna("") + "||" + base_text.fillna("")).tolist()
        out["scenario_id"] = [hashlib.sha1(k.encode("utf-8")).hexdigest()[:16] for k in keys]
    else:
        out["scenario_id"] = [f"row_{i}" for i in range(len(out))]
    return out


def dataset_fingerprint(path: Path) -> dict:
    """Return a deterministic fingerprint for an input file used by cached artifacts."""
    if not path.exists():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_manifest_model_match(manifest_path: Path, model_id: str, kind: str) -> None:
    """Raise if a manifest exists and declares a different model_id."""
    manifest = read_json_if_exists(manifest_path)
    if manifest is None:
        return
    declared = manifest.get("model_id")
    if declared is None:
        return
    if str(declared) != str(model_id):
        raise ValueError(
            f"{kind} manifest mismatch: expected model_id={model_id}, found {declared} in {manifest_path}"
        )


def manifest_matches(actual: dict | None, expected: dict) -> tuple[bool, str]:
    """Return whether a manifest contains the expected key/value pairs."""
    if actual is None:
        return False, "missing manifest"
    for key, expected_value in expected.items():
        if isinstance(expected_value, dict):
            ok, reason = manifest_matches(actual.get(key), expected_value)
            if not ok:
                return False, f"{key}.{reason}"
            continue
        actual_value = actual.get(key)
        if actual_value != expected_value:
            return False, f"{key} mismatch: expected {expected_value!r}, got {actual_value!r}"
    return True, ""


def resolve_text_columns(df: pd.DataFrame) -> tuple[str, str | None]:
    """Return `(base_text_col, prompted_text_col)` for the canonical combined dataset."""
    if "situation" not in df.columns:
        raise ValueError("Combined dataset must contain a 'situation' column.")
    prompted_col = "hidden_emo_text" if "hidden_emo_text" in df.columns else None
    return "situation", prompted_col


def build_prompted_texts(
    situations: list[str],
    emotions_list: list[str],
    prompt_index: int = 1,
    shots: tuple[str, ...] = DEFAULT_PROMPT_SHOTS,
) -> list[str]:
    """Generate prompted texts deterministically using the same prompt manager as probe training."""
    from prompt_manager import build_prompt

    fn = build_prompt(shots=shots, prompt_index=prompt_index, emotions_list=emotions_list)
    return [fn(str(s)) for s in situations]


def macro_ovr_roc_auc_from_scores(y_true, scores) -> float:
    """
    Compute macro one-vs-rest ROC-AUC from per-class scores.

    Unlike sklearn's multiclass ROC helper, this does not require each row of `scores`
    to sum to 1. That matters in this pipeline because top-k fusion combines
    independently trained binary OVA probe scores rather than a calibrated multiclass
    probability distribution.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=np.float64)
    if y_true.ndim != 1:
        raise ValueError("y_true must be 1D.")
    if scores.ndim != 2:
        raise ValueError("scores must be 2D with shape (n_samples, n_classes).")
    if scores.shape[0] != y_true.shape[0]:
        raise ValueError("scores and y_true must have the same number of rows.")

    present = np.unique(y_true)
    aucs: list[float] = []
    for label in present.tolist():
        label = int(label)
        if label < 0 or label >= scores.shape[1]:
            continue
        y_bin = (y_true == label).astype(np.int64)
        if y_bin.min() == y_bin.max():
            continue
        try:
            aucs.append(float(roc_auc_score(y_bin, scores[:, label])))
        except Exception:
            continue
    if not aucs:
        return float("nan")
    return float(np.mean(aucs))


def supported_emotion_stats(
    df: pd.DataFrame,
    selection_frac: float,
    test_frac: float,
    random_state: int,
    min_train_count: int,
    min_selection_count: int,
    label_col: str = "emotion",
    group_col: str = "scenario_id",
) -> dict:
    """
    Compute the main supported emotion set from the rigorous split protocol.

    Supported emotions must have enough positive examples in both train and selection.
    """
    split_bundle = split_combined_dataset(
        df,
        selection_frac=selection_frac,
        test_frac=test_frac,
        random_state=random_state,
        label_col=label_col,
        group_col=group_col,
    )
    train_counts = split_bundle["train"][label_col].value_counts().sort_index()
    selection_counts = split_bundle["selection"][label_col].value_counts().sort_index()
    test_counts = split_bundle["test"][label_col].value_counts().sort_index()
    all_labels = sorted(set(train_counts.index).union(selection_counts.index).union(test_counts.index))
    rows = []
    supported = []
    for label in all_labels:
        tr = int(train_counts.get(label, 0))
        sel = int(selection_counts.get(label, 0))
        tst = int(test_counts.get(label, 0))
        is_supported = tr >= min_train_count and sel >= min_selection_count
        rows.append(
            {
                "emotion": str(label),
                "train_count": tr,
                "selection_count": sel,
                "test_count": tst,
                "is_supported": bool(is_supported),
            }
        )
        if is_supported:
            supported.append(str(label))
    return {
        "supported_emotions": supported,
        "counts_df": pd.DataFrame(rows),
        "split_bundle": split_bundle,
    }


def _stable_group_frame(df: pd.DataFrame, label_col: str, group_col: str) -> pd.DataFrame:
    groups = df[[group_col, label_col]].dropna().drop_duplicates().copy()
    dup_counts = groups.groupby(group_col)[label_col].nunique()
    ambiguous = set(dup_counts[dup_counts > 1].index.tolist())
    if ambiguous:
        groups["split_group_id"] = groups.apply(
            lambda r: f"{r[group_col]}::{r[label_col]}" if r[group_col] in ambiguous else r[group_col],
            axis=1,
        )
    else:
        groups["split_group_id"] = groups[group_col]
    groups = groups.drop_duplicates(subset=["split_group_id"]).rename(columns={label_col: "label"})
    return groups[["split_group_id", "label"]]


def _safe_group_train_test_split(
    groups: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    # Coerce to plain numpy object arrays: sklearn indexing fails on PyArrow-backed
    # ExtensionArray values from pandas 2.x (TypeError: only integer scalar arrays...).
    split_ids = groups["split_group_id"].astype(str).to_numpy(dtype=object)
    labels = groups["label"].astype(str).to_numpy(dtype=object)
    counts = pd.Series(labels).value_counts()
    if len(counts) == 0:
        return np.array([], dtype=object), np.array([], dtype=object)
    stratify = labels if counts.min() >= 2 else None
    train_ids, test_ids = train_test_split(
        split_ids,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify,
    )
    return np.asarray(train_ids), np.asarray(test_ids)


def split_combined_dataset(
    df: pd.DataFrame,
    selection_frac: float,
    test_frac: float,
    random_state: int,
    label_col: str = "emotion",
    group_col: str = "scenario_id",
) -> dict[str, pd.DataFrame]:
    """
    Split the combined dataset at the scenario-group level into train / selection / test.

    This avoids prompt leakage where multiple prompt realizations of the same scenario
    could otherwise land in both train and validation/test.
    """
    if selection_frac < 0 or test_frac < 0 or selection_frac + test_frac >= 1.0:
        raise ValueError("selection_frac and test_frac must be non-negative and sum to < 1.")
    groups = _stable_group_frame(df, label_col=label_col, group_col=group_col)
    if len(groups) == 0:
        raise ValueError("Cannot split an empty dataset.")

    holdout_frac = selection_frac + test_frac
    if holdout_frac == 0:
        return {"train": df.copy(), "selection": df.iloc[0:0].copy(), "test": df.iloc[0:0].copy()}

    train_ids, holdout_ids = _safe_group_train_test_split(groups, holdout_frac, random_state)
    holdout = groups[groups["split_group_id"].isin(holdout_ids)].copy()

    if selection_frac == 0 or len(holdout) == 0:
        selection_ids = np.array([], dtype=object)
        test_ids = holdout_ids
    elif test_frac == 0:
        selection_ids = holdout_ids
        test_ids = np.array([], dtype=object)
    else:
        selection_share = selection_frac / holdout_frac
        selection_ids, test_ids = _safe_group_train_test_split(
            holdout,
            test_size=(1.0 - selection_share),
            random_state=random_state + 1,
        )

    group_map = _stable_group_frame(df, label_col=label_col, group_col=group_col).set_index("split_group_id")
    data = df.copy()
    if set(group_map.index) != set(groups["split_group_id"]):
        # Recompute the same split-group IDs on the full frame when ambiguous groups exist.
        split_groups = _stable_group_frame(df, label_col=label_col, group_col=group_col)
        lookup = split_groups.set_index(group_col if group_col in split_groups.columns else "split_group_id")
        if "split_group_id" in split_groups.columns and group_col in df.columns:
            ambiguous_groups = set(groups.loc[groups["split_group_id"].str.contains("::", regex=False), "split_group_id"])
            data["split_group_id"] = data.apply(
                lambda r: f"{r[group_col]}::{r[label_col]}" if f"{r[group_col]}::{r[label_col]}" in ambiguous_groups else r[group_col],
                axis=1,
            )
        else:
            data["split_group_id"] = data[group_col]
    else:
        ambiguous_ids = set(groups.loc[groups["split_group_id"].str.contains("::", regex=False), "split_group_id"])
        data["split_group_id"] = data.apply(
            lambda r: f"{r[group_col]}::{r[label_col]}" if f"{r[group_col]}::{r[label_col]}" in ambiguous_ids else r[group_col],
            axis=1,
        )

    train_df = data[data["split_group_id"].isin(set(train_ids))].drop(columns=["split_group_id"]).copy()
    selection_df = data[data["split_group_id"].isin(set(selection_ids))].drop(columns=["split_group_id"]).copy()
    test_df = data[data["split_group_id"].isin(set(test_ids))].drop(columns=["split_group_id"]).copy()
    return {"train": train_df, "selection": selection_df, "test": test_df}


def make_split_manifest(
    model_id: str,
    dataset_path: Path,
    selection_frac: float,
    test_frac: float,
    random_state: int,
    text_policy: str,
    split_name: str,
    n_rows: int,
) -> dict:
    """Create a manifest payload describing a canonical research split."""
    return {
        "version": 1,
        "model_id": model_id,
        "dataset": dataset_fingerprint(dataset_path),
        "split_protocol": {
            "selection_frac": selection_frac,
            "test_frac": test_frac,
            "random_state": random_state,
            "group_key": "scenario_id",
            "text_policy": text_policy,
        },
        "split_name": split_name,
        "n_rows": int(n_rows),
    }


def emotion_probe_direction_raw(rec, hidden_size: int | None = None) -> np.ndarray | None:
    """Convert a binary emotion probe record into a direction in raw hidden-state space."""
    if rec is None or "error" in rec:
        return None
    w = rec.get("weights")
    if w is None:
        clf = rec.get("classifier")
        if clf is None or not hasattr(clf, "coef_"):
            return None
        w = clf.coef_.ravel()
    vec = np.asarray(w, dtype=np.float64).reshape(-1)
    scaler = rec.get("scaler")
    if scaler is not None and hasattr(scaler, "scale_"):
        scale = np.asarray(scaler.scale_, dtype=np.float64).reshape(-1)
        if scale.shape == vec.shape:
            vec = vec / np.where(np.abs(scale) < 1e-12, 1.0, scale)
    if hidden_size is not None and vec.shape[0] != hidden_size:
        return None
    return vec


def appraisal_probe_direction_raw(rec, hidden_size: int | None = None) -> np.ndarray | None:
    """Convert an appraisal probe record into a direction in raw hidden-state space."""
    if rec is None:
        return None
    if "weights" in rec and rec.get("weights") is not None:
        vec = np.asarray(rec["weights"], dtype=np.float64).reshape(-1)
    else:
        ridge = rec.get("ridge")
        if ridge is None or not hasattr(ridge, "coef_"):
            return None
        vec = np.asarray(ridge.coef_, dtype=np.float64).reshape(-1)
    scaler = rec.get("scaler")
    if scaler is not None and hasattr(scaler, "scale_"):
        scale = np.asarray(scaler.scale_, dtype=np.float64).reshape(-1)
        if scale.shape == vec.shape:
            vec = vec / np.where(np.abs(scale) < 1e-12, 1.0, scale)
    if hidden_size is not None and vec.shape[0] != hidden_size:
        return None
    return vec


def weighted_mean(df: pd.DataFrame, value_col: str, weight_col: str) -> float | None:
    if df.empty or value_col not in df.columns or weight_col not in df.columns:
        return None
    w = df[weight_col].astype(float).to_numpy()
    v = df[value_col].astype(float).to_numpy()
    total = float(w.sum())
    if total <= 0:
        return None
    return float(np.sum(v * w) / total)


def canonical_pair(a: str, b: str) -> tuple[str, str]:
    """Return a symmetric canonical pair tuple."""
    a = str(a)
    b = str(b)
    return (a, b) if a <= b else (b, a)


def build_pair_category_lookup(contrastive_pairs: list[tuple[str, str]], similar_pairs: list[tuple[str, str]]) -> dict[tuple[str, str], str]:
    """Build a symmetric pair-category lookup."""
    lookup: dict[tuple[str, str], str] = {}
    for a, b in contrastive_pairs:
        lookup[canonical_pair(a, b)] = "contrastive"
    for a, b in similar_pairs:
        lookup[canonical_pair(a, b)] = "similar"
    return lookup


def pair_category(a: str, b: str, lookup: dict[tuple[str, str], str]) -> str:
    """Return pair category using a symmetric lookup, defaulting to 'other'."""
    return lookup.get(canonical_pair(a, b), "other")


def wrap_hover_text(text: str, width: int = 60) -> str:
    """Wrap long hover text for HTML plots using <br> breaks."""
    text = str(text)
    parts = []
    while len(text) > width:
        split_at = text.rfind(" ", 0, width)
        if split_at <= 0:
            split_at = width
        parts.append(text[:split_at].strip())
        text = text[split_at:].strip()
    if text:
        parts.append(text)
    return "<br>".join(parts)

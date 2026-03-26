"""
Probe training robustness: compare dataset variants (prompted only, prompted+unprompted,
2–3 prompt variations + no prompt) with fair OVA splits and optional probe performance comparison.

Uses BinaryOvaDatasetProcessor with even_negative_distribution=True for fair train/val splits.
Logs each step and dataset stats (per emotion, split sizes, negative distribution).
"""
from pathlib import Path
import sys
import pandas as pd
import numpy as np

from pipeline.core.config import (
    REPO_ROOT,
    COMBINED_CSV,
    DATA_DIR,
    EXTRACTION_TOKENS,
    PROMPT_INDICES_FOR_VARIANTS,
    SELECTION_SPLIT,
    FINAL_TEST_SPLIT,
    BALANCE_RATIO,
    RANDOM_STATE,
    MIN_SUPPORTED_EMOTION_TRAIN_COUNT,
    MIN_SUPPORTED_EMOTION_SELECTION_COUNT,
    PROBE_C_GRID,
    get_probe_paths,
    get_probe_robustness_dir,
    DEFAULT_MODEL_ID,
)
from pipeline.core.model_config import get_extraction_layers, get_extraction_locs, get_extraction_batch_size, get_default_probe_n_jobs
from pipeline.core.research_contracts import canonicalize_combined_dataset, split_combined_dataset, write_json, dataset_fingerprint, supported_emotion_stats
def _ensure_repo():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def _get_emotions_list(df: pd.DataFrame) -> list:
    return sorted(df["emotion"].dropna().astype(str).str.strip().str.lower().unique().tolist())


def _choose_best_variant(comparison_df: pd.DataFrame) -> dict | None:
    """Pick the default training variant by selection ROC-AUC, then accuracy."""
    if comparison_df is None or comparison_df.empty:
        return None
    df = comparison_df.copy()
    df["_roc_sort"] = df["mean_selection_roc_auc"].fillna(-1.0)
    df["_acc_sort"] = df["mean_selection_accuracy"].fillna(-1.0)
    df = df.sort_values(["_roc_sort", "_acc_sort"], ascending=False)
    best = df.iloc[0]
    return {
        "variant": str(best["variant"]),
        "mean_selection_roc_auc": float(best["mean_selection_roc_auc"]) if pd.notna(best["mean_selection_roc_auc"]) else None,
        "mean_selection_accuracy": float(best["mean_selection_accuracy"]) if pd.notna(best["mean_selection_accuracy"]) else None,
    }


def _build_data_variant(
    name: str,
    df_raw: pd.DataFrame,
    prompt_indices: list[int],
    include_unprompted: bool,
    prompt_fn_registry: dict,
    logger,
) -> pd.DataFrame:
    """
    Build a single dataset variant with one text column 'text_for_probe' and 'emotion'.
    - prompt_indices: which prompt_index to use (e.g. [1], [1,4], [1,4,7])
    - include_unprompted: if True, add one row per scenario with raw situation.
    """
    rows = []
    for _, row in df_raw.iterrows():
        situation = row["situation"]
        emotion = row["emotion"]
        scenario_id = row.get("scenario_id")
        for prompt_index in prompt_indices:
            fn = prompt_fn_registry.get(prompt_index)
            if fn is None:
                continue
            text = fn(situation)
            rows.append({
                "emotion": emotion,
                "text_for_probe": text,
                "scenario_id": scenario_id,
                "source_text_type": f"prompt_{prompt_index}",
            })
        if include_unprompted:
            rows.append({
                "emotion": emotion,
                "text_for_probe": situation,
                "scenario_id": scenario_id,
                "source_text_type": "unprompted",
            })
    out = pd.DataFrame(rows)
    if logger:
        logger.info(
            f"  [Dataset variant '{name}'] rows={len(out)}, "
            f"prompts={prompt_indices}, include_unprompted={include_unprompted} "
            f"-> {len(out) / max(1, len(df_raw)):.1f} rows per scenario"
        )
    return out


def _run_processor_and_log(
    variant_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: Path,
    emotions_list: list[str],
    logger,
):
    """
    Run BinaryOvaDatasetProcessor with even_negative_distribution=True.
    Returns (train_summary_df, val_summary_df) and saves per-emotion stats.
    """
    _ensure_repo()
    from experiments.utils.data_utils import BinaryOvaDatasetProcessor

    # Use text_for_probe as the column name for OVA datasets (saved as-is; trainer will use it)
    train_df = train_df.rename(columns={"text_for_probe": "hidden_emo_text"})
    val_df = val_df.rename(columns={"text_for_probe": "hidden_emo_text"})
    processor = BinaryOvaDatasetProcessor(
        output_dir=str(output_dir),
        emotions_list=emotions_list,
        logger=logger,
    )
    result = processor.process_pre_split_datasets(
        train_data=train_df,
        val_data=val_df,
        emotion_column="emotion",
        text_column="hidden_emo_text",
        filter_valid=True,
        balance_ratio=BALANCE_RATIO,
        random_state=RANDOM_STATE,
        even_negative_distribution=True,
    )
    train_summary = result.get("train_summary")
    val_summary = result.get("val_summary")
    if train_summary is not None and not train_summary.empty:
        train_summary.to_csv(output_dir / "train_summary.csv", index=False)
    if val_summary is not None and not val_summary.empty:
        val_summary.to_csv(output_dir / "val_summary.csv", index=False)
    return (
        pd.DataFrame(train_summary) if train_summary is not None and len(train_summary) else pd.DataFrame(),
        pd.DataFrame(val_summary) if val_summary is not None and len(val_summary) else pd.DataFrame(),
    )


def _best_layer_loc_from_probe_summary(model_id: str) -> tuple[int, int]:
    """Return (best_layer, best_loc) by mean test_roc_auc from probe_summary for this model."""
    paths = get_probe_paths(model_id)
    if not paths.probe_summary_csv.exists():
        layers = get_extraction_layers(model_id)
        locs = get_extraction_locs(model_id)
        return layers[len(layers) // 2], locs[0]
    df = pd.read_csv(paths.probe_summary_csv)
    by = df.groupby(["layer", "loc"])["test_roc_auc"].mean().reset_index().sort_values("test_roc_auc", ascending=False)
    r = by.iloc[0]
    return int(r["layer"]), int(r["loc"])


def _run_quick_probe_comparison(
    variant_dirs: dict[str, Path],
    model_id: str,
    emotions_list: list[str],
    logger,
) -> pd.DataFrame:
    """
    Train OVA probes at best (layer, loc) only for each dataset variant;
    return DataFrame with variant_name, mean_selection_roc_auc, mean_selection_accuracy.
    """
    _ensure_repo()
    from utils import Log
    from experiments.utils.training_utils import OvaLogisticRegressionTrainer

    best_layer, best_loc = _best_layer_loc_from_probe_summary(model_id)
    if logger:
        logger.info(f"  Quick probe comparison: training at layer={best_layer}, loc={best_loc} only (best from probe_summary).")

    trainer = OvaLogisticRegressionTrainer(model_id, logger=logger or Log("probe_robustness").logger)
    trainer.load_model_and_tokenizer()

    extraction_layers = [best_layer]
    extraction_locs = [best_loc]
    extraction_batch_size = get_extraction_batch_size(model_id)
    probe_n_jobs = get_default_probe_n_jobs()

    rows = []
    for variant_name, datasets_dir in variant_dirs.items():
        if not datasets_dir.exists():
            if logger:
                logger.warning(f"  Skip variant '{variant_name}': dir not found {datasets_dir}")
            continue
        try:
            out_dir = datasets_dir.parent / f"probes_{variant_name.replace(' ', '_')}"
            out_dir.mkdir(parents=True, exist_ok=True)
            result = trainer.train(
                datasets_dir=str(datasets_dir),
                emotions_list=emotions_list,
                output_dir=str(out_dir),
                extraction_layers=extraction_layers,
                extraction_locs=extraction_locs,
                extraction_tokens=EXTRACTION_TOKENS,
                text_column="hidden_emo_text",
                save_hidden_states=False,
                save_by_location=False,
                C_grid=PROBE_C_GRID,
                batch_size=extraction_batch_size,
                n_jobs_probes=probe_n_jobs,
            )
            summary = result.get("summary_df")
            if summary is not None and not summary.empty:
                mean_roc = float(summary["test_roc_auc"].mean())
                mean_acc = float(summary["test_accuracy"].mean())
                rows.append({
                    "variant": variant_name,
                    "mean_selection_roc_auc": mean_roc,
                    "mean_selection_accuracy": mean_acc,
                    "n_emotions": len(summary),
                })
            else:
                rows.append({"variant": variant_name, "mean_selection_roc_auc": np.nan, "mean_selection_accuracy": np.nan, "n_emotions": 0})
        except Exception as e:
            if logger:
                logger.warning(f"  Variant '{variant_name}' probe training failed: {e}")
            rows.append({"variant": variant_name, "mean_selection_roc_auc": np.nan, "mean_selection_accuracy": np.nan, "n_emotions": 0})
    return pd.DataFrame(rows)


def run_probe_training_robustness(
    model_id: str = DEFAULT_MODEL_ID,
    run_quick_probes: bool = True,
    logger=None,
) -> dict:
    """
    Build dataset variants (prompted only; prompted+unprompted; 2 and 3 prompt variations + unprompted),
    run BinaryOvaDatasetProcessor with fair split and even negative distribution, log everything,
    optionally run quick probe comparison at best layer/loc and save comparison.
    """
    _ensure_repo()
    from utils import Log
    from prompt_manager import build_prompt

    if logger is None:
        logger = Log("probe_robustness").logger

    out_root = get_probe_robustness_dir(model_id)
    out_root.mkdir(parents=True, exist_ok=True)

    # ----- Load raw data -----
    logger.info("=" * 60)
    logger.info("STEP 1: Load combined data")
    logger.info("=" * 60)
    if not COMBINED_CSV.exists():
        raise FileNotFoundError(f"Combined CSV not found: {COMBINED_CSV}")
    df_raw = canonicalize_combined_dataset(pd.read_csv(COMBINED_CSV))
    df_raw = df_raw.dropna(subset=["situation", "emotion"])
    support_info = supported_emotion_stats(
        df_raw,
        selection_frac=SELECTION_SPLIT,
        test_frac=FINAL_TEST_SPLIT,
        random_state=RANDOM_STATE,
        min_train_count=MIN_SUPPORTED_EMOTION_TRAIN_COUNT,
        min_selection_count=MIN_SUPPORTED_EMOTION_SELECTION_COUNT,
    )
    emotions_list = support_info["supported_emotions"]
    split_bundle = support_info["split_bundle"]
    raw_train = split_bundle["train"]
    raw_selection = split_bundle["selection"]
    raw_test = split_bundle["test"]
    logger.info(f"  Loaded {len(df_raw)} rows, emotions: {emotions_list}")
    logger.info(
        f"  Scenario-level split -> train={len(raw_train)}, selection={len(raw_selection)}, "
        f"final_test={len(raw_test)} rows"
    )
    logger.info(
        "  Supported emotions for main benchmark: "
        f"{emotions_list} (min_train={MIN_SUPPORTED_EMOTION_TRAIN_COUNT}, "
        f"min_selection={MIN_SUPPORTED_EMOTION_SELECTION_COUNT})"
    )
    logger.info("")

    # ----- Build prompt functions -----
    logger.info("STEP 2: Build prompt functions (prompt_manager.build_prompt)")
    logger.info("-" * 60)
    SHOTS = ("fear", "anger", "sadness")
    prompt_fn_registry = {}
    for idx in set([1] + PROMPT_INDICES_FOR_VARIANTS):
        prompt_fn_registry[idx] = build_prompt(shots=SHOTS, prompt_index=idx, emotions_list=emotions_list)
    logger.info(f"  Prompt indices registered: {sorted(prompt_fn_registry.keys())}")
    logger.info("")

    # ----- Define variants -----
    variants = [
        ("prompted_only", [1], False),
        ("prompted_plus_unprompted", [1], True),
        ("two_prompts_plus_unprompted", PROMPT_INDICES_FOR_VARIANTS[:2], True),
        ("three_prompts_plus_unprompted", PROMPT_INDICES_FOR_VARIANTS[:3], True),
    ]

    logger.info("STEP 3: Build dataset variants and run BinaryOvaDatasetProcessor (fair split, even negative distribution)")
    logger.info("-" * 60)
    variant_dirs = {}
    all_train = []
    all_val = []
    for variant_name, prompt_indices, include_unprompted in variants:
        logger.info(f"  Variant: {variant_name}")
        train_var = _build_data_variant(
            variant_name,
            raw_train,
            prompt_indices,
            include_unprompted,
            prompt_fn_registry,
            logger,
        )
        val_var = _build_data_variant(
            variant_name,
            raw_selection,
            prompt_indices,
            include_unprompted,
            prompt_fn_registry,
            None,
        )
        variant_dir = out_root / variant_name.replace(" ", "_")
        variant_dir.mkdir(parents=True, exist_ok=True)
        train_var.to_csv(variant_dir / "data_expanded_train.csv", index=False)
        val_var.to_csv(variant_dir / "data_expanded_selection.csv", index=False)
        write_json(
            variant_dir / "split_manifest.json",
            {
                "version": 1,
                "split_protocol": "scenario_level_train_selection_plus_heldout_test",
                "selection_frac": SELECTION_SPLIT,
                "test_frac": FINAL_TEST_SPLIT,
                "random_state": RANDOM_STATE,
                "combined_csv": dataset_fingerprint(COMBINED_CSV),
                "n_raw_train": int(len(raw_train)),
                "n_raw_selection": int(len(raw_selection)),
                "n_raw_test": int(len(raw_test)),
                "n_expanded_train": int(len(train_var)),
                "n_expanded_selection": int(len(val_var)),
            },
        )
        train_summary, val_summary = _run_processor_and_log(
            variant_name,
            train_var,
            val_var,
            variant_dir,
            emotions_list,
            logger,
        )
        if not train_summary.empty:
            train_summary["variant"] = variant_name
            all_train.append(train_summary)
            # Log per-emotion split for this variant (readable table)
            logger.info(f"  [{variant_name}] Train split (n_positive, n_negative_sampled, n_total) per emotion:")
            for _, r in train_summary.iterrows():
                logger.info(f"      {r['emotion']}: pos={r['n_positive']}, neg_sampled={r['n_negative_sampled']}, total={r['n_total']}")
        if not val_summary.empty:
            val_summary["variant"] = variant_name
            all_val.append(val_summary)
            logger.info(f"  [{variant_name}] Val split (n_positive, n_negative_sampled, n_total) per emotion:")
            for _, r in val_summary.iterrows():
                logger.info(f"      {r['emotion']}: pos={r['n_positive']}, neg_sampled={r['n_negative_sampled']}, total={r['n_total']}")
        variant_dirs[variant_name] = variant_dir
        logger.info("")

    # ----- Aggregate and save dataset stats -----
    logger.info("STEP 4: Save dataset split statistics")
    logger.info("-" * 60)
    if all_train:
        train_agg = pd.concat(all_train, ignore_index=True)
        train_agg.to_csv(out_root / "dataset_stats_train_per_emotion_per_variant.csv", index=False)
        logger.info(f"  Saved dataset_stats_train_per_emotion_per_variant.csv ({len(train_agg)} rows)")
    if all_val:
        val_agg = pd.concat(all_val, ignore_index=True)
        val_agg.to_csv(out_root / "dataset_stats_val_per_emotion_per_variant.csv", index=False)
        logger.info(f"  Saved dataset_stats_val_per_emotion_per_variant.csv ({len(val_agg)} rows)")
    support_info["counts_df"].to_csv(out_root / "supported_emotion_counts.csv", index=False)
    write_json(
        out_root / "supported_emotions.json",
        {
            "supported_emotions": emotions_list,
            "min_train_count": MIN_SUPPORTED_EMOTION_TRAIN_COUNT,
            "min_selection_count": MIN_SUPPORTED_EMOTION_SELECTION_COUNT,
        },
    )
    logger.info("")

    # ----- Quick probe comparison (optional) -----
    comparison_df = None
    if run_quick_probes and variant_dirs:
        logger.info("STEP 5: Quick probe comparison (best layer/loc only per variant)")
        logger.info("-" * 60)
        comparison_df = _run_quick_probe_comparison(variant_dirs, model_id, emotions_list, logger)
        comparison_df.to_csv(out_root / "probe_robustness_comparison.csv", index=False)
        logger.info(f"  Saved probe_robustness_comparison.csv")
        for _, r in comparison_df.iterrows():
            logger.info(
                f"    {r['variant']}: mean_selection_roc_auc={r['mean_selection_roc_auc']:.4f}, "
                f"mean_selection_accuracy={r['mean_selection_accuracy']:.4f}"
            )
        logger.info("")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            x = np.arange(len(comparison_df))
            ax.bar(x - 0.2, comparison_df["mean_selection_roc_auc"], width=0.35, label="Mean selection ROC-AUC")
            ax.bar(x + 0.2, comparison_df["mean_selection_accuracy"], width=0.35, label="Mean selection accuracy")
            ax.set_xticks(x)
            ax.set_xticklabels(comparison_df["variant"], rotation=15, ha="right")
            ax.set_ylabel("Score")
            ax.set_title("Probe robustness: dataset variant comparison (best layer/loc only)")
            ax.legend()
            ax.set_ylim(0, 1.05)
            fig.tight_layout()
            fig.savefig(out_root / "probe_robustness_comparison.png", bbox_inches="tight")
            fig.savefig(out_root / "probe_robustness_comparison.pdf", bbox_inches="tight")
            plt.close(fig)
            logger.info(f"  Saved probe_robustness_comparison.png / .pdf")
        except Exception as e:
            logger.warning(f"  Could not save comparison figure: {e}")
        best_variant = _choose_best_variant(comparison_df)
        if best_variant is not None:
            write_json(out_root / "best_variant_selection.json", best_variant)
            logger.info(
                "  Selected default training variant: "
                f"{best_variant['variant']} (selection ROC-AUC={best_variant['mean_selection_roc_auc']}, "
                f"selection accuracy={best_variant['mean_selection_accuracy']})"
            )
    else:
        logger.info("STEP 5: Skipped (run_quick_probes=False or no variant dirs)")

    # ----- Summary for human reader -----
    summary_md = [
        "# Probe training robustness",
        "",
        "## Dataset variants",
        "",
        "Each variant uses **BinaryOvaDatasetProcessor** with leakage-safe scenario-level train/selection splits:",
        "- `even_negative_distribution=True` (fair negative examples across emotions)",
        f"- `selection_split={SELECTION_SPLIT}`, `final_test_split={FINAL_TEST_SPLIT}`, `balance_ratio={BALANCE_RATIO}`",
        "",
        "| Variant | Description |",
        "|---------|-------------|",
        "| prompted_only | One row per scenario: prompt_index=1 only |",
        "| prompted_plus_unprompted | Two rows per scenario: prompted + raw situation |",
        "| two_prompts_plus_unprompted | Three rows per scenario: 2 prompt indices + raw |",
        "| three_prompts_plus_unprompted | Four rows per scenario: 3 prompt indices + raw |",
        "",
        "## Outputs",
        "",
        "- `dataset_stats_train_per_emotion_per_variant.csv` — per emotion, per variant: n_positive, n_negative_sampled, n_total",
        "- `dataset_stats_val_per_emotion_per_variant.csv` — same for validation",
        "- `<variant>/train_summary.csv`, `<variant>/val_summary.csv` — per-emotion counts for that variant",
        "- `probe_robustness_comparison.csv` — mean selection ROC-AUC and accuracy when training at best layer/loc only (if run_quick_probes=True)",
        "",
    ]
    (out_root / "summary.md").write_text("\n".join(summary_md), encoding="utf-8")
    logger.info(f"  Wrote {out_root / 'summary.md'}")

    return {
        "out_root": out_root,
        "variant_dirs": variant_dirs,
        "comparison_df": comparison_df,
        "emotions_list": emotions_list,
    }


if __name__ == "__main__":
    import argparse
    _ensure_repo()
    from utils import Log
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--no_quick_probes", action="store_true", help="Skip probe training comparison")
    args = p.parse_args()
    run_probe_training_robustness(
        model_id=args.model_id,
        run_quick_probes=not args.no_quick_probes,
        logger=Log("probe_robustness").logger,
    )

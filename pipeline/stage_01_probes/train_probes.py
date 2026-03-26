"""
Train binary OVA emotion probes across all layers and locs for comprehensive circuit analysis.

Uses EXTRACTION_LAYERS and EXTRACTION_LOCS from config (full grid: 16 layers × 3 locs).
Reuses datasets from probe_robustness if available; otherwise builds from COMBINED_CSV.
Outputs to outputs/<model_id>/01_probes/binary_ova_probes/ (probe_summary.csv + .pt).
"""
from pathlib import Path
import sys
import pandas as pd

from pipeline.core.config import (
    REPO_ROOT,
    COMBINED_CSV,
    EXTRACTION_TOKENS,
    SELECTION_SPLIT,
    FINAL_TEST_SPLIT,
    BALANCE_RATIO,
    RANDOM_STATE,
    MIN_SUPPORTED_EMOTION_TRAIN_COUNT,
    MIN_SUPPORTED_EMOTION_SELECTION_COUNT,
    PROBE_C_GRID,
    get_probes_dir,
    get_probe_robustness_dir,
    DEFAULT_MODEL_ID,
)
from pipeline.core.model_config import (
    get_extraction_layers,
    get_extraction_locs,
    get_extraction_batch_size,
    get_default_probe_n_jobs,
)
from pipeline.core.research_contracts import (
    canonicalize_combined_dataset,
    split_combined_dataset,
    read_json_if_exists,
    write_json,
    dataset_fingerprint,
    supported_emotion_stats,
)


def _ensure_repo():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def _get_emotions_list(df: pd.DataFrame) -> list:
    return sorted(df["emotion"].dropna().astype(str).str.strip().str.lower().unique().tolist())


def _resolve_default_variant_dir(model_id: str, logger) -> Path:
    robustness_dir = get_probe_robustness_dir(model_id)
    selection = read_json_if_exists(robustness_dir / "best_variant_selection.json")
    if selection is not None:
        candidate = robustness_dir / str(selection.get("variant", "")).replace(" ", "_")
        if candidate.exists() and (candidate / "anger.csv").exists():
            logger.info(f"Using best variant from probe_robustness: {candidate}")
            return candidate
    return robustness_dir / "prompted_plus_unprompted"


def run_train_probes(
    model_id: str = DEFAULT_MODEL_ID,
    datasets_dir: Path | str | None = None,
    probe_n_jobs: int | None = None,
    logger=None,
    output_dir: Path | str | None = None,
    extraction_locs_override: list | None = None,
    extraction_tokens_override: list | None = None,
    emotions_filter: list[str] | None = None,
) -> dict:
    """
    Train binary OVA probes across all layers and locs.
    Saves probe_summary.csv and binary_ova_probes_*.pt under `output_dir` (default:
    outputs/<model_id>/01_probes/binary_ova_probes/).

    Args:
        model_id: Model identifier.
        datasets_dir: Directory with OVA datasets (emotion.csv, emotion_val.csv per emotion).
            If None, uses probe_robustness/prompted_plus_unprompted if it exists,
            otherwise builds from COMBINED_CSV.
        logger: Optional logger.
        output_dir: If set, save probes here (e.g. grid ablation under `01_probes_grid_ablation/binary_ova_probes`).
            Default: `outputs/<model_id>/01_probes/binary_ova_probes`.
        extraction_locs_override / extraction_tokens_override: Override main-pipeline extraction grid (e.g. ablation).
        emotions_filter: If set, only train these emotions (names matched case-insensitively).

    Returns:
        Dict with 'results', 'summary_df', 'output_dir'.
    """
    _ensure_repo()
    from utils import Log
    from experiments.utils.training_utils import OvaLogisticRegressionTrainer

    if logger is None:
        logger = Log("train_probes").logger

    out_dir = Path(output_dir) if output_dir is not None else (get_probes_dir(model_id) / "binary_ova_probes")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve datasets_dir
    if datasets_dir is None:
        candidate = _resolve_default_variant_dir(model_id, logger)
        manifest = read_json_if_exists(candidate / "split_manifest.json")
        if (
            candidate.exists()
            and (candidate / "anger.csv").exists()
            and manifest is not None
            and manifest.get("split_protocol") == "scenario_level_train_selection_plus_heldout_test"
            and manifest.get("combined_csv", {}).get("sha256") == dataset_fingerprint(COMBINED_CSV)["sha256"]
        ):
            datasets_dir = str(candidate)
            logger.info(f"Using existing datasets from probe_robustness: {datasets_dir}")
        else:
            # Build datasets from COMBINED_CSV
            from pipeline.stage_01_probes.probe_training_robustness import _build_data_variant, _run_processor_and_log
            from prompt_manager import build_prompt

            if not COMBINED_CSV.exists():
                raise FileNotFoundError(f"COMBINED_CSV not found: {COMBINED_CSV}. Run probe_robustness first or provide datasets_dir.")
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
            SHOTS = ("fear", "anger", "sadness")
            prompt_fn_registry = {1: build_prompt(shots=SHOTS, prompt_index=1, emotions_list=emotions_list)}
            train_var = _build_data_variant(
                "prompted_plus_unprompted",
                split_bundle["train"],
                [1],
                True,
                prompt_fn_registry,
                logger,
            )
            val_var = _build_data_variant(
                "prompted_plus_unprompted",
                split_bundle["selection"],
                [1],
                True,
                prompt_fn_registry,
                None,
            )
            datasets_dir = robustness_dir / "prompted_plus_unprompted"
            datasets_dir.mkdir(parents=True, exist_ok=True)
            train_var.to_csv(datasets_dir / "data_expanded_train.csv", index=False)
            val_var.to_csv(datasets_dir / "data_expanded_selection.csv", index=False)
            _run_processor_and_log(
                "prompted_plus_unprompted",
                train_var,
                val_var,
                datasets_dir,
                emotions_list,
                logger,
            )
            write_json(
                datasets_dir / "split_manifest.json",
                {
                    "version": 1,
                    "split_protocol": "scenario_level_train_selection_plus_heldout_test",
                    "selection_frac": SELECTION_SPLIT,
                    "test_frac": FINAL_TEST_SPLIT,
                    "random_state": RANDOM_STATE,
                    "combined_csv": dataset_fingerprint(COMBINED_CSV),
                    "n_raw_train": int(len(split_bundle["train"])),
                    "n_raw_selection": int(len(split_bundle["selection"])),
                    "n_raw_test": int(len(split_bundle["test"])),
                },
            )
            write_json(
                get_probe_robustness_dir(model_id) / "best_variant_selection.json",
                {"variant": "prompted_plus_unprompted", "mean_selection_roc_auc": None, "mean_selection_accuracy": None},
            )
            datasets_dir = str(datasets_dir)
            logger.info(f"Built datasets from COMBINED_CSV: {datasets_dir}")
    else:
        datasets_dir = str(Path(datasets_dir))

    # Load emotions from datasets
    processor_output = Path(datasets_dir)
    train_summary_path = processor_output / "train_summary.csv"
    if train_summary_path.exists():
        train_summary = pd.read_csv(train_summary_path)
        emotions_list = train_summary["emotion"].astype(str).tolist()
    else:
        raise FileNotFoundError(f"No train_summary.csv in {datasets_dir}. Run probe_robustness first or provide valid datasets_dir.")

    if emotions_filter is not None:
        filt = {str(e).strip().lower() for e in emotions_filter}
        emotions_list = [e for e in emotions_list if str(e).strip().lower() in filt]
        if not emotions_list:
            raise ValueError("emotions_filter removed all emotions; check names against train_summary.csv.")

    extraction_layers = get_extraction_layers(model_id)
    extraction_locs = extraction_locs_override if extraction_locs_override is not None else get_extraction_locs(model_id)
    extraction_tokens_eff = extraction_tokens_override if extraction_tokens_override is not None else EXTRACTION_TOKENS
    extraction_batch_size = get_extraction_batch_size(model_id)
    probe_n_jobs = get_default_probe_n_jobs() if probe_n_jobs is None else int(probe_n_jobs)
    logger.info("=" * 60)
    logger.info("Training binary OVA probes across ALL layers and locs")
    logger.info("=" * 60)
    logger.info(f"  Layers: {extraction_layers} ({len(extraction_layers)} total)")
    logger.info(f"  Locs: {extraction_locs} ({len(extraction_locs)} total)")
    logger.info(f"  Tokens: {extraction_tokens_eff}")
    logger.info(f"  Emotions: {len(emotions_list)}")
    logger.info(f"  Hidden-state extraction batch size: {extraction_batch_size}")
    logger.info(f"  Probe CPU parallel jobs: {probe_n_jobs}")
    logger.info(f"  Output: {out_dir}")
    logger.info("")

    trainer = OvaLogisticRegressionTrainer(model_id, logger=logger)
    result = trainer.train(
        datasets_dir=datasets_dir,
        emotions_list=emotions_list,
        output_dir=str(out_dir),
        extraction_layers=extraction_layers,
        extraction_locs=extraction_locs,
        extraction_tokens=extraction_tokens_eff,
        C_grid=PROBE_C_GRID,
        text_column="hidden_emo_text",
        save_hidden_states=False,
        save_by_location=False,
        batch_size=extraction_batch_size,
        n_jobs_probes=probe_n_jobs,
    )

    summary_df = result.get("summary_df")
    if summary_df is not None and not summary_df.empty:
        logger.info(f"Saved probe_summary.csv ({len(summary_df)} rows)")
        logger.info(f"Saved binary_ova_probes_layers_*.pt")
        by_layer_loc = summary_df.groupby(["layer", "loc"])["test_roc_auc"].mean().reset_index()
        by_layer_loc = by_layer_loc.sort_values("test_roc_auc", ascending=False)
        best = by_layer_loc.iloc[0]
        logger.info(f"Best (layer, loc): ({int(best['layer'])}, {int(best['loc'])}) mean ROC-AUC={best['test_roc_auc']:.4f}")

    return result


if __name__ == "__main__":
    import argparse
    _ensure_repo()
    from utils import Log
    p = argparse.ArgumentParser(description="Train binary OVA probes across all layers and locs")
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--datasets_dir", type=Path, default=None, help="OVA datasets dir (default: probe_robustness/prompted_plus_unprompted)")
    p.add_argument("--probe_n_jobs", type=int, default=None, help="Parallel jobs for per-(layer,loc,token) probe fitting (default: auto from CPU count)")
    args = p.parse_args()
    run_train_probes(
        model_id=args.model_id,
        datasets_dir=args.datasets_dir,
        probe_n_jobs=args.probe_n_jobs,
        logger=Log("train_probes").logger,
    )

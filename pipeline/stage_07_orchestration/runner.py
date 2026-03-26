"""
Run the full pipeline for one or more models, with optional VRAM-based filtering.

Use --model_ids to run specific models, or leave unset and use --max_vram_gb (or
PIPELINE_MAX_VRAM_GB) to run only models that fit within the given VRAM.

With --skip_complete, models that have already finished (06_synthesis/SUMMARY.md) are skipped.
With --resume, for each model the runner starts from the first step whose output is missing.

LLM judge (GENERATION_BENCHMARK_RUN_JUDGE defaults to False):
  Run all models first, then score in one pass with a stronger judge model:
    python -m pipeline.run_behavior_judges --auto --judge_model_id Llama3.1_8B --re_run_synthesis
  Or let the pipeline trigger it automatically at the end:
    python -m pipeline.run_pipeline_for_models --model_ids ... --run_judges_after --judge_model_id Llama3.1_8B

Probe grid ablation (optional, long-running; writes outputs/<model_id>/01_probes_grid_ablation/):
    python -m pipeline.run_pipeline_for_models --model_ids ... --run_grid_ablation
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pipeline.core.config import REPO_ROOT, DEFAULT_MODEL_ID, get_model_output_dir, OUTPUTS_ROOT
from pipeline.core.model_config import get_models_within_vram, get_estimated_vram_gb
# Ordered pipeline steps: (name, module, run_fn, completion_path_relative_to_model_output)
# Completion path is used to detect "this step finished" for resume.
PIPELINE_STEPS = [
    ("probe_robustness", "pipeline.stage_01_probes.probe_training_robustness", "run_probe_training_robustness", "01_probe_robustness/dataset_stats_train_per_emotion_per_variant.csv"),
    ("train_probes", "pipeline.stage_01_probes.train_probes", "run_train_probes", "01_probes/binary_ova_probes/probe_summary.csv"),
    (
        "probe_grid_ablation",
        "pipeline.stage_01_probes.run_probe_grid_ablation",
        "run_probe_grid_ablation",
        "01_probes_grid_ablation/binary_ova_probes/probe_summary.csv",
    ),
    ("train_appraisal_probes", "pipeline.stage_01_probes.train_appraisal_probes", "run_train_appraisal_probes", "01_probes/appraisal_regression_probes.pt"),
    ("train_appraisal_binary_probes", "pipeline.stage_01_probes.train_appraisal_binary_probes", "run_train_appraisal_binary_probes", "01_probes/appraisal_binary_ova_probes/appraisal_binary_summary.csv"),
    # circuit_evidence first: val-based auto-k + circuit_top_k_selection.json; phase1 then maps pairs -> circuits.json
    ("circuit_evidence", "pipeline.stage_02_circuit.circuit_evidence", "run_classification_experiment", "02_circuit/circuit_evidence_classification.csv"),
    ("phase1_circuits", "pipeline.stage_02_circuit.phase1_circuits", "run_phase1", "02_circuit/circuits.json"),
    ("appraisal_structure", "pipeline.stage_03_appraisal_structure.appraisal_structure", "run_appraisal_structure", "03_appraisal_structure/appraisal_zscore_by_emotion.csv"),
    ("phase2_summary", "pipeline.stage_04_appraisal_in_circuit.phase2_summary", "write_phase2_summary", "04_appraisal_in_circuit/SUMMARY.md"),
    ("appraisal_theory", "pipeline.stage_05_appraisal_theory.appraisal_theory", "run_appraisal_theory", "04_appraisal_theory/SUMMARY.md"),
    ("steering_benchmark", "pipeline.stage_06_benchmarks.steering_benchmark", "run_steering_benchmark", "05_steering/steering_benchmark.csv"),
    ("generation_behavior_benchmark", "pipeline.stage_06_benchmarks.generation_behavior_benchmark", "run_generation_behavior_benchmark", "05_generation_behavior/summary.md"),
    ("mental_health_steering", "pipeline.stage_06_benchmarks.mental_health_steering_benchmark", "run_mental_health_steering_benchmark", "05_mental_health_steering/summary.md"),
    ("synthesis", "pipeline.stage_07_orchestration.synthesis", "run_synthesis", "06_synthesis/SUMMARY.md"),
]

SYNTHESIS_STEP_INDEX = 13

logger = logging.getLogger(__name__)


def _ensure_repo():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def _run_step(module_name: str, run_fn: str, model_id: str, **kwargs) -> None:
    """Import and run a pipeline step for the given model_id."""
    _ensure_repo()
    import importlib
    mod = importlib.import_module(module_name)
    fn = getattr(mod, run_fn)
    fn(model_id=model_id, **kwargs)


def _is_pipeline_complete(model_id: str) -> bool:
    """True if this model has finished the full pipeline (synthesis SUMMARY exists)."""
    out = get_model_output_dir(model_id)
    return (out / PIPELINE_STEPS[SYNTHESIS_STEP_INDEX][3]).exists()


def _existing_model_outputs() -> list[str]:
    """Return model output directories currently present in outputs/."""
    if not OUTPUTS_ROOT.exists():
        return []
    return sorted(
        p.name for p in OUTPUTS_ROOT.iterdir()
        if p.is_dir() and p.name != "synthesis"
    )


def _completed_model_ids() -> list[str]:
    """Subset of output dir names that have finished synthesis (06_synthesis/SUMMARY.md)."""
    return [mid for mid in _existing_model_outputs() if _is_pipeline_complete(mid)]


def _aggregate_synthesis_model_ids(aggregate_all_output_dirs: bool) -> list[str]:
    """Model IDs to pass to multi-model synthesis (completed only unless legacy flag)."""
    if aggregate_all_output_dirs:
        return _existing_model_outputs()
    completed = _completed_model_ids()
    skipped = len(_existing_model_outputs()) - len(completed)
    if skipped:
        logger.info(
            "Aggregate synthesis: skipping %d incomplete output dir(s); using %d completed model(s).",
            skipped,
            len(completed),
        )
    return completed


def _train_probes_complete(model_id: str) -> bool:
    """train_probes is complete only when both probe_summary.csv and a combined .pt exist."""
    out = get_model_output_dir(model_id)
    probes_dir = out / "01_probes" / "binary_ova_probes"
    if not (probes_dir / "probe_summary.csv").exists():
        return False
    # circuit_evidence needs the combined .pt; don't consider step done without it
    any_pt = next(probes_dir.glob("binary_ova_probes_*.pt"), None)
    return any_pt is not None


def _first_incomplete_step_index(
    model_id: str,
    skip_appraisal_probes: bool = False,
    run_grid_ablation: bool = False,
) -> int | None:
    """Return the index of the first step whose completion path is missing, or None if all complete."""
    out = get_model_output_dir(model_id)
    for i, (name, _, _, rel) in enumerate(PIPELINE_STEPS):
        if name == "probe_grid_ablation" and not run_grid_ablation:
            continue  # optional step; not part of resume when disabled
        if name == "train_appraisal_probes" and skip_appraisal_probes:
            continue  # treat as always complete when skipping
        if name == "train_probes":
            if not _train_probes_complete(model_id):
                return i
        elif not (out / rel).exists():
            return i
    return None


def _step_kwargs(
    step_index: int,
    skip_probe_robustness: bool,
    skip_phase2_summary: bool,
    circuit_skip_extract: bool,
    steering_behavioral: bool = False,
    max_gen_rows: int | None = None,
    max_mh_posts: int | None = None,
) -> dict:
    """Extra kwargs for _run_step for the given step index."""
    name = PIPELINE_STEPS[step_index][0]
    if name == "probe_robustness":
        return {"run_quick_probes": True}
    if name == "circuit_evidence":
        return {"skip_extract": circuit_skip_extract}
    if name == "steering_benchmark":
        return {"run_behavioral": steering_behavioral}
    if name == "generation_behavior_benchmark" and max_gen_rows is not None:
        return {"max_rows": max_gen_rows}
    if name == "mental_health_steering" and max_mh_posts is not None:
        return {"max_posts_per_condition": max_mh_posts}
    return {}


# Step index for steering_benchmark (used with --from_steering); synthesis is SYNTHESIS_STEP_INDEX
STEERING_STEP_INDEX = 10


def run_pipeline_for_models(
    model_ids: list[str] | None = None,
    max_vram_gb: float | None = None,
    skip_probe_robustness: bool = False,
    skip_phase2_summary: bool = False,
    skip_appraisal_probes: bool = False,
    skip_generation_behavior: bool = False,
    circuit_skip_extract: bool = False,
    aggregate_synthesis: bool = True,
    skip_complete: bool = True,
    resume: bool = True,
    overwrite: bool = False,
    from_steering: bool = False,
    steering_behavioral: bool = False,
    run_judges_after: bool = False,
    judge_model_id: str | None = None,
    max_gen_rows: int | None = None,
    max_mh_posts: int | None = None,
    run_grid_ablation: bool = False,
    aggregate_all_output_dirs: bool = False,
) -> list[str]:
    """
    Run the full pipeline for each model in model_ids (or models within max_vram_gb).

    - skip_complete: if True, skip models that already have 06_synthesis/SUMMARY.md.
    - resume: if True, for each model start from the first step whose output is missing.
    - overwrite: if True, run all models from step 0 (same as skip_complete=False and resume=False).
    - from_steering: if True, run only steering_benchmark and synthesis for each selected model
      (no skip_complete; use to refresh steering after code changes without re-running earlier steps).
    - steering_behavioral: if True, steering_benchmark also runs real forward pass with in-pass steering
      (slower; saves steering_benchmark_behavioral.csv and steering_curves_behavioral.csv).
    - run_judges_after: if True, run run_behavior_judges on all completed models after the main loop,
      then re-run synthesis to incorporate judge scores.
    - judge_model_id: model to use as judge (defaults to the last pipeline model_id if not set).
    - max_gen_rows: cap on generation-behavior benchmark prompts (stratified sample). None = full CSV.
    - max_mh_posts: cap on mental-health posts per condition. None = config default.
    - run_grid_ablation: if True, after canonical train_probes run probe loc-by-token grid ablation
      (outputs under 01_probes_grid_ablation/; slow). Ignored with --from_steering.
    - aggregate_all_output_dirs: if True, multi-model synthesis includes every outputs/* dir
      (legacy). Default False: only dirs with a complete pipeline (06_synthesis/SUMMARY.md).

    Returns the list of model_ids that were run.
    """
    if overwrite:
        skip_complete = False
        resume = False
    if model_ids is not None and len(model_ids) > 0:
        to_run = list(model_ids)
    else:
        to_run = get_models_within_vram(max_vram_gb)
        if not to_run:
            print("No models selected. Set --model_ids or --max_vram_gb (or PIPELINE_MAX_VRAM_GB).")
            return []
        print(f"Running pipeline for models within VRAM: {to_run}")

    if from_steering:
        print("Running from steering step only (steering_benchmark + synthesis) for each model.")
    elif skip_complete:
        original_count = len(to_run)
        to_run = [m for m in to_run if not _is_pipeline_complete(m)]
        skipped = original_count - len(to_run)
        if skipped:
            print(f"Skipping {skipped} model(s) that have already completed the pipeline.")

    ran = []
    for i, model_id in enumerate(to_run):
        print(f"\n{'='*60}\nPipeline for model: {model_id} ({i+1}/{len(to_run)})\n{'='*60}")
        try:
            if from_steering:
                start_index = STEERING_STEP_INDEX
                print(f"Running from step: {PIPELINE_STEPS[start_index][0]} (steering) through synthesis")
            else:
                start_index = 0
                if resume:
                    first_incomplete = _first_incomplete_step_index(
                        model_id,
                        skip_appraisal_probes=skip_appraisal_probes,
                        run_grid_ablation=run_grid_ablation,
                    )
                    if first_incomplete is None:
                        print(f"Skipping {model_id} (already complete).")
                        continue
                    start_index = first_incomplete
                    if start_index > 0:
                        print(f"Resuming from step: {PIPELINE_STEPS[start_index][0]}")

            for step_index in range(start_index, len(PIPELINE_STEPS)):
                name, module_name, run_fn, _ = PIPELINE_STEPS[step_index]
                if name == "probe_robustness" and skip_probe_robustness:
                    continue
                if name == "phase2_summary" and skip_phase2_summary:
                    continue
                if name == "train_appraisal_probes" and skip_appraisal_probes:
                    continue
                if name == "generation_behavior_benchmark" and skip_generation_behavior:
                    continue
                if name == "probe_grid_ablation" and not run_grid_ablation:
                    continue
                kwargs = _step_kwargs(step_index, skip_probe_robustness, skip_phase2_summary, circuit_skip_extract, steering_behavioral, max_gen_rows, max_mh_posts)
                _run_step(module_name, run_fn, model_id, **kwargs)
            ran.append(model_id)
        except Exception as e:
            print(f"Pipeline failed for {model_id}: {e}")
            raise

    if aggregate_synthesis and ran:
        aggregate_ids = _aggregate_synthesis_model_ids(aggregate_all_output_dirs)
        if not aggregate_ids:
            logger.warning(
                "Skipping aggregate synthesis: no %s model output dirs found.",
                "matching" if aggregate_all_output_dirs else "pipeline-complete",
            )
        else:
            print(f"\nAggregating synthesis for {len(aggregate_ids)} model(s)...")
            _ensure_repo()
            from pipeline.stage_07_orchestration.synthesis import run_synthesis
            run_synthesis(model_ids=aggregate_ids)

    if run_judges_after and ran:
        _ensure_repo()
        judge_mid = judge_model_id or ran[-1]
        judge_ids = (
            _existing_model_outputs()
            if aggregate_all_output_dirs
            else _completed_model_ids()
        )
        if not judge_ids:
            logger.warning("Skipping post-pipeline judges: no model IDs selected.")
        else:
            print(f"\n{'='*60}\nPost-pipeline judge pass ({judge_mid}) for {len(judge_ids)} model(s)\n{'='*60}")
            from pipeline.stage_06_benchmarks.run_behavior_judges import run_behavior_judges
            run_behavior_judges(
                model_ids=judge_ids,
                judge_model_id=judge_mid,
                re_run_synthesis=aggregate_synthesis,
            )

    return ran


def main():
    parser = argparse.ArgumentParser(
        description="Run full pipeline for one or more models (optionally filter by VRAM)."
    )
    parser.add_argument(
        "--model_ids", nargs="*", default=None,
        help="Model IDs to run (e.g. Llama3.2_1B Gemma2_2B). If unset, use --max_vram_gb or PIPELINE_MAX_VRAM_GB.",
    )
    parser.add_argument(
        "--max_vram_gb", type=float, default=None,
        help="Only run models with estimated_vram_gb <= this (ignored if --model_ids is set).",
    )
    parser.add_argument("--skip_probe_robustness", action="store_true", help="Skip probe_training_robustness step.")
    parser.add_argument("--skip_phase2_summary", action="store_true", help="Skip phase2_summary step.")
    parser.add_argument("--skip_appraisal_probes", action="store_true", help="Skip train_appraisal_probes (use existing or input_probes appraisal_regression_probes.pt).")
    parser.add_argument("--skip_generation_behavior", action="store_true", help="Skip generation_behavior_benchmark.")
    parser.add_argument("--circuit_skip_extract", action="store_true", help="Use cached val_hidden_states in circuit_evidence.")
    parser.add_argument("--no_aggregate_synthesis", action="store_true", help="Do not run synthesis across all models at the end.")
    parser.add_argument("--no_skip_complete", action="store_true", help="Do not skip models that already have 06_synthesis/SUMMARY.md.")
    parser.add_argument("--no_resume", action="store_true", help="For incomplete models, start from step 0 instead of resuming.")
    parser.add_argument("--overwrite", action="store_true", help="Run every model from step 0 (no skip complete, no resume). Use to redo everything.")
    parser.add_argument("--from_steering", action="store_true", help="Run only steering_benchmark and synthesis for each model (re-run steering without redoing earlier steps).")
    parser.add_argument("--steering_behavioral", action="store_true", help="Also run real forward pass with in-pass steering (slower; writes steering_benchmark_behavioral.csv).")
    parser.add_argument(
        "--run_judges_after",
        action="store_true",
        help="After all models finish, run LLM judge on generation and MH outputs, then re-run synthesis.",
    )
    parser.add_argument(
        "--judge_model_id",
        default=None,
        help="Model to use as judge (defaults to the last model run). Used with --run_judges_after.",
    )
    parser.add_argument(
        "--max_gen_rows", type=int, default=None,
        help="Cap generation-behavior benchmark to N prompts (stratified sample). Default: full CSV.",
    )
    parser.add_argument(
        "--max_mh_posts", type=int, default=None,
        help="Cap mental-health benchmark to N posts per condition. Default: config value (15).",
    )
    parser.add_argument(
        "--run_grid_ablation",
        action="store_true",
        help=(
            "After canonical train_probes, run probe grid ablation (wider loc-by-token grid) into "
            "01_probes_grid_ablation/. Slow; does not replace 01_probes/. Ignored with --from_steering."
        ),
    )
    parser.add_argument(
        "--aggregate_all_output_dirs",
        action="store_true",
        help=(
            "When aggregating synthesis (and optional judges), include every outputs/* folder "
            "instead of only pipeline-complete models (06_synthesis/SUMMARY.md per model)."
        ),
    )
    args = parser.parse_args()

    from pipeline.core.logutil import configure_logging
    from pipeline.core.runtime_env import apply_cpu_thread_limits
    configure_logging()
    apply_cpu_thread_limits()

    run_pipeline_for_models(
        model_ids=args.model_ids,
        max_vram_gb=args.max_vram_gb,
        skip_probe_robustness=args.skip_probe_robustness,
        skip_phase2_summary=args.skip_phase2_summary,
        skip_appraisal_probes=args.skip_appraisal_probes,
        skip_generation_behavior=args.skip_generation_behavior,
        circuit_skip_extract=args.circuit_skip_extract,
        aggregate_synthesis=not args.no_aggregate_synthesis,
        skip_complete=not args.no_skip_complete,
        resume=not args.no_resume,
        overwrite=args.overwrite,
        from_steering=args.from_steering,
        steering_behavioral=args.steering_behavioral,
        run_judges_after=args.run_judges_after,
        judge_model_id=args.judge_model_id,
        max_gen_rows=args.max_gen_rows,
        max_mh_posts=args.max_mh_posts,
        run_grid_ablation=args.run_grid_ablation and not args.from_steering,
        aggregate_all_output_dirs=args.aggregate_all_output_dirs,
    )


if __name__ == "__main__":
    main()

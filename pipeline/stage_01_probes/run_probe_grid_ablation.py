"""
Probe-only grid ablation (scope A): train emotion OVA + appraisal Ridge probes on a wider
(layer presets unchanged), loc × token grid. Writes under outputs/<model_id>/01_probes_grid_ablation/
only — canonical 01_probes/ is untouched.

Does not run circuit_evidence, phase1_circuits, or steering.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PIPELINE_ROOT.parent


def _ensure_paths():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _parse_locs(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_tokens(s: str) -> list:
    out = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        if p.lower() == "mid":
            out.append("mid")
        else:
            out.append(int(p))
    return out


def _parse_emotions(s: str | None) -> list[str] | None:
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def _tokens_json_safe(tokens: list) -> list:
    return [t if isinstance(t, str) else int(t) for t in tokens]


def run_probe_grid_ablation(
    model_id: str,
    datasets_dir: Path | str | None = None,
    locs: list[int] | None = None,
    tokens: list | None = None,
    skip_appraisal: bool = False,
    probe_n_jobs: int | None = None,
    max_samples: int | None = None,
    emotions_filter: list[str] | None = None,
) -> Path:
    _ensure_paths()
    from pipeline.config import (
        PROBE_GRID_ABLATION_LOCS,
        PROBE_GRID_ABLATION_TOKENS,
        get_probe_grid_ablation_dir,
        REPO_ROOT as CFG_ROOT,
    )
    from pipeline.research_contracts import write_json
    from pipeline.train_appraisal_probes import run_train_appraisal_probes
    from pipeline.train_probes import run_train_probes
    from utils import Log

    logger = Log("probe_grid_ablation").logger
    locs_eff = list(locs) if locs is not None else list(PROBE_GRID_ABLATION_LOCS)
    tokens_eff = list(tokens) if tokens is not None else list(PROBE_GRID_ABLATION_TOKENS)

    root = get_probe_grid_ablation_dir(model_id)
    root.mkdir(parents=True, exist_ok=True)
    emotion_out = root / "binary_ova_probes"
    emotion_out.mkdir(parents=True, exist_ok=True)

    ds_str = str(Path(datasets_dir).resolve()) if datasets_dir is not None else None
    manifest = {
        "version": 1,
        "scope": "A_probe_grid_only",
        "model_id": model_id,
        "locs": locs_eff,
        "tokens": _tokens_json_safe(tokens_eff),
        "datasets_dir": ds_str,
        "emotion_probe_subdir": "binary_ova_probes",
        "git_commit": _git_head(),
        "repo_root": str(CFG_ROOT),
        "skip_appraisal": bool(skip_appraisal),
        "max_samples_appraisal": max_samples,
    }
    write_json(root / "grid_manifest.json", manifest)
    logger.info(f"Wrote {root / 'grid_manifest.json'}")

    logger.info("Training emotion OVA probes (grid ablation)...")
    run_train_probes(
        model_id=model_id,
        datasets_dir=datasets_dir,
        probe_n_jobs=probe_n_jobs,
        logger=logger,
        output_dir=emotion_out,
        extraction_locs_override=locs_eff,
        extraction_tokens_override=tokens_eff,
        emotions_filter=emotions_filter,
    )

    if not skip_appraisal:
        logger.info("Training appraisal Ridge probes (grid ablation)...")
        run_train_appraisal_probes(
            model_id=model_id,
            max_samples=max_samples,
            logger=logger,
            output_dir=root,
            extraction_locs_override=locs_eff,
            extraction_tokens_override=tokens_eff,
        )
    else:
        logger.info("Skipping appraisal training (--skip_appraisal).")

    logger.info(f"Grid ablation artifacts under: {root}")
    return root


def main():
    _ensure_paths()
    from pipeline.config import (
        DEFAULT_MODEL_ID,
        PROBE_GRID_ABLATION_LOCS,
        PROBE_GRID_ABLATION_TOKENS,
    )

    p = argparse.ArgumentParser(description="Probe-only loc×token grid ablation (scope A)")
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--datasets_dir", type=Path, default=None, help="OVA datasets dir (same as train_probes)")
    p.add_argument(
        "--locs",
        type=str,
        default=None,
        help=f"Comma-separated loc ids (default: config PROBE_GRID_ABLATION_LOCS = {PROBE_GRID_ABLATION_LOCS})",
    )
    p.add_argument(
        "--tokens",
        type=str,
        default=None,
        help='Comma-separated token positions; use mid for masked middle (default: config grid)',
    )
    p.add_argument("--skip_appraisal", action="store_true", help="Only train emotion probes")
    p.add_argument("--probe_n_jobs", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None, help="Cap appraisal training rows (dev smoke)")
    p.add_argument("--emotions", type=str, default=None, help="Comma-separated emotion subset for OVA training")
    args = p.parse_args()

    locs = _parse_locs(args.locs) if args.locs else None
    tokens = _parse_tokens(args.tokens) if args.tokens else None
    emotions = _parse_emotions(args.emotions)

    run_probe_grid_ablation(
        model_id=args.model_id,
        datasets_dir=args.datasets_dir,
        locs=locs,
        tokens=tokens,
        skip_appraisal=args.skip_appraisal,
        probe_n_jobs=args.probe_n_jobs,
        max_samples=args.max_samples,
        emotions_filter=emotions,
    )


if __name__ == "__main__":
    main()

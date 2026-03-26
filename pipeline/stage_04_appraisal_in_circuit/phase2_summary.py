"""
Phase 2 summary: compute (if missing) and read Phase 2 outputs per model; write SUMMARY.md
to pipeline/outputs/<model_id>/04_appraisal_in_circuit/.

Geometry and correlation are now computed at runtime for each model (no external notebook).
"""
from pathlib import Path
import pandas as pd

from pipeline.core.config import (
    get_appraisal_in_circuit_dir,
    DEFAULT_MODEL_ID,
)
def write_phase2_summary(
    model_id: str = DEFAULT_MODEL_ID,
    phase2_dir: Path | None = None,
    run_compute_if_missing: bool = True,
) -> Path:
    """
    Read geometry_circuit_layers.csv and correlation_circuit_vs_default.csv from the
    per-model 04_appraisal_in_circuit dir (or phase2_dir if provided). If files are
    missing and run_compute_if_missing is True, run phase2_compute for this model first.
    Write SUMMARY.md to pipeline 04_appraisal_in_circuit. Returns path to SUMMARY.md.
    """
    out_dir = get_appraisal_in_circuit_dir(model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Per-model: read from out_dir unless caller passed a custom phase2_dir (e.g. legacy)
    read_dir = phase2_dir if phase2_dir is not None else out_dir

    geo_path = read_dir / "geometry_circuit_layers.csv"
    corr_path = read_dir / "correlation_circuit_vs_default.csv"
    ablation_path = read_dir / "appraisal_ablation_summary.csv"
    if run_compute_if_missing and (not geo_path.exists() or not corr_path.exists()):
        from pipeline.stage_04_appraisal_in_circuit.phase2_compute import run_phase2_compute
        run_phase2_compute(model_id)
        read_dir = out_dir
        geo_path = read_dir / "geometry_circuit_layers.csv"
        corr_path = read_dir / "correlation_circuit_vs_default.csv"
        ablation_path = read_dir / "appraisal_ablation_summary.csv"

    summary_path = out_dir / "SUMMARY.md"
    lines = [
        "# Appraisal in circuit (Phase 2) — Summary",
        "",
        "One-page summary for the principal researcher.",
        "",
    ]

    # Geometry
    if geo_path.exists():
        geo_df = pd.read_csv(geo_path)
        if len(geo_df) > 0 and "cos_sim" in geo_df.columns:
            mean_cos = float(geo_df["cos_sim"].mean())
            by_emotion = geo_df.groupby("emotion")["cos_sim"].mean()
            lines.extend([
                "## Geometry (appraisal vs emotion directions at circuit layers)",
                "",
                f"- **Mean cosine similarity** (across selected emotion-specific circuit sites): **{mean_cos:.4f}**.",
                f"- Per-emotion mean cosine sim: min = {by_emotion.min():.4f}, max = {by_emotion.max():.4f}.",
                "",
                "Interpretation: This is an internal representational alignment diagnostic only. Positive cosine means the appraisal and emotion directions point similarly in raw hidden-state space at the selected circuit sites.",
                "",
            ])
        else:
            lines.extend([
                "## Geometry",
                "",
                "No geometry data (empty or missing appraisal probes).",
                "",
            ])
    else:
        lines.extend([
            "## Geometry",
            "",
            "Geometry not computed (circuit or probes missing).",
            "",
        ])

    # Correlation
    if corr_path.exists():
        corr_df = pd.read_csv(corr_path)
        if len(corr_df) > 0 and "mean_corr_with_default" in corr_df.columns:
            mean_corr = float(corr_df["mean_corr_with_default"].mean())
            lines.extend([
                "## Representation content (circuit-layer appraisal vs default layer)",
                "",
                f"- **Mean correlation** (selected circuit-site appraisal predictions vs default-layer appraisal predictions): **{mean_corr:.4f}**.",
                "",
                "Interpretation: This measures agreement with the model's default-layer appraisal predictions on held-out activations; it is not a direct human-label-grounded validity metric.",
                "",
            ])
        else:
            lines.extend([
                "## Representation content",
                "",
                "No correlation data.",
                "",
            ])
    else:
        lines.extend([
            "## Representation content",
            "",
            "Correlation not computed (val hidden states or appraisal probes missing).",
            "",
        ])

    # Ablations
    if ablation_path.exists():
        abl_df = pd.read_csv(ablation_path)
        if len(abl_df) > 0 and "delta_self_logit" in abl_df.columns:
            full_df = abl_df[abl_df["ablation_type"] == "full_appraisal_signature"]
            dim_df = abl_df[abl_df["ablation_type"] == "single_dimension"]
            lines.extend([
                "## Appraisal ablations",
                "",
            ])
            if not full_df.empty:
                mean_full = float(full_df["delta_self_logit"].mean())
                lines.append(
                    f"- **Full appraisal-signature erasure:** mean self-logit change **{mean_full:.4f}**."
                )
            if not dim_df.empty:
                strongest = dim_df.loc[dim_df["delta_self_logit"].abs().idxmax()]
                lines.append(
                    f"- **Strongest single-dimension effect:** `{strongest['dimension']}` on `{strongest['emotion']}` "
                    f"with delta self-logit **{strongest['delta_self_logit']:.4f}**."
                )
            lines.extend([
                "",
                "Interpretation: If erasing appraisal directions shifts the circuit readout for an emotion, that is stronger evidence that appraisal structure contributes to the circuit's emotion representation.",
                "",
            ])

    lines.extend([
        "## Conclusion",
        "",
        "These phase-2 outputs should be interpreted as a ladder of evidence. Geometry and default-layer agreement are internal diagnostics; the appraisal ablations provide stronger intervention-style evidence that appraisal structure matters for the circuit readout. None of these outputs alone fully establish human-label-grounded causal validity, but together they make the theory test more explicit.",
        "",
    ])

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {summary_path}")
    return summary_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--phase2_dir", type=Path, default=None, help="Legacy: read CSVs from this dir instead of per-model 04_appraisal_in_circuit")
    p.add_argument("--no_compute", action="store_true", help="Do not run phase2_compute if CSVs missing")
    args = p.parse_args()
    write_phase2_summary(
        model_id=args.model_id,
        phase2_dir=args.phase2_dir,
        run_compute_if_missing=not args.no_compute,
    )

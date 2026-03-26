"""
Synthesis: read pipeline outputs and produce summary figures, metrics table, and SUMMARY.md.

Reads from 02_circuit, 03_appraisal_structure, 04_appraisal_in_circuit, 05_steering, 01_probes.
Writes to 06_synthesis/ (or outputs/synthesis/ for multi-model): figures, synthesis_metrics.csv, and SUMMARY.md.
"""
from __future__ import annotations

import logging
import math
import os
import shutil
from pathlib import Path

from pipeline.core.logutil import configure_logging
from pipeline.core.runtime_env import apply_cpu_thread_limits
logger = logging.getLogger(__name__)

from pipeline.core.config import (
    OUTPUTS_ROOT,
    get_model_output_dir,
    get_circuit_dir,
    get_appraisal_structure_dir,
    get_appraisal_in_circuit_dir,
    get_appraisal_theory_dir,
    get_steering_dir,
    get_generation_behavior_dir,
    get_mental_health_steering_dir,
    get_synthesis_dir,
    get_probes_dir,
    get_probe_paths,
    DEFAULT_MODEL_ID,
    MENTAL_HEALTH_REPORT_ALPHA,
)
from pipeline.core.research_contracts import weighted_mean


def _mh_pick_report_alpha_from_series(alphas_in_data, preferred: float) -> float:
    """Match mental_health `_pick_report_alpha`: preferred if present, else nearest positive alpha."""
    import pandas as pd

    alphas = sorted({float(a) for a in pd.unique(alphas_in_data)})
    if not alphas:
        return float(preferred)
    pref = float(preferred)
    for a in alphas:
        if math.isclose(a, pref, rel_tol=0.0, abs_tol=1e-9):
            return float(a)
    positive = [a for a in alphas if a > 0]
    if not positive:
        return 0.0
    return float(min(positive, key=lambda x: abs(x - pref)))


def _mh_best_alpha_mean_tone(mdf, method: str) -> tuple[float | None, float | None]:
    """
    Pick alpha (among alpha>0) that maximizes aggregate mean_emotional_tone; for
    appraisal_elicitation, minimize (stronger negative-affect proxy on this metric).
    """
    import pandas as pd

    sub = mdf[mdf["method"] == method]
    if sub.empty:
        return None, None
    g = sub.groupby("alpha", as_index=False)["mean_emotional_tone"].mean()
    g = g[g["alpha"] > 0]
    if g.empty:
        return None, None
    if method == "appraisal_elicitation":
        pick = g.loc[g["mean_emotional_tone"].idxmin()]
    else:
        pick = g.loc[g["mean_emotional_tone"].idxmax()]
    return float(pick["alpha"]), float(pick["mean_emotional_tone"])


def _collect_metrics_for_model(mid: str) -> dict:
    """Build one row of metrics from this model's CSVs."""
    import pandas as pd
    row = {"model_id": mid}
    circuit_dir = get_circuit_dir(mid)
    circuit_csv = circuit_dir / "circuit_evidence_classification.csv"
    if circuit_csv.exists():
        df = pd.read_csv(circuit_csv)
        sb = df[df["method"] == "single_best"]
        tk = df[df["method"] == "topk_fusion"]
        tk_global = df[df["method"] == "topk_fusion_global"]
        if not sb.empty:
            row["circuit_single_best_accuracy"] = float(sb["accuracy"].iloc[0])
            row["circuit_single_best_roc_auc"] = float(sb["roc_auc"].iloc[0]) if pd.notna(sb["roc_auc"].iloc[0]) else None
        if not tk.empty:
            row["circuit_topk_accuracy"] = float(tk["accuracy"].iloc[0])
            row["circuit_topk_roc_auc"] = float(tk["roc_auc"].iloc[0]) if pd.notna(tk["roc_auc"].iloc[0]) else None
            row["circuit_topk_k_pairs"] = float(tk["k_pairs"].iloc[0]) if "k_pairs" in tk.columns and pd.notna(tk["k_pairs"].iloc[0]) else None
        if not tk_global.empty:
            row["circuit_topk_global_accuracy"] = float(tk_global["accuracy"].iloc[0])
            row["circuit_topk_global_roc_auc"] = float(tk_global["roc_auc"].iloc[0]) if pd.notna(tk_global["roc_auc"].iloc[0]) else None
    steer_csv = get_steering_dir(mid) / "steering_benchmark.csv"
    if steer_csv.exists():
        df = pd.read_csv(steer_csv)
        if "n_source_samples" not in df.columns:
            df["n_source_samples"] = 1.0
        agg_rows = []
        for method, sub in df.groupby("method"):
            agg_rows.append({
                "method": method,
                "success_rate": weighted_mean(sub, "success_rate", "n_source_samples"),
                "mean_delta_target_logit": weighted_mean(sub, "mean_delta_target_logit", "n_source_samples"),
            })
        agg = pd.DataFrame(agg_rows)
        for _, r in agg.iterrows():
            m = r.get("method", "")
            if m == "appraisal_steer":
                row["appraisal_steer_success_rate"] = float(r.get("success_rate", 0))
                row["appraisal_steer_mean_delta_logit"] = float(r.get("mean_delta_target_logit", 0))
            elif m == "emotion_steer":
                row["emotion_steer_success_rate"] = float(r.get("success_rate", 0))
                row["emotion_steer_mean_delta_logit"] = float(r.get("mean_delta_target_logit", 0))
    behavioral_csv = get_steering_dir(mid) / "steering_benchmark_behavioral.csv"
    if behavioral_csv.exists():
        df = pd.read_csv(behavioral_csv)
        if "n_source_samples" not in df.columns:
            df["n_source_samples"] = 1.0
        agg_rows = []
        for method, sub in df.groupby("method"):
            agg_rows.append({
                "method": method,
                "success_rate": weighted_mean(sub, "success_rate", "n_source_samples"),
                "mean_delta_target_logit": weighted_mean(sub, "mean_delta_target_logit", "n_source_samples"),
            })
        agg = pd.DataFrame(agg_rows)
        for _, r in agg.iterrows():
            m = r.get("method", "")
            if m == "appraisal_steer":
                row["behavioral_appraisal_steer_success_rate"] = float(r.get("success_rate", 0))
            elif m == "emotion_steer":
                row["behavioral_emotion_steer_success_rate"] = float(r.get("success_rate", 0))
    gen_behavior_csv = get_generation_behavior_dir(mid) / "generation_behavior_summary_by_condition.csv"
    if gen_behavior_csv.exists():
        df = pd.read_csv(gen_behavior_csv)
        row["generation_behavior_rows"] = float(df["prompts"].sum()) if "prompts" in df.columns and not df.empty else 0.0
        for _, r in df.iterrows():
            condition = str(r.get("intervention_type", ""))
            safe_name = condition.replace("-", "_")
            if pd.notna(r.get("assistantlike_proxy")):
                row[f"gen_behavior_{safe_name}_assistantlike_proxy"] = float(r["assistantlike_proxy"])
            if pd.notna(r.get("unsafe_compliance_proxy")):
                row[f"gen_behavior_{safe_name}_unsafe_compliance_proxy"] = float(r["unsafe_compliance_proxy"])
    judge_summary_csv = get_generation_behavior_dir(mid) / "generation_behavior_judge_summary_by_condition.csv"
    if judge_summary_csv.exists():
        df = pd.read_csv(judge_summary_csv)
        for _, r in df.iterrows():
            condition = str(r.get("intervention_type", ""))
            safe_name = condition.replace("-", "_")
            for metric in [
                "assistantlike_score",
                "hostility_score",
                "blame_score",
                "empathy_score",
                "deescalation_score",
                "unsafe_compliance_score",
                "refusal_quality_score",
                "persona_stability_score",
            ]:
                if metric in r and pd.notna(r.get(metric)):
                    row[f"gen_behavior_{safe_name}_{metric}"] = float(r[metric])
    probe_summary = get_probe_paths(mid).probe_summary_csv
    if probe_summary.exists():
        ps = pd.read_csv(probe_summary)
        if "test_roc_auc" in ps.columns:
            row["mean_probe_test_roc_auc"] = float(ps["test_roc_auc"].mean())
    theory_dir = get_appraisal_theory_dir(mid)
    recon_csv = theory_dir / "reconstruction_by_layer_loc.csv"
    if recon_csv.exists():
        rdf = pd.read_csv(recon_csv)
        if not rdf.empty:
            row["theory_best_reconstruction_accuracy"] = float(rdf["reconstruction_accuracy"].max())
    onset_csv = theory_dir / "onset_comparison.csv"
    if onset_csv.exists():
        odf = pd.read_csv(onset_csv)
        valid = odf[odf["onset_layer"] >= 0]
        ap_med = valid.loc[valid["type"] == "appraisal", "onset_layer"].median()
        em_med = valid.loc[valid["type"] == "emotion", "onset_layer"].median()
        if pd.notna(ap_med):
            row["theory_appraisal_onset_median"] = float(ap_med)
        if pd.notna(em_med):
            row["theory_emotion_onset_median"] = float(em_med)
    cos_csv = theory_dir / "ridge_vs_binary_cosine.csv"
    if cos_csv.exists():
        cdf = pd.read_csv(cos_csv)
        if not cdf.empty:
            row["theory_ridge_binary_mean_cosine"] = float(cdf["cosine_sim"].mean())
    mh_dir = get_mental_health_steering_dir(mid)
    mh_curves = mh_dir / "dose_response_curves.csv"
    if mh_curves.exists():
        mdf = pd.read_csv(mh_curves)
        if not mdf.empty:
            ref_alpha = _mh_pick_report_alpha_from_series(mdf["alpha"].values, MENTAL_HEALTH_REPORT_ALPHA)
            row["mh_synthesis_reference_alpha"] = ref_alpha
            methods = [
                "appraisal_targeted",
                "appraisal_full",
                "emotion_steer",
                "combined",
                "appraisal_elicitation",
            ]
            for method in methods:
                sub = mdf[(mdf["method"] == method) & mdf["alpha"].apply(
                    lambda a: math.isclose(float(a), ref_alpha, rel_tol=0.0, abs_tol=1e-9)
                )]
                if not sub.empty:
                    tr = float(sub["mean_emotional_tone"].mean())
                    row[f"mh_{method}_mean_tone_ref"] = tr
                    row[f"mh_{method}_tone_alpha5"] = tr
                ba, bt = _mh_best_alpha_mean_tone(mdf, method)
                if ba is not None and bt is not None:
                    row[f"mh_{method}_best_alpha_mean_tone"] = ba
                    row[f"mh_{method}_mean_tone_best"] = bt
            baseline = mdf[mdf["method"] == "baseline"]
            if not baseline.empty:
                row["mh_baseline_tone"] = float(baseline["mean_emotional_tone"].mean())
    return row


def run_synthesis(
    model_id: str | None = None,
    model_ids: list[str] | None = None,
) -> Path:
    """
    If model_id is set, synthesize that model. If model_ids is set, synthesize all and
    write multi-model summary. Otherwise use DEFAULT_MODEL_ID.
    """
    configure_logging()
    apply_cpu_thread_limits()
    import pandas as pd

    if model_ids is not None:
        ids = list(model_ids)
    elif model_id is not None:
        ids = [model_id]
    else:
        ids = [DEFAULT_MODEL_ID]

    if len(ids) == 1:
        out_dir = get_synthesis_dir(ids[0])
    else:
        out_dir = get_synthesis_dir(None)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure per-model PCA/EDA analysis exists for each requested model.
    from pipeline.stage_03_appraisal_structure.representation_analysis import run_representation_analysis

    for mid in ids:
        analysis_dir = get_appraisal_structure_dir(mid) / "pca_eda"
        if not analysis_dir.exists():
            try:
                run_representation_analysis(model_id=mid)
            except Exception:
                logger.warning(
                    "Could not run representation_analysis for %s",
                    mid,
                    exc_info=True,
                )
                strict = os.environ.get("PIPELINE_STRICT_SYNTHESIS", "").strip().lower()
                if strict in ("1", "true", "yes"):
                    raise

    # Metrics table (one row per model)
    metrics_rows = [_collect_metrics_for_model(mid) for mid in ids]
    metrics_df = pd.DataFrame(metrics_rows)
    if not metrics_df.empty:
        metrics_df.to_csv(out_dir / "synthesis_metrics.csv", index=False)
        print(f"Wrote {out_dir / 'synthesis_metrics.csv'}")

    # Copy or collect key figures and tables
    for mid in ids:
        circuit_dir = get_circuit_dir(mid)
        app_struct_dir = get_appraisal_structure_dir(mid)
        analysis_dir = app_struct_dir / "pca_eda"
        app_circuit_dir = get_appraisal_in_circuit_dir(mid)
        steer_dir = get_steering_dir(mid)
        gen_behavior_dir = get_generation_behavior_dir(mid)
        for src_name, dest_name in [
            (circuit_dir / "circuit_evidence_classification.pdf", f"circuit_evidence_{mid}.pdf"),
            (circuit_dir / "circuit_evidence_classification.csv", f"circuit_evidence_{mid}.csv"),
            (circuit_dir / "circuit_top_k_selection.json", f"circuit_top_k_selection_{mid}.json"),
            (app_struct_dir / "appraisal_zscore_heatmap.pdf", f"appraisal_heatmap_{mid}.pdf"),
            (app_struct_dir / "appraisal_zscore_by_emotion.csv", f"appraisal_zscore_{mid}.csv"),
            (analysis_dir / "pca_site_summary.png", f"pca_site_summary_{mid}.png"),
            (analysis_dir / "pca_explained_variance.csv", f"pca_explained_variance_{mid}.csv"),
            (analysis_dir / "eda" / "eda_counts.png", f"eda_counts_{mid}.png"),
            (steer_dir / "steering_benchmark.pdf", f"steering_comparison_{mid}.pdf"),
            (steer_dir / "steering_benchmark.csv", f"steering_benchmark_{mid}.csv"),
            (steer_dir / "steering_benchmark_behavioral.csv", f"steering_benchmark_behavioral_{mid}.csv"),
            (steer_dir / "steering_benchmark_by_pair.png", f"steering_benchmark_by_pair_{mid}.png"),
            (steer_dir / "steering_benchmark_behavioral_by_pair.png", f"steering_benchmark_behavioral_by_pair_{mid}.png"),
            (steer_dir / "steering_benchmark_behavioral_by_text_type.png", f"steering_benchmark_behavioral_by_text_type_{mid}.png"),
            (steer_dir / "behavioral_appraisal_ablation.csv", f"behavioral_appraisal_ablation_{mid}.csv"),
            (steer_dir / "behavioral_appraisal_ablation.png", f"behavioral_appraisal_ablation_{mid}.png"),
            (gen_behavior_dir / "generation_behavior_outputs.csv", f"generation_behavior_outputs_{mid}.csv"),
            (gen_behavior_dir / "generation_behavior_scores.csv", f"generation_behavior_scores_{mid}.csv"),
            (gen_behavior_dir / "generation_behavior_judge_scores.csv", f"generation_behavior_judge_scores_{mid}.csv"),
            (gen_behavior_dir / "generation_behavior_summary_by_condition.csv", f"generation_behavior_summary_by_condition_{mid}.csv"),
            (gen_behavior_dir / "generation_behavior_summary_by_family.csv", f"generation_behavior_summary_by_family_{mid}.csv"),
            (gen_behavior_dir / "generation_behavior_judge_summary_by_condition.csv", f"generation_behavior_judge_summary_by_condition_{mid}.csv"),
            (app_circuit_dir / "geometry_by_emotion_dimension.png", f"geometry_by_emotion_dimension_{mid}.png"),
            (app_circuit_dir / "geometry_pair_category_comparison.png", f"geometry_pair_category_comparison_{mid}.png"),
            (app_circuit_dir / "emotion_appraisal_mapping.png", f"emotion_appraisal_mapping_{mid}.png"),
            (app_circuit_dir / "appraisal_ablation_summary.csv", f"appraisal_ablation_summary_{mid}.csv"),
            (app_circuit_dir / "appraisal_ablation_full_signature.png", f"appraisal_ablation_full_signature_{mid}.png"),
            (app_circuit_dir / "appraisal_ablation_by_dimension.png", f"appraisal_ablation_by_dimension_{mid}.png"),
        ]:
            if src_name.exists():
                shutil.copy2(src_name, out_dir / dest_name)
        theory_dir = get_appraisal_theory_dir(mid)
        for src_name, dest_name in [
            (theory_dir / "onset_comparison.png", f"theory_onset_comparison_{mid}.png"),
            (theory_dir / "reconstruction_curves.png", f"theory_reconstruction_{mid}.png"),
            (theory_dir / "cross_layer_asymmetry.png", f"theory_cross_layer_{mid}.png"),
            (theory_dir / "ridge_vs_binary_comparison.png", f"theory_ridge_vs_binary_{mid}.png"),
            (theory_dir / "circuit_location_distribution.png", f"theory_circuit_location_{mid}.png"),
            (theory_dir / "circuit_overlap_heatmap.png", f"theory_circuit_overlap_{mid}.png"),
            (theory_dir / "location_ordering_heatmaps.png", f"theory_location_ordering_{mid}.png"),
            (theory_dir / "appraisal_space_biplot.png", f"theory_appraisal_space_{mid}.png"),
            (theory_dir / "appraisal_space_3d.html", f"theory_appraisal_space_3d_{mid}.html"),
            (theory_dir / "reconstruction_by_layer_loc.csv", f"theory_reconstruction_{mid}.csv"),
            (theory_dir / "onset_comparison.csv", f"theory_onset_{mid}.csv"),
            (theory_dir / "cross_layer_prediction.csv", f"theory_cross_layer_{mid}.csv"),
            (theory_dir / "circuit_structure_summary.csv", f"theory_circuit_structure_{mid}.csv"),
            (theory_dir / "appraisal_space_coordinates.csv", f"theory_appraisal_coords_{mid}.csv"),
        ]:
            if src_name.exists():
                shutil.copy2(src_name, out_dir / dest_name)
        mh_dir = get_mental_health_steering_dir(mid)
        for src_name, dest_name in [
            (mh_dir / "dose_response_curves.csv", f"mh_dose_response_{mid}.csv"),
            (mh_dir / "condition_method_heatmap.png", f"mh_condition_heatmap_{mid}.png"),
            (mh_dir / "mental_health_steering_scores.csv", f"mh_steering_scores_{mid}.csv"),
            (mh_dir / "mental_health_judge_scores.csv", f"mh_judge_scores_{mid}.csv"),
            (mh_dir / "mental_health_judge_summary.csv", f"mh_judge_summary_{mid}.csv"),
        ]:
            if src_name.exists():
                shutil.copy2(src_name, out_dir / dest_name)
        for png in mh_dir.glob("dose_response_*.png"):
            shutil.copy2(png, out_dir / f"mh_{png.name.replace('.png', '')}_{mid}.png")

        if (app_circuit_dir / "SUMMARY.md").exists() and len(ids) == 1:
            shutil.copy2(app_circuit_dir / "SUMMARY.md", out_dir / "appraisal_in_circuit_summary.md")
        if (gen_behavior_dir / "summary.md").exists() and len(ids) == 1:
            shutil.copy2(gen_behavior_dir / "summary.md", out_dir / "generation_behavior_summary.md")

    # SUMMARY.md narrative
    lines = [
        "# Pipeline synthesis",
        "",
        "Summary of emotion circuit and appraisal steering results.",
        "Metrics are reported from each stage's saved outputs; selection artifacts and held-out evaluations are distinguished where available.",
        "",
    ]
    for mid in ids:
        lines.append(f"## Model: {mid}")
        lines.append("")
        row = next((r for r in metrics_rows if r.get("model_id") == mid), {})
        if row.get("circuit_single_best_accuracy") is not None:
            sb_acc = row.get("circuit_single_best_accuracy")
            tk_acc = row.get("circuit_topk_accuracy")
            tk_gl = row.get("circuit_topk_global_accuracy")
            if tk_gl is not None:
                lines.append(f"- **Circuit evidence (held-out test):** single-best accuracy {sb_acc:.3f}, per-emotion top-k {tk_acc:.3f}, global top-k {tk_gl:.3f} (see circuit_evidence_*.csv).")
            else:
                lines.append(f"- **Circuit evidence:** single-best accuracy {sb_acc:.3f}, top-k fusion {tk_acc:.3f} (see circuit_evidence_*.csv).")
            lines.append("")
        if row.get("emotion_steer_success_rate") is not None:
            app_sr = row.get("appraisal_steer_success_rate", 0)
            emo_sr = row.get("emotion_steer_success_rate", 0)
            lines.append(f"- **Cached steering (held-out test):** appraisal success {app_sr:.3f}, emotion success {emo_sr:.3f} (weighted by source-sample count).")
            lines.append("")
        if row.get("behavioral_emotion_steer_success_rate") is not None:
            app_b = row.get("behavioral_appraisal_steer_success_rate", 0)
            emo_b = row.get("behavioral_emotion_steer_success_rate", 0)
            lines.append(f"- **Behavioral steering (held-out test):** appraisal success {app_b:.3f}, emotion success {emo_b:.3f} (see steering_benchmark_behavioral_*.csv).")
            if emo_b - app_b > 0.1:
                lines.append(f"  This run shows a sizable behavioral gap in favor of direct emotion steering ({emo_b - app_b:.3f}), so pair-level outputs should be checked before treating appraisal steering as broadly effective.")
            lines.append("")
        if row.get("generation_behavior_rows") is not None:
            lines.append(f"- **Generation-behavior benchmark:** {int(row['generation_behavior_rows'])} prompt-condition summaries were recorded under `05_generation_behavior`, connecting steering to actual generated text.")
            app_assistant = row.get("gen_behavior_appraisal_steer_assistantlike_proxy")
            emo_assistant = row.get("gen_behavior_emotion_steer_assistantlike_proxy")
            if app_assistant is not None and emo_assistant is not None:
                lines.append(f"  Assistantlike proxy: appraisal {app_assistant:.3f}, emotion {emo_assistant:.3f}.")
            app_judge = row.get("gen_behavior_appraisal_steer_assistantlike_score")
            emo_judge = row.get("gen_behavior_emotion_steer_assistantlike_score")
            if app_judge is not None and emo_judge is not None:
                lines.append(f"  Judge assistantlike score: appraisal {app_judge:.3f}, emotion {emo_judge:.3f}.")
            lines.append("")
        if (get_appraisal_in_circuit_dir(mid) / "appraisal_ablation_summary.csv").exists():
            lines.append("- **Appraisal ablations:** see `appraisal_ablation_summary_*.csv` and related figures for intervention-style evidence about appraisal dimensions inside the circuit.")
            lines.append("")
        if (get_appraisal_structure_dir(mid) / "pca_eda" / "pca_explained_variance.csv").exists():
            lines.append("- **Representation analysis:** PCA / EDA outputs are available under the copied `pca_*` and `eda_*` artifacts for layer/loc-level structure inspection.")
            lines.append("")
        if (get_appraisal_theory_dir(mid) / "SUMMARY.md").exists():
            recon_acc = row.get("theory_best_reconstruction_accuracy")
            ap_onset = row.get("theory_appraisal_onset_median")
            em_onset = row.get("theory_emotion_onset_median")
            cosine = row.get("theory_ridge_binary_mean_cosine")
            lines.append("- **Appraisal theory analysis:**")
            if recon_acc is not None:
                lines.append(f"  Best appraisal-to-emotion reconstruction accuracy: {recon_acc:.1%}.")
            if ap_onset is not None and em_onset is not None:
                lines.append(f"  Appraisal onset median layer: {ap_onset:.0f}, emotion onset: {em_onset:.0f}.")
            if cosine is not None:
                lines.append(f"  Ridge vs binary probe mean cosine: {cosine:.3f}.")
            lines.append("  See `04_appraisal_theory/SUMMARY.md` for full analysis.")
            lines.append("")
        if (get_mental_health_steering_dir(mid) / "summary.md").exists():
            mh_baseline = row.get("mh_baseline_tone")
            ref_a = row.get("mh_synthesis_reference_alpha")
            mh_targeted = row.get("mh_appraisal_targeted_mean_tone_ref")
            mh_elicitation = row.get("mh_appraisal_elicitation_mean_tone_ref")
            mh_emotion = row.get("mh_emotion_steer_mean_tone_ref")
            lines.append("- **Mental health steering benchmark:**")
            if ref_a is not None:
                lines.append(
                    f"  Reference α for rollup (nearest `MENTAL_HEALTH_REPORT_ALPHA` in dose curves): {ref_a:g}."
                )
            if mh_baseline is not None:
                lines.append(f"  Baseline emotional tone: {mh_baseline:.2f}.")
            if mh_targeted is not None:
                lines.append(f"  Targeted appraisal steering (mean tone at reference α): {mh_targeted:.2f}.")
            if mh_elicitation is not None:
                lines.append(f"  Appraisal elicitation steering (mean tone at reference α): {mh_elicitation:.2f}.")
            if mh_emotion is not None:
                lines.append(f"  Emotion steering (mean tone at reference α): {mh_emotion:.2f}.")
            lines.append(
                "  Best-α columns in `synthesis_metrics.csv` pick the dose that maxes mean emotional tone "
                "(mins for `appraisal_elicitation`). See `05_mental_health_steering/summary.md` for curves."
            )
            lines.append("")
        lines.append("---")
        lines.append("")

    if len(ids) > 1 and not metrics_df.empty:
        lines.append("## Comparison")
        lines.append("")
        if "circuit_topk_accuracy" in metrics_df.columns and metrics_df["circuit_topk_accuracy"].notna().any():
            best_circuit = metrics_df.loc[metrics_df["circuit_topk_accuracy"].idxmax()]
            lines.append(f"- **Best circuit top-k accuracy:** {best_circuit['circuit_topk_accuracy']:.3f} ({best_circuit['model_id']}).")
        if "emotion_steer_success_rate" in metrics_df.columns and metrics_df["emotion_steer_success_rate"].notna().any():
            best_steer = metrics_df.loc[metrics_df["emotion_steer_success_rate"].idxmax()]
            lines.append(f"- **Best emotion steering success:** {best_steer['emotion_steer_success_rate']:.3f} ({best_steer['model_id']}).")
        lines.append("")
        lines.append("See synthesis_metrics.csv for full table.")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.extend([
        "## Narrative (1 → 2 → 3 → 4)",
        "",
        "1. **Circuit evidence:** The selected circuit is a performance-tuned combination of probe sites, with metrics reported on the held-out test split.",
        "2. **Native appraisal structure:** The appraisal heatmap summarizes train-split label structure rather than a model performance claim.",
        "3. **Representation analysis:** PCA / EDA outputs expose how emotion and appraisal structure evolve across layers and locations.",
        "4. **Appraisal in circuit:** Phase 2 now includes geometry, pair-category comparisons, and appraisal ablations; read these as progressively stronger evidence rather than one monolithic claim.",
        "5. **Steering:** Compare cached pair-level steering and behavioral forward-pass steering separately; large gaps between appraisal and emotion steering should be interpreted as evidence that appraisal-based control is pair-dependent rather than uniformly strong.",
        "6. **Generation behavior:** Treat the generation benchmark as the bridge from latent-state steering to user-visible assistant behavior; use it to test whether appraisal/emotion interventions actually alter tone, refusals, blame, and de-escalation in produced text.",
        "",
        "**Conclusion:** Use this synthesis as an index into the saved artifacts, not as a substitute for the underlying methodological caveats. Prefer held-out test metrics and behavioral steering over tuned or purely internal diagnostics when making claims.",
        "",
    ])

    summary_path = out_dir / "SUMMARY.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {summary_path}")
    return out_dir


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default=None)
    p.add_argument("--model_ids", nargs="*", default=None)
    args = p.parse_args()
    run_synthesis(model_id=args.model_id, model_ids=args.model_ids)

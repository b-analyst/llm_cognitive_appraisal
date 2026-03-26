from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.core.research_contracts import build_pair_category_lookup, pair_category


def _save_fig(fig, out_base: Path):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")


def add_pair_annotations(df: pd.DataFrame, contrastive_pairs: list[tuple[str, str]], similar_pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Annotate steering rows with pair label and pair category."""
    out = df.copy()
    lookup = build_pair_category_lookup(contrastive_pairs, similar_pairs)
    if {"source_emotion", "target_emotion"}.issubset(out.columns):
        out["pair_label"] = out["source_emotion"].astype(str) + " -> " + out["target_emotion"].astype(str)
        out["pair_category"] = [
            pair_category(s, t, lookup)
            for s, t in zip(out["source_emotion"], out["target_emotion"])
        ]
    return out


def save_steering_pair_outputs(
    benchmark_df: pd.DataFrame,
    curves_df: pd.DataFrame,
    out_dir: Path,
    contrastive_pairs: list[tuple[str, str]],
    similar_pairs: list[tuple[str, str]],
    behavioral: bool = False,
) -> None:
    """Save per-pair steering CSVs and informative static figures."""
    if benchmark_df is None or benchmark_df.empty:
        return
    annotated = add_pair_annotations(benchmark_df, contrastive_pairs, similar_pairs)
    curves_annotated = add_pair_annotations(curves_df, contrastive_pairs, similar_pairs) if curves_df is not None and not curves_df.empty else pd.DataFrame()

    stem = "steering_benchmark_behavioral_by_pair" if behavioral else "steering_benchmark_by_pair"
    annotated.to_csv(out_dir / f"{stem}.csv", index=False)
    if not curves_annotated.empty:
        curve_stem = "steering_curves_behavioral_by_pair" if behavioral else "steering_curves_by_pair"
        curves_annotated.to_csv(out_dir / f"{curve_stem}.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pivot_sr = annotated.pivot_table(
            index="pair_label", columns="method", values="success_rate", aggfunc="mean"
        ).sort_index()
        pivot_dl = annotated.pivot_table(
            index="pair_label", columns="method", values="mean_delta_target_logit", aggfunc="mean"
        ).sort_index()

        fig, axes = plt.subplots(1, 2, figsize=(14, max(4, 0.4 * len(pivot_sr.index) + 2)))
        pivot_sr.plot(kind="barh", ax=axes[0])
        axes[0].set_title("Per-pair success rate")
        axes[0].set_xlabel("Success rate")
        axes[0].set_ylabel("Emotion pair")
        axes[0].set_xlim(0, 1.05)

        pivot_dl.plot(kind="barh", ax=axes[1])
        axes[1].set_title("Per-pair delta target logit")
        axes[1].set_xlabel("Mean delta target logit")
        axes[1].set_ylabel("")

        _save_fig(fig, out_dir / stem)
        plt.close(fig)

        if "pair_category" in annotated.columns and annotated["pair_category"].notna().any():
            cat = annotated.groupby(["pair_category", "method"]).agg(
                success_rate=("success_rate", "mean"),
                mean_delta_target_logit=("mean_delta_target_logit", "mean"),
            ).reset_index()
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for idx, metric in enumerate(["success_rate", "mean_delta_target_logit"]):
                pivot = cat.pivot(index="pair_category", columns="method", values=metric)
                pivot.plot(kind="bar", ax=axes[idx])
                axes[idx].set_title(metric.replace("_", " "))
                axes[idx].set_ylabel(metric.replace("_", " "))
            _save_fig(fig, out_dir / f"{stem}_by_category")
            plt.close(fig)

        if behavioral and "text_type" in annotated.columns:
            by_text = annotated.groupby(["text_type", "method"]).agg(
                success_rate=("success_rate", "mean"),
                mean_delta_target_logit=("mean_delta_target_logit", "mean"),
            ).reset_index()
            by_text.to_csv(out_dir / "steering_benchmark_behavioral_by_text_type.csv", index=False)
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for idx, metric in enumerate(["success_rate", "mean_delta_target_logit"]):
                pivot = by_text.pivot(index="text_type", columns="method", values=metric)
                pivot.plot(kind="bar", ax=axes[idx])
                axes[idx].set_title(metric.replace("_", " "))
                axes[idx].set_ylabel(metric.replace("_", " "))
            _save_fig(fig, out_dir / "steering_benchmark_behavioral_by_text_type")
            plt.close(fig)

        if not curves_annotated.empty:
            methods = curves_annotated["method"].unique().tolist()
            fig, axes = plt.subplots(len(methods), 1, figsize=(10, max(4, 3 * len(methods))), sharex=False)
            if len(methods) == 1:
                axes = [axes]
            for ax, method in zip(axes, methods):
                sub = curves_annotated[curves_annotated["method"] == method].copy()
                for pair_label, pair_df in sub.groupby("pair_label"):
                    pair_df = pair_df.sort_values("strength")
                    ax.plot(pair_df["strength"], pair_df["success_rate"], marker="o", label=pair_label)
                ax.set_title(f"{method} success curves by pair")
                ax.set_ylabel("Success rate")
            axes[-1].set_xlabel("Strength")
            if len(methods) > 0 and len(curves_annotated["pair_label"].unique()) <= 10:
                axes[0].legend(loc="best", fontsize=8)
            curve_name = "steering_curves_behavioral_by_pair" if behavioral else "steering_curves_by_pair"
            _save_fig(fig, out_dir / curve_name)
            plt.close(fig)
    except Exception as e:
        print(f"Could not save steering theory plots: {e}")


def save_phase2_theory_outputs(
    geo_df: pd.DataFrame,
    corr_df: pd.DataFrame | None,
    appraisal_zscore_df: pd.DataFrame | None,
    out_dir: Path,
    contrastive_pairs: list[tuple[str, str]],
    similar_pairs: list[tuple[str, str]],
) -> None:
    """Save theory-facing Phase 2 plots and grouped comparison CSVs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lookup = build_pair_category_lookup(contrastive_pairs, similar_pairs)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        if geo_df is not None and not geo_df.empty:
            geo_mean = geo_df.groupby(["emotion", "dimension"])["cos_sim"].mean().reset_index()
            geo_pivot = geo_mean.pivot(index="emotion", columns="dimension", values="cos_sim").fillna(0.0)
            fig, ax = plt.subplots(figsize=(12, max(5, 0.35 * len(geo_pivot.index) + 2)))
            sns.heatmap(geo_pivot, cmap="RdBu_r", center=0, ax=ax)
            ax.set_title("Circuit geometry by emotion and appraisal dimension")
            _save_fig(fig, out_dir / "geometry_by_emotion_dimension")
            plt.close(fig)

            by_site = geo_df.groupby(["layer", "loc"])["cos_sim"].mean().reset_index()
            site_labels = [f"L{int(r.layer)}@{int(r.loc)}" for r in by_site.itertuples()]
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.bar(site_labels, by_site["cos_sim"])
            ax.set_title("Mean appraisal-emotion geometry by circuit site")
            ax.set_ylabel("Mean cosine similarity")
            ax.tick_params(axis="x", rotation=90)
            _save_fig(fig, out_dir / "geometry_by_site")
            plt.close(fig)

            if appraisal_zscore_df is not None and not appraisal_zscore_df.empty:
                emotions = geo_pivot.index.tolist()
                rows = []
                for i, e1 in enumerate(emotions):
                    for e2 in emotions[i + 1:]:
                        geom1 = geo_pivot.loc[e1].to_numpy(dtype=float)
                        geom2 = geo_pivot.loc[e2].to_numpy(dtype=float)
                        z1 = appraisal_zscore_df.loc[e1].to_numpy(dtype=float) if e1 in appraisal_zscore_df.index else None
                        z2 = appraisal_zscore_df.loc[e2].to_numpy(dtype=float) if e2 in appraisal_zscore_df.index else None
                        def _cos(a, b):
                            denom = np.linalg.norm(a) * np.linalg.norm(b)
                            return 0.0 if denom < 1e-12 else float(np.dot(a, b) / denom)
                        rows.append(
                            {
                                "emotion_a": e1,
                                "emotion_b": e2,
                                "pair_category": pair_category(e1, e2, lookup),
                                "geometry_cosine": _cos(geom1, geom2),
                                "appraisal_profile_cosine": _cos(z1, z2) if z1 is not None and z2 is not None else np.nan,
                            }
                        )
                pair_df = pd.DataFrame(rows)
                pair_df.to_csv(out_dir / "geometry_pair_comparison.csv", index=False)
                if not pair_df.empty:
                    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                    sns.boxplot(data=pair_df, x="pair_category", y="geometry_cosine", ax=axes[0])
                    axes[0].set_title("Geometry cosine by pair category")
                    sns.boxplot(data=pair_df, x="pair_category", y="appraisal_profile_cosine", ax=axes[1])
                    axes[1].set_title("Appraisal-profile cosine by pair category")
                    _save_fig(fig, out_dir / "geometry_pair_category_comparison")
                    plt.close(fig)

        if corr_df is not None and not corr_df.empty:
            corr_pivot = corr_df.pivot_table(
                index="dimension", columns="loc", values="mean_corr_with_default", aggfunc="mean"
            ).fillna(0.0)
            fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(corr_pivot.index) + 2)))
            sns.heatmap(corr_pivot, cmap="viridis", ax=ax)
            ax.set_title("Default-layer appraisal agreement by dimension and loc")
            _save_fig(fig, out_dir / "correlation_by_dimension_loc")
            plt.close(fig)

        if appraisal_zscore_df is not None and not appraisal_zscore_df.empty:
            fig, ax = plt.subplots(figsize=(12, max(5, 0.35 * len(appraisal_zscore_df.index) + 2)))
            sns.heatmap(appraisal_zscore_df, cmap="RdBu_r", center=0, ax=ax)
            ax.set_title("Emotion mapping in appraisal space")
            _save_fig(fig, out_dir / "emotion_appraisal_mapping")
            plt.close(fig)
    except Exception as e:
        print(f"Could not save phase2 theory plots: {e}")

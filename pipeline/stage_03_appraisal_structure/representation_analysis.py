from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from pipeline.core.config import (
    COMBINED_CSV,
    COMMON_APPRAISAL,
    DEFAULT_MODEL_ID,
    SELECTION_SPLIT,
    FINAL_TEST_SPLIT,
    RANDOM_STATE,
    get_circuit_dir,
    get_appraisal_structure_dir,
    CONTRASTIVE_EMOTION_PAIRS,
    SIMILAR_EMOTION_PAIRS,
)
from pipeline.core.model_config import get_extraction_layers, get_extraction_locs
from pipeline.core.research_contracts import (
    canonicalize_combined_dataset,
    split_combined_dataset,
    wrap_hover_text,
)
def _load_test_frame(model_id: str) -> pd.DataFrame:
    df = canonicalize_combined_dataset(pd.read_csv(COMBINED_CSV))
    split_bundle = split_combined_dataset(
        df,
        selection_frac=SELECTION_SPLIT,
        test_frac=FINAL_TEST_SPLIT,
        random_state=RANDOM_STATE,
    )
    return split_bundle["test"].reset_index(drop=True)


def _save_fig(fig, out_base: Path):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")


def _decision_surface(ax, coords: np.ndarray, y_codes: np.ndarray, labels: list[str], title: str):
    if len(np.unique(y_codes)) < 2:
        return
    clf = LogisticRegression(max_iter=2000)
    clf.fit(coords[:, :2], y_codes)
    x_min, x_max = coords[:, 0].min() - 1.0, coords[:, 0].max() + 1.0
    y_min, y_max = coords[:, 1].min() - 1.0, coords[:, 1].max() + 1.0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = clf.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, zz, alpha=0.15, levels=np.arange(len(labels) + 1) - 0.5, cmap="tab20")
    ax.set_title(title)


def run_representation_analysis(model_id: str = DEFAULT_MODEL_ID) -> Path:
    """
    Generate default PCA, explained-variance, appraisal-space, and EDA outputs.

    Outputs live under `03_appraisal_structure/pca_eda/`.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px

    out_dir = get_appraisal_structure_dir(model_id) / "pca_eda"
    out_dir.mkdir(parents=True, exist_ok=True)
    circuit_dir = get_circuit_dir(model_id)
    hs_path = circuit_dir / "test_hidden_states.pt"
    labels_path = circuit_dir / "test_labels.csv"
    if not hs_path.exists() or not labels_path.exists():
        raise FileNotFoundError("Run circuit_evidence first so test hidden states exist.")

    hidden_states = torch.load(hs_path, weights_only=False)
    if isinstance(hidden_states, torch.Tensor):
        hs = hidden_states.float().cpu().numpy()
    else:
        hs = np.asarray(hidden_states, dtype=np.float32)
    test_df = _load_test_frame(model_id)
    test_labels = list(pd.read_csv(labels_path)["emotion"].astype(str))
    if len(test_df) != len(test_labels) or hs.shape[0] != len(test_labels):
        raise ValueError("Test split metadata and hidden states are misaligned.")
    test_df = test_df.copy()
    test_df["emotion"] = test_labels

    extraction_layers = get_extraction_layers(model_id)
    extraction_locs = get_extraction_locs(model_id)
    appraisal_cols = [c for c in COMMON_APPRAISAL if c in test_df.columns]

    # EDA outputs
    eda_dir = out_dir / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    counts = test_df["emotion"].value_counts().sort_values(ascending=False).reset_index()
    counts.columns = ["emotion", "count"]
    counts.to_csv(eda_dir / "emotion_counts_test.csv", index=False)
    if "source" in test_df.columns:
        source_counts = test_df.groupby(["source", "emotion"]).size().reset_index(name="count")
        source_counts.to_csv(eda_dir / "source_emotion_counts_test.csv", index=False)
    token_lengths = test_df["situation"].astype(str).str.split().map(len)
    pd.DataFrame({"token_length_words": token_lengths}).to_csv(eda_dir / "scenario_length_summary.csv", index=False)
    if appraisal_cols:
        appraisal_means = test_df.groupby("emotion")[appraisal_cols].mean()
        appraisal_means.to_csv(eda_dir / "appraisal_means_by_emotion_test.csv")
    try:
        fig, axes = plt.subplots(1, 2 if "source" in test_df.columns else 1, figsize=(14, 4))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        sns.barplot(data=counts, x="emotion", y="count", ax=axes[0], color="tab:blue")
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].set_title("Held-out test emotion counts")
        if "source" in test_df.columns:
            pivot = source_counts.pivot(index="source", columns="emotion", values="count").fillna(0.0)
            sns.heatmap(pivot, cmap="Blues", ax=axes[1])
            axes[1].set_title("Test counts by source and emotion")
        _save_fig(fig, eda_dir / "eda_counts")
        plt.close(fig)
    except Exception:
        pass

    # PCA outputs per site
    pca_dir = out_dir / "pca"
    pca_dir.mkdir(parents=True, exist_ok=True)
    explained_rows = []
    labels = sorted(test_df["emotion"].astype(str).unique().tolist())
    label_to_code = {label: idx for idx, label in enumerate(labels)}
    y_codes = np.array([label_to_code[e] for e in test_df["emotion"].astype(str)], dtype=int)
    hover_cols = [c for c in appraisal_cols if c in test_df.columns]

    for layer_idx, layer in enumerate(extraction_layers):
        for loc_idx, loc in enumerate(extraction_locs):
            X = hs[:, layer_idx, loc_idx, 0, :].astype(np.float32)
            n_components = max(2, min(10, X.shape[0], X.shape[1]))
            pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
            coords = pca.fit_transform(X)
            site_name = f"layer_{layer}_loc_{loc}"

            for comp_idx, ratio in enumerate(pca.explained_variance_ratio_, start=1):
                explained_rows.append(
                    {
                        "layer": int(layer),
                        "loc": int(loc),
                        "component": int(comp_idx),
                        "explained_variance_ratio": float(ratio),
                        "cumulative_explained_variance": float(np.sum(pca.explained_variance_ratio_[:comp_idx])),
                    }
                )

            plot_df = pd.DataFrame(
                {
                    "pc1": coords[:, 0],
                    "pc2": coords[:, 1],
                    "emotion": test_df["emotion"].astype(str),
                    "scenario": test_df["situation"].astype(str),
                }
            )
            for col in hover_cols:
                plot_df[col] = test_df[col]
            plot_df["hover_text"] = plot_df.apply(
                lambda r: "<b>Emotion:</b> "
                + str(r["emotion"])
                + "<br><b>Scenario:</b> "
                + wrap_hover_text(r["scenario"], width=70)
                + (
                    "<br><b>Appraisals:</b><br>"
                    + "<br>".join(f"{c}: {r[c]:.2f}" for c in hover_cols if pd.notna(r[c]))
                    if hover_cols
                    else ""
                ),
                axis=1,
            )

            fig_html = px.scatter(
                plot_df,
                x="pc1",
                y="pc2",
                color="emotion",
                hover_name="emotion",
                hover_data={"hover_text": True, "scenario": False, "pc1": False, "pc2": False},
                title=f"PCA at layer {layer}, loc {loc}",
                template="plotly_white",
            )
            fig_html.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")
            fig_html.write_html(str(pca_dir / f"{site_name}.html"))

            fig, ax = plt.subplots(figsize=(7, 6))
            _decision_surface(ax, coords, y_codes, labels, f"PCA + decision regions at layer {layer}, loc {loc}")
            sns.scatterplot(data=plot_df, x="pc1", y="pc2", hue="emotion", ax=ax, s=35, alpha=0.85)
            ax.legend(loc="best", fontsize=7, ncol=2)
            _save_fig(fig, pca_dir / site_name)
            plt.close(fig)

    explained_df = pd.DataFrame(explained_rows)
    explained_df.to_csv(out_dir / "pca_explained_variance.csv", index=False)
    site_summary = (
        explained_df[explained_df["component"] <= 3]
        .pivot_table(index=["layer", "loc"], columns="component", values="explained_variance_ratio", aggfunc="first")
        .reset_index()
    )
    site_summary.to_csv(out_dir / "pca_site_summary.csv", index=False)

    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        summary_plot = site_summary.copy()
        summary_plot["site"] = summary_plot.apply(lambda r: f"L{int(r['layer'])}@{int(r['loc'])}", axis=1)
        ax.bar(summary_plot["site"], summary_plot.get(1, 0), label="PC1")
        if 2 in summary_plot.columns:
            ax.bar(summary_plot["site"], summary_plot.get(2, 0), bottom=summary_plot.get(1, 0), label="PC2")
        if 3 in summary_plot.columns:
            ax.bar(summary_plot["site"], summary_plot.get(3, 0), bottom=summary_plot.get(1, 0) + summary_plot.get(2, 0), label="PC3")
        ax.set_ylabel("Explained variance ratio")
        ax.set_title("Top PCA explained variance by site")
        ax.tick_params(axis="x", rotation=90)
        ax.legend()
        _save_fig(fig, out_dir / "pca_site_summary")
        plt.close(fig)
    except Exception:
        pass

    return out_dir


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    args = p.parse_args()
    run_representation_analysis(model_id=args.model_id)

# Stage 3 — Appraisal structure

Describe how **labeled emotions** sit in **appraisal space** on held-out data: baselines, clustering on probe logits, z-score heatmaps, and optional **label coupling** diagnostics.

## Modules

- `appraisal_structure` — main structure run; invokes label coupling when configured.
- `appraisal_label_coupling` — pairwise overlap / independence dashboards (`label_coupling/`).
- `representation_analysis` — PCA / representation EDA (often under `03_appraisal_structure/pca_eda/`).

## Documentation in this folder

- [`docs/APPRAISAL_LABEL_COUPLING.md`](docs/APPRAISAL_LABEL_COUPLING.md) — metrics and interpretation for label coupling outputs.

## Typical outputs

- `outputs/<model_id>/03_appraisal_structure/` — CSVs, heatmaps, `summary.md`, `label_coupling/`, optional `pca_eda/`.

CLI: `python -m pipeline.appraisal_structure`, `python -m pipeline.appraisal_label_coupling`, `python -m pipeline.representation_analysis`.

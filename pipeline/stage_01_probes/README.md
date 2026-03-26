# Stage 1 — Probes

Train and evaluate **emotion** and **appraisal** linear probes across internal model sites, plus optional robustness and grid ablation studies.

## Modules

- `probe_training_robustness` — dataset variants and robustness stats.
- `train_probes` — binary OvA emotion probes.
- `train_appraisal_probes` — Ridge regressors on appraisal dimensions.
- `train_appraisal_binary_probes` — median-split binary OvA appraisal probes.
- `run_probe_grid_ablation` / `analyze_probe_grid_ablation` — optional layer/loc grid search.

## Typical outputs

- `outputs/<model_id>/01_probe_robustness/`
- `outputs/<model_id>/01_probes/`
- `outputs/<model_id>/01_probes_grid_ablation/` (if run)

CLI entry points stay at the package root, e.g. `python -m pipeline.train_probes`.

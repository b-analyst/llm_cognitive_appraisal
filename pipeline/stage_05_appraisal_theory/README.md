# Stage 5 — Appraisal theory

Analyses aimed at **cognitive appraisal theory** claims: layer onset, reconstruction, cross-layer prediction, appraisal space geometry, and related CSV / HTML summaries.

## Modules

- `appraisal_theory` — orchestrates theory-facing runs into `04_appraisal_theory/`.
- `theory_analysis` — shared plotting and export helpers (also used from benchmarks / phase 2).
- `appraisal_overlap_diagnostic` — optional overlap diagnostics between probe directions.

## Typical outputs

- `outputs/<model_id>/04_appraisal_theory/` — multiple analysis CSVs, figures, `SUMMARY.md`.

See also repo doc `docs/APPRAISAL_THEORY.md`.

CLI: `python -m pipeline.appraisal_theory`, `python -m pipeline.appraisal_overlap_diagnostic`.

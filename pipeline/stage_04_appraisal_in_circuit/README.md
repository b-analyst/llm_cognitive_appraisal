# Stage 4 — Appraisal in circuit

Test whether **appraisal geometry** aligns with **emotion directions** at circuit sites, whether circuit-site appraisal readouts match default-layer readouts, and whether **ablating appraisal** changes the circuit’s emotion readout.

## Modules

- `phase2_compute` — geometry, correlation, cache-based ablations (invokes theory helpers where needed).
- `phase2_summary` — writes `04_appraisal_in_circuit/SUMMARY.md`; runs compute if outputs are missing.
- `phase2_isolate_directions` — optional direction isolation utilities.

## Typical outputs

- `outputs/<model_id>/04_appraisal_in_circuit/` — `geometry_circuit_layers.csv`, `correlation_circuit_vs_default.csv`, `appraisal_ablation_summary.csv`, `SUMMARY.md`.

CLI: `python -m pipeline.phase2_compute`, `python -m pipeline.phase2_summary`, `python -m pipeline.phase2_isolate_directions`.

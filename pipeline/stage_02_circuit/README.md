# Stage 2 — Circuit evidence and definition

Select **top-k internal sites per emotion** using probe evidence, compare multi-site fusion to single-site baselines, cache hidden states for later stages, and write reusable **circuit definition** files.

## Modules

- `circuit_evidence` — selection split, auto-k, test metrics, hidden-state caches.
- `phase1_circuits` — `circuits.json` / `circuit_sites.json` for downstream code.

## Typical outputs

- `outputs/<model_id>/02_circuit/` — `circuit_top_k_selection.json`, classification tables/plots, `selection_hidden_states.pt`, `test_hidden_states.pt`, etc.

CLI: `python -m pipeline.circuit_evidence`, `python -m pipeline.phase1_circuits`.

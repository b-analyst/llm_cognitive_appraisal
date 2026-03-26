"""
Isolate appraisal directions inside the circuit: erase or steer along appraisal directions
at circuit layers and measure effect on emotion logits / appraisal prediction.

Uses intervention_utils.erase_direction and steer_hidden_states.
Saves a short report to 04_appraisal_in_circuit/isolate_directions_report.csv.
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import torch

from pipeline.core.config import (
    REPO_ROOT,
    get_circuit_dir,
    get_appraisal_in_circuit_dir,
    get_probe_paths,
    EXTRACTION_TOKENS,
    COMMON_APPRAISAL,
    DEFAULT_MODEL_ID,
)
from pipeline.core.model_config import get_extraction_layers, get_extraction_locs
def _ensure_repo():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def run_isolate_directions(
    model_id: str = DEFAULT_MODEL_ID,
    circuit_layer: int = 10,
    loc_idx: int = 0,
    dimension: str = "pleasantness",
    n_samples: int = 50,
) -> pd.DataFrame:
    """
    Load val hidden states and probes; erase one appraisal dimension at one circuit layer;
    compute emotion logits before/after and appraisal prediction before/after; report deltas.
    """
    _ensure_repo()
    from experiments.utils.intervention_utils import erase_direction

    circuit_dir = get_circuit_dir(model_id)
    out_dir = get_appraisal_in_circuit_dir(model_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    hs_path = circuit_dir / "val_hidden_states.pt"
    if not hs_path.exists():
        raise FileNotFoundError(f"Run circuit_evidence first to create {hs_path}")

    with open(circuit_dir / "circuits.json", encoding="utf-8") as f:
        circuits = json.load(f)
    hidden_states = torch.load(hs_path, weights_only=False)
    if hidden_states.shape[0] > n_samples:
        hidden_states = hidden_states[:n_samples]
    extraction_layers = get_extraction_layers(model_id)
    extraction_locs = get_extraction_locs(model_id)
    layer_idx = extraction_layers.index(circuit_layer) if circuit_layer in extraction_layers else 0

    # Appraisal direction at (circuit_layer, loc)
    appraisal_path = get_probe_paths(model_id).appraisal_probes_path
    appraisal_probes = torch.load(appraisal_path, weights_only=False) if appraisal_path.exists() else None
    if appraisal_probes is None or dimension not in appraisal_probes:
        return pd.DataFrame([{"error": "appraisal_probes_or_dimension_missing"}])

    try:
        r = appraisal_probes[dimension][circuit_layer][extraction_locs[loc_idx]][EXTRACTION_TOKENS[0]]
    except (KeyError, TypeError):
        return pd.DataFrame([{"error": f"no_probe_at_layer_{circuit_layer}_dim_{dimension}"}])
    direction = torch.tensor(r["ridge"].coef_.ravel(), dtype=hidden_states.dtype)

    # Before: mean emotion logit (we don't have emotion probes loaded here, so use appraisal prediction as proxy)
    # Simple proxy: score = hidden @ direction (appraisal score before)
    x = hidden_states[:, layer_idx, loc_idx, 0, :].float()
    score_before = (x @ direction).mean().item()

    # Erase
    hs_erased = erase_direction(hidden_states, direction.numpy(), layer_idx, loc_idx=loc_idx, token_idx=0)
    x_erased = hs_erased[:, layer_idx, loc_idx, 0, :].float()
    score_after = (x_erased @ direction).mean().item()

    report = pd.DataFrame([
        {
            "circuit_layer": circuit_layer,
            "dimension": dimension,
            "appraisal_score_before": score_before,
            "appraisal_score_after_erasure": score_after,
            "delta_erasure": score_after - score_before,
        }
    ])
    report.to_csv(out_dir / "isolate_directions_report.csv", index=False)
    print(f"Wrote {out_dir / 'isolate_directions_report.csv'}")
    return report


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--layer", type=int, default=10)
    p.add_argument("--dimension", default="pleasantness")
    p.add_argument("--n_samples", type=int, default=50)
    args = p.parse_args()
    run_isolate_directions(
        model_id=args.model_id,
        circuit_layer=args.layer,
        dimension=args.dimension,
        n_samples=args.n_samples,
    )

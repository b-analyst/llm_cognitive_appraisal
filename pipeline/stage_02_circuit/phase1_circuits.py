"""
Phase 1: Produce circuits.json from probe performance (per model).

Primary path: read circuit_top_k_selection.json from 02_circuit/ (written by
circuit_evidence) and map each emotion's selected (layer, loc) pairs to an ordered
list of unique circuit layers.

Fallback (e.g. circuit_evidence not run yet): build from probe_summary.csv for this
model via get_probe_paths(model_id), using a probe-summary heuristic (penalized mean
test_roc_auc) capped at CIRCUIT_TOP_K_MAX — not a fixed k for every emotion.

Writes circuits.json to pipeline/outputs/<model_id>/02_circuit/ and optionally
experiments/outputs/phase_1_circuits/<model_id>_circuits.json.
"""
from pathlib import Path
import json
import pandas as pd

from pipeline.core.config import (
    get_probe_paths,
    CIRCUIT_TOP_K_MAX,
    get_circuit_dir,
    PHASE1_CIRCUITS_DIR_LEGACY,
    DEFAULT_MODEL_ID,
)
# Must match pipeline.stage_02_circuit.circuit_evidence.CIRCUIT_TOP_K_SELECTION_FILENAME
CIRCUIT_TOP_K_SELECTION_FILENAME = "circuit_top_k_selection.json"


def build_circuits_from_selection(sel: dict) -> dict:
    """emotion -> ordered unique layers from per_emotion_pairs (first occurrence order)."""
    per = sel.get("per_emotion_pairs") or {}
    circuits: dict[str, list[int]] = {}
    for emotion, pair_list in per.items():
        layers_ordered: list[int] = []
        seen: set[int] = set()
        for p in pair_list:
            L = int(p[0])
            if L not in seen:
                seen.add(L)
                layers_ordered.append(L)
        circuits[str(emotion)] = layers_ordered
    return circuits


def build_circuits_from_probe_summary_heuristic(
    probe_summary_path: Path,
    max_k: int = CIRCUIT_TOP_K_MAX,
    penalty: float = 0.01,
) -> dict:
    """
    Per emotion: rank layers by mean test_roc_auc over locs; choose k in 1..max_k that
    maximizes mean(top-k layer ROC) - penalty * k (no fixed k).
    """
    if not probe_summary_path.exists():
        raise FileNotFoundError(f"Probe summary not found: {probe_summary_path}")
    df = pd.read_csv(probe_summary_path)
    df["layer"] = df["layer"].astype(int)
    by_emotion_layer = df.groupby(["emotion", "layer"])["test_roc_auc"].mean().reset_index()
    circuits: dict[str, list[int]] = {}
    for emotion in by_emotion_layer["emotion"].unique():
        sub = by_emotion_layer[by_emotion_layer["emotion"] == emotion].sort_values(
            "test_roc_auc", ascending=False
        )
        vals = sub["test_roc_auc"].astype(float).values
        n = min(max_k, len(vals))
        best_k, best_score = 1, -1e9
        for k in range(1, n + 1):
            score = float(vals[:k].mean()) - penalty * k
            if score > best_score:
                best_score = score
                best_k = k
        top_layers = sub.head(best_k)["layer"].astype(int).tolist()
        circuits[str(emotion)] = top_layers
    return circuits


def run_phase1(
    model_id: str = DEFAULT_MODEL_ID,
    probe_summary_path: Path | None = None,
    max_k_fallback: int = CIRCUIT_TOP_K_MAX,
    write_legacy_path: bool = True,
    ignore_selection_file: bool = False,
) -> dict:
    """
    Build circuits and write circuits.json. Prefers circuit_top_k_selection.json when present.
    """
    probe_summary_path = probe_summary_path or get_probe_paths(model_id).probe_summary_csv
    circuit_dir = get_circuit_dir(model_id)
    sel_path = circuit_dir / CIRCUIT_TOP_K_SELECTION_FILENAME

    if not ignore_selection_file and sel_path.exists():
        with open(sel_path, encoding="utf-8") as f:
            sel = json.load(f)
        circuits = build_circuits_from_selection(sel)
        pair_sites = sel.get("per_emotion_pairs") or {}
        source = "circuit_top_k_selection.json"
        top_k_note = "layers from auto/fixed (layer,loc) selection"
    else:
        circuits = build_circuits_from_probe_summary_heuristic(
            probe_summary_path, max_k=max_k_fallback
        )
        pair_sites = {}
        source = "probe_summary_heuristic"
        top_k_note = f"heuristic cap max_k={max_k_fallback}"

    out_dir = circuit_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "circuits.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(circuits, f, indent=2)
    if pair_sites:
        with open(out_dir / "circuit_sites.json", "w", encoding="utf-8") as f:
            json.dump(pair_sites, f, indent=2)
    (out_dir / "summary.md").write_text(
        f"# Circuit (Phase 1)\n\ncircuits.json: {len(circuits)} emotions ({source}). {top_k_note}. "
        f"{'circuit_sites.json preserves exact (layer, loc) sites.' if pair_sites else ''}\n",
        encoding="utf-8",
    )
    print(f"Wrote {out_file} ({len(circuits)} emotions, source={source})")

    if write_legacy_path:
        PHASE1_CIRCUITS_DIR_LEGACY.mkdir(parents=True, exist_ok=True)
        legacy_file = PHASE1_CIRCUITS_DIR_LEGACY / f"{model_id}_circuits.json"
        with open(legacy_file, "w", encoding="utf-8") as f:
            json.dump(circuits, f, indent=2)
        print(f"Wrote legacy {legacy_file}.")

    return circuits


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p.add_argument(
        "--ignore_selection_file",
        action="store_true",
        help="Ignore circuit_top_k_selection.json; use probe-summary heuristic only",
    )
    p.add_argument(
        "--max_k_fallback",
        type=int,
        default=CIRCUIT_TOP_K_MAX,
        help="Max layers per emotion for heuristic fallback",
    )
    args = p.parse_args()
    run_phase1(
        model_id=args.model_id,
        max_k_fallback=args.max_k_fallback,
        ignore_selection_file=args.ignore_selection_file,
    )

"""
Model architecture registry: per-model extraction layers, locs, and VRAM estimates.
Used to auto-configure pipeline runs and skip models that exceed available VRAM.
"""
from __future__ import annotations

import os
from pathlib import Path

# Import config for fallback and EXTRACTION_TOKENS
from .config import EXTRACTION_LAYERS, EXTRACTION_LOCS, EXTRACTION_TOKENS
# Keys must match experiments.utils.training_utils.OvaLogisticRegressionTrainer.AVAILABLE_HF_MODELS
MODEL_ARCH_REGISTRY: dict[str, dict] = {
    "Llama3.2_1B": {"num_layers": 16, "extraction_locs": [3, 6, 7], "estimated_vram_gb": 6.0, "extraction_batch_size": 32, "behavioral_batch_size": 8},
    "Llama3.1_8B": {"num_layers": 32, "extraction_locs": [3, 6, 7], "estimated_vram_gb": 20.0, "extraction_batch_size": 8, "behavioral_batch_size": 2},
    "Gemma2_2B": {"num_layers": 26, "extraction_locs": [3, 6, 7], "estimated_vram_gb": 8.0, "extraction_batch_size": 24, "behavioral_batch_size": 6},
    "Gemma2_9B": {"num_layers": 42, "extraction_locs": [3, 6, 7], "estimated_vram_gb": 22.0, "extraction_batch_size": 6, "behavioral_batch_size": 2},
    "Phi3_4B": {"num_layers": 32, "extraction_locs": [3, 6, 7], "estimated_vram_gb": 12.0, "extraction_batch_size": 16, "behavioral_batch_size": 4},
    "Phi3_14B": {"num_layers": 40, "extraction_locs": [3, 6, 7], "estimated_vram_gb": 32.0, "extraction_batch_size": 4, "behavioral_batch_size": 1},
    "Mistral_8B": {"num_layers": 32, "extraction_locs": [3, 6, 7], "estimated_vram_gb": 18.0, "extraction_batch_size": 8, "behavioral_batch_size": 2},
    "Mistral_12B": {"num_layers": 36, "extraction_locs": [3, 6, 7], "estimated_vram_gb": 26.0, "extraction_batch_size": 4, "behavioral_batch_size": 1},
    "OLMo2_7B": {"num_layers": 32, "extraction_locs": [3, 6, 7], "estimated_vram_gb": 16.0, "extraction_batch_size": 8, "behavioral_batch_size": 2},
    "OLMo2_13B": {"num_layers": 40, "extraction_locs": [3, 6, 7], "estimated_vram_gb": 28.0, "extraction_batch_size": 4, "behavioral_batch_size": 1},
}


def get_extraction_layers(model_id: str) -> list[int]:
    """Return extraction layer indices for this model. Fallback to config default if unknown."""
    if model_id in MODEL_ARCH_REGISTRY:
        n = MODEL_ARCH_REGISTRY[model_id]["num_layers"]
        return list(range(n))
    return list(EXTRACTION_LAYERS)


def get_extraction_locs(model_id: str) -> list:
    """Return extraction location indices for this model. Fallback to config default if unknown."""
    if model_id in MODEL_ARCH_REGISTRY:
        return list(MODEL_ARCH_REGISTRY[model_id]["extraction_locs"])
    return list(EXTRACTION_LOCS)


def get_estimated_vram_gb(model_id: str) -> float | None:
    """Return estimated VRAM in GB for this model, or None if unknown."""
    if model_id in MODEL_ARCH_REGISTRY:
        return MODEL_ARCH_REGISTRY[model_id]["estimated_vram_gb"]
    return None


def get_extraction_batch_size(model_id: str) -> int:
    """Return a model-aware hidden-state extraction batch size, overridable by env."""
    env_bs = os.environ.get("PIPELINE_EXTRACTION_BATCH_SIZE")
    if env_bs is not None:
        try:
            return max(1, int(env_bs))
        except ValueError:
            pass
    if model_id in MODEL_ARCH_REGISTRY:
        return int(MODEL_ARCH_REGISTRY[model_id].get("extraction_batch_size", 8))
    return 8


def get_behavioral_batch_size(model_id: str) -> int:
    """Return a conservative batch size for real forward-pass steering, overridable by env."""
    env_bs = os.environ.get("PIPELINE_BEHAVIORAL_BATCH_SIZE")
    if env_bs is not None:
        try:
            return max(1, int(env_bs))
        except ValueError:
            pass
    if model_id in MODEL_ARCH_REGISTRY:
        return int(MODEL_ARCH_REGISTRY[model_id].get("behavioral_batch_size", 2))
    return 2


def get_default_probe_n_jobs() -> int:
    """Return a sensible CPU parallelism default for probe fitting."""
    env_jobs = os.environ.get("PIPELINE_PROBE_N_JOBS")
    if env_jobs is not None:
        try:
            return max(1, int(env_jobs))
        except ValueError:
            pass
    cpu_total = os.cpu_count() or 4
    # Leave some headroom for tokenization / OS / BLAS bookkeeping.
    return max(1, min(16, cpu_total - 2))


def _get_available_vram_gb() -> float | None:
    """Return available GPU VRAM in GB, or None if CUDA unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:
        pass
    return None


def get_models_within_vram(max_vram_gb: float | None = None) -> list[str]:
    """
    Return model_ids from the registry that fit within the given VRAM (GB).
    If max_vram_gb is None: use PIPELINE_MAX_VRAM_GB env, then try CUDA total memory.
    If still None (no CUDA / no env), return all registered models.
    """
    if max_vram_gb is None:
        env_gb = os.environ.get("PIPELINE_MAX_VRAM_GB")
        if env_gb is not None:
            try:
                max_vram_gb = float(env_gb)
            except ValueError:
                pass
        if max_vram_gb is None:
            max_vram_gb = _get_available_vram_gb()
    if max_vram_gb is None:
        return list(MODEL_ARCH_REGISTRY.keys())
    result = []
    for mid, cfg in MODEL_ARCH_REGISTRY.items():
        need = cfg.get("estimated_vram_gb")
        if need is not None and need <= max_vram_gb:
            result.append(mid)
    return result


def probe_filename_suffix(model_id: str) -> str:
    """Return a filesystem-safe layers/locs/tokens suffix for binary_ova_probes_*.pt filename."""
    layers = get_extraction_layers(model_id)
    locs = get_extraction_locs(model_id)
    tokens = list(EXTRACTION_TOKENS)
    safe_model = "".join(ch if ch.isalnum() else "_" for ch in str(model_id))
    return f"{safe_model}_{_probe_suffix_safe(layers, locs, tokens)}"


def _probe_suffix_safe(layers: list, locs: list, tokens: list) -> str:
    """
    Filesystem-safe suffix (no brackets/commas) for Windows and other OSes.

    Must stay in sync with experiments.utils.training_utils._probe_suffix_safe.
    """
    if len(layers) == 0:
        layer_part = "layers_unknown"
    else:
        layer_part = f"layers_{layers[0]}-{layers[-1]}"
    loc_part = f"locs_{'_'.join(str(x) for x in locs)}" if locs else "locs_unknown"
    tok_part = f"tokens_{'_'.join(str(t) for t in tokens)}" if tokens else "tokens_unknown"
    return f"{layer_part}_{loc_part}_{tok_part}"

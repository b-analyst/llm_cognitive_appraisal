"""
Intervention utilities for mechanistic interpretability on extracted hidden states.

Operates on tensors of shape (batch, n_layers, n_locs, n_tokens, hidden_size)
as returned by extract_hidden_states / OvaProbeInference.extract_hidden_states.
Model-agnostic: no hooks or model calls; all interventions are tensor operations.

Used for: activation patching, steering, concept erasure, and causal mediation.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import List, Optional, Union


def patch_hidden_states(
    hs_target: torch.Tensor,
    hs_source: torch.Tensor,
    layers_to_patch: List[int],
    extraction_layers: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    Replace activations at specified layers in the target with those from the source.
    Used for: causal tracing, layer attribution, and "hold mediator fixed" mediation.

    Args:
        hs_target: (batch_t, n_layers, n_locs, n_tokens, dim) target run.
        hs_source: (batch_s, n_layers, n_locs, n_tokens, dim) source run.
            If batch sizes differ, source is broadcast (e.g. one source for many targets).
        layers_to_patch: layer indices (0-based into the tensor) to replace.
        extraction_layers: if given, layers_to_patch are logical layer ids;
            they are mapped to tensor indices. If None, layers_to_patch are tensor indices.

    Returns:
        New tensor same shape as hs_target with patched layers.
    """
    out = hs_target.clone()
    if extraction_layers is not None:
        layer_indices = [extraction_layers.index(l) for l in layers_to_patch if l in extraction_layers]
    else:
        layer_indices = [l for l in layers_to_patch if 0 <= l < out.shape[1]]
    for li in layer_indices:
        # Broadcast source to target batch if needed
        src = hs_source if hs_source.shape[0] == out.shape[0] else hs_source[[0]].expand(out.shape[0], -1, -1, -1, -1)
        out[:, li, :, :, :] = src[:, li, :, :, :].to(out.device)
    return out


def steer_hidden_states(
    hs: torch.Tensor,
    steering_vector: Union[torch.Tensor, np.ndarray],
    alpha: float,
    layer_idx: int,
    loc_idx: int = 0,
    token_idx: int = 0,
) -> torch.Tensor:
    """
    Add a steering vector to hidden states at one (layer, loc, token) position.
    steering_vector: (hidden_size,) or (1, 1, 1, 1, hidden_size).
    Used for: activation steering (ActAdd-style) to test causal effect of a direction.
    """
    out = hs.clone()
    v = steering_vector if isinstance(steering_vector, torch.Tensor) else torch.tensor(steering_vector, dtype=out.dtype, device=out.device)
    v = v.reshape(-1)
    # Slice has shape (batch, hidden_size); v (hidden_size,) broadcasts correctly to add same vector to each row
    out[:, layer_idx, loc_idx, token_idx, :] = out[:, layer_idx, loc_idx, token_idx, :] + alpha * v.to(out.device)
    return out


def compute_steering_vector(
    hs_high: torch.Tensor,
    hs_low: torch.Tensor,
    layer_idx: int,
    loc_idx: int = 0,
    token_idx: int = 0,
    aggregate: str = "mean",
) -> torch.Tensor:
    """
    Compute steering vector as difference between two sets of activations at one position.
    hs_high: (batch_high, n_layers, n_locs, n_tokens, dim) e.g. "high other-responsibility".
    hs_low: (batch_low, n_layers, ...) e.g. "low other-responsibility".
    aggregate: "mean" -> mean(high) - mean(low); "diff_mean" same.
    Returns: (hidden_size,) vector.
    """
    h = hs_high[:, layer_idx, loc_idx, token_idx, :].float().mean(dim=0)
    l = hs_low[:, layer_idx, loc_idx, token_idx, :].float().mean(dim=0)
    return (h - l).to(hs_high.device)


def erase_direction(
    hs: torch.Tensor,
    direction: Union[torch.Tensor, np.ndarray],
    layer_idx: int,
    loc_idx: int = 0,
    token_idx: int = 0,
) -> torch.Tensor:
    """
    Remove the component of hidden states along the given direction (projection onto direction).
    LEACE-style linear erasure: out = x - (x·d/||d||^2) * d.
    direction: (hidden_size,) e.g. Ridge probe coefficient for one appraisal dimension.
    """
    out = hs.clone()
    d = direction if isinstance(direction, torch.Tensor) else torch.tensor(direction, dtype=out.dtype, device=out.device)
    d = d.reshape(-1).float().to(out.device)
    x = out[:, layer_idx, loc_idx, token_idx, :].float()
    # Project onto d: proj_d(x) = (x @ d / (d @ d)) * d
    d_sq = (d @ d).clamp(min=1e-8)
    coef = (x @ d) / d_sq
    out[:, layer_idx, loc_idx, token_idx, :] = (x - coef.unsqueeze(1) * d.unsqueeze(0)).to(out.dtype)
    return out


def mediation_hold_mediator_fixed(
    hs_treatment: torch.Tensor,
    hs_neutral: torch.Tensor,
    mediator_layer_idx: int,
) -> torch.Tensor:
    """
    Replace the representation at the mediator layer in the treatment run with that from the neutral run.
    Used for causal mediation: NDE (natural direct effect) style — effect of text when mediator is fixed.
    hs_treatment: hidden states for the "treatment" sentence (e.g. high other-responsibility).
    hs_neutral: hidden states for a neutral sentence (same batch size or single batch).
    mediator_layer_idx: which layer to treat as the mediator (0-based index into tensor).
    Returns: new tensor, same shape as hs_treatment.
    """
    return patch_hidden_states(
        hs_treatment,
        hs_neutral,
        layers_to_patch=[mediator_layer_idx],
        extraction_layers=None,
    )

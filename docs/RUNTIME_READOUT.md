# Runtime emotion readout (generation behavior + mental health)

This document is the **single reference** for how `RUNTIME_READOUT_*` settings turn hidden states into a **ranked emotion list** and optional JSON logs. Implementation: `pipeline/runtime_emotion_readout.py`.

## Why two “circuit” score spaces?

| Branch | Per site | Across sites for emotion *e* | Typical scale | Use when… |
|--------|----------|-------------------------------|---------------|-----------|
| **Primary (circuit-evidence aligned)** | Linear OvA logit for class *e* → **σ(logit)** | **Mean** of those σ values | ≈ [0, 1] per emotion after fusion | You want runtime rank-1 to match **`circuit_evidence`** / **`_circuit_logits`** / steering-style probe fusion. |
| **Auxiliary (`circuit_linear_mean` or aux columns)** | Linear OvA logit for *e* (no σ) | **Mean** of linear values | Margin / logit space | You care about **raw separability** or want ranks comparable to “pre-sigmoid” probe training objectives. |

**Important:** Both are still **one score per emotion** (multilabel OvA vector). The mean is only over **spatial sites** for the **same** emotion’s head — not mixing different emotions’ probes.

## ASCII: primary sigmoid-mean circuit (default)

```
For each unique (layer, loc) in union of topk_per_emotion[*]:
        h(layer,loc)
              |
              v
    linear OvA vector  -->  take component for emotion e  -->  z_e
              |
              v
           sigma(z_e)   =  p_e,site
              |
              +-------- accumulate for emotion e
              |
              v
    fused_e = mean_sites( p_e,site )
```

Rank-1 / top-k: same **top-k + margin** policy as `BASELINE_PROBE_STUDY_*`, applied to the **fused** vector (`RUNTIME_READOUT_TOP_K`, `RUNTIME_READOUT_MIN_MARGIN`, `RUNTIME_READOUT_MIN_TOP1_LOGIT`). Thresholds are evaluated on the **same numeric scale** as the ranked scores (σ-mean vs linear-mean).

## ASCII: auxiliary linear-mean circuit

```
For each unique (layer, loc):
        h(layer,loc)  -->  linear OvA  -->  z_e  -->  (no sigma)
                              |
                              v
              fused_e = mean_sites( z_e )
```

## Modes (`RUNTIME_READOUT_EMOTION_MODE`)

| Config value | Sites | Fusion | Notes |
|--------------|-------|--------|-------|
| `circuit_sigmoid_mean` | `topk_per_emotion` union | σ then mean | **Default** in `config.py`. Aliases: `circuit_evidence`, `circuit`, `circuit_mean`. |
| `single_site` | probe-summary best `(layer,loc)` | σ(linear) only | No circuit union; `runtime_readout_union_n_sites` = 1. |
| `circuit_linear_mean` | `topk_per_emotion` union | mean linear | Optional **primary** if you explicitly set this mode. |

## Auxiliary CSV columns (linear circuit alongside primary σ-mean)

When primary mode is a **sigmoid-mean** circuit family and **`RUNTIME_READOUT_LOG_LINEAR_CIRCUIT_AUX = True`**:

- `runtime_linear_circuit_rank1_emotion`
- `runtime_linear_circuit_ranked_top_k_json` (if `RUNTIME_READOUT_LOG_RANK_JSON`) — JSON includes `readout_role: auxiliary_linear_circuit`, `readout_mode: circuit_linear_mean`, `score_kind: linear_mean_fused`, and may include `all_emotions_linear_scores_json` when full spectrum is enabled.

Generation behavior and mental health steering both emit these columns for parity.

## JSON fields (`ranked_top_k_json`)

Stable keys:

- `readout_mode` — string from config
- `score_kind` — `sigmoid_mean_fused` \| `sigmoid_single_site` \| `linear_mean_fused`
- `ranked_top_k` — list of `{ "emotion", "logit" }` (**`logit` is legacy key**; value is the **score** in the active space, not always a logit)
- `union_n_sites` — optional
- `all_emotions_scores_json` — optional truncated spectrum for the **primary** readout (sorted `{emotion, score}` list)

Auxiliary JSON (second column) may use `all_emotions_linear_scores_json` for the linear branch.

## Related code

- Fusion: `circuit_sigmoid_mean_fused_scores`, `circuit_linear_mean_fused_scores`
- Ranking: `readout_ranked_emotions_auto`, `readout_ranked_emotions_circuit_sigmoid_mean`, `readout_ranked_emotions_circuit_linear_mean`
- Call sites: `pipeline/generation_behavior_benchmark.py`, `pipeline/mental_health_steering_benchmark.py`

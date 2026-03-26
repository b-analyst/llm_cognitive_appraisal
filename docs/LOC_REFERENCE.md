# Extraction Locations: Where Are We Reading From?

This document explains what a **loc** (location) is inside a transformer model, why we pick the specific ones we do, and what it means to "read from" or "steer at" each point.

See also:
- `docs/GLOSSARY.md` for definitions of layer, token, and probe
- `docs/EXPERIMENTAL_SETUP.md` for how locs are configured per model
- `pipeline/model_config.py` for the exact loc list per model

---

## The Short Version

A transformer layer is not one single thing. It is a pipeline of subcomputations. The **loc** number says *which step inside that pipeline* we are tapping into.

All models in this pipeline currently use **locs 3, 6, and 7**.

- **loc 3** = after the attention block
- **loc 6** = after the feedforward (MLP) block
- **loc 7** = after the full layer is done (residual output)

---

## One Transformer Layer, Step by Step

Every transformer block processes the same internal vector in roughly the same order. Here is what that looks like in plain English:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          ONE TRANSFORMER LAYER                          │
│                                                                         │
│  Input from previous layer                                              │
│        │                                                                │
│        ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  SELF-ATTENTION BLOCK                                           │   │
│  │  (the model decides which other tokens in the sequence          │   │
│  │   are important to look at for this token)                      │   │
│  └────────────────────────────────┬────────────────────────────────┘   │
│                                   │                                     │
│                    ◄── loc 3 ─────┘                                     │
│           (output of attention, BEFORE adding back to stream)          │
│                                   │                                     │
│                                   ▼                                     │
│          add attention output back into the main residual stream        │
│                                   │                                     │
│                                   ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  FEEDFORWARD BLOCK (MLP)                                        │   │
│  │  (the model does nonlinear transformation to mix and process    │   │
│  │   what was learned from attention)                              │   │
│  └────────────────────────────────┬────────────────────────────────┘   │
│                                   │                                     │
│                    ◄── loc 6 ─────┘                                     │
│           (output of MLP, BEFORE adding back to stream)                │
│                                   │                                     │
│                                   ▼                                     │
│          add MLP output back into the main residual stream              │
│                                   │                                     │
│                    ◄── loc 7 ─────┘                                     │
│           (full layer output, AFTER both blocks are done)              │
│                                   │                                     │
│                                   ▼                                     │
│  Output to next layer                                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## What Each Loc Means

### loc 3 — After Attention

**In the code:** `model.layers.[L].self_attn` forward output

**Plain English:**
This is the moment right after the model has decided what to *pay attention to* in the sequence. The attention block figures out "which parts of what I've seen are relevant to this token right now." The output at loc 3 captures that relational, context-sensitive signal, before it is merged back into the main information stream.

**Why we read here:**
Attention is where the model builds its context-sensitive interpretation. If the model understands emotional valence or social relationship between the current token and earlier parts of the conversation, that may be visible here.

---

### loc 6 — After Feedforward (MLP)

**In the code:** `model.layers.[L].mlp` forward output

**Plain English:**
After attention, the feedforward network processes what was found. It adds nonlinear transformation and is thought to be where much of the model's "knowledge" is stored. The output at loc 6 captures the MLP's contribution before it is merged back.

**Why we read here:**
The MLP is often where semantic and conceptual content lives. If appraisal-style information (control, blame, urgency, certainty) is stored as a learnable concept in the model, it may be most accessible here.

---

### loc 7 — Full Layer Output (Residual Stream)

**In the code:** `model.layers.[L]` forward output

**Plain English:**
This is the final output of the complete transformer layer: attention + feedforward both done, both added back into the main information flow. It represents everything the model has computed up to and including this layer.

**Why we read here:**
Loc 7 is the "summary" of what a full layer contributed. It is the most integrated view available inside a layer and is the most commonly used extraction point in mechanistic interpretability research.

---

## Why These Three Specifically

Using locs 3, 6, and 7 together lets the pipeline ask:

- Did the emotion/appraisal signal come from the attention step? (loc 3)
- Did it come from the feedforward memory/knowledge step? (loc 6)
- What is the combined signal after both? (loc 7)

This lets the probe analysis distinguish *where inside a block* the signal is strongest, not just *which layer* it comes from.

---

## Full Location Numbering Reference

For completeness, here is the full set of named locations defined in `utils.py`:

| loc | Name | What it is |
|-----|------|------------|
| 1 | `hook_initial_hs` | Input to the layer before any processing |
| 2 | `hook_after_attn_normalization` | After layer norm before attention |
| **3** | `hook_after_attn` | **After the self-attention block** |
| 4 | `hook_after_attn_hs` | After attention + residual add |
| 5 | `hook_after_mlp_normalization` | After layer norm before MLP |
| **6** | `hook_after_mlp` | **After the feedforward (MLP) block** |
| **7** | `hook_after_mlp_hs` | **After MLP + residual add (full layer output)** |
| 8 | `self_attn.hook_attn_heads` | Individual attention head outputs |
| 9 | `model.final_hook` | After the final layer |
| 10 | `hook_attn_weights` | Raw attention weight matrices |

Bolded rows are the ones this pipeline uses by default.

---

## A Simple Analogy for Non-Technical Readers

Imagine reading a sentence with the goal of understanding the speaker's emotional state.

**Step 1 (attention):**
You scan back through the conversation to see which earlier words or context seem most relevant right now. You notice the speaker said "threatened" two sentences ago. That changes how you read the current sentence.

→ *loc 3 reads the output of that scanning step.*

**Step 2 (feedforward):**
You apply your stored knowledge and experience. You recognize that "threatened" combined with the current context implies fear and loss of control. This is your semantic processing, your "memory" of how language works.

→ *loc 6 reads the output of that knowledge application step.*

**Step 3 (layer output):**
You now have a fully updated understanding of the sentence given both context and knowledge. This is your complete interpretation of this passage.

→ *loc 7 reads that final integrated result.*

---

## How Many Extraction Points Does That Give?

For a given model, the total number of extraction sites is:

```
sites = num_layers × num_locs × num_tokens
```

For `Llama3.2_1B`:
- 16 layers × 3 locs × 1 token = **48 sites**

For `Llama3.1_8B`:
- 32 layers × 3 locs × 1 token = **96 sites**

Each site gets its own separately trained emotion probe and appraisal regressor. The pipeline's job is to find which sites carry the strongest emotion and appraisal signal.

---

## What "Steering at a Loc" Means

When we steer or ablate at a given `(layer, loc)`, we are:
- intercepting the hidden state vector at that exact point
- adding or removing a direction before the computation continues

For example, if we steer at `(layer 7, loc 3)`, we are modifying the output of the attention block in layer 7 before it gets added back into the residual stream and before the MLP runs.

This is important because it means the intervention happens *within* the layer's computation, not just between layers.

---

## Model-Specific Layer Counts

Different models have different depths. The same loc is used across all of them, but the total number of layers changes:

| Model | Layers | Locs | Sites |
|-------|--------|------|-------|
| Llama3.2_1B | 16 | 3, 6, 7 | 48 |
| Llama3.1_8B | 32 | 3, 6, 7 | 96 |
| Gemma2_2B | 26 | 3, 6, 7 | 78 |
| Gemma2_9B | 42 | 3, 6, 7 | 126 |
| Phi3_4B | 32 | 3, 6, 7 | 96 |
| Phi3_14B | 40 | 3, 6, 7 | 120 |
| Mistral_8B | 32 | 3, 6, 7 | 96 |
| OLMo2_7B | 32 | 3, 6, 7 | 96 |
| OLMo2_13B | 40 | 3, 6, 7 | 120 |

The circuit-selection step then picks a small subset of these sites that actually carry strong emotion signal.

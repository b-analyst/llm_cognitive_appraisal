"""
Baseline probing (prompted vs unprompted) + top-k emotion-ranked appraisal steering.

Implements the study design in docs/BASELINE_PROBE_STEERING_STUDY.md: frozen top-m appraisal
dimensions per emotion from appraisal_zscore_by_emotion, ridge directions at circuit sites,
multilabel-style ranking with margin, null/specificity controls, and behavior + probe deltas.

This stage is optional: run with `python -m pipeline.baseline_probe_steering_study` or after
prerequisites (probes, circuit, appraisal_structure). Not part of the default full pipeline
runner unless you invoke this module explicitly.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pipeline.core.config import (
    DEFAULT_MODEL_ID,
    GENERATION_BEHAVIOR_BENCHMARK_CSV,
    GENERATION_BEHAVIOR_TAXONOMY_CSV,
    GENERATION_BEHAVIOR_MAX_ROWS,
    EXTRACTION_TOKENS,
    COMMON_APPRAISAL,
    get_probe_paths,
    get_circuit_dir,
    get_appraisal_structure_dir,
    get_baseline_probe_study_dir,
    BASELINE_PROBE_STUDY_PRIMARY_TEXT,
    BASELINE_PROBE_STUDY_PRIMARY_OUTCOME,
    BASELINE_PROBE_STUDY_TOP_K,
    BASELINE_PROBE_STUDY_TOP_M_APPRAISAL_DIMS,
    BASELINE_PROBE_STUDY_RANK_MIN_MARGIN,
    BASELINE_PROBE_STUDY_MIN_TOP1_LOGIT,
    BASELINE_PROBE_STUDY_STEER_ALPHA,
    BASELINE_PROBE_STUDY_STEERING_UNIT_NORM,
    BASELINE_PROBE_STUDY_GEN_INTERVENTION_DURING_DECODE,
    BASELINE_PROBE_STUDY_MAX_ROWS,
    BASELINE_PROBE_STUDY_MAX_NEW_TOKENS,
    BASELINE_PROBE_STUDY_TEMPERATURE,
    BASELINE_PROBE_STUDY_TOP_P,
)
from pipeline.core.model_config import get_extraction_layers, get_extraction_locs
from pipeline.stage_02_circuit.circuit_evidence import load_circuit_top_k_selection, topk_per_emotion_from_selection
from pipeline.stage_06_benchmarks.steering_benchmark import _find_probes_pt, _circuit_logits
from pipeline.core.research_contracts import build_prompted_texts, ensure_manifest_model_match, appraisal_probe_direction_raw
from pipeline.stage_06_benchmarks.generation_behavior_benchmark import (
    _load_behavior_benchmark,
    _rule_based_scores,
    _spec_from_vectors,
    _unit_normalize_steering_vec,
)
from pipeline.stage_06_benchmarks.utils.probe_latent_scoring import (
    emotion_linear_logits_at_site,
    appraisal_ridge_values_at_site,
    top_m_appraisal_dims_for_emotion,
    rank_emotions_top_k_margin,
    resolve_probe_layer_loc,
)
from pipeline.stage_06_benchmarks.utils.runtime_emotion_readout import select_probe_readout_site, hidden_vector_at_site
def _progress(msg: str) -> None:
    print(f"[baseline_probe_steering] {msg}", flush=True)


# Readout site + hidden-at-site: shared with generation/MH benchmarks (runtime_emotion_readout).


def _appraisal_vec_restricted(
    target_emotion: str,
    source_emotion: str | None,
    layer: int,
    loc: int,
    hidden_size: int,
    token_key: int,
    appraisal_probes: dict,
    zscore_df: pd.DataFrame,
    dims_subset: list[str],
) -> np.ndarray:
    """Steer in span of ridge directions for dims_subset; weights = z(target)-z(source)."""
    vec = np.zeros(hidden_size, dtype=np.float32)
    if target_emotion not in zscore_df.index:
        return vec
    used = 0
    for dim in dims_subset:
        if dim not in appraisal_probes:
            continue
        L, lc = resolve_probe_layer_loc(appraisal_probes[dim], layer, loc)
        if L is None or lc is None:
            continue
        try:
            rec = appraisal_probes[dim][L][lc][token_key]
        except (KeyError, TypeError):
            continue
        direction = appraisal_probe_direction_raw(rec, hidden_size=hidden_size)
        if direction is None:
            continue
        zt = float(zscore_df.loc[target_emotion, dim])
        if source_emotion and source_emotion in zscore_df.index:
            zs = float(zscore_df.loc[source_emotion, dim])
        else:
            zs = 0.0
        w = zt - zs
        vec += np.asarray(w, dtype=np.float32) * direction.astype(np.float32)
        used += 1
    if used == 0:
        return np.zeros(hidden_size, dtype=np.float32)
    return vec


def _random_appraisal_mix_vector(
    layer: int,
    loc: int,
    hidden_size: int,
    token_key: int,
    appraisal_probes: dict,
    n_dims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    dims = [d for d in COMMON_APPRAISAL if d in appraisal_probes]
    if len(dims) < n_dims:
        pick = dims
    else:
        pick = list(rng.choice(dims, size=n_dims, replace=False))
    vec = np.zeros(hidden_size, dtype=np.float32)
    for dim in pick:
        L, lc = resolve_probe_layer_loc(appraisal_probes[dim], layer, loc)
        if L is None or lc is None:
            continue
        try:
            rec = appraisal_probes[dim][L][lc][token_key]
        except (KeyError, TypeError):
            continue
        direction = appraisal_probe_direction_raw(rec, hidden_size=hidden_size)
        if direction is None:
            continue
        vec += float(rng.standard_normal()) * direction.astype(np.float32)
    return vec


def _shuffle_weights_vector(
    target_emotion: str,
    source_emotion: str | None,
    layer: int,
    loc: int,
    hidden_size: int,
    token_key: int,
    appraisal_probes: dict,
    zscore_df: pd.DataFrame,
    dims_subset: list[str],
    rng: np.random.Generator,
) -> np.ndarray:
    """Same directions as restricted steer but weights permuted across dims (null structure)."""
    weights = []
    dirs = []
    if target_emotion not in zscore_df.index:
        return np.zeros(hidden_size, dtype=np.float32)
    for dim in dims_subset:
        if dim not in appraisal_probes:
            continue
        L, lc = resolve_probe_layer_loc(appraisal_probes[dim], layer, loc)
        if L is None or lc is None:
            continue
        try:
            rec = appraisal_probes[dim][L][lc][token_key]
        except (KeyError, TypeError):
            continue
        direction = appraisal_probe_direction_raw(rec, hidden_size=hidden_size)
        if direction is None:
            continue
        zt = float(zscore_df.loc[target_emotion, dim])
        if source_emotion and source_emotion in zscore_df.index:
            zs = float(zscore_df.loc[source_emotion, dim])
        else:
            zs = 0.0
        weights.append(zt - zs)
        dirs.append(direction.astype(np.float32))
    if not dirs:
        return np.zeros(hidden_size, dtype=np.float32)
    perm = rng.permutation(len(weights))
    vec = np.zeros(hidden_size, dtype=np.float32)
    for i, j in enumerate(perm):
        vec += float(weights[j]) * dirs[i]
    return vec


def run_baseline_probe_steering_study(
    model_id: str = DEFAULT_MODEL_ID,
    benchmark_csv: str | Path | None = None,
    max_rows: int | None = None,
    primary_text: str | None = None,
    top_k: int | None = None,
    top_m_dims: int | None = None,
    rank_min_margin: float | None = None,
    steer_alpha: float | None = None,
    unit_norm: bool | None = None,
    gen_intervention_during_decode: bool | None = None,
) -> dict:
    """
    Run baseline readouts (prompted + unprompted) and steering conditions on the generation
    behavior benchmark prompts.
    """
    from utils import Log, TextDataset, run_forward_with_steering, generate_with_steering
    from experiments.utils.training_utils import OvaLogisticRegressionTrainer

    primary_text = primary_text or BASELINE_PROBE_STUDY_PRIMARY_TEXT
    if primary_text not in ("prompted", "unprompted"):
        raise ValueError("primary_text must be 'prompted' or 'unprompted'")
    top_k = int(top_k if top_k is not None else BASELINE_PROBE_STUDY_TOP_K)
    top_m = int(top_m_dims if top_m_dims is not None else BASELINE_PROBE_STUDY_TOP_M_APPRAISAL_DIMS)
    rank_min_margin = float(rank_min_margin if rank_min_margin is not None else BASELINE_PROBE_STUDY_RANK_MIN_MARGIN)
    steer_alpha = float(steer_alpha if steer_alpha is not None else BASELINE_PROBE_STUDY_STEER_ALPHA)
    unit_norm = BASELINE_PROBE_STUDY_STEERING_UNIT_NORM if unit_norm is None else unit_norm
    decode_steer = (
        BASELINE_PROBE_STUDY_GEN_INTERVENTION_DURING_DECODE
        if gen_intervention_during_decode is None
        else gen_intervention_during_decode
    )
    max_rows = max_rows if max_rows is not None else BASELINE_PROBE_STUDY_MAX_ROWS

    out_dir = get_baseline_probe_study_dir(model_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    bench_path = Path(benchmark_csv) if benchmark_csv is not None else GENERATION_BEHAVIOR_BENCHMARK_CSV
    if not bench_path.exists():
        summary = f"# 05_baseline_probe_steering\n\nMissing benchmark CSV: `{bench_path}`\n"
        (out_dir / "summary.md").write_text(summary, encoding="utf-8")
        return {"skipped": True, "reason": "missing_benchmark_csv"}

    benchmark_df = _load_behavior_benchmark(bench_path)
    cap = max_rows if max_rows is not None else GENERATION_BEHAVIOR_MAX_ROWS
    if cap is not None and len(benchmark_df) > int(cap):
        benchmark_df = benchmark_df.head(int(cap)).reset_index(drop=True)

    logger = Log("baseline_probe_steering").logger
    trainer = OvaLogisticRegressionTrainer(model_id, logger=logger)
    trainer.load_model_and_tokenizer()
    model, tokenizer = trainer.model, trainer.tokenizer
    if model is None or tokenizer is None:
        raise RuntimeError("Failed to load model/tokenizer.")

    paths = get_probe_paths(model_id)
    ensure_manifest_model_match(paths.probe_manifest_path, model_id, "probe")
    probes = torch.load(_find_probes_pt(paths.probes_dir_v2, model_id), weights_only=False)
    if not paths.appraisal_probes_path.exists():
        (out_dir / "summary.md").write_text(
            "# 05_baseline_probe_steering\n\nMissing appraisal_regression_probes.pt\n", encoding="utf-8"
        )
        return {"skipped": True, "reason": "missing_appraisal_probes"}
    ensure_manifest_model_match(paths.appraisal_manifest_path, model_id, "appraisal probe")
    appraisal_probes = torch.load(paths.appraisal_probes_path, weights_only=False)

    zpath = get_appraisal_structure_dir(model_id) / "appraisal_zscore_by_emotion.csv"
    if not zpath.exists():
        (out_dir / "summary.md").write_text(
            "# 05_baseline_probe_steering\n\nRun appraisal_structure first (appraisal_zscore_by_emotion.csv).\n",
            encoding="utf-8",
        )
        return {"skipped": True, "reason": "missing_zscore_csv"}
    zscore_df = pd.read_csv(zpath, index_col=0)
    zscore_df.index = zscore_df.index.astype(str).str.lower()

    summary_df = pd.read_csv(paths.probe_summary_csv)
    emotions_list = sorted(summary_df["emotion"].astype(str).unique().tolist())
    emotions_list = [e for e in emotions_list if e.lower() != "no-emotion"]
    label_to_idx = {e: i for i, e in enumerate(emotions_list)}
    token_key = EXTRACTION_TOKENS[0]
    token_idx = 0
    extraction_layers = get_extraction_layers(model_id)
    extraction_locs = get_extraction_locs(model_id)

    hidden_size = None
    for layer in probes:
        for loc in probes[layer]:
            if token_key in probes[layer][loc]:
                for rec in probes[layer][loc][token_key].values():
                    if isinstance(rec, dict) and rec.get("weights") is not None:
                        hidden_size = len(np.asarray(rec["weights"]).ravel())
                        break
            if hidden_size is not None:
                break
        if hidden_size is not None:
            break
    if hidden_size is None:
        raise RuntimeError("Could not infer hidden size from probes.")

    read_layer, read_loc = _select_readout_site(summary_df)
    sel = load_circuit_top_k_selection(get_circuit_dir(model_id))
    if sel is None:
        raise FileNotFoundError("Missing circuit selection; run circuit_evidence first.")
    topk_per_emotion = topk_per_emotion_from_selection(sel, emotions_list, summary_df=summary_df)

    # Frozen dim policy: top-m by |z| per emotion (logged with zscore file hash)
    z_blob = zpath.read_bytes()
    z_fingerprint = hashlib.sha256(z_blob).hexdigest()[:16]
    emotion_to_dims: dict[str, list[str]] = {}
    for em in emotions_list:
        emotion_to_dims[str(em).lower()] = top_m_appraisal_dims_for_emotion(
            str(em).lower(), zscore_df, top_m, COMMON_APPRAISAL
        )

    policy_payload = {
        "version": 1,
        "source": "appraisal_zscore_by_emotion_top_abs_z",
        "zscore_csv": str(zpath),
        "zscore_sha256_prefix": z_fingerprint,
        "top_m": top_m,
        "emotion_to_dims": emotion_to_dims,
    }
    (out_dir / "dim_selection_policy.json").write_text(json.dumps(policy_payload, indent=2), encoding="utf-8")

    rng = np.random.default_rng(42)
    readout_rows = []
    run_rows = []

    shots_emotions = sorted(
        {e for e in summary_df["emotion"].astype(str).unique().tolist() if str(e).lower() != "no-emotion"}
    )

    records = benchmark_df.to_dict(orient="records")
    _progress(
        f"model={model_id} prompts={len(records)} readout=({read_layer},{read_loc}) "
        f"primary_text={primary_text} top_k={top_k} top_m={top_m} margin={rank_min_margin}"
    )

    behavioral_batch_size = 1
    for row in tqdm(records, desc="baseline_probe_steering", unit="prompt"):
        prompt_id = str(row.get("prompt_id") or "")
        raw_text = str(row["prompt_text"])
        prompted_text = build_prompted_texts([raw_text], emotions_list=shots_emotions, prompt_index=1)[0]
        texts = {"unprompted": raw_text, "prompted": prompted_text}

        baseline_scores_primary: dict | None = None
        baseline_latent_primary: dict | None = None
        rank1_source: str | None = None
        top_emotions: list[str] = []
        baseline_hs_np: np.ndarray | None = None

        for text_type, ptext in texts.items():
            ds = TextDataset(texts=[ptext], labels=[0])
            dl = DataLoader(ds, batch_size=behavioral_batch_size, shuffle=False)
            hs, _ = run_forward_with_steering(
                dl,
                tokenizer,
                model,
                extraction_layers,
                extraction_locs,
                list(EXTRACTION_TOKENS),
                steering_spec={},
                logger=logger,
                show_progress=False,
            )
            if isinstance(hs, torch.Tensor):
                hs_np = hs.cpu().numpy()
            else:
                hs_np = np.asarray(hs)
            h_vec = hidden_vector_at_site(hs_np, extraction_layers, extraction_locs, read_layer, read_loc, token_idx)
            logits = emotion_linear_logits_at_site(probes, h_vec, read_layer, read_loc, token_key, emotions_list)
            fused = _circuit_logits(
                hs_np, probes, topk_per_emotion, extraction_layers, extraction_locs, emotions_list, token_key, token_idx
            )
            pred_i = int(np.argmax(fused[0]))
            pred_em = emotions_list[pred_i]
            appr = appraisal_ridge_values_at_site(
                appraisal_probes, h_vec, read_layer, read_loc, token_key, COMMON_APPRAISAL
            )
            ranked_e, ranked_v = rank_emotions_top_k_margin(
                logits,
                emotions_list,
                top_k=top_k,
                min_margin=rank_min_margin,
                min_top1_logit=BASELINE_PROBE_STUDY_MIN_TOP1_LOGIT,
            )

            readout_rows.append(
                {
                    "prompt_id": prompt_id,
                    "benchmark_family": row.get("benchmark_family"),
                    "text_type": text_type,
                    "readout_layer": read_layer,
                    "readout_loc": read_loc,
                    "fused_argmax_emotion": pred_em,
                    "ranked_top_k": json.dumps(ranked_e),
                    "ranked_top_k_linear_logits": json.dumps(ranked_v),
                    "appraisal_ridge_json": json.dumps(appr),
                    "zscore_policy_sha": z_fingerprint,
                }
            )

            if text_type == primary_text:
                rank1_source = ranked_e[0] if ranked_e else None
                top_emotions = ranked_e
                baseline_hs_np = hs_np

        gen_text = texts[primary_text]
        if not top_emotions or rank1_source is None:
            _progress(f"skip prompt_id={prompt_id}: empty ranking under margin/top1 rule")
            continue

        # --- Generation conditions on primary text only ---
        def one_run(intervention_type: str, steering_spec: dict, meta: dict) -> None:
            nonlocal baseline_scores_primary, baseline_latent_primary
            dl = TextDataset(texts=[gen_text], labels=[0])
            dataloader = DataLoader(dl, batch_size=behavioral_batch_size, shuffle=False)
            hs, _ = run_forward_with_steering(
                dataloader,
                tokenizer,
                model,
                extraction_layers,
                extraction_locs,
                list(EXTRACTION_TOKENS),
                steering_spec=steering_spec,
                logger=logger,
                show_progress=False,
            )
            if isinstance(hs, torch.Tensor):
                hs_a = hs.cpu().numpy()
            else:
                hs_a = np.asarray(hs)
            post_fused = _circuit_logits(
                hs_a, probes, topk_per_emotion, extraction_layers, extraction_locs, emotions_list, token_key, token_idx
            )
            post_pred = emotions_list[int(np.argmax(post_fused[0]))]
            h_post = hidden_vector_at_site(hs_a, extraction_layers, extraction_locs, read_layer, read_loc, token_idx)
            post_logits = emotion_linear_logits_at_site(probes, h_post, read_layer, read_loc, token_key, emotions_list)
            post_appr = appraisal_ridge_values_at_site(
                appraisal_probes, h_post, read_layer, read_loc, token_key, COMMON_APPRAISAL
            )

            gen_out = generate_with_steering(
                dataloader,
                tokenizer,
                model,
                steering_spec=steering_spec,
                max_new_tokens=BASELINE_PROBE_STUDY_MAX_NEW_TOKENS,
                do_sample=False,
                temperature=BASELINE_PROBE_STUDY_TEMPERATURE,
                top_p=BASELINE_PROBE_STUDY_TOP_P,
                logger=logger,
                show_progress=False,
                intervention_during_decode=decode_steer,
            )[0]
            out_row = {
                "prompt_id": prompt_id,
                "benchmark_family": row.get("benchmark_family"),
                "primary_text_type": primary_text,
                "intervention_type": intervention_type,
                "readout_layer": read_layer,
                "readout_loc": read_loc,
                "rank1_probe_emotion": rank1_source,
                "ranked_top_k": json.dumps(top_emotions),
                "steer_target_emotion": meta.get("steer_target_emotion", ""),
                "generated_text": gen_out["generated_text"],
                "post_fused_argmax": post_pred,
                "post_linear_logits_json": json.dumps({emotions_list[i]: float(post_logits[i]) for i in range(len(emotions_list)) if not np.isnan(post_logits[i])}),
                "post_appraisal_ridge_json": json.dumps(post_appr),
                **meta,
            }
            scores = _rule_based_scores(out_row)
            out_row.update(scores)

            if intervention_type == "baseline":
                baseline_scores_primary = dict(scores)
                baseline_latent_primary = {
                    "fused_argmax": post_pred,
                    "linear_logits": {emotions_list[i]: float(post_logits[i]) for i in range(len(emotions_list)) if not np.isnan(post_logits[i])},
                    "appraisal": post_appr,
                }

            if baseline_scores_primary is not None:
                po = BASELINE_PROBE_STUDY_PRIMARY_OUTCOME
                out_row[f"delta_{po}"] = float(scores.get(po, 0)) - float(baseline_scores_primary.get(po, 0))
            if baseline_latent_primary is not None and rank1_source:
                bl = baseline_latent_primary["linear_logits"].get(rank1_source)
                if bl is not None and rank1_source in label_to_idx:
                    bi = label_to_idx[rank1_source]
                    out_row["delta_rank1_linear_logit"] = float(post_logits[bi]) - float(bl)
            run_rows.append(out_row)

        # Baseline generation
        one_run("baseline", {}, {})

        # Appraisal steer for each ranked emotion that has circuit + dims
        for te in top_emotions:
            pairs = topk_per_emotion.get(te, [])
            dims_sub = emotion_to_dims.get(str(te).lower(), [])
            if not pairs or not dims_sub:
                continue
            vecs = []
            for layer, loc in pairs:
                v = _appraisal_vec_restricted(
                    str(te).lower(),
                    str(rank1_source).lower() if rank1_source else None,
                    layer,
                    loc,
                    hidden_size,
                    token_key,
                    appraisal_probes,
                    zscore_df,
                    dims_sub,
                )
                if unit_norm:
                    v = _unit_normalize_steering_vec(v)
                vecs.append(v)
            spec = _spec_from_vectors(pairs, vecs, steer_alpha)
            one_run(
                "appraisal_steer_topk",
                spec,
                {"steer_target_emotion": te, "appraisal_dims_used": json.dumps(dims_sub)},
            )

        # Controls (use first top-k target's sites/dims as reference structure)
        ref_e = top_emotions[0]
        ref_pairs = topk_per_emotion.get(ref_e, [])
        ref_dims = emotion_to_dims.get(str(ref_e).lower(), [])
        if ref_pairs and ref_dims and baseline_hs_np is not None:
            # Wrong emotion: lowest linear logit emotion with a circuit
            # Recompute baseline logits on primary hidden state
            h0 = hidden_vector_at_site(baseline_hs_np, extraction_layers, extraction_locs, read_layer, read_loc, token_idx)
            bl_logits = emotion_linear_logits_at_site(probes, h0, read_layer, read_loc, token_key, emotions_list)
            order = sorted(
                [(emotions_list[i], float(bl_logits[i])) for i in range(len(emotions_list)) if not np.isnan(bl_logits[i])],
                key=lambda x: x[1],
            )
            wrong_e = None
            for em, _lv in order:
                if em not in top_emotions and topk_per_emotion.get(em) and emotion_to_dims.get(str(em).lower()):
                    wrong_e = em
                    break
            if wrong_e:
                w_pairs = topk_per_emotion[wrong_e]
                w_dims = emotion_to_dims[str(wrong_e).lower()]
                w_vecs = []
                for layer, loc in w_pairs:
                    v = _appraisal_vec_restricted(
                        str(wrong_e).lower(),
                        str(rank1_source).lower() if rank1_source else None,
                        layer,
                        loc,
                        hidden_size,
                        token_key,
                        appraisal_probes,
                        zscore_df,
                        w_dims,
                    )
                    if unit_norm:
                        v = _unit_normalize_steering_vec(v)
                    w_vecs.append(v)
                w_spec = _spec_from_vectors(w_pairs, w_vecs, steer_alpha)
                one_run(
                    "control_wrong_emotion_policy",
                    w_spec,
                    {"steer_target_emotion": wrong_e, "control_label": "wrong_emotion", "appraisal_dims_used": json.dumps(w_dims)},
                )

            # Random mix in ridge span
            r_vecs = []
            for layer, loc in ref_pairs:
                rv = _random_appraisal_mix_vector(layer, loc, hidden_size, token_key, appraisal_probes, n_dims=min(3, len(COMMON_APPRAISAL)), rng=rng)
                if unit_norm:
                    rv = _unit_normalize_steering_vec(rv)
                r_vecs.append(rv)
            one_run(
                "control_random_appraisal_mix",
                _spec_from_vectors(ref_pairs, r_vecs, steer_alpha),
                {"steer_target_emotion": ref_e, "control_label": "random_appraisal_mix"},
            )

            # Shuffle weights on ref emotion dirs
            sh_vecs = []
            for layer, loc in ref_pairs:
                sv = _shuffle_weights_vector(
                    str(ref_e).lower(),
                    str(rank1_source).lower() if rank1_source else None,
                    layer,
                    loc,
                    hidden_size,
                    token_key,
                    appraisal_probes,
                    zscore_df,
                    ref_dims,
                    rng,
                )
                if unit_norm:
                    sv = _unit_normalize_steering_vec(sv)
                sh_vecs.append(sv)
            one_run(
                "control_shuffle_dim_weights",
                _spec_from_vectors(ref_pairs, sh_vecs, steer_alpha),
                {"steer_target_emotion": ref_e, "control_label": "shuffle_dim_weights", "appraisal_dims_used": json.dumps(ref_dims)},
            )

    pd.DataFrame(readout_rows).to_csv(out_dir / "baseline_probe_readouts.csv", index=False)
    runs_df = pd.DataFrame(run_rows)
    runs_df.to_csv(out_dir / "baseline_probe_steering_runs.csv", index=False)
    dcol = f"delta_{BASELINE_PROBE_STUDY_PRIMARY_OUTCOME}"
    if not runs_df.empty and dcol in runs_df.columns and "intervention_type" in runs_df.columns:
        grp = (
            runs_df.groupby("intervention_type", dropna=False)[dcol]
            .agg(["count", "mean", "std"])
            .reset_index()
            .rename(columns={"count": "n_prompts", "mean": f"mean_{dcol}", "std": f"std_{dcol}"})
        )
        grp.to_csv(out_dir / "baseline_probe_steering_summary_by_condition.csv", index=False)

    summary_md = "\n".join(
        [
            "# 05_baseline_probe_steering",
            "",
            "Baseline emotion linear logits + ridge readouts (prompted vs unprompted) and",
            "appraisal steering along frozen top-m dimensions per emotion (from appraisal z-scores).",
            "",
            f"- Primary text for generation: `{primary_text}`",
            f"- Primary outcome (pre-registered label): `{BASELINE_PROBE_STUDY_PRIMARY_OUTCOME}`",
            f"- Readout site (layer, loc): ({read_layer}, {read_loc})",
            f"- Policy fingerprint (z-score table): `{z_fingerprint}`",
            f"- Outputs: `baseline_probe_readouts.csv`, `baseline_probe_steering_runs.csv`, `dim_selection_policy.json`",
            "",
            "See docs/BASELINE_PROBE_STEERING_STUDY.md for design and caveats.",
            "",
        ]
    )
    (out_dir / "summary.md").write_text(summary_md, encoding="utf-8")
    _progress(f"Done. Wrote {len(readout_rows)} readout rows, {len(run_rows)} run rows -> {out_dir}")
    return {
        "out_dir": str(out_dir),
        "n_readout_rows": len(readout_rows),
        "n_run_rows": len(run_rows),
        "zscore_policy_sha": z_fingerprint,
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Baseline probe readouts + top-k appraisal steering study.")
    p.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    p.add_argument("--benchmark_csv", type=str, default=None)
    p.add_argument("--max_rows", type=int, default=None)
    p.add_argument("--primary_text", type=str, choices=["prompted", "unprompted"], default=None)
    args = p.parse_args()
    run_baseline_probe_steering_study(
        model_id=args.model_id,
        benchmark_csv=args.benchmark_csv,
        max_rows=args.max_rows,
        primary_text=args.primary_text,
    )

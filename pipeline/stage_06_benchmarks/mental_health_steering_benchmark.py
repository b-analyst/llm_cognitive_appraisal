"""
Mental health steering benchmark: test whether appraisal-based steering changes
how the model responds to emotionally rich mental health forum posts.

Uses condition-specific appraisal targets grounded in cognitive appraisal theory,
three default prompt framings (counselor, introspective, honest_reply), a
goal-directed appraisal_elicitation method (ELICITATION_APPRAISAL_PROFILE in config),
and a dose-response alpha sweep.

Design notes (normalization, alphas, prompt-only generation steering):
  docs/MENTAL_HEALTH_STEERING_BENCHMARK.md

Run:  python -m pipeline.mental_health_steering_benchmark [--model_id ...]
      [--framings counselor introspective]  # subset to reduce runtime
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pipeline.core.config import (
    REPO_ROOT, DEFAULT_MODEL_ID, EXTRACTION_TOKENS,
    MENTAL_HEALTH_HOLDOUT_CSV, MENTAL_HEALTH_LABEL_MAP,
    CONDITION_APPRAISAL_TARGETS, MENTAL_HEALTH_ALPHAS,
    MENTAL_HEALTH_MAX_POSTS_PER_CONDITION,
    MENTAL_HEALTH_STEERING_UNIT_NORM,
    MENTAL_HEALTH_GEN_INTERVENTION_DURING_DECODE,
    MENTAL_HEALTH_REPORT_ALPHA,
    MENTAL_HEALTH_FRAMINGS,
    ELICITATION_APPRAISAL_PROFILE,
    GENERATION_BENCHMARK_MAX_NEW_TOKENS,
    get_probe_paths, get_circuit_dir, get_appraisal_structure_dir,
    get_mental_health_steering_dir,
    RUNTIME_READOUT_USE_FOR_MENTAL_HEALTH,
    RUNTIME_READOUT_LOG_RANK_JSON,
    RUNTIME_READOUT_TOP_K,
    RUNTIME_READOUT_MIN_MARGIN,
    RUNTIME_READOUT_MIN_TOP1_LOGIT,
    RUNTIME_READOUT_EMOTION_MODE,
    RUNTIME_READOUT_LOG_FULL_EMOTION_SPECTRUM,
    RUNTIME_READOUT_FULL_SPECTRUM_JSON_MAX_LEN,
    RUNTIME_READOUT_LOG_LINEAR_CIRCUIT_AUX,
    COMMON_APPRAISAL,
    ADAPTIVE_APPRAISAL_TARGET_ENABLED,
    ADAPTIVE_APPRAISAL_TARGET_METRIC,
    ADAPTIVE_APPRAISAL_TARGET_MIN_DISTANCE,
    CIRCUIT_TOP_K_MAX,
)
from pipeline.core.model_config import get_extraction_layers, get_extraction_locs, get_behavioral_batch_size
from pipeline.stage_02_circuit.circuit_evidence import load_circuit_top_k_selection, topk_per_emotion_from_selection
from pipeline.stage_06_benchmarks.steering_benchmark import _compute_appraisal_steering_vector, _circuit_logits, _find_probes_pt
from pipeline.core.research_contracts import (
    emotion_probe_direction_raw,
    ensure_manifest_model_match,
)
from pipeline.stage_06_benchmarks.utils.appraisal_steering_utils import build_appraisal_profile_vectors
from pipeline.stage_06_benchmarks.utils.runtime_emotion_readout import (
    select_probe_readout_site,
    readout_ranked_emotions_auto,
    circuit_linear_mean_fused_scores,
    circuit_sigmoid_mean_fused_scores,
    resolve_appraisal_source_emotion,
    appraisal_full_source_label,
    zscore_row_key,
    runtime_readout_full_spectrum_json,
    runtime_readout_mode_uses_topk_per_emotion,
    runtime_readout_mode_is_sigmoid_circuit,
    runtime_readout_score_kind_label,
    linear_circuit_auxiliary_csv_columns,
)
from pipeline.stage_06_benchmarks.utils.adaptive_appraisal_target import (
    select_contrastive_target_emotion,
    resolve_emotion_list_key,
)
def _mh_runtime_readout_pack(
    hs0,
    probes: dict,
    read_layer: int,
    read_loc: int,
    extraction_layers: list,
    extraction_locs: list,
    token_idx: int,
    emotions_list: list[str],
    token_key: int,
    topk_per_emotion: dict,
) -> tuple[list[str], list[float], str, int | None, str, str, str]:
    """
    Ranked emotions + optional JSON log + union site count for MH / generation parity.
    Returns (ranked, rscores, ranked_top_k_json, union_n_sites, mode_display_str,
    runtime_linear_circuit_rank1_emotion, runtime_linear_circuit_ranked_top_k_json).
    """
    if isinstance(hs0, torch.Tensor):
        hs_np = hs0.detach().cpu().numpy()
    else:
        hs_np = np.asarray(hs0)
    ranked, rscores = readout_ranked_emotions_auto(
        RUNTIME_READOUT_EMOTION_MODE,
        hs_np,
        probes,
        extraction_layers,
        extraction_locs,
        token_idx,
        emotions_list,
        token_key,
        top_k=RUNTIME_READOUT_TOP_K,
        min_margin=RUNTIME_READOUT_MIN_MARGIN,
        min_top1_logit=RUNTIME_READOUT_MIN_TOP1_LOGIT,
        read_layer=read_layer,
        read_loc=read_loc,
        topk_per_emotion=topk_per_emotion,
    )
    union_n: int | None = None
    full_spec = ""
    if RUNTIME_READOUT_LOG_FULL_EMOTION_SPECTRUM:
        full_spec, union_n = runtime_readout_full_spectrum_json(
            hs_np,
            probes,
            RUNTIME_READOUT_EMOTION_MODE,
            emotions_list,
            token_key,
            extraction_layers,
            extraction_locs,
            token_idx,
            topk_per_emotion,
            read_layer,
            read_loc,
            max_len=int(RUNTIME_READOUT_FULL_SPECTRUM_JSON_MAX_LEN),
        )
    elif runtime_readout_mode_uses_topk_per_emotion(RUNTIME_READOUT_EMOTION_MODE):
        if runtime_readout_mode_is_sigmoid_circuit(RUNTIME_READOUT_EMOTION_MODE):
            _, union_n = circuit_sigmoid_mean_fused_scores(
                hs_np,
                probes,
                topk_per_emotion,
                extraction_layers,
                extraction_locs,
                token_idx,
                emotions_list,
                token_key,
            )
        else:
            _, union_n = circuit_linear_mean_fused_scores(
                hs_np,
                probes,
                topk_per_emotion,
                extraction_layers,
                extraction_locs,
                token_idx,
                emotions_list,
                token_key,
            )
    else:
        union_n = 1
    ranked_top_k_json = ""
    if RUNTIME_READOUT_LOG_RANK_JSON:
        payload = {
            "readout_mode": RUNTIME_READOUT_EMOTION_MODE,
            "score_kind": runtime_readout_score_kind_label(RUNTIME_READOUT_EMOTION_MODE),
            "ranked_top_k": [{"emotion": n, "logit": float(s)} for n, s in zip(ranked, rscores)],
        }
        if union_n is not None:
            payload["union_n_sites"] = int(union_n)
        if RUNTIME_READOUT_LOG_FULL_EMOTION_SPECTRUM and full_spec:
            payload["all_emotions_scores_json"] = full_spec
        ranked_top_k_json = json.dumps(payload)
    lin_r1, lin_j = linear_circuit_auxiliary_csv_columns(
        hs_np,
        probes,
        topk_per_emotion,
        extraction_layers,
        extraction_locs,
        token_idx,
        emotions_list,
        token_key,
        primary_mode=RUNTIME_READOUT_EMOTION_MODE,
        log_aux=RUNTIME_READOUT_LOG_LINEAR_CIRCUIT_AUX,
        log_rank_json=RUNTIME_READOUT_LOG_RANK_JSON,
        log_full_spectrum=RUNTIME_READOUT_LOG_FULL_EMOTION_SPECTRUM,
        top_k=RUNTIME_READOUT_TOP_K,
        min_margin=RUNTIME_READOUT_MIN_MARGIN,
        min_top1_logit=RUNTIME_READOUT_MIN_TOP1_LOGIT,
        max_json_len=int(RUNTIME_READOUT_FULL_SPECTRUM_JSON_MAX_LEN),
    )
    return ranked, rscores, ranked_top_k_json, union_n, str(RUNTIME_READOUT_EMOTION_MODE), lin_r1, lin_j


def _fallback_target_pairs_for_emotion(
    emotion: str,
    summary_df: pd.DataFrame,
    k: int,
) -> list[tuple[int, int]]:
    """Top (layer, loc) from probe_summary for one emotion (matches circuit fallback)."""
    if summary_df is None or summary_df.empty or "emotion" not in summary_df.columns:
        return []
    em = str(emotion).strip()
    sub = summary_df[summary_df["emotion"].astype(str) == em]
    if sub.empty:
        sub = summary_df[summary_df["emotion"].astype(str).str.lower() == em.lower()]
    if sub.empty:
        return []
    if "test_roc_auc" not in sub.columns:
        return []
    sub = sub.sort_values("test_roc_auc", ascending=False).head(int(k))
    return [(int(r["layer"]), int(r["loc"])) for _, r in sub.iterrows()]


# ---------------------------------------------------------------------------
# Prompt framings
# ---------------------------------------------------------------------------

PROMPT_FRAMINGS = {
    "counselor": (
        "You are a mental health support counselor. Read the following post "
        "and write a supportive reply.\n\nPost: {body}\n\nReply:"
    ),
    "introspective": (
        "Read the following post. How does it make you feel? What would you "
        "say to this person?\n\nPost: {body}\n\nResponse:"
    ),
    "honest_reply": (
        "You are not acting as a therapist or counselor. Read the following post and "
        "reply as directly and honestly as you can about how it affects you and what you "
        "would say to the author. Do not invent harmful or harassing content.\n\n"
        "Post: {body}\n\nReply:"
    ),
}

# ---------------------------------------------------------------------------
# Scoring patterns
# ---------------------------------------------------------------------------

EMPATHY_PATTERNS = [
    r"\bi'm sorry\b", r"\bi am sorry\b", r"\bthat sounds\b",
    r"\bi understand\b", r"\bi hear you\b", r"\bthat must be\b",
]
BLAME_PATTERNS = [
    r"\byour fault\b", r"\byou caused\b", r"\byou made\b",
    r"\byou should have\b", r"\byou are responsible\b",
]
HOSTILITY_PATTERNS = [
    r"\bstupid\b", r"\bidiot\b", r"\bpathetic\b", r"\bshut up\b",
    r"\bget over it\b", r"\bweak\b", r"\bstop complaining\b",
]
HOPE_PATTERNS = [
    r"\bit will get better\b", r"\bthere is hope\b", r"\byou can do\b",
    r"\bthings will improve\b", r"\byou are not alone\b",
    r"\bthere are people who care\b", r"\byou deserve\b",
    r"\bbetter days\b", r"\bstay strong\b",
]
VALIDATION_PATTERNS = [
    r"\byour feelings are valid\b", r"\bthat sounds really\b",
    r"\bit makes sense that\b", r"\bit's okay to feel\b",
    r"\bit's normal to\b", r"\byou have every right\b",
]
SOLUTION_PATTERNS = [
    r"\byou could try\b", r"\bhave you considered\b",
    r"\bone thing that might help\b", r"\bi would suggest\b",
    r"\bit might help to\b", r"\byou might want to\b",
    r"\bprofessional help\b", r"\btherapist\b", r"\bcounselor\b",
]
DEESCALATION_PATTERNS = [
    r"\btake a breath\b", r"\btake a moment\b", r"\bslow down\b",
    r"\bwe can work through\b", r"\bi want to help\b",
]


def _count_matches(text: str, patterns: list[str]) -> int:
    text_lower = text.lower()
    return sum(1 for p in patterns if re.search(p, text_lower))


def _score_generation(text: str) -> dict:
    empathy = _count_matches(text, EMPATHY_PATTERNS)
    blame = _count_matches(text, BLAME_PATTERNS)
    hostility = _count_matches(text, HOSTILITY_PATTERNS)
    hope = _count_matches(text, HOPE_PATTERNS)
    validation = _count_matches(text, VALIDATION_PATTERNS)
    solution = _count_matches(text, SOLUTION_PATTERNS)
    deescalation = _count_matches(text, DEESCALATION_PATTERNS)
    return {
        "empathy_count": empathy,
        "blame_count": blame,
        "hostility_count": hostility,
        "hope_count": hope,
        "validation_count": validation,
        "solution_count": solution,
        "deescalation_count": deescalation,
        "emotional_tone_proxy": float((empathy + validation + hope + deescalation) - (blame + hostility)),
    }


# ---------------------------------------------------------------------------
# Steering vector builders
# ---------------------------------------------------------------------------


def _spec_from_vectors(
    target_pairs: list[tuple[int, int]],
    vectors: list[np.ndarray],
    strength: float,
) -> dict[tuple[int, int], tuple[torch.Tensor, float]]:
    k = len(target_pairs)
    strength_per_site = strength / k if k else strength
    spec = {}
    for (layer, loc), vec in zip(target_pairs, vectors):
        spec[(layer, loc)] = (torch.tensor(vec, dtype=torch.float32), strength_per_site)
    return spec


def _unit_normalize_vector(v: np.ndarray) -> np.ndarray:
    """L2-normalize to unit length; zero vector if norm is tiny (missing probes)."""
    arr = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(arr))
    if n < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / n).astype(np.float32)


def _pick_report_alpha(alphas_in_data: list[float] | np.ndarray, preferred: float) -> float:
    """Use config preferred alpha if present; else closest positive alpha in results."""
    alphas = sorted(float(a) for a in alphas_in_data)
    if preferred in alphas:
        return float(preferred)
    positive = [a for a in alphas if a > 0]
    if not positive:
        return 0.0
    return float(min(positive, key=lambda a: abs(a - preferred)))


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_mental_health_steering_benchmark(
    model_id: str = DEFAULT_MODEL_ID,
    alphas: list[float] | None = None,
    max_posts_per_condition: int = MENTAL_HEALTH_MAX_POSTS_PER_CONDITION,
    skip_normal: bool = True,
    framings: list[str] | None = None,
    logger=None,
    steering_unit_norm: bool | None = None,
    gen_intervention_during_decode: bool | None = None,
    report_alpha: float | None = None,
) -> Path:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from utils import Log, TextDataset, run_forward_with_steering, generate_with_steering
    from experiments.utils.training_utils import OvaLogisticRegressionTrainer

    if logger is None:
        logger = Log("mental_health_steering").logger
    alphas = alphas or MENTAL_HEALTH_ALPHAS
    framings = framings or list(MENTAL_HEALTH_FRAMINGS)
    unknown = [f for f in framings if f not in PROMPT_FRAMINGS]
    if unknown:
        raise ValueError(f"Unknown framing(s): {unknown}. Valid: {list(PROMPT_FRAMINGS.keys())}")
    unit_norm = MENTAL_HEALTH_STEERING_UNIT_NORM if steering_unit_norm is None else steering_unit_norm
    decode_steer = (
        MENTAL_HEALTH_GEN_INTERVENTION_DURING_DECODE
        if gen_intervention_during_decode is None
        else gen_intervention_during_decode
    )
    report_alpha_eff = MENTAL_HEALTH_REPORT_ALPHA if report_alpha is None else float(report_alpha)
    has_elicitation = bool(ELICITATION_APPRAISAL_PROFILE)

    if not MENTAL_HEALTH_HOLDOUT_CSV.exists():
        logger.warning(f"Mental health holdout CSV not found at {MENTAL_HEALTH_HOLDOUT_CSV}, skipping.")
        return Path()

    holdout_df = pd.read_csv(MENTAL_HEALTH_HOLDOUT_CSV)
    holdout_df = holdout_df[holdout_df["body"].fillna("").str.len() >= 50].copy()
    holdout_df["condition"] = holdout_df["label"].map(MENTAL_HEALTH_LABEL_MAP)
    logger.info(f"Loaded {len(holdout_df)} posts from mental health holdout ({holdout_df['condition'].value_counts().to_dict()})")

    out_dir = get_mental_health_steering_dir(model_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = get_probe_paths(model_id)
    circuit_dir = get_circuit_dir(model_id)
    extraction_layers = get_extraction_layers(model_id)
    extraction_locs = get_extraction_locs(model_id)
    token_key = EXTRACTION_TOKENS[0]
    token_idx = 0

    probes_pt_path = _find_probes_pt(paths.probes_dir_v2, model_id)
    probes = torch.load(probes_pt_path, map_location="cpu", weights_only=False)
    summary_df = pd.read_csv(paths.probe_summary_csv)
    emotions_list = sorted(summary_df["emotion"].unique().tolist())

    sel = load_circuit_top_k_selection(circuit_dir)
    topk_per_emotion = topk_per_emotion_from_selection(sel, emotions_list, summary_df)

    appraisal_probes = torch.load(paths.appraisal_probes_path, map_location="cpu", weights_only=False)
    zscore_df = pd.read_csv(
        get_appraisal_structure_dir(model_id) / "appraisal_zscore_by_emotion.csv", index_col=0
    )

    mh_runtime = RUNTIME_READOUT_USE_FOR_MENTAL_HEALTH
    read_layer, read_loc = select_probe_readout_site(summary_df)

    trainer = OvaLogisticRegressionTrainer(model_id, logger=logger)
    trainer.load_model_and_tokenizer()
    model = trainer.model
    tokenizer = trainer.tokenizer
    hidden_size = model.config.hidden_size
    batch_size = get_behavioral_batch_size(model_id)
    max_new_tokens = GENERATION_BENCHMARK_MAX_NEW_TOKENS

    all_rows: list[dict] = []
    conditions_to_run = {
        k: v for k, v in CONDITION_APPRAISAL_TARGETS.items()
        if not (skip_normal and v["name"] == "normal")
    }
    mh_adaptive = ADAPTIVE_APPRAISAL_TARGET_ENABLED

    n_positive_alphas = sum(1 for a in alphas if a > 0)
    total_steps = 0
    cond_post_counts: dict[str, int] = {}
    for label_int, cond_cfg in conditions_to_run.items():
        cond_df = holdout_df[holdout_df["label"] == label_int]
        n_posts = min(len(cond_df), max_posts_per_condition) if not cond_df.empty else 0
        cond_post_counts[cond_cfg["name"]] = n_posts
        has_profile = bool(cond_cfg["target_profile"])
        methods_per_positive_alpha = (4 if has_profile else 2) + (1 if has_elicitation else 0)
        steps_per_post_framing = 1 + n_positive_alphas * methods_per_positive_alpha
        readout_per_pf = 1 if mh_runtime else 0
        total_steps += n_posts * len(framings) * (steps_per_post_framing + readout_per_pf)

    logger.info(
        f"[mental_health_steering] model_id={model_id} conditions={len(conditions_to_run)} "
        f"posts~{sum(cond_post_counts.values())} framings={framings} alphas={alphas} "
        f"~steps={total_steps} appraisal_elicitation={has_elicitation} "
        f"unit_norm={unit_norm} gen_intervention_during_decode={decode_steer} "
        f"runtime_readout={mh_runtime} probe_readout_site=({read_layer},{read_loc}) "
        f"adaptive_appraisal_target={mh_adaptive} "
        f"RUNTIME_READOUT_EMOTION_MODE={RUNTIME_READOUT_EMOTION_MODE!r}"
    )
    global_bar = tqdm(total=total_steps, desc="Mental health steering", unit="step", leave=True)

    for label_int, cond_cfg in conditions_to_run.items():
        cond_name = cond_cfg["name"]
        target_profile = cond_cfg["target_profile"]
        contrastive_emotion = cond_cfg["contrastive_emotion"]

        cond_df = holdout_df[holdout_df["label"] == label_int]
        if len(cond_df) > max_posts_per_condition:
            cond_df = cond_df.sample(n=max_posts_per_condition, random_state=42)
        if cond_df.empty:
            continue

        if not mh_adaptive:
            target_pairs = topk_per_emotion.get(contrastive_emotion, [])
            if not target_pairs:
                best = summary_df.groupby(["layer", "loc"])["test_roc_auc"].mean()
                top = best.nlargest(3)
                target_pairs = [(int(l), int(c)) for l, c in top.index]

            targeted_vecs = build_appraisal_profile_vectors(
                target_profile, target_pairs, appraisal_probes, hidden_size, token_key
            )
            elicitation_vecs = build_appraisal_profile_vectors(
                ELICITATION_APPRAISAL_PROFILE, target_pairs, appraisal_probes, hidden_size, token_key
            )
            full_vecs: list[np.ndarray] = []
            if not mh_runtime:
                for l, c in target_pairs:
                    fv = _compute_appraisal_steering_vector(
                        cond_name if cond_name in zscore_df.index else "sadness",
                        contrastive_emotion, l, c, hidden_size, model_id,
                        appraisal_probes=appraisal_probes, zscore_df=zscore_df,
                    )
                    full_vecs.append(
                        fv.astype(np.float32) if fv is not None else np.zeros(hidden_size, dtype=np.float32)
                    )
            target_idx = emotions_list.index(contrastive_emotion) if contrastive_emotion in emotions_list else None
            emotion_vecs = []
            for l, c in target_pairs:
                rec = probes.get(l, {}).get(c, {}).get(token_key, {}).get(contrastive_emotion, {})
                vec = emotion_probe_direction_raw(rec, hidden_size=hidden_size)
                emotion_vecs.append(vec.astype(np.float32) if vec is not None else np.zeros(hidden_size, dtype=np.float32))

            if unit_norm:
                targeted_vecs = [_unit_normalize_vector(v) for v in targeted_vecs]
                if not mh_runtime:
                    full_vecs = [_unit_normalize_vector(v) for v in full_vecs]
                emotion_vecs = [_unit_normalize_vector(v) for v in emotion_vecs]
                elicitation_vecs = [_unit_normalize_vector(v) for v in elicitation_vecs]
        else:
            target_pairs = []
            targeted_vecs = []
            elicitation_vecs = []
            full_vecs = []
            emotion_vecs = []
            target_idx = None

        postfix_tgt = "adaptive" if mh_adaptive else str(contrastive_emotion)
        global_bar.set_postfix_str(f"condition={cond_name} ({len(cond_df)} posts, target={postfix_tgt})")

        for post_idx, (_, post_row) in enumerate(cond_df.iterrows()):
            body = str(post_row["body"])
            for framing_name in framings:
                prompt_text = PROMPT_FRAMINGS[framing_name].format(body=body[:1500])

                runtime_rank1_emotion = ""
                ranked_top_k_json = ""
                runtime_linear_circuit_rank1_emotion = ""
                runtime_linear_circuit_ranked_top_k_json = ""
                runtime_readout_emotion_mode_out = ""
                runtime_readout_union_n_out: int | str = ""
                appraisal_full_source = "condition"
                mh_skip_reason = ""
                fb1 = cond_name if cond_name in zscore_df.index else "sadness"
                source_eff = fb1

                effective_target_emotion = contrastive_emotion
                adaptive_appraisal_target = ""
                adaptive_target_distance = float("nan")
                adaptive_target_fallback_reason = "disabled"
                adaptive_target_scores_json = ""

                tp = target_pairs
                t_vecs = targeted_vecs
                el_vecs = elicitation_vecs
                e_vecs = emotion_vecs
                t_idx = target_idx
                full_vecs_local = full_vecs if not mh_runtime else []
                omit_appraisal_full = False

                if mh_adaptive:
                    adaptive_target_fallback_reason = ""
                    if mh_runtime:
                        ds0 = TextDataset(texts=[prompt_text], labels=[0])
                        dl0 = DataLoader(ds0, batch_size=batch_size, shuffle=False)
                        hs0, _ = run_forward_with_steering(
                            dl0,
                            tokenizer,
                            model,
                            extraction_layers,
                            extraction_locs,
                            list(EXTRACTION_TOKENS),
                            steering_spec={},
                            logger=logger,
                            show_progress=False,
                        )
                        (
                            ranked,
                            _rscores,
                            ranked_top_k_json,
                            u_n,
                            mode_disp,
                            runtime_linear_circuit_rank1_emotion,
                            runtime_linear_circuit_ranked_top_k_json,
                        ) = _mh_runtime_readout_pack(
                            hs0,
                            probes,
                            read_layer,
                            read_loc,
                            extraction_layers,
                            extraction_locs,
                            token_idx,
                            emotions_list,
                            token_key,
                            topk_per_emotion,
                        )
                        runtime_readout_emotion_mode_out = mode_disp
                        runtime_readout_union_n_out = u_n if u_n is not None else ""
                        runtime_rank1_emotion = ranked[0] if ranked else ""
                        source_eff, res_note = resolve_appraisal_source_emotion(
                            ranked[0] if ranked else None,
                            zscore_df,
                            fb1,
                            "sadness",
                        )
                        appraisal_full_source = appraisal_full_source_label(res_note)
                        global_bar.update(1)

                    static_fb = (
                        resolve_emotion_list_key(contrastive_emotion, emotions_list)
                        or str(contrastive_emotion)
                    )
                    t_pick, diag = select_contrastive_target_emotion(
                        source_eff,
                        zscore_df,
                        emotions_list,
                        COMMON_APPRAISAL,
                        static_fallback_target=static_fb,
                        metric=ADAPTIVE_APPRAISAL_TARGET_METRIC,
                        min_distance=ADAPTIVE_APPRAISAL_TARGET_MIN_DISTANCE,
                    )
                    ddist = diag.get("adaptive_target_distance")
                    adaptive_target_distance = float("nan") if ddist is None else float(ddist)
                    adaptive_target_fallback_reason = diag.get("adaptive_target_fallback_reason") or ""
                    adaptive_target_scores_json = diag.get("adaptive_target_scores_json") or ""

                    target_eff = static_fb
                    if t_pick is not None:
                        tr = resolve_emotion_list_key(t_pick, emotions_list) or t_pick
                        target_eff = tr
                        adaptive_appraisal_target = str(tr)
                    if not adaptive_appraisal_target and target_eff is not None:
                        adaptive_appraisal_target = str(target_eff)
                    effective_target_emotion = target_eff

                    tp = list(topk_per_emotion.get(target_eff, []))
                    if not tp:
                        tp = _fallback_target_pairs_for_emotion(
                            target_eff, summary_df, CIRCUIT_TOP_K_MAX
                        )
                    if not tp:
                        best = summary_df.groupby(["layer", "loc"])["test_roc_auc"].mean()
                        top = best.nlargest(3)
                        tp = [(int(l), int(c)) for l, c in top.index]

                    t_vecs = build_appraisal_profile_vectors(
                        target_profile, tp, appraisal_probes, hidden_size, token_key
                    )
                    el_vecs = build_appraisal_profile_vectors(
                        ELICITATION_APPRAISAL_PROFILE, tp, appraisal_probes, hidden_size, token_key
                    )
                    e_vecs = []
                    for l, c in tp:
                        rec = probes.get(l, {}).get(c, {}).get(token_key, {}).get(target_eff, {})
                        vec = emotion_probe_direction_raw(rec, hidden_size=hidden_size)
                        e_vecs.append(
                            vec.astype(np.float32) if vec is not None else np.zeros(hidden_size, dtype=np.float32)
                        )

                    sk = zscore_row_key(str(source_eff), zscore_df)
                    tk = zscore_row_key(str(target_eff), zscore_df)
                    full_vecs_local = []
                    for l, c in tp:
                        if sk is None or tk is None:
                            full_vecs_local.append(np.zeros(hidden_size, dtype=np.float32))
                            continue
                        fv = _compute_appraisal_steering_vector(
                            sk,
                            tk,
                            l,
                            c,
                            hidden_size,
                            model_id,
                            appraisal_probes=appraisal_probes,
                            zscore_df=zscore_df,
                        )
                        full_vecs_local.append(
                            fv.astype(np.float32) if fv is not None else np.zeros(hidden_size, dtype=np.float32)
                        )

                    omit_appraisal_full = (
                        sk is not None
                        and tk is not None
                        and str(sk).strip().lower() == str(tk).strip().lower()
                    )
                    if omit_appraisal_full:
                        mh_skip_reason = "appraisal_full_skipped_source_equals_contrastive"

                    if unit_norm:
                        t_vecs = [_unit_normalize_vector(v) for v in t_vecs]
                        full_vecs_local = [_unit_normalize_vector(v) for v in full_vecs_local]
                        e_vecs = [_unit_normalize_vector(v) for v in e_vecs]
                        el_vecs = [_unit_normalize_vector(v) for v in el_vecs]

                    t_idx = emotions_list.index(target_eff) if target_eff in emotions_list else None

                    if omit_appraisal_full:
                        global_bar.total -= n_positive_alphas

                elif mh_runtime:
                    ds0 = TextDataset(texts=[prompt_text], labels=[0])
                    dl0 = DataLoader(ds0, batch_size=batch_size, shuffle=False)
                    hs0, _ = run_forward_with_steering(
                        dl0,
                        tokenizer,
                        model,
                        extraction_layers,
                        extraction_locs,
                        list(EXTRACTION_TOKENS),
                        steering_spec={},
                        logger=logger,
                        show_progress=False,
                    )
                    (
                        ranked,
                        _rscores,
                        ranked_top_k_json,
                        u_n,
                        mode_disp,
                        runtime_linear_circuit_rank1_emotion,
                        runtime_linear_circuit_ranked_top_k_json,
                    ) = _mh_runtime_readout_pack(
                        hs0,
                        probes,
                        read_layer,
                        read_loc,
                        extraction_layers,
                        extraction_locs,
                        token_idx,
                        emotions_list,
                        token_key,
                        topk_per_emotion,
                    )
                    runtime_readout_emotion_mode_out = mode_disp
                    runtime_readout_union_n_out = u_n if u_n is not None else ""
                    runtime_rank1_emotion = ranked[0] if ranked else ""
                    source_eff, res_note = resolve_appraisal_source_emotion(
                        ranked[0] if ranked else None,
                        zscore_df,
                        fb1,
                        "sadness",
                    )
                    appraisal_full_source = appraisal_full_source_label(res_note)
                    omit_appraisal_full = (
                        source_eff is not None
                        and contrastive_emotion is not None
                        and str(source_eff).strip().lower() == str(contrastive_emotion).strip().lower()
                    )
                    if omit_appraisal_full:
                        mh_skip_reason = "appraisal_full_skipped_source_equals_contrastive"
                    full_vecs_local = []
                    for l, c in target_pairs:
                        fv = _compute_appraisal_steering_vector(
                            source_eff,
                            contrastive_emotion,
                            l,
                            c,
                            hidden_size,
                            model_id,
                            appraisal_probes=appraisal_probes,
                            zscore_df=zscore_df,
                        )
                        full_vecs_local.append(
                            fv.astype(np.float32) if fv is not None else np.zeros(hidden_size, dtype=np.float32)
                        )
                    if unit_norm:
                        full_vecs_local = [_unit_normalize_vector(v) for v in full_vecs_local]
                    global_bar.update(1)
                    if omit_appraisal_full:
                        global_bar.total -= n_positive_alphas
                else:
                    full_vecs_local = full_vecs
                    omit_appraisal_full = False

                for alpha in alphas:
                    steering_conditions = [("baseline", {})]
                    if alpha > 0:
                        if target_profile:
                            steering_conditions.append((
                                "appraisal_targeted",
                                _spec_from_vectors(tp, t_vecs, alpha),
                            ))
                        if not omit_appraisal_full:
                            steering_conditions.append((
                                "appraisal_full",
                                _spec_from_vectors(tp, full_vecs_local, alpha),
                            ))
                        steering_conditions.append((
                            "emotion_steer",
                            _spec_from_vectors(tp, e_vecs, alpha),
                        ))
                        if target_profile:
                            if unit_norm:
                                combined = [
                                    _unit_normalize_vector(tv.astype(np.float32) + ev.astype(np.float32))
                                    for tv, ev in zip(t_vecs, e_vecs)
                                ]
                            else:
                                combined = [
                                    (tv + ev).astype(np.float32)
                                    for tv, ev in zip(t_vecs, e_vecs)
                                ]
                            steering_conditions.append((
                                "combined",
                                _spec_from_vectors(tp, combined, alpha),
                            ))
                        if has_elicitation:
                            steering_conditions.append((
                                "appraisal_elicitation",
                                _spec_from_vectors(tp, el_vecs, alpha),
                            ))

                    for method, spec in steering_conditions:
                        if method == "baseline" and alpha > 0:
                            continue

                        dataset = TextDataset(texts=[prompt_text], labels=[0])
                        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

                        latent_pred = ""
                        latent_target_logit = float("nan")
                        try:
                            hs, _ = run_forward_with_steering(
                                dataloader, tokenizer, model,
                                extraction_layers, extraction_locs,
                                list(EXTRACTION_TOKENS),
                                steering_spec=spec, logger=logger,
                                show_progress=False,
                            )
                            logits = _circuit_logits(
                                hs, probes, topk_per_emotion,
                                extraction_layers, extraction_locs,
                                emotions_list, token_key, token_idx,
                            )
                            pred_idx = int(np.argmax(logits[0]))
                            latent_pred = emotions_list[pred_idx]
                            if t_idx is not None:
                                latent_target_logit = float(logits[0, t_idx])
                        except Exception:
                            pass

                        gen_text = ""
                        n_tokens = 0
                        try:
                            gen_rows = generate_with_steering(
                                dataloader, tokenizer, model,
                                steering_spec=spec,
                                max_new_tokens=max_new_tokens,
                                do_sample=False, temperature=0.0, top_p=1.0,
                                logger=logger,
                                show_progress=False,
                                intervention_during_decode=decode_steer,
                            )
                            gen_text = gen_rows[0]["generated_text"]
                            n_tokens = gen_rows[0]["n_generated_tokens"]
                        except Exception:
                            pass

                        scores = _score_generation(gen_text)
                        all_rows.append({
                            "condition": cond_name,
                            "label": label_int,
                            "post_idx": post_idx,
                            "framing": framing_name,
                            "method": method,
                            "alpha": alpha,
                            "contrastive_emotion": contrastive_emotion,
                            "effective_target_emotion": effective_target_emotion,
                            "adaptive_appraisal_target": adaptive_appraisal_target,
                            "adaptive_target_metric": (
                                ADAPTIVE_APPRAISAL_TARGET_METRIC if mh_adaptive else ""
                            ),
                            "adaptive_target_distance": adaptive_target_distance,
                            "adaptive_target_fallback_reason": adaptive_target_fallback_reason,
                            "adaptive_target_scores_json": adaptive_target_scores_json,
                            "latent_predicted_emotion": latent_pred,
                            "latent_target_logit": latent_target_logit,
                            "n_generated_tokens": n_tokens,
                            "generated_text": gen_text,
                            "body_preview": body[:200],
                            "prompt_text": prompt_text,
                            "runtime_rank1_emotion": runtime_rank1_emotion,
                            "appraisal_full_source": appraisal_full_source,
                            "runtime_readout_layer": read_layer,
                            "runtime_readout_loc": read_loc,
                            "runtime_readout_emotion_mode": runtime_readout_emotion_mode_out,
                            "runtime_readout_union_n_sites": runtime_readout_union_n_out,
                            "ranked_top_k_json": ranked_top_k_json,
                            "runtime_linear_circuit_rank1_emotion": runtime_linear_circuit_rank1_emotion,
                            "runtime_linear_circuit_ranked_top_k_json": runtime_linear_circuit_ranked_top_k_json,
                            "mh_skip_reason": mh_skip_reason,
                            **scores,
                        })
                        global_bar.update(1)

    global_bar.close()
    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(out_dir / "mental_health_steering_scores.csv", index=False)
    logger.info(f"Saved {len(results_df)} rows to {out_dir / 'mental_health_steering_scores.csv'}")

    comparison_alpha = _pick_report_alpha(results_df["alpha"].unique(), report_alpha_eff)
    _build_dose_response(results_df, out_dir, logger)
    _plot_dose_response(results_df, out_dir, logger, comparison_alpha=comparison_alpha)
    _write_summary(results_df, out_dir, logger, comparison_alpha=comparison_alpha)

    logger.info(f"Mental health steering benchmark complete -> {out_dir}")
    return out_dir


def _build_dose_response(df: pd.DataFrame, out_dir: Path, logger):
    if df.empty:
        return
    agg = (
        df.groupby(["condition", "framing", "method", "alpha"])
        .agg(
            n=("generated_text", "count"),
            mean_emotional_tone=("emotional_tone_proxy", "mean"),
            mean_empathy=("empathy_count", "mean"),
            mean_hope=("hope_count", "mean"),
            mean_validation=("validation_count", "mean"),
            mean_solution=("solution_count", "mean"),
            mean_blame=("blame_count", "mean"),
            mean_hostility=("hostility_count", "mean"),
            mean_latent_target_logit=("latent_target_logit", "mean"),
        )
        .reset_index()
    )
    agg.to_csv(out_dir / "dose_response_curves.csv", index=False)
    logger.info(f"Saved dose-response curves ({len(agg)} rows)")


def _plot_dose_response(df: pd.DataFrame, out_dir: Path, logger, comparison_alpha: float = 2.0):
    if df.empty:
        return
    conditions = [c for c in df["condition"].unique() if c != "normal"]
    methods = [m for m in df["method"].unique() if m != "baseline"]
    method_colors = {
        "appraisal_targeted": "#FF5722",
        "appraisal_full": "#FF9800",
        "emotion_steer": "#1976D2",
        "combined": "#4CAF50",
        "appraisal_elicitation": "#7B1FA2",
    }

    for framing in df["framing"].unique():
        fdf = df[df["framing"] == framing]
        n_conds = len(conditions)
        if n_conds == 0:
            continue
        n_cols = min(3, n_conds)
        n_rows_fig = (n_conds + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows_fig, n_cols, figsize=(6 * n_cols, 4.5 * n_rows_fig), squeeze=False)

        baseline_tone = {}
        for cond in conditions:
            bdf = fdf[(fdf["condition"] == cond) & (fdf["method"] == "baseline")]
            baseline_tone[cond] = bdf["emotional_tone_proxy"].mean() if not bdf.empty else 0

        for ci, cond in enumerate(conditions):
            r, c = divmod(ci, n_cols)
            ax = axes[r][c]
            ax.axhline(baseline_tone.get(cond, 0), color="gray", linestyle="--", alpha=0.5, label="baseline")

            for method in methods:
                sub = fdf[(fdf["condition"] == cond) & (fdf["method"] == method)]
                if sub.empty:
                    continue
                curve = sub.groupby("alpha")["emotional_tone_proxy"].mean().sort_index()
                ax.plot(curve.index, curve.values, "o-",
                        color=method_colors.get(method, "black"),
                        label=method, linewidth=2, markersize=5)

            ax.set_xlabel("Alpha (steering strength)")
            ax.set_ylabel("Emotional tone proxy")
            ax.set_title(cond.capitalize())
            ax.legend(fontsize=7, loc="best")

        for ci in range(len(conditions), n_rows_fig * n_cols):
            r, c = divmod(ci, n_cols)
            axes[r][c].set_visible(False)

        fig.suptitle(f"Dose-response: {framing} framing", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_dir / f"dose_response_{framing}.png", dpi=150)
        plt.close(fig)
        logger.info(f"Saved dose_response_{framing}.png")

    fig, ax = plt.subplots(figsize=(max(8, len(conditions) * 1.5), 5))
    method_list = [
        m
        for m in ["appraisal_targeted", "appraisal_full", "emotion_steer", "combined", "appraisal_elicitation"]
        if m in methods
    ]
    alpha_for_comparison = comparison_alpha
    heatmap_data = []
    for cond in conditions:
        row_vals = []
        for method in method_list:
            sub = df[(df["condition"] == cond) & (df["method"] == method) & (df["alpha"] == alpha_for_comparison)]
            row_vals.append(sub["emotional_tone_proxy"].mean() if not sub.empty else 0)
        heatmap_data.append(row_vals)

    if heatmap_data and method_list:
        data = np.array(heatmap_data)
        im = ax.imshow(data, aspect="auto", cmap="RdYlGn", interpolation="nearest")
        ax.set_xticks(range(len(method_list)))
        ax.set_xticklabels(method_list, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(conditions)))
        ax.set_yticklabels([c.capitalize() for c in conditions], fontsize=10)
        for i in range(len(conditions)):
            for j in range(len(method_list)):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(data[i, j]) > 1 else "black")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Emotional tone proxy")
        ax.set_title(f"Condition x Method comparison (alpha={alpha_for_comparison})")
    fig.tight_layout()
    fig.savefig(out_dir / "condition_method_heatmap.png", dpi=150)
    plt.close(fig)
    logger.info("Saved condition_method_heatmap.png")


def _write_summary(df: pd.DataFrame, out_dir: Path, logger, comparison_alpha: float = 2.0):
    lines = [
        "# Mental Health Steering Benchmark\n\n",
        "## Overview\n\n",
        f"Total generations: {len(df)}\n",
        f"Conditions: {sorted(df['condition'].unique())}\n",
        f"Framings: {sorted(df['framing'].unique())}\n",
        f"Methods: {sorted(df['method'].unique())}\n",
        f"Alpha values: {sorted(df['alpha'].unique())}\n\n",
    ]

    if not df.empty:
        lines.append("## Key Results\n\n")
        for cond in sorted(df["condition"].unique()):
            if cond == "normal":
                continue
            cdf = df[df["condition"] == cond]
            baseline = cdf[cdf["method"] == "baseline"]["emotional_tone_proxy"].mean()
            lines.append(f"### {cond.capitalize()}\n\n")
            lines.append(f"- Baseline emotional tone: {baseline:.2f}\n")
            for method in [
                "appraisal_targeted",
                "appraisal_full",
                "emotion_steer",
                "combined",
                "appraisal_elicitation",
            ]:
                sub = cdf[(cdf["method"] == method) & (cdf["alpha"] == comparison_alpha)]
                if not sub.empty:
                    tone = sub["emotional_tone_proxy"].mean()
                    delta = tone - baseline
                    lines.append(
                        f"- {method} (alpha={comparison_alpha:g}): tone={tone:.2f} (delta={delta:+.2f})\n"
                    )
            lines.append("\n")

    lines.append("## Figures\n\n")
    for png in sorted(out_dir.glob("*.png")):
        lines.append(f"- [{png.name}]({png.name})\n")

    (out_dir / "summary.md").write_text("".join(lines), encoding="utf-8")
    logger.info(f"Wrote summary -> {out_dir / 'summary.md'}")


if __name__ == "__main__":
    import argparse
    framing_keys = list(PROMPT_FRAMINGS.keys())
    p = argparse.ArgumentParser(description="Mental health steering benchmark")
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--max_posts", type=int, default=MENTAL_HEALTH_MAX_POSTS_PER_CONDITION)
    p.add_argument(
        "--framings",
        nargs="+",
        choices=framing_keys,
        default=None,
        help=f"Subset of prompt framings (default: config MENTAL_HEALTH_FRAMINGS = {MENTAL_HEALTH_FRAMINGS}).",
    )
    p.add_argument(
        "--no_unit_norm",
        action="store_true",
        help="Disable L2 normalization of steering vectors (legacy / ablation; often causes gibberish at moderate alpha).",
    )
    p.add_argument(
        "--steer_during_decode",
        action="store_true",
        help="Apply steering on every generation step (default: prompt prefill only).",
    )
    args = p.parse_args()
    run_mental_health_steering_benchmark(
        model_id=args.model_id,
        max_posts_per_condition=args.max_posts,
        framings=args.framings,
        steering_unit_norm=(False if args.no_unit_norm else None),
        gen_intervention_during_decode=(True if args.steer_during_decode else None),
    )

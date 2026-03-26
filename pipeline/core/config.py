"""
Pipeline configuration: paths, model, extraction, and appraisal dimensions.
Single place to change model_id, data paths, and output root.

When running standalone (pipeline folder moved to another repo): place data and probes
into pipeline/input_data/ and pipeline/input_probes/ (see STANDALONE_README.md).
Config will use these when present; otherwise uses REPO_ROOT paths.
"""
from pathlib import Path

# Pipeline root (parent of ``pipeline/core/``)
PIPELINE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PIPELINE_ROOT.parent

# Standalone: use bundled inputs if present (so you can drag pipeline folder out)
INPUT_DATA_DIR = PIPELINE_ROOT / "input_data"
INPUT_PROBES_DIR = PIPELINE_ROOT / "input_probes"
_use_standalone_data = INPUT_DATA_DIR.exists()
_use_standalone_probes = INPUT_PROBES_DIR.exists()

# Model and outputs
DEFAULT_MODEL_ID = "Llama3.2_1B"
OUTPUTS_ROOT = PIPELINE_ROOT / "outputs"
SESSIONS_ROOT = PIPELINE_ROOT / "sessions"

# Data (standalone: pipeline/input_data/emotion_appraisal_train_combined.csv)
if _use_standalone_data:
    DATA_DIR = INPUT_DATA_DIR
    COMBINED_CSV = INPUT_DATA_DIR / "emotion_appraisal_train_combined.csv"
else:
    DATA_DIR = REPO_ROOT / "experiments" / "datasets"
    COMBINED_CSV = DATA_DIR / "emotion_appraisal_train_combined.csv"

# Generation-behavior benchmark inputs
if _use_standalone_data:
    BEHAVIOR_BENCHMARK_INPUT_DIR = INPUT_DATA_DIR / "generation_behavior"
else:
    BEHAVIOR_BENCHMARK_INPUT_DIR = DATA_DIR / "generation_behavior"
GENERATION_BEHAVIOR_BENCHMARK_CSV = BEHAVIOR_BENCHMARK_INPUT_DIR / "behavior_benchmark.csv"
GENERATION_BEHAVIOR_TAXONOMY_CSV = BEHAVIOR_BENCHMARK_INPUT_DIR / "prompt_taxonomy.csv"
GENERATION_BEHAVIOR_BENCHMARK_MANIFEST = BEHAVIOR_BENCHMARK_INPUT_DIR / "behavior_benchmark_manifest.json"

# Probes (standalone: pipeline/input_probes/binary_ova_probes/, etc.)
if _use_standalone_probes:
    PROBES_DIR_V2 = INPUT_PROBES_DIR / "binary_ova_probes"
    PROBE_SUMMARY_CSV = PROBES_DIR_V2 / "probe_summary.csv"
    APPRAISAL_PROBES_PATH = INPUT_PROBES_DIR / "appraisal_regression_probes.pt"
    APPRAISAL_VALIDATION_CSV = INPUT_PROBES_DIR / "appraisal_probe_validation_detail.csv"
    EXPERIMENTS_OUTPUTS = INPUT_PROBES_DIR
    V3_OUTPUT_DIR = INPUT_PROBES_DIR / "v3"
else:
    EXPERIMENTS_OUTPUTS = REPO_ROOT / "experiments" / "outputs"
    PROBES_DIR_V2 = EXPERIMENTS_OUTPUTS / "combined_dataset_probe_training_v2" / "binary_ova_probes"
    PROBE_SUMMARY_CSV = PROBES_DIR_V2 / "probe_summary.csv"
    APPRAISAL_PROBES_PATH = EXPERIMENTS_OUTPUTS / "combined_dataset_probe_training_v2" / "appraisal_regression_probes.pt"
    APPRAISAL_VALIDATION_CSV = EXPERIMENTS_OUTPUTS / "combined_dataset_probe_training_v2" / "appraisal_probe_validation_detail.csv"
    V3_OUTPUT_DIR = EXPERIMENTS_OUTPUTS / "combined_dataset_probe_training_v3"

# Optional v3 outputs (layer/loc rankings)
LAYERS_RANKED_EMOTION_CSV = V3_OUTPUT_DIR / "layers_ranked_emotion.csv"

# Extraction
EXTRACTION_LAYERS = list(range(16))
EXTRACTION_LOCS = [3, 6, 7]
EXTRACTION_TOKENS = [-1]

# Probe grid ablation (scope A): wider loc × token grid; outputs live under a **separate** subtree
# so canonical `01_probes/` defaults are unchanged. Layers still follow `get_extraction_layers(model_id)`.
PROBE_GRID_ABLATION_LOCS = [1, 2, 3, 5, 6, 7]
PROBE_GRID_ABLATION_TOKENS = [-1, 0, "mid"]
PROBE_GRID_ABLATION_OUTPUT_SUBDIR = "01_probes_grid_ablation"

# Appraisal dimensions (must match combined CSV columns)
COMMON_APPRAISAL = [
    "pleasantness", "unpleasantness", "attention", "effort",
    "self_responsibility", "other_responsibility", "control", "certainty",
    "goal_relevance", "urgency", "self_control", "other_control",
    "responsibility", "circumstance",
]

# Circuit / circuit-evidence: upper bound for (layer, loc) pair count when auto-selecting k.
# Actual k is chosen by validation performance in circuit_evidence (macro ROC-AUC, then accuracy).
# phase1_circuits builds circuits.json from that selection when present; otherwise uses a probe-summary heuristic capped by this max.
CIRCUIT_TOP_K_MAX = 16
# Deprecated alias (fixed k); prefer auto via circuit_evidence + circuit_top_k_selection.json
CIRCUIT_TOP_K_LAYERS = CIRCUIT_TOP_K_MAX

# Steering: contrastive emotion pairs (source -> target) for robust evaluation.
# Use all pairs that have enough val samples; emotions must exist in probe summary.
CONTRASTIVE_EMOTION_PAIRS = [
    ("anger", "joy"),
    ("sadness", "joy"),
    ("fear", "relief"),
    ("disgust", "trust"),
    ("boredom", "joy"),
    ("guilt", "relief"),
    ("shame", "trust"),
    ("surprise", "joy"),
]

# Similar / comorbid emotion relationships used in theory-facing analyses.
SIMILAR_EMOTION_PAIRS = [
    ("anger", "disgust"),
    ("joy", "relief"),
    ("guilt", "shame"),
    ("pride", "trust"),
    ("fear", "surprise"),
    ("sadness", "shame"),
    ("boredom", "sadness"),
]

# Appraisal label coupling (03_appraisal_structure/label_coupling/): pairs (dim_a, dim_b) for overlap / independence diagnostics.
APPRAISAL_LABEL_COUPLING_PAIRS = [
    ("self_control", "other_control"),
    ("pleasantness", "unpleasantness"),
    ("self_responsibility", "other_responsibility"),
]
# How to bin the conditioning dimension for conditional KDEs: "median" or "tertile".
APPRAISAL_LABEL_COUPLING_SPLIT_METHOD = "median"

# Probe-score coupling uses one shared (layer, loc) for all ridge readouts: last extraction layer, first configured loc.
# Override by setting explicit ints if needed (None = derive from model_config).
APPRAISAL_PROBE_COUPLING_LAYER: int | None = None
APPRAISAL_PROBE_COUPLING_LOC: int | None = None

# Probe training robustness: prompt variants for dataset-variant analysis
# Prompt indices from prompt_manager.build_prompt (0-9). Use 1, 4, 7 as distinct prompts.
PROMPT_INDICES_FOR_VARIANTS = [1, 4, 7]
VAL_SPLIT = 0.15
SELECTION_SPLIT = 0.15
FINAL_TEST_SPLIT = 0.15
BALANCE_RATIO = 1.0
RANDOM_STATE = 42
PROMPT_TEXT_POLICY = "generate_if_missing"
MIN_SUPPORTED_EMOTION_TRAIN_COUNT = 50
MIN_SUPPORTED_EMOTION_SELECTION_COUNT = 20
PROBE_C_GRID = [0.01, 0.1, 1.0, 10.0]
GENERATION_BENCHMARK_MAX_NEW_TOKENS = 250
GENERATION_BENCHMARK_TEMPERATURE = 0.0
GENERATION_BENCHMARK_TOP_P = 1.0
GENERATION_BENCHMARK_RUN_JUDGE = False    # Deferred: use run_behavior_judges.py after all models finish
GENERATION_BENCHMARK_JUDGE_MAX_NEW_TOKENS = 200
GENERATION_BEHAVIOR_MAX_ROWS: int | None = None  # None = full CSV; set e.g. 200 for dev runs
# Match mental-health steering calibration (see docs/MENTAL_HEALTH_STEERING_BENCHMARK.md).
GENERATION_BEHAVIOR_STEERING_UNIT_NORM = True
GENERATION_BEHAVIOR_GEN_INTERVENTION_DURING_DECODE = False
# Add appraisal_elicitation_steer using ELICITATION_APPRAISAL_PROFILE (same circuit sites as appraisal steer).
# Default True increases output rows ~25% vs four conditions; set False to skip.
GENERATION_BEHAVIOR_INCLUDE_ELICITATION_STEER = True

# Backward compatibility: Phase 2 notebook expects this path (only when running inside full repo)
PHASE1_CIRCUITS_DIR_LEGACY = REPO_ROOT / "experiments" / "outputs" / "phase_1_circuits" if not _use_standalone_probes else PIPELINE_ROOT / "outputs" / "_legacy_phase_1_circuits"


def get_model_output_dir(model_id: str) -> Path:
    return OUTPUTS_ROOT / model_id


def get_probes_dir(model_id: str) -> Path:
    """Output dir for pipeline-trained probes: outputs/<model_id>/01_probes/."""
    return get_model_output_dir(model_id) / "01_probes"


def get_probe_grid_ablation_dir(model_id: str) -> Path:
    """Output dir for probe-only grid ablation (scope A): sibling to 01_probes/, not a replacement."""
    return get_model_output_dir(model_id) / PROBE_GRID_ABLATION_OUTPUT_SUBDIR


def get_probe_paths(model_id: str):
    """
    Return probe-related paths for this model. Checks 01_probes first (after train_probes),
    then input_probes (standalone), then experiments/outputs.
    """
    from types import SimpleNamespace
    probes_root = get_probes_dir(model_id)
    has_01 = (probes_root / "binary_ova_probes" / "probe_summary.csv").exists()
    if has_01:
        return SimpleNamespace(
            probes_dir_v2=probes_root / "binary_ova_probes",
            probe_summary_csv=probes_root / "binary_ova_probes" / "probe_summary.csv",
            probe_manifest_path=probes_root / "binary_ova_probes" / "probe_manifest.json",
            appraisal_probes_path=probes_root / "appraisal_regression_probes.pt",
            appraisal_manifest_path=probes_root / "appraisal_regression_probes_manifest.json",
            appraisal_validation_csv=probes_root / "appraisal_probe_validation_detail.csv",
            v3_output_dir=probes_root / "v3",
            layers_ranked_emotion_csv=probes_root / "v3" / "layers_ranked_emotion.csv",
        )
    if _use_standalone_probes:
        return SimpleNamespace(
            probes_dir_v2=INPUT_PROBES_DIR / "binary_ova_probes",
            probe_summary_csv=INPUT_PROBES_DIR / "binary_ova_probes" / "probe_summary.csv",
            probe_manifest_path=INPUT_PROBES_DIR / "binary_ova_probes" / "probe_manifest.json",
            appraisal_probes_path=INPUT_PROBES_DIR / "appraisal_regression_probes.pt",
            appraisal_manifest_path=INPUT_PROBES_DIR / "appraisal_regression_probes_manifest.json",
            appraisal_validation_csv=INPUT_PROBES_DIR / "appraisal_probe_validation_detail.csv",
            v3_output_dir=INPUT_PROBES_DIR / "v3",
            layers_ranked_emotion_csv=INPUT_PROBES_DIR / "v3" / "layers_ranked_emotion.csv",
        )
    exp_v2 = REPO_ROOT / "experiments" / "outputs" / "combined_dataset_probe_training_v2"
    exp_v3 = REPO_ROOT / "experiments" / "outputs" / "combined_dataset_probe_training_v3"
    return SimpleNamespace(
        probes_dir_v2=exp_v2 / "binary_ova_probes",
        probe_summary_csv=exp_v2 / "binary_ova_probes" / "probe_summary.csv",
        probe_manifest_path=exp_v2 / "binary_ova_probes" / "probe_manifest.json",
        appraisal_probes_path=exp_v2 / "appraisal_regression_probes.pt",
        appraisal_manifest_path=exp_v2 / "appraisal_regression_probes_manifest.json",
        appraisal_validation_csv=exp_v2 / "appraisal_probe_validation_detail.csv",
        v3_output_dir=exp_v3,
        layers_ranked_emotion_csv=exp_v3 / "layers_ranked_emotion.csv",
    )


def get_probe_robustness_dir(model_id: str) -> Path:
    """Dataset variants and probe robustness analysis (prompted/unprompted, multi-prompt)."""
    return get_model_output_dir(model_id) / "01_probe_robustness"


def get_circuit_dir(model_id: str) -> Path:
    return get_model_output_dir(model_id) / "02_circuit"


def get_appraisal_structure_dir(model_id: str) -> Path:
    return get_model_output_dir(model_id) / "03_appraisal_structure"


def get_appraisal_in_circuit_dir(model_id: str) -> Path:
    return get_model_output_dir(model_id) / "04_appraisal_in_circuit"


def get_appraisal_theory_dir(model_id: str) -> Path:
    return get_model_output_dir(model_id) / "04_appraisal_theory"


def get_appraisal_binary_probes_dir(model_id: str) -> Path:
    return get_probes_dir(model_id) / "appraisal_binary_ova_probes"


# Appraisal theory analysis thresholds
APPRAISAL_BINARY_THRESHOLD = "median"
ONSET_THRESHOLD_EMOTION_AUC = 0.70
ONSET_THRESHOLD_APPRAISAL_CORR = 0.25

# Phase 2: now computed per model in 04_appraisal_in_circuit/. Legacy path only if you pass phase2_dir explicitly.
PHASE2_OUTPUT_DIR = (INPUT_PROBES_DIR / "phase_2_circuit_appraisal") if _use_standalone_probes else (REPO_ROOT / "experiments" / "outputs" / "phase_2_circuit_appraisal")


def get_steering_dir(model_id: str) -> Path:
    return get_model_output_dir(model_id) / "05_steering"


def get_generation_behavior_dir(model_id: str) -> Path:
    return get_model_output_dir(model_id) / "05_generation_behavior"


def get_mental_health_steering_dir(model_id: str) -> Path:
    return get_model_output_dir(model_id) / "05_mental_health_steering"


MENTAL_HEALTH_HOLDOUT_CSV = INPUT_DATA_DIR / "mental_health_dataset_holdout.csv"

MENTAL_HEALTH_LABEL_MAP = {
    0: "normal", 1: "depression", 2: "suicidal",
    3: "anxiety", 4: "bipolar", 5: "stress",
}

CONDITION_APPRAISAL_TARGETS = {
    1: {
        "name": "depression",
        "target_profile": {
            "pleasantness": 1.5, "control": 2.0, "certainty": 1.0,
            "self_responsibility": -1.0, "goal_relevance": 1.0,
        },
        "contrastive_emotion": "joy",
    },
    3: {
        "name": "anxiety",
        "target_profile": {
            "certainty": 2.0, "control": 1.5, "urgency": -1.5,
            "attention": -1.0,
        },
        "contrastive_emotion": "relief",
    },
    2: {
        "name": "suicidal",
        "target_profile": {
            "goal_relevance": 2.0, "control": 2.0, "certainty": 1.5,
            "pleasantness": 1.0,
        },
        "contrastive_emotion": "trust",
    },
    5: {
        "name": "stress",
        "target_profile": {
            "effort": -1.5, "urgency": -1.5, "self_control": 1.5,
            "control": 1.0,
        },
        "contrastive_emotion": "relief",
    },
    4: {
        "name": "bipolar",
        "target_profile": {
            "certainty": 1.5, "self_control": 2.0, "control": 1.0,
        },
        "contrastive_emotion": "trust",
    },
    0: {
        "name": "normal",
        "target_profile": {},
        "contrastive_emotion": "joy",
    },
}

# Dose-response grid for mental-health steering. Alphas multiply **L2-unit** directions at each
# circuit site (see docs/MENTAL_HEALTH_STEERING_BENCHMARK.md). Typical fluent range: ~0.5–4.
MENTAL_HEALTH_ALPHAS = [0.0, 0.5, 1.0, 2.0, 4.0, 5.0]
# Default alpha sweep for `run_steering_benchmark` (same unit-norm dose grid as mental-health benchmark).
STEERING_BENCHMARK_ALPHAS = MENTAL_HEALTH_ALPHAS
# Wider sweep for large models / ablations (still use unit-normalized vectors if MENTAL_HEALTH_STEERING_UNIT_NORM).
MENTAL_HEALTH_ALPHAS_FULL = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0]
MENTAL_HEALTH_MAX_POSTS_PER_CONDITION = 15
# L2-normalize each site's steering vector before applying alpha (strongly recommended).
MENTAL_HEALTH_STEERING_UNIT_NORM = True
# If False, steering hooks run only during prompt prefill, not on each decode step (reduces gibberish).
MENTAL_HEALTH_GEN_INTERVENTION_DURING_DECODE = False
# Alpha used in summary.md bullets and condition_method_heatmap when that alpha exists in results.
# Also used as the preferred dose for steering-benchmark row picks, generation-behavior default appraisal strength,
# baseline-probe steer alpha, and synthesis aggregation from dose_response_curves (nearest alpha in data).
MENTAL_HEALTH_REPORT_ALPHA = 2.0

# Generation behavior: defaults when CSV omits strength columns (aligned with MENTAL_HEALTH_REPORT_ALPHA).
GENERATION_BEHAVIOR_DEFAULT_APPRAISAL_STRENGTH = MENTAL_HEALTH_REPORT_ALPHA
GENERATION_BEHAVIOR_DEFAULT_EMOTION_STRENGTH = 1.0
GENERATION_BEHAVIOR_DEFAULT_COMBINED_STRENGTH = 1.5

# Default prompt framings for mental health benchmark (order stable; see PROMPT_FRAMINGS in mental_health_steering_benchmark.py).
MENTAL_HEALTH_FRAMINGS = ["counselor", "introspective", "honest_reply"]

# Goal-directed steering: push internal appraisal readout toward a fixed profile (threat / frustration / annoyance
# toward the *post text* — research construct, not a clinical assessment). Used as method `appraisal_elicitation`
# in MH benchmark and optionally in generation_behavior_benchmark. Keys must be subset of COMMON_APPRAISAL.
ELICITATION_APPRAISAL_PROFILE = {
    "unpleasantness": 1.2,
    "urgency": 1.0,
    "other_responsibility": 0.8,
    "attention": 0.6,
    "pleasantness": -1.0,
    "self_responsibility": -0.6,
    "control": -0.5,
    "certainty": -0.3,
}

# --- Baseline probe + top-k appraisal steering study (optional stage; see docs/BASELINE_PROBE_STEERING_STUDY.md) ---
# Primary text condition for steering and generation (both are always logged for probe readouts).
BASELINE_PROBE_STUDY_PRIMARY_TEXT = "unprompted"  # "unprompted" | "prompted"
# Pre-registered primary outcome proxies (for reporting; all metrics still logged).
BASELINE_PROBE_STUDY_PRIMARY_OUTCOME = "assistantlike_proxy"  # from generation_behavior heuristics
BASELINE_PROBE_STUDY_TOP_K = 3
BASELINE_PROBE_STUDY_TOP_M_APPRAISAL_DIMS = 3
# Gap between consecutive ranked linear logits must be >= this to include the next emotion (top-k + margin).
BASELINE_PROBE_STUDY_RANK_MIN_MARGIN = 0.15
BASELINE_PROBE_STUDY_MIN_TOP1_LOGIT: float | None = None  # e.g. 0.0 to drop flat readouts
BASELINE_PROBE_STUDY_STEER_ALPHA = MENTAL_HEALTH_REPORT_ALPHA
BASELINE_PROBE_STUDY_STEERING_UNIT_NORM = True
BASELINE_PROBE_STUDY_GEN_INTERVENTION_DURING_DECODE = False
BASELINE_PROBE_STUDY_MAX_ROWS: int | None = None  # None = all prompts; use small int for smoke tests
# Match generation behavior defaults unless overridden here.
BASELINE_PROBE_STUDY_MAX_NEW_TOKENS = GENERATION_BENCHMARK_MAX_NEW_TOKENS
BASELINE_PROBE_STUDY_TEMPERATURE = GENERATION_BENCHMARK_TEMPERATURE
BASELINE_PROBE_STUDY_TOP_P = GENERATION_BENCHMARK_TOP_P

# --- Runtime emotion readout for adaptive appraisal z-source (generation behavior + mental health) ---
# When True, run one unsteered forward per prompt (or per post×framing for MH), rank emotions at
# probe_summary-optimal (layer, loc), and use rank-1 as appraisal **source** for z(target)−z(source).
# Does not change emotion-only or elicitation steering; combined inherits only when appraisal uses runtime source.
RUNTIME_READOUT_USE_FOR_GENERATION_BEHAVIOR = True
RUNTIME_READOUT_USE_FOR_MENTAL_HEALTH = True
# Optional: log top-k ranked names/scores as JSON in generation_behavior CSV (can bloat files).
RUNTIME_READOUT_LOG_RANK_JSON = True
# Match baseline probe study ranking policy (top-k + margin gate).
RUNTIME_READOUT_TOP_K = BASELINE_PROBE_STUDY_TOP_K
RUNTIME_READOUT_MIN_MARGIN = BASELINE_PROBE_STUDY_RANK_MIN_MARGIN
RUNTIME_READOUT_MIN_TOP1_LOGIT: float | None = BASELINE_PROBE_STUDY_MIN_TOP1_LOGIT
# How to build the ranked emotion list for runtime rank-1 (see docs/RUNTIME_READOUT.md):
#   "circuit_sigmoid_mean" / "circuit_evidence" / "circuit" / "circuit_mean" — PRIMARY: per-site
#       sigmoid(linear OvA), then mean over each emotion's circuit sites (matches circuit_evidence).
#   "single_site" — sigmoid(linear OvA) at probe_summary-optimal (layer, loc) only.
#   "circuit_linear_mean" — mean **pre-sigmoid** linear OvA over sites (margin space; optional primary
#       if you explicitly want linear fusion).
RUNTIME_READOUT_EMOTION_MODE = "circuit_sigmoid_mean"
# When primary mode is sigmoid-mean circuit fusion, also log linear-mean circuit as separate CSV
# columns + optional JSON (readout_role=auxiliary_linear_circuit). See docs/RUNTIME_READOUT.md.
RUNTIME_READOUT_LOG_LINEAR_CIRCUIT_AUX = True
# When True and RUNTIME_READOUT_LOG_RANK_JSON, append truncated all-emotion fused scores JSON.
RUNTIME_READOUT_LOG_FULL_EMOTION_SPECTRUM = True
RUNTIME_READOUT_FULL_SPECTRUM_JSON_MAX_LEN = 8000

# After each steered generation, run an extra **unsteered** forward on ``full_text`` (prompt + assistant
# completion) and log probe readout at the **last token** — comparable to ``generated_text``.
# Adds one forward pass per intervention row (increases runtime and CSV width).
GENERATION_BEHAVIOR_POSTGEN_READOUT = True

# --- Adaptive contrastive appraisal target (generation behavior + mental health) ---
# When True, choose target emotion by max L2 distance between z-score profiles vs resolved source
# (see select_contrastive_target_emotion). Circuit sites and emotion probes use this target.
# Static CSV / taxonomy target_emotion (gen) or contrastive_emotion (MH) is fallback only.
ADAPTIVE_APPRAISAL_TARGET_ENABLED = True
ADAPTIVE_APPRAISAL_TARGET_METRIC = "l2"  # v1: L2 only
# If set, use static fallback when best contrast distance is below this threshold (None = no threshold).
ADAPTIVE_APPRAISAL_TARGET_MIN_DISTANCE: float | None = None


def get_baseline_probe_study_dir(model_id: str) -> Path:
    return get_model_output_dir(model_id) / "05_baseline_probe_steering"


def get_synthesis_dir(model_id: str | None = None) -> Path:
    """Per-model synthesis, or use OUTPUTS_ROOT / 'synthesis' for multi-model."""
    if model_id is None:
        return OUTPUTS_ROOT / "synthesis"
    return get_model_output_dir(model_id) / "06_synthesis"

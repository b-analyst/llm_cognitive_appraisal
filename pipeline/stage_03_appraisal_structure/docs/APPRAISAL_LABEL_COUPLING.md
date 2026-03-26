# Appraisal label coupling

This stage measures how **pairs of appraisal dimensions** co-vary in the **dataset labels** (and optionally in **ridge probe predictions** on the test split). It lives under:

`pipeline/outputs/<model_id>/03_appraisal_structure/label_coupling/`

It is invoked automatically at the end of `run_appraisal_structure()` (no extra pipeline step).

## What gets computed

### Label space (train split only)

For each configured pair `(A, B)` in `APPRAISAL_LABEL_COUPLING_PAIRS` in `pipeline/config.py`:

1. **Conditional overlap:** median (or tertile) split on `B`; compare the distribution of `A` in the low vs high group (KDE overlays). Same with roles swapped (`B` split on `A`).
2. **Bivariate shape:** scatter of `A` vs `B` with 2D density contours and an OLS line — a diagonal band vs a filled 2D region is visible at a glance.
3. **Quantitative metrics** (one CSV row per pair):
   - Pearson and Spearman correlation between `A` and `B`
   - Partial correlation between `A` and `B` controlling all other `COMMON_APPRAISAL` columns present (requires enough complete rows)
   - Cohen’s d (high vs low conditioning group), two-sided Mann–Whitney p-value, and two-sample KS statistic for the conditional distributions

### Probe space (test split, optional)

If **all** of the following hold:

- `02_circuit/test_hidden_states.pt` exists
- `02_circuit/test_labels.csv` contains a **`dataset_row_idx`** column. If it is missing (e.g. an older cache), the pipeline tries to **backfill** row indices from the current combined CSV and the same train/selection/test protocol as `circuit_evidence`, verifying that the `emotion` column matches the canonical test split row-for-row. If that check fails, re-run `circuit_evidence` **without** `--circuit_skip_extract` to refresh caches.
- `01_probes/appraisal_regression_probes.pt` exists for the model

then the same metrics and **matching 2×2 dashboard figures** are produced for **ridge predictions** at a **single shared** `(layer, loc)`:

- Default: last extraction layer and first configured loc (`APPRAISAL_PROBE_COUPLING_LAYER` / `APPRAISAL_PROBE_COUPLING_LOC` in `config.py` override this).

Partial correlation in probe space is usually undefined (no other appraisal columns in the synthetic dataframe) and appears as NaN.

## How to read the dashboard figures

Each pair produces one **2×2** PNG and PDF under `label_coupling/figures/`:

| Panel | Content |
|--------|--------|
| Top left | Distribution of **A** when **B** is low vs high (overlap ⇒ strong coupling / little independent variation of A given B). |
| Top right | **B** split on **A** (check asymmetry). |
| Bottom left | Scatter + 2D density + OLS line (shared axis vs two degrees of freedom). |
| Bottom right | Numeric summary and a short plain-language line. |

## Interpretation limits

- Results are **descriptive** and concern **identifiability** of dimensions in the annotations (and in linear readouts), **not** causal claims about internal model computation.
- **Heavy overlap** in the top row supports the idea that probes may struggle to separate the two constructs **if** labels do not vary independently.
- Comparing **label** dashboards to **probe** dashboards can suggest whether the model **compresses** two constructs into one factor when labels look two-dimensional.

## Standalone rerun

```bash
python -m pipeline.appraisal_label_coupling --model_id YourModelId
python -m pipeline.appraisal_label_coupling --model_id YourModelId --label_only   # skip probe coupling
```

## Related config keys

- `APPRAISAL_LABEL_COUPLING_PAIRS`
- `APPRAISAL_LABEL_COUPLING_SPLIT_METHOD` (`median` or `tertile`)
- `APPRAISAL_PROBE_COUPLING_LAYER`, `APPRAISAL_PROBE_COUPLING_LOC`

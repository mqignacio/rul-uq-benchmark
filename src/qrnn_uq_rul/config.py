"""Configuration constants for the QRNN-UQ-RUL project (Phase 2: reproduction + pipeline fixes)."""
from __future__ import annotations

from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"
CMAPSS_RAW_DIR = PROJECT_ROOT.parent / "resources" / "cmapss"
MODEL_DIR = PROJECT_ROOT / "model"
RESULTS_DIR = PROJECT_ROOT / "results"
EVAL_DIR = RESULTS_DIR / "eval"
PRED_DIR = RESULTS_DIR / "pred"

# ── N-CMAPSS Dataset ───────────────────────────────────────────────────
# Included subsets (DS08b/DS08d excluded — see doc/data_provenance.md)
FILENAMES: dict[str, str] = {
    "DS01": "N-CMAPSS_DS01-005.h5",
    "DS02": "N-CMAPSS_DS02-006.h5",
    "DS03": "N-CMAPSS_DS03-012.h5",
    "DS04": "N-CMAPSS_DS04.h5",
    "DS05": "N-CMAPSS_DS05.h5",
    "DS06": "N-CMAPSS_DS06.h5",
    "DS07": "N-CMAPSS_DS07.h5",
    "DS08a": "N-CMAPSS_DS08a-009.h5",
    "DS08c": "N-CMAPSS_DS08c-008.h5",
}

# DS08 units excluded from prior work (audit finding M6 — must be documented)
# Format: {subset: [excluded_unit_ids]}
DS08_EXCLUDED_UNITS: dict[str, list[int]] = {
    # Populated during preprocessing; see doc/data_provenance.md for full list
}

# ── C-MAPSS Dataset ────────────────────────────────────────────────────
CMAPSS_SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
CMAPSS_N_FEATURES = 26  # columns per time step in raw C-MAPSS txt files

# ── Split Mode ─────────────────────────────────────────────────────────
# "legacy"      → old random-split logic (Stage 1 port validation)
# "group_aware" → split by engine unit BEFORE sliding windows (Stage 2 fix)
SPLIT_MODE: str = "legacy"

# ── Quantiles ──────────────────────────────────────────────────────────
QUANTILES = [0.1, 0.5, 0.9]

# ── QRNN Hyperparameters ───────────────────────────────────────────────
LR = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 64
EPOCHS = 150

# ── GPR Hyperparameters ────────────────────────────────────────────────
N_RESTARTS_OPTIMIZER = 10
NORMALIZE_Y = True

# ── Efficiency Metrics ─────────────────────────────────────────────────
N_SAMPLES = 10  # Number of examples for latency measurement

# ── Sliding Window ─────────────────────────────────────────────────────
WINDOW_LENGTH = 3
STRIDE = 1

# ── Reproducibility ────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Statistical Testing ────────────────────────────────────────────────
BOOTSTRAP_N = 1000  # Number of bootstrap resamples
BOOTSTRAP_CONFIDENCE = 0.95  # Confidence level for bootstrap CIs
SIGNIFICANCE_ALPHA = 0.05  # Alpha for paired tests


# ═══════════════════════════════════════════════════════════════════════
# Phase-3 (Path-B comparative method set) constants — tasks/03-implement.md
# ═══════════════════════════════════════════════════════════════════════

# ── Nominal level (single source of truth, fixes M1) ───────────────────
# Reviewer M1: the rejected manuscript mixed 80% PI / 90% calibration /
# 95% metric target, making ranking + validity claims unjustified. Everything
# in this study — calibration, CQR adjustment, and the coverage-reference
# metrics below — uses ONE nominal level: 90% PI (miscoverage α = 0.10).
NOMINAL_COVERAGE: float = 0.90  # 1 - α; the target PI coverage probability
NOMINAL_ALPHA: float = 0.10     # α; the miscoverage rate

# ── UQ Metrics (re-anchored to NOMINAL_COVERAGE above) ─────────────────
TAU = 50  # Penalty factor for CWC
# NOTE: despite the legacy name, `ALPHA` here is the target *coverage*
# probability that RS compares against (|PICP - coverage|), NOT a miscoverage
# rate. It is set to NOMINAL_COVERAGE (0.90) so it is consistent with
# CQR_ALPHA (NOMINAL_ALPHA = 0.10). Rename to TARGET_COVERAGE in Phase 6.
KAPPA = NOMINAL_COVERAGE  # Desired coverage probability for CWC  (was 0.95, M1)
ALPHA = NOMINAL_COVERAGE  # Expected coverage probability for RS   (was 0.95, M1)

# ── CQR (Conformalized Quantile Regression) ────────────────────────────
CQR_ALPHA = NOMINAL_ALPHA  # Miscoverage rate → 90% nominal (M1)
CQR_CALIBRATION_SPLIT = 0.80  # 80% train+val, 20% calibration (of training data)

# ── Multi-Seed Experiments (raw QRNN multi-seed, fixes M4) ─────────────
SEED_LIST = [42, 123, 456, 789, 1024]  # ≥5 seeds (design §9)

# ── MDN (Mixture Density Network) — C-MDN series arm ───────────────────
# Backward-compatible port of the project's reference myMDN
# (resources/mdn_reference_pilario.py): scalar target, 5-component Gaussian
# mixture, tanh MLP backbone (12 -> 5).
MDN_N_COMPONENTS: int = 5
MDN_HIDDEN_DIMS: tuple[int, ...] = (12, 5)
MDN_SIGMA_MIN: float = 1e-6      # reference sigma floor (softplus + clamp)
MDN_DENSITY_FLOOR: float = 1e-6  # reference loss log-floor on mixture density
MDN_LR: float = 0.01
MDN_EPOCHS: int = 150
MDN_BATCH_SIZE: int = 64
MDN_VAL_SPLIT: float = 0.10
# Numerical mixture-CDF inversion (quantile extraction from MDN output).
MDN_CDF_GRID: int = 4096         # CDF evaluation grid resolution
MDN_CDF_RANGE_SIGMAS: float = 8.0  # bracket = mixture mean ± this * std
MDN_QUANTILE_TOL: float = 1e-7   # bisection tolerance on the CDF

# ── UACQR-S-style (epistemic-localized CQR), design §4.1 ───────────────
UACQR_ENSEMBLE_SIZE: int = 5   # QRNN ensemble members for the epistemic scalar
UACQR_SCORE_EPS: float = 1e-8  # floor on ĝ_ep to avoid divide-by-zero
UACQR_DISAGREEMENT_METRIC: str = "std"  # between-member spread of medians

# ── Weighted Conformal Prediction (fixes M2), [Tibshirani2020] ─────────
# Density ratio w(x)=p_test(x)/p_cal(x) estimated by a probabilistic domain
# classifier (cal-vs-test covariates), then clipped. Guarantee is marginal
# under weighted exchangeability up to weight-estimation error (audit F7).
WCP_DENSITY_RATIO_MODEL: str = "logistic"  # {"logistic", "gbm"}
WCP_WEIGHT_CLIP_QUANTILE: float = 0.99     # clip weight at this quantile
WCP_MIN_ESS_FRACTION: float = 0.10         # if ESS/n < this -> collapse, fall back

# ── WQLCP-style (loss-scaled score + weighted quantile), design §4.1 ───
# Simplified tabular port (audit F13/F11: native domain is vision/VAE).
WQLCP_RECON_HIDDEN_DIM: int = 32
WQLCP_RECON_LATENT_DIM: int = 8
WQLCP_RECON_LR: float = 1e-3
WQLCP_RECON_EPOCHS: int = 100
WQLCP_SCORE_FLOOR: float = 1e-6  # floor on the loss-scaled denominator
WQLCP_FEASIBLE: bool = True      # False -> drop the arm, retain as literature row

# ── EDL (Deep Evidential Regression, Amini et al. NeurIPS 2020) ─────────
# Normal-Inverse-Gamma (NIG) conjugate prior. Network emits (γ, ν, α, β).
EDL_NIG_NU_MIN: float = 1e-8      # ν > 0 (precision of the precision)
EDL_NIG_ALPHA_MIN: float = 1.0 + 1e-8  # α > 1 so predictive variance is finite
EDL_EVIDENCE_REG_COEF: float = 0.10  # λ annealed OOD-evidence regularizer
EDL_EVIDENCE_ANNEAL_EPOCHS: int = 60
EDL_HNN_HIDDEN_DIMS: tuple[int, ...] = (64, 32)  # header MLP backbone

# ── Quantile non-crossing (fixes M7), MCQRNN-style penalty ─────────────
# Cannon (2018) monotone composite quantile penalty: penalize
# max(0, q̂_{τ_j} - q̂_{τ_{j+1}}) between consecutive predicted quantiles.
CROSSING_PENALTY_ENABLED: bool = True
CROSSING_PENALTY_WEIGHT: float = 1.0  # λ on the hinge penalty term

# ── Statistics (fixes M5 / A6) ─────────────────────────────────────────
# TOST equivalence replaces the rejected "CI contains target" logic.
TOST_EQUIV_MARGIN: float = 0.03  # ±3 pp equivalence margin (design H1 / F12)
TOST_ALPHA: float = 0.05         # one-sided level for each TOST test
# Mixed-effects model: metric ~ method + (1|unit) + (1|seed nested in unit).
MIXED_EFFECTS_REFormula: str = "metric ~ C(method)"
MIXED_EFFECTS_GROUPS: tuple[str, str] = ("unit", "seed")  # seed ⊂ unit

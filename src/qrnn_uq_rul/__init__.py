"""
QRNN-UQ-RUL: Uncertainty quantification for aero-engine Remaining Useful Life.

Phase 2: Baseline reproduction with pipeline fixes (group-aware split, C-MAPSS loader).
Phase 3: Path-B comparative method set + methodological fixes.
"""
from .conformal import (
    calibrate_cqr_from_quantiles,
    conformal_quantile,
    cqr_max_score,
    cqr_predict,
    effective_sample_size,
    empirical_coverage,
    normalized_max_score,
    normalized_predict,
    weighted_conformal_quantile,
)
from .config import (
    ALPHA,
    BATCH_SIZE,
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_N,
    CMAPSS_SUBSETS,
    CQR_ALPHA,
    CQR_CALIBRATION_SPLIT,
    EPOCHS,
    FILENAMES,
    KAPPA,
    LR,
    N_RESTARTS_OPTIMIZER,
    QUANTILES,
    RANDOM_SEED,
    SEED_LIST,
    SIGNIFICANCE_ALPHA,
    SPLIT_MODE,
    STRIDE,
    TAU,
    WEIGHT_DECAY,
    WINDOW_LENGTH,
)
from .metrics import (
    bootstrap_ci_per_unit,
    calculate_cwc,
    calculate_nmpiw,
    calculate_picp,
    calculate_rs,
    cohens_d_per_unit,
    compute_per_unit_metrics,
    evaluate_point_estimate,
    evaluate_uq_metrics,
    mixed_effects_model,
    paired_test_per_unit,
    quantile_crossing_frequency,
    rul_asymmetric_score,
    tost_equivalence,
    winkler_score,
)
from .models import (
    QRNN,
    RULDataset,
    calibrate_cqr,
    compute_nonconformity_scores,
    predict_cqr,
    qrnn_loss,
    train_qrnn,
    train_test_gpr,
    train_test_qrnn_cqr,
    train_test_xgbq,
)
from .models_calibration import (
    raw_qrnn_multiseed,
    raw_qrnn_predict,
    train_test_cgpr,
    train_test_cxgbq,
)
from .models_edl import (
    EvidentialMLP,
    edl_loss,
    edl_predict,
    nig_predictive,
    train_edl,
    train_test_edl,
)
from .models_mdn import (
    MDN,
    fit_mdn_standardized,
    mdn_cdf,
    mdn_log_pdf,
    mdn_loss,
    mdn_quantiles,
    predict_params_data,
    scale_params_to_data,
    standardize_y,
    train_mdn,
    train_test_mdn_cqr,
)
from .models_uacqr import (
    QuantileEnsemble,
    epistemic_scale,
    fit_quantile_ensemble,
    train_test_uacqr,
)
from .models_variants import (
    estimate_density_ratio,
    train_test_distcp,
    train_test_weighted_cqr,
    train_test_wqlcp,
)
from .preprocess import (
    balance_train_dataset,
    create_intercycle_sliding_window,
    group_aware_unit_split,
    load_cmapss_subset,
    load_ncmapss_dataset,
    load_preprocessed_from_csv,
    preprocess_cmapss,
    preprocess_ds,
    preprocess_save_ncmapss,
    standardize_features,
    transform_intracycle_features,
    transform_piecewise_rul,
)
from .utils import ensure_dirs, get_device, set_seed

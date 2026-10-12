"""
Evaluation metrics for RUL prediction and uncertainty quantification.

Metrics:
- Point estimate: RMSE, MAE, R-squared
- UQ: PICP, NMPIW, CWC, RS
- Efficiency: Model size, inference latency
- Statistical: Bootstrap CIs, paired tests, Cohen's d
"""
from __future__ import annotations

import logging
import pickle
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import (
    ALPHA,
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_N,
    KAPPA,
    N_SAMPLES,
    QUANTILES,
    SIGNIFICANCE_ALPHA,
    TAU,
)

logger = logging.getLogger(__name__)


# ── Point Estimate Metrics ─────────────────────────────────────────────


def evaluate_point_estimate(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> pd.DataFrame:
    """
    Evaluate point estimate predictions using RMSE, MAE, and R-squared.

    Args:
        y_true: True RUL values.
        y_pred: Predicted RUL values.

    Returns:
        DataFrame with metric names and values.
    """
    y_true_arr = np.asarray(y_true).ravel()
    y_pred_arr = np.asarray(y_pred).ravel()

    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    r2 = float(r2_score(y_true_arr, y_pred_arr))

    logger.info("RMSE: %.2f, MAE: %.2f, R²: %.2f", rmse, mae, r2)

    return pd.DataFrame({"Metric": ["RMSE", "MAE", "R²"], "Value": [rmse, mae, r2]})


# ── UQ Metrics ─────────────────────────────────────────────────────────


def calculate_picp(
    y_true: np.ndarray | pd.Series,
    y_pred_lower: np.ndarray | pd.Series,
    y_pred_upper: np.ndarray | pd.Series,
) -> float:
    """
    Prediction Interval Coverage Probability (PICP): fraction of true values in the PI.

    Args:
        y_true: True RUL values.
        y_pred_lower: Lower bound of prediction interval.
        y_pred_upper: Upper bound of prediction interval.

    Returns:
        PICP value (0-1). Higher is better.
    """
    y_true_arr = np.asarray(y_true).ravel()
    lower = np.asarray(y_pred_lower).ravel()
    upper = np.asarray(y_pred_upper).ravel()
    in_pi = np.logical_and(y_true_arr >= lower, y_true_arr <= upper)
    return float(np.mean(in_pi))


def calculate_nmpiw(
    y_true: np.ndarray | pd.Series,
    y_pred_lower: np.ndarray | pd.Series,
    y_pred_upper: np.ndarray | pd.Series,
) -> float:
    """
    Normalized Mean Prediction Interval Width (NMPIW), normalized by mean |y_true|.

    Args:
        y_true: True RUL values.
        y_pred_lower: Lower bound.
        y_pred_upper: Upper bound.

    Returns:
        NMPIW value. Lower is better.
    """
    y_true_arr = np.asarray(y_true).ravel()
    lower = np.asarray(y_pred_lower).ravel()
    upper = np.asarray(y_pred_upper).ravel()
    interval_width = upper - lower
    return float(np.mean(interval_width) / np.mean(np.abs(y_true_arr)))


def calculate_cwc(
    y_true: np.ndarray | pd.Series,
    y_pred_lower: np.ndarray | pd.Series,
    y_pred_upper: np.ndarray | pd.Series,
    tau: float = TAU,
    kappa: float = KAPPA,
) -> float:
    """
    Coverage Width-based Criterion (CWC): combines PICP and NMPIW with under-coverage penalty.

    Args:
        y_true: True RUL values.
        y_pred_lower: Lower bound.
        y_pred_upper: Upper bound.
        tau: Penalty factor (default 50).
        kappa: Desired coverage (default 0.95).

    Returns:
        CWC value. Lower is better.
    """
    picp = calculate_picp(y_true, y_pred_lower, y_pred_upper)
    nmpiw = calculate_nmpiw(y_true, y_pred_lower, y_pred_upper)
    penalty = max(0, kappa - picp)
    return float(nmpiw * (1 + tau * penalty))


def calculate_rs(
    y_true: np.ndarray | pd.Series,
    y_pred_lower: np.ndarray | pd.Series,
    y_pred_upper: np.ndarray | pd.Series,
    alpha: float = ALPHA,
) -> float:
    """
    Reliability Score (RS): |PICP - alpha|. Measures deviation from expected coverage.

    Args:
        y_true: True RUL values.
        y_pred_lower: Lower bound.
        y_pred_upper: Upper bound.
        alpha: Expected coverage probability (default 0.95).

    Returns:
        RS value. Lower is better.
    """
    picp = calculate_picp(y_true, y_pred_lower, y_pred_upper)
    return float(abs(picp - alpha))


def evaluate_uq_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: Dict[str, np.ndarray | pd.Series],
    tau: float = TAU,
    kappa: float = KAPPA,
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """
    Evaluate all UQ metrics (PICP, NMPIW, CWC, RS) for a model's predictions.

    Args:
        y_true: True RUL values.
        y_pred: Dict with keys 'lower' and 'upper'.
        tau: CWC penalty factor.
        kappa: CWC desired coverage.
        alpha: RS expected coverage.

    Returns:
        DataFrame with PICP, NMPIW, CWC, and RS values.
    """
    lower = y_pred["lower"]
    upper = y_pred["upper"]

    picp = calculate_picp(y_true, lower, upper)
    nmpiw = calculate_nmpiw(y_true, lower, upper)
    cwc = calculate_cwc(y_true, lower, upper, tau=tau, kappa=kappa)
    rs = calculate_rs(y_true, lower, upper, alpha=alpha)

    # `alpha` here is the COVERAGE reference for RS (0.90); winkler_score
    # expects the MISOVERAGE rate (0.10) for its 2*alpha/(1-alpha) penalty
    # (methodology sec.5.2 "at alpha=0.10"). Passing the coverage reference
    # would inflate the miss penalty 81x (18 vs 0.222).
    wink = winkler_score(y_true, lower, upper, alpha=1.0 - alpha)

    return pd.DataFrame({
        "Metric": ["PICP", "NMPIW", "CWC", "RS", "Winkler"],
        "Value": [picp, nmpiw, cwc, rs, wink],
    })


# ── Winkler Score (fixes A5) ───────────────────────────────────────────────


def winkler_score(
    y_true: np.ndarray | pd.Series,
    y_pred_lower: np.ndarray | pd.Series,
    y_pred_upper: np.ndarray | pd.Series,
    alpha: float = 1.0 - ALPHA,  # miscoverage rate (default 0.10)
) -> float:
    """
    Winkler score (interval score): jointly rewards narrow intervals and penalizes misses.

    For a (1-alpha)*100% prediction interval [L, U]:
        S_W = (U - L) + (2*alpha/(1-alpha)) * [(L - y) * 1{y < L} + (y - U) * 1{y > U}]

    Reference: Winkler (1972); Gneiting & Raftery (2007).

    Args:
        y_true: True RUL values.
        y_pred_lower: Lower bound of PI.
        y_pred_upper: Upper bound of PI.
        alpha: Miscoverage rate (e.g. 0.10 for 90% PI). Default: 1 - ALPHA.

    Returns:
        Mean Winkler score. Lower is better.
    """
    y_arr = np.asarray(y_true).ravel()
    lower = np.asarray(y_pred_lower).ravel()
    upper = np.asarray(y_pred_upper).ravel()

    width = upper - lower
    penalty_coeff = 2.0 * alpha / (1.0 - alpha)
    below_penalty = np.maximum(lower - y_arr, 0.0)
    above_penalty = np.maximum(y_arr - upper, 0.0)

    scores = width + penalty_coeff * (below_penalty + above_penalty)
    return float(np.mean(scores))


# ── RUL Asymmetric Score (fixes M7) ─────────────────────────────────────────


def rul_asymmetric_score(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    a_late: float = 13.0,
    a_early: float = 10.0,
) -> float:
    """
    PHM2008-style asymmetric RUL scoring function.

    Penalizes over-prediction (predicting longer life than actual) more heavily
    than under-prediction in late-life; the reverse in early life.

    Piecewise exponential:
        if d = y_true - y_pred >= 0 (under-prediction):
            s = exp(d / a_early) - 1
        else (over-prediction):
            s = exp(-d / a_late) - 1

    Args:
        y_true: True RUL values.
        y_pred: Predicted RUL point estimates.
        a_late: Scale parameter for late-life over-prediction penalty.
        a_early: Scale parameter for early-life under-prediction penalty.

    Returns:
        Mean asymmetric score. Lower is better.
    """
    y_arr = np.asarray(y_true).ravel()
    p_arr = np.asarray(y_pred).ravel()
    d = y_arr - p_arr

    scores = np.where(d >= 0, np.exp(d / a_early) - 1.0, np.exp(-d / a_late) - 1.0)
    return float(np.mean(scores))


# ── Quantile Crossing Frequency (fixes M7) ──────────────────────────────────


def quantile_crossing_frequency(
    predictions: np.ndarray,
    quantile_levels: Optional[List[float]] = None,
) -> float:
    """
    Fraction of samples where predicted quantiles violate monotonicity.

    A crossing occurs when q_hat_{tau_j} > q_hat_{tau_{j+1}} for tau_j < tau_{j+1}.
    Reports the fraction of samples with at least one crossing.

    Args:
        predictions: Array (N, n_quantiles) of predicted quantiles, ordered by tau.
        quantile_levels: Sorted list of quantile levels (for validation only).

    Returns:
        Crossing frequency in [0, 1]. Lower is better (0 = no crossings).
    """
    preds = np.asarray(predictions)
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)

    if quantile_levels is not None:
        assert sorted(quantile_levels) == list(quantile_levels), \
            "quantile_levels must be sorted ascending"
        assert preds.shape[1] == len(quantile_levels), \
            f"predictions cols ({preds.shape[1]}) != len(quantile_levels) ({len(quantile_levels)})"

    if preds.shape[1] < 2:
        return 0.0  # single quantile cannot cross

    # Check adjacent pairs for violation
    diffs = np.diff(preds, axis=1)  # (N, n_quantiles-1)
    has_crossing = np.any(diffs < 0, axis=1)  # (N,) — any pair violates monotonicity
    return float(np.mean(has_crossing))


# ── Per-Unit Metrics ───────────────────────────────────────────────────


def compute_per_unit_metrics(
    A_test: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Compute UQ metrics per test unit (grouped by engine unit).

    Args:
        A_test: Auxiliary data with 'unit' column.
        y_test: True RUL values (N,).
        y_pred: Dict with 'lower', 'upper' keys.

    Returns:
        DataFrame indexed by unit with per-unit UQ metrics.
    """
    units = A_test["unit"].unique()
    results = []

    y_test_arr = np.asarray(y_test).ravel()
    lower_arr = np.asarray(y_pred["lower"]).ravel()
    upper_arr = np.asarray(y_pred["upper"]).ravel()

    for unit in units:
        mask = (A_test["unit"] == unit).values
        y_unit = y_test_arr[mask]
        lower_unit = lower_arr[mask]
        upper_unit = upper_arr[mask]

        if len(y_unit) < 2:
            continue

        results.append({
            "unit": unit,
            "PICP": calculate_picp(y_unit, lower_unit, upper_unit),
            "NMPIW": calculate_nmpiw(y_unit, lower_unit, upper_unit),
            "CWC": calculate_cwc(y_unit, lower_unit, upper_unit),
            "RS": calculate_rs(y_unit, lower_unit, upper_unit),
            "n_samples": len(y_unit),
        })

    return pd.DataFrame(results).set_index("unit")


# ── Statistical Significance Tests ─────────────────────────────────────


def bootstrap_ci_per_unit(
    A_test: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: Dict[str, np.ndarray],
    metric_name: str = "PICP",
    n_bootstrap: int = BOOTSTRAP_N,
    confidence: float = BOOTSTRAP_CONFIDENCE,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for a global metric via unit-level resampling.

    Args:
        A_test: Auxiliary data with 'unit' column.
        y_test: True RUL values (N,).
        y_pred: Dict with 'lower', 'upper' keys.
        metric_name: Metric to compute CI for.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level (default 0.95).
        seed: Random seed.

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper).
    """
    rng = np.random.default_rng(seed)
    units = A_test["unit"].unique()
    n_units = len(units)

    metric_func = {
        "PICP": calculate_picp, "NMPIW": calculate_nmpiw,
        "CWC": calculate_cwc, "RS": calculate_rs,
    }[metric_name]

    y_test_arr = np.asarray(y_test).ravel()
    lower_arr = np.asarray(y_pred["lower"]).ravel()
    upper_arr = np.asarray(y_pred["upper"]).ravel()

    point_est = metric_func(y_test_arr, lower_arr, upper_arr)

    bootstrap_estimates: List[float] = []
    for _ in range(n_bootstrap):
        sampled_units = rng.choice(units, size=n_units, replace=True)
        sampled_mask = A_test["unit"].isin(sampled_units).values
        boot_est = metric_func(
            y_test_arr[sampled_mask], lower_arr[sampled_mask], upper_arr[sampled_mask]
        )
        bootstrap_estimates.append(boot_est)

    a = 1 - confidence
    ci_lower = float(np.percentile(bootstrap_estimates, 100 * a / 2))
    ci_upper = float(np.percentile(bootstrap_estimates, 100 * (1 - a / 2)))

    return point_est, ci_lower, ci_upper


def paired_test_per_unit(
    per_unit_A: pd.DataFrame,
    per_unit_B: pd.DataFrame,
    metric_name: str = "PICP",
    alpha: float = SIGNIFICANCE_ALPHA,
) -> Tuple[float, float, bool]:
    """
    Paired t-test between two methods on per-unit metrics.

    Args:
        per_unit_A: Per-unit metrics for method A.
        per_unit_B: Per-unit metrics for method B.
        metric_name: Metric to test.
        alpha: Significance level.

    Returns:
        Tuple of (t_statistic, p_value, is_significant).
    """
    common_units = per_unit_A.index.intersection(per_unit_B.index)
    values_A = per_unit_A.loc[common_units, metric_name].values
    values_B = per_unit_B.loc[common_units, metric_name].values

    t_stat, p_value = scipy_stats.ttest_rel(values_A, values_B)
    return float(t_stat), float(p_value), bool(p_value < alpha)


def cohens_d_per_unit(
    per_unit_A: pd.DataFrame,
    per_unit_B: pd.DataFrame,
    metric_name: str = "PICP",
) -> float:
    """
    Cohen's d effect size between two methods on per-unit metrics (paired).

    Args:
        per_unit_A: Per-unit metrics for method A.
        per_unit_B: Per-unit metrics for method B.
        metric_name: Metric to compute effect size for.

    Returns:
        Cohen's d (positive means A > B for that metric).
    """
    common_units = per_unit_A.index.intersection(per_unit_B.index)
    values_A = per_unit_A.loc[common_units, metric_name].values
    values_B = per_unit_B.loc[common_units, metric_name].values

    diff = values_A - values_B
    s_pooled = np.sqrt((np.var(values_A, ddof=1) + np.var(values_B, ddof=1)) / 2)

    if s_pooled == 0:
        return 0.0
    return float(np.mean(diff) / s_pooled)


# ── TOST Equivalence Test (fixes M5) ─────────────────────────────────────


def tost_equivalence(
    per_unit_A: pd.DataFrame,
    per_unit_B: pd.DataFrame,
    metric_name: str = "PICP",
    margin: Optional[float] = None,
    alpha: float = 0.05,
) -> Dict[str, object]:
    """
    Two One-Sided Tests (TOST) equivalence test on paired per-unit metrics.

    Tests whether the mean difference between methods A and B falls within
    [-margin, +margin]. This replaces the rejected "CI contains target" logic.

    H0: |mean(A - B)| >= margin  (not equivalent)
    H1: |mean(A - B)| <   margin  (equivalent)

    Rejection requires BOTH one-sided tests to be significant at level alpha.

    Args:
        per_unit_A: Per-unit metrics for method A (indexed by unit).
        per_unit_B: Per-unit metrics for method B (indexed by unit).
        metric_name: Metric column to compare.
        margin: Equivalence margin. Default: TOST_EQUIV_MARGIN from config (0.03).
        alpha: Significance level for each one-sided test.

    Returns:
        Dict with keys: 'mean_diff', 't_lower', 'p_lower', 't_upper', 'p_upper',
        'equivalent' (bool), 'margin'.
    """
    from .config import TOST_EQUIV_MARGIN

    if margin is None:
        margin = TOST_EQUIV_MARGIN

    common_units = per_unit_A.index.intersection(per_unit_B.index)
    values_A = per_unit_A.loc[common_units, metric_name].values
    values_B = per_unit_B.loc[common_units, metric_name].values

    diff = values_A - values_B
    n = len(diff)
    if n < 2:
        return {
            "mean_diff": float(np.mean(diff)) if n == 1 else 0.0,
            "t_lower": 0.0, "p_lower": 1.0,
            "t_upper": 0.0, "p_upper": 1.0,
            "equivalent": False, "margin": margin,
        }

    mean_d = float(np.mean(diff))
    se_d = float(np.std(diff, ddof=1) / np.sqrt(n))

    if se_d == 0.0:
        # Perfectly identical differences — trivially equivalent if within margin
        equivalent = abs(mean_d) < margin
        return {
            "mean_diff": mean_d, "t_lower": 0.0, "p_lower": 0.0 if equivalent else 1.0,
            "t_upper": 0.0, "p_upper": 0.0 if equivalent else 1.0,
            "equivalent": equivalent, "margin": margin,
        }

    # Lower bound test: H0: mean_d <= -margin  vs  H1: mean_d > -margin
    t_lower = (mean_d - (-margin)) / se_d
    p_lower = float(1.0 - scipy_stats.t.cdf(t_lower, df=n - 1))

    # Upper bound test: H0: mean_d >= +margin  vs  H1: mean_d < +margin
    t_upper = (mean_d - margin) / se_d
    p_upper = float(scipy_stats.t.cdf(t_upper, df=n - 1))

    equivalent = (p_lower < alpha) and (p_upper < alpha)

    return {
        "mean_diff": mean_d,
        "t_lower": float(t_lower), "p_lower": p_lower,
        "t_upper": float(t_upper), "p_upper": p_upper,
        "equivalent": bool(equivalent), "margin": margin,
    }


# ── Mixed-Effects Model (fixes M5 / A6) ──────────────────────────────────


def mixed_effects_model(
    results_df: pd.DataFrame,
    metric_col: str = "value",
    method_col: str = "method",
    unit_col: str = "unit",
    seed_col: str = "seed",
) -> Dict[str, object]:
    """
    Linear mixed-effects model for hierarchical evaluation: seed nested in unit.

    Model: metric ~ C(method) + (1|unit) + (1|unit:seed)

    The random-intercept structure accounts for repeated measurements (seeds)
    within engine units. Uses statsmodels MixedLM with variance components.

    Args:
        results_df: DataFrame with columns [metric_col, method_col, unit_col, seed_col].
            Each row is one observation (one method x unit x seed cell).
        metric_col: Name of the numeric response column.
        method_col: Name of the categorical method/factor column.
        unit_col: Name of the grouping (random intercept) column.
        seed_col: Name of the nested factor (seed within unit) column.

    Returns:
        Dict with keys: 'fixed_effects' (Series), 'p_values' (Series),
        'random_effect_variances' (dict), 'converged' (bool), 'aic' (float),
        'bic' (float), 'model_summary' (str).
    """
    import statsmodels.api as sm

    # Build nesting key: seed nested within unit
    df = results_df.copy()
    df["unit_seed"] = df[unit_col].astype(str) + ":" + df[seed_col].astype(str)

    formula = "{} ~ C({})".format(metric_col, method_col)

    try:
        # Primary model: random intercept by unit + seed-nested-in-unit variance
        # component, i.e. metric ~ C(method) + (1|unit) + (1|unit:seed).
        # MixedLM.from_formula supports `vc_formula`; the smf formula wrapper does
        # not accept `vc`, so we use the class API directly.
        model = sm.MixedLM.from_formula(
            formula, df,
            groups=df[unit_col],
            vc_formula={"unit_seed": "0 + C(unit_seed)"},
        )
        fit = model.fit(reml=True, maxiter=200)
        converged = bool(fit.converged)
    except Exception as exc:
        logger.warning("MixedLM with nested vc failed (%s); falling back to unit-only groups", exc)
        try:
            model = sm.MixedLM.from_formula(formula, df, groups=df[unit_col])
            fit = model.fit(reml=True, maxiter=200)
            converged = bool(fit.converged)
        except Exception as exc2:
            logger.error("MixedLM fit completely failed: %s", exc2)
            return {
                "fixed_effects": pd.Series(dtype=float),
                "p_values": pd.Series(dtype=float),
                "random_effect_variances": {},
                "converged": False,
                "aic": float("nan"),
                "bic": float("nan"),
                "model_summary": str(exc2),
            }

    fe = fit.fe_params
    pvals = fit.pvalues

    # Extract variance components: group random intercept + nested seed term
    re_var = {}
    if getattr(fit, "cov_re", None) is not None:
        cov = np.asarray(fit.cov_re)
        re_var["group_var"] = float(cov.ravel()[0]) if cov.size else 0.0
    else:
        re_var["group_var"] = 0.0
    if getattr(fit, "vcomp", None) is not None and len(np.atleast_1d(fit.vcomp)) > 0:
        re_var["vc_unit_seed"] = float(np.sum(np.atleast_1d(fit.vcomp)))
    else:
        re_var["vc_unit_seed"] = 0.0

    return {
        "fixed_effects": fe,
        "p_values": pvals,
        "random_effect_variances": re_var,
        "converged": converged,
        "aic": float(fit.aic) if np.isfinite(fit.aic) else float("nan"),
        "bic": float(fit.bic) if np.isfinite(fit.bic) else float("nan"),
        "model_summary": str(fit.summary()),
    }


# ── Efficiency Metrics ─────────────────────────────────────────────────


def get_model_size(model, quantile: bool = False, pytorch: bool = False) -> float:
    """
    Get model size in MB.

    Args:
        model: The model object.
        quantile: Whether it's a dict of quantile models.
        pytorch: Whether it's a PyTorch model.

    Returns:
        Model size in MB.
    """
    if quantile:
        return sum(sys.getsizeof(pickle.dumps(model[q])) for q in QUANTILES) / (1024 * 1024)
    elif pytorch:
        return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    else:
        return sys.getsizeof(pickle.dumps(model)) / (1024 * 1024)


def measure_latency(
    model,
    X,
    quantiles: Optional[List[float]] = None,
    pytorch: bool = False,
    device: Optional[torch.device] = None,
) -> float:
    """
    Measure average inference latency in milliseconds over N_SAMPLES runs.

    Args:
        model: The trained model.
        X: Input data.
        quantiles: Quantiles for multi-model quantile regression.
        pytorch: Whether the model is a PyTorch model.
        device: PyTorch device.

    Returns:
        Average latency in milliseconds.
    """
    if device is None:
        device = torch.device("cpu")

    start_time = time.time()
    for _ in range(N_SAMPLES):
        if quantiles is not None:
            for q in quantiles:
                model[q].predict(X)
        elif pytorch:
            X_tensor = torch.FloatTensor(X.to_numpy() if hasattr(X, "to_numpy") else X).to(device)
            model.eval()
            with torch.no_grad():
                model(X_tensor)
        else:
            model.predict(X)
    end_time = time.time()

    return (end_time - start_time) / N_SAMPLES * 1000


def evaluate_efficiency_metrics(
    model,
    X_test,
    quantiles: Optional[List[float]] = None,
    pytorch_model=None,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """
    Evaluate model efficiency metrics (size and latency).

    Args:
        model: The trained model.
        X_test: Test input data.
        quantiles: Quantiles for quantile regression models.
        pytorch_model: PyTorch model instance (if applicable).
        device: PyTorch device.

    Returns:
        DataFrame with model size and latency values.
    """
    model_size = get_model_size(
        model, quantile=(quantiles is not None), pytorch=(pytorch_model is not None)
    )
    latency = measure_latency(
        model, X_test, quantiles=quantiles,
        pytorch=(pytorch_model is not None), device=device,
    )

    return pd.DataFrame({
        "Metric": ["Model Size (MB)", f"Prediction Latency (ms) @ {N_SAMPLES} samples"],
        "Value": [model_size, latency],
    })

"""
Generic, base-predictor-agnostic conformal-prediction utilities.

This module is the single conformal *stage* shared by every distribution-free
method arm in the Path-B comparative study (C-QRNN, C-MDN series, UACQR-S,
distributional-CP/HPD, Weighted-CP CQR, WQLCP-style, and the calibrated
model-internal baselines C-XGBQ / GPR / EDL). Routing all arms through these
functions is the mechanical implementation of fix **M4** ("calibrate *all*
baselines with the same procedure").

Conventions
-----------
* ``alpha`` is always the **miscoverage rate** (e.g. 0.10 → 90% PI). Fix **M1**
  is enforced by defaulting to ``NOMINAL_ALPHA`` from :mod:`qrnn_uq_rul.config`.
* Nonconformity scores are oriented so that *smaller means more conforming*;
  the conformal quantile is the ``ceil((n+1)(1-alpha))/n`` empirical quantile.
* Intervals are returned as a dict with ``lower`` / ``median`` / ``upper``
  NumPy arrays, matching the Phase-2 prediction-dict contract.
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np

from .config import NOMINAL_ALPHA

logger = logging.getLogger(__name__)


# ── Split-conformal calibration ────────────────────────────────────────


def conformal_quantile(scores: np.ndarray, alpha: float = NOMINAL_ALPHA) -> float:
    """
    Split-conformal calibration quantile q_hat for nonconformity scores.

    Canonical finite-sample split-conformal threshold: the smallest order
    statistic whose empirical CDF reaches ``min(ceil((n+1)(1-alpha))/n, 1)``.
    Delegates to :func:`weighted_conformal_quantile` with uniform weights so
    every arm shares ONE convention (fix M4) and the weighted/unweighted paths
    are consistent by construction.

    Args:
        scores: Nonconformity scores on the calibration set, shape (n,).
        alpha: Miscoverage rate (default 0.10 → 90% PI, fix M1).

    Returns:
        The conformal adjustment / threshold q_hat (a scalar float).
    """
    s = np.asarray(scores, dtype=float).ravel()
    if s.size == 0:
        raise ValueError("conformal_quantile requires at least one calibration score")
    return weighted_conformal_quantile(s, np.ones_like(s), alpha=alpha)


def empirical_coverage(scores: np.ndarray, q_hat: float) -> float:
    """
    Empirical coverage implied by a threshold: fraction of scores <= q_hat.

    ``scores`` here are nonconformity scores in raw (non-normalized) space, so
    ``score <= q_hat`` is a covered point. Returns a value in [0, 1].
    """
    s = np.asarray(scores, dtype=float).ravel()
    if s.size == 0:
        return float("nan")
    return float(np.mean(s <= q_hat))


# ── CQR (conformalized quantile regression) score + interval ───────────


def cqr_max_score(
    y: np.ndarray,
    q_lower: np.ndarray,
    q_upper: np.ndarray,
) -> np.ndarray:
    """
    Conformalized-quantile-regression max nonconformity score [Romano2019].

    E_i = max(q_lower_i - y_i, y_i - q_upper_i). Points inside the raw interval
    yield E_i <= 0; points outside yield a positive distance-to-interval.

    Args:
        y: True target values, shape (n,).
        q_lower: Predicted lower quantile (e.g. tau=alpha/2), shape (n,).
        q_upper: Predicted upper quantile (e.g. tau=1-alpha/2), shape (n,).

    Returns:
        Score array, shape (n,).
    """
    y = np.asarray(y, dtype=float).ravel()
    lo = np.asarray(q_lower, dtype=float).ravel()
    hi = np.asarray(q_upper, dtype=float).ravel()
    return np.maximum(lo - y, y - hi)


def cqr_predict(
    q_lower: np.ndarray,
    q_median: np.ndarray,
    q_upper: np.ndarray,
    e_hat: float,
) -> Dict[str, np.ndarray]:
    """
    Build CQR-adjusted intervals [q_lower - e_hat, q_upper + e_hat].

    The median is left unadjusted (it is the point estimate). Returns a dict
    with ``lower`` / ``median`` / ``upper`` NumPy arrays.
    """
    lo = np.asarray(q_lower, dtype=float).ravel()
    med = np.asarray(q_median, dtype=float).ravel()
    hi = np.asarray(q_upper, dtype=float).ravel()
    return {"lower": lo - e_hat, "median": med, "upper": hi + e_hat}


def calibrate_cqr_from_quantiles(
    y_cal: np.ndarray,
    q_lower_cal: np.ndarray,
    q_upper_cal: np.ndarray,
    alpha: float = NOMINAL_ALPHA,
) -> float:
    """
    One-stop CQR calibration: compute max-scores on the calibration set and
    return the conformal adjustment E_hat. Shared by every quantile-based arm
    (QRNN, MDN series, XGBQ) so they use an identical procedure (fix M4).
    """
    scores = cqr_max_score(y_cal, q_lower_cal, q_upper_cal)
    return conformal_quantile(scores, alpha=alpha)


# ── Normalized / locally-adaptive conformal score ──────────────────────


def normalized_max_score(
    y: np.ndarray,
    q_lower: np.ndarray,
    q_upper: np.ndarray,
    scale: np.ndarray,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Epistemic/locally-adaptive normalized CQR score (UACQR-S family).

    s_i = max(q_lower_i - y_i, y_i - q_upper_i) / max(g_i, eps)

    where ``g`` is a strictly positive local dispersion estimate (ensemble
    disagreement, density width, etc.). The same ``g`` scales the test-time
    interval reconstruction so the procedure stays a valid monotone transform
    of the score.

    Args:
        y: True targets (n,).
        q_lower: Predicted lower quantile (n,).
        q_upper: Predicted upper quantile (n,).
        scale: Local dispersion g_i > 0 (n,).
        eps: Floor on the denominator to avoid divide-by-zero.

    Returns:
        Tuple (scores, floored_scale).
    """
    y = np.asarray(y, dtype=float).ravel()
    lo = np.asarray(q_lower, dtype=float).ravel()
    hi = np.asarray(q_upper, dtype=float).ravel()
    g = np.maximum(np.asarray(scale, dtype=float).ravel(), eps)
    scores = np.maximum(lo - y, y - hi) / g
    return scores, g


def normalized_predict(
    q_lower: np.ndarray,
    q_median: np.ndarray,
    q_upper: np.ndarray,
    scale: np.ndarray,
    q_hat: float,
    eps: float = 1e-8,
) -> Dict[str, np.ndarray]:
    """
    Reconstruct intervals for a normalized CQR score:
    [q_lower - q_hat*g, q_upper + q_hat*g] with g the same local scale used in
    ``normalized_max_score`` (floored at ``eps``).
    """
    lo = np.asarray(q_lower, dtype=float).ravel()
    med = np.asarray(q_median, dtype=float).ravel()
    hi = np.asarray(q_upper, dtype=float).ravel()
    g = np.maximum(np.asarray(scale, dtype=float).ravel(), eps)
    return {"lower": lo - q_hat * g, "median": med, "upper": hi + q_hat * g}


# ── Weighted-conformal quantile (fix M2) ───────────────────────────────


def weighted_conformal_quantile(
    scores: np.ndarray,
    weights: np.ndarray,
    alpha: float = NOMINAL_ALPHA,
) -> float:
    """
    Weighted split-conformal quantile under covariate shift [Tibshirani2020].

    Returns the smallest q such that the *weighted* empirical CDF of the
    calibration scores reaches the ``min(ceil((n+1)(1-alpha))/n, 1)`` level,
    where each calibration point contributes mass proportional to its
    likelihood-ratio weight w_i (already clipped upstream). With all weights
    equal this collapses to :func:`conformal_quantile`.

    Args:
        scores: Calibration nonconformity scores (n,).
        weights: Non-negative likelihood-ratio weights (n,).
        alpha: Miscoverage rate (fix M1).

    Returns:
        The weighted conformal threshold q_hat.
    """
    s = np.asarray(scores, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    if s.size != w.size:
        raise ValueError("scores and weights must have equal length")
    if s.size == 0:
        raise ValueError("weighted_conformal_quantile requires at least one score")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")

    n = s.size
    level = min(np.ceil((n + 1) * (1.0 - alpha)) / n, 1.0)

    # Weighted empirical CDF: sort by score, accumulate normalized weights.
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    w_sorted = w[order]
    total = w_sorted.sum()
    if total <= 0:
        # Degenerate weights -> fall back to unweighted split conformal.
        logger.warning("weighted_conformal_quantile: non-positive total weight; "
                       "falling back to unweighted quantile")
        return conformal_quantile(s, alpha=alpha)

    cdf = np.cumsum(w_sorted) / total
    idx = int(np.searchsorted(cdf, level, side="left"))
    idx = min(idx, n - 1)
    return float(s_sorted[idx])


def effective_sample_size(weights: np.ndarray) -> float:
    """
    Kish effective sample size of a weight vector: (sum w)^2 / sum w^2.

    Reported alongside Weighted-CP coverage as a weight-collapse diagnostic
    (audit F7). Equals n for uniform weights; → 1 when a single weight dominates.
    """
    w = np.asarray(weights, dtype=float).ravel()
    denom = np.sum(w ** 2)
    if denom == 0:
        return 0.0
    return float((np.sum(w) ** 2) / denom)

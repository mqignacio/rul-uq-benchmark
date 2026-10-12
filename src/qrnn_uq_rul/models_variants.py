"""
The remaining distribution-free arms of the Path-B comparative study
(design §4.1): **Weighted-CP CQR** (fix M2), **Distributional-CP / HPD-split**
(density level-set, [Izbicki2022b]/[Harris2026]), and a simplified
**WQLCP-style** arm (reconstruction-loss-scaled score + weighted quantile,
[alijani_wqlcp_2025]).

All three route their calibration through :mod:`qrnn_uq_rul.conformal`, so every
arm shares ONE split-/weighted-conformal procedure (fix M4). Coverage
guarantees are marginal-under-exchangeability (split-CP) or marginal-under-
weighted-exchangeability with *estimated* weights (Weighted-CP/WQLCP, so
approximate — audit F7, design §8); both are reported, not asserted as
unconditional.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .conformal import (
    conformal_quantile,
    cqr_max_score,
    effective_sample_size,
    normalized_max_score,
    normalized_predict,
    weighted_conformal_quantile,
)
from .config import (
    CQR_CALIBRATION_SPLIT,
    NOMINAL_ALPHA,
    RANDOM_SEED,
    WCP_DENSITY_RATIO_MODEL,
    WCP_MIN_ESS_FRACTION,
    WCP_WEIGHT_CLIP_QUANTILE,
    WQLCP_FEASIBLE,
    WQLCP_RECON_EPOCHS,
    WQLCP_RECON_HIDDEN_DIM,
    WQLCP_RECON_LATENT_DIM,
    WQLCP_RECON_LR,
    WQLCP_SCORE_FLOOR,
)
from .models import train_qrnn
from .models_mdn import MDN, fit_mdn_standardized, predict_params_data
from .utils import get_device

logger = logging.getLogger(__name__)


# ── QRNN quantile helper (shared by the QRNN-based arms) ───────────────


def _qrnn_quantiles(
    model: nn.Module, X: np.ndarray, device: torch.device
) -> np.ndarray:
    """Return QRNN quantile predictions ``(N, 3)`` for the columns QUANTILES."""
    model.eval()
    with torch.no_grad():
        out = model(torch.FloatTensor(np.asarray(X, dtype=np.float32)).to(device))
    return out.cpu().numpy()


# ── Density-ratio estimation for weighted CP (fix M2) ──────────────────


def estimate_density_ratio(
    X_ref: np.ndarray,
    X_target: np.ndarray,
    model: str = WCP_DENSITY_RATIO_MODEL,
    clip_quantile: float = WCP_WEIGHT_CLIP_QUANTILE,
    min_ess_fraction: float = WCP_MIN_ESS_FRACTION,
    seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Estimate likelihood-ratio weights ``w(x) = p_target(x) / p_ref(x)`` on the
    reference (calibration) points via a probabilistic domain classifier trained
    to separate reference from target covariates [Tibshirani2020].

    The classifier's posterior ``P(target | x)`` is converted to a density ratio
    (with the class-prior correction ``n_ref / n_target``) and clipped at the
    ``clip_quantile`` to control variance from mis-estimation (audit F7). Weight
    diagnostics — effective sample size, clip rate, and a ``collapsed`` flag when
    ESS/n drops below ``min_ess_fraction`` — are returned so callers can fall
    back to unweighted split-CP when the shift is too severe to reweight.

    Args:
        X_ref: Reference (calibration) covariates ``(n_ref, d)``.
        X_target: Target (test) covariates ``(n_target, d)``.
        model: ``"logistic"`` or ``"gbm"`` domain classifier.
        clip_quantile: Upper quantile at which to clip weights.
        min_ess_fraction: ESS/n below which weights are deemed collapsed.
        seed: Random seed for the classifier.

    Returns:
        ``(weights, diagnostics)`` where ``weights`` has shape ``(n_ref,)`` and
        ``diagnostics`` has keys ``ess, ess_fraction, clip_rate, max_weight,
        collapsed, n_ref, n_target``.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier

    X_ref = np.asarray(X_ref, dtype=float)
    X_target = np.asarray(X_target, dtype=float)
    n_ref, n_target = len(X_ref), len(X_target)
    if n_ref == 0 or n_target == 0:
        raise ValueError("estimate_density_ratio needs non-empty ref and target sets")

    X = np.vstack([X_ref, X_target])
    y = np.concatenate([np.zeros(n_ref), np.ones(n_target)])

    if model == "logistic":
        clf = LogisticRegression(max_iter=1000, random_state=seed)
    elif model == "gbm":
        clf = HistGradientBoostingClassifier(random_state=seed, max_iter=150)
    else:
        raise ValueError(f"unknown density-ratio model {model!r} (use 'logistic'/'gbm')")
    clf.fit(X, y)

    p_target = clf.predict_proba(X_ref)[:, 1]
    p_target = np.clip(p_target, 1e-6, 1 - 1e-6)
    # odds * class-prior ratio = p_target(x) / p_ref(x)
    w = (p_target / (1.0 - p_target)) * (n_ref / n_target)

    cap = float(np.quantile(w, clip_quantile))
    clipped = np.minimum(w, cap)
    clip_rate = float(np.mean(w > cap))
    ess = effective_sample_size(clipped)
    ess_fraction = ess / n_ref
    collapsed = bool(ess_fraction < min_ess_fraction)

    diagnostics = {
        "ess": ess, "ess_fraction": ess_fraction, "clip_rate": clip_rate,
        "max_weight": float(np.max(w)), "collapsed": collapsed,
        "n_ref": n_ref, "n_target": n_target,
    }
    if collapsed:
        logger.warning("Density-ratio weights collapsed (ESS/n=%.3f < %.2f); "
                       "caller should fall back to unweighted split-CP",
                       ess_fraction, min_ess_fraction)
    return clipped, diagnostics


# ── Weighted-CP CQR arm (fix M2) ───────────────────────────────────────


def train_test_weighted_cqr(
    X_train_all: np.ndarray,
    y_train_all: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float = NOMINAL_ALPHA,
    device: Optional[torch.device] = None,
    seed: int = RANDOM_SEED,
    cal_fraction: float = CQR_CALIBRATION_SPLIT,
    density_ratio_model: str = WCP_DENSITY_RATIO_MODEL,
    save_model: bool = False,
    model_dir: Optional[str] = None,
    suffix: str = "",
) -> Tuple[nn.Module, float, Dict[str, np.ndarray], Dict[str, object]]:
    """
    Weighted-CP CQR: CQR max-scores calibrated by the likelihood-ratio-weighted
    quantile, so coverage is valid under covariate shift up to weight-estimation
    error (design §8, audit F7). If the weights collapse, falls back to plain
    split-CP and flags ``fell_back=True`` in diagnostics.

    Returns ``(model, q_hat, {"lower","median","upper"} on test, diagnostics)``.
    """
    from sklearn.model_selection import train_test_split

    device = device or get_device()
    torch.manual_seed(seed); np.random.seed(seed)
    y_all = np.asarray(y_train_all, dtype=float).ravel()
    y_test = np.asarray(y_test, dtype=float).ravel()

    X_trval, X_cal, y_trval, y_cal = train_test_split(
        X_train_all, y_all, test_size=1.0 - cal_fraction, random_state=seed)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_trval, y_trval, test_size=0.10, random_state=seed)

    model, _, _ = train_qrnn(
        X_tr, y_tr, X_val, y_val, device=device, seed=seed,
        save_model=save_model, model_dir=model_dir, suffix=suffix)

    q_cal = _qrnn_quantiles(model, X_cal, device)
    scores = cqr_max_score(y_cal, q_cal[:, 0], q_cal[:, 2])

    weights, diag = estimate_density_ratio(
        X_cal, X_test, model=density_ratio_model, seed=seed)
    if diag["collapsed"]:
        q_hat = conformal_quantile(scores, alpha=alpha)
        diag["fell_back"] = True
    else:
        q_hat = weighted_conformal_quantile(scores, weights, alpha=alpha)
        diag["fell_back"] = False

    q_te = _qrnn_quantiles(model, X_test, device)
    y_pred = {"lower": q_te[:, 0] - q_hat, "median": q_te[:, 1], "upper": q_te[:, 2] + q_hat}
    diag["mae"] = float(np.mean(np.abs(y_test - y_pred["median"])))
    logger.info("Weighted-CP CQR: q_hat=%.4f fell_back=%s ESS/n=%.3f",
                q_hat, diag["fell_back"], diag["ess_fraction"])
    return model, q_hat, y_pred, diag


# ── Distributional-CP / HPD-split arm (density level-set score) ─────────


def _density_level_set_interval(
    params: Tuple[np.ndarray, np.ndarray, np.ndarray],
    q_hat: float,
    grid: int = 2048,
    range_sigmas: float = 8.0,
    chunk: int = 256,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    For score ``s = -log p(y|x)``, the conformal set is the density superlevel
    set ``{y : p(y|x) >= exp(-q_hat)}``. Return the enclosing interval
    ``[lo, hi]`` per sample (min/max y meeting the level) plus the fraction of
    samples whose superlevel set is *disconnected* (multimodal), the diagnostic
    that motivates the density-fused family over a single central band.

    Processes samples in ``chunk`` blocks to bound the ``(B, grid, K)`` broadcast.
    """
    from scipy.stats import norm

    pi, mu, sigma = params
    sigma = np.maximum(sigma, 1e-6)
    n = pi.shape[0]
    level = float(np.exp(-q_hat))
    los = np.empty(n); his = np.empty(n)
    disconnected = 0

    t = np.linspace(0.0, 1.0, grid)[None, :]
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        s = slice(start, stop)
        half = range_sigmas * sigma[s].max(axis=1)
        lo = mu[s].min(axis=1) - half
        hi = mu[s].max(axis=1) + half
        ys = lo[:, None] + (hi - lo)[:, None] * t  # (B, grid)
        z = (ys[..., None] - mu[s][:, None, :]) / sigma[s][:, None, :]  # (B,grid,K)
        dens = np.sum(pi[s][:, None, :] * norm.pdf(z), axis=-1)  # (B, grid)
        mask = dens >= level  # superlevel membership

        B = stop - start
        any_pts = mask.any(axis=1)
        idx_first = np.argmax(mask, axis=1)                     # first True
        idx_last = (grid - 1) - np.argmax(mask[:, ::-1], axis=1)  # last True
        lo_out = np.where(any_pts, ys[np.arange(B), idx_first], lo)
        hi_out = np.where(any_pts, ys[np.arange(B), idx_last], hi)
        # multimodality: count disjoint True-runs; a single run (contiguous
        # interval) has runs == 1; a disconnected (HPD-multimodal) set has >= 2.
        for b in range(B):
            m = np.concatenate(([0], mask[b].astype(np.int8), [0]))
            runs = int(np.sum(np.diff(m) == 1))
            if runs >= 2:
                disconnected += 1
        los[s] = lo_out
        his[s] = hi_out
    return los, his, float(disconnected / max(1, n))


def train_test_distcp(
    X_train_all: np.ndarray,
    y_train_all: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float = NOMINAL_ALPHA,
    n_components: int = 5,
    device: Optional[torch.device] = None,
    seed: int = RANDOM_SEED,
    cal_fraction: float = CQR_CALIBRATION_SPLIT,
    epochs: int = 150,
    save_model: bool = False,
    model_dir: Optional[str] = None,
    suffix: str = "",
) -> Tuple[MDN, float, Dict[str, np.ndarray], Dict[str, float]]:
    """
    Distributional-CP / HPD-split arm: MDN conditional density with the
    ``-log p(y|x)`` nonconformity score, calibrated by split-CP. Intervals are
    the enclosing interval of the density superlevel set (design §4.1).

    Returns ``(model, q_hat, {"lower","median","upper"} on test,
    {"mae", "multimodal_frac"})``.
    """
    from sklearn.model_selection import train_test_split

    device = device or get_device()
    torch.manual_seed(seed); np.random.seed(seed)
    y_all = np.asarray(y_train_all, dtype=float).ravel()
    y_test = np.asarray(y_test, dtype=float).ravel()

    X_trval, X_cal, y_trval, y_cal = train_test_split(
        X_train_all, y_all, test_size=1.0 - cal_fraction, random_state=seed)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_trval, y_trval, test_size=0.10, random_state=seed)

    model, y_mean, y_std = fit_mdn_standardized(
        X_tr, y_tr, X_val, y_val, n_components=n_components, device=device,
        epochs=epochs, seed=seed, save_model=save_model, model_dir=model_dir, suffix=suffix)

    # calibration scores: -log density of the observed y under the (data-scale) MDN
    cal_params = predict_params_data(model, X_cal, y_mean, y_std, device=device)
    scores = -_log_density_at_points(cal_params, y_cal)
    q_hat = conformal_quantile(scores, alpha=alpha)

    te_params = predict_params_data(model, X_test, y_mean, y_std, device=device)
    lo, hi, multimodal = _density_level_set_interval(te_params, q_hat)
    pi, mu, sigma = te_params
    median = np.sum(pi * mu, axis=1)  # mixture mean as point estimate
    y_pred = {"lower": lo, "median": median, "upper": hi}
    mae = float(np.mean(np.abs(y_test - median)))
    logger.info("DistCP/HPD: q_hat=%.4f multimodal=%.3f MAE=%.4f", q_hat, multimodal, mae)
    return model, q_hat, y_pred, {"mae": mae, "multimodal_frac": multimodal}


def _log_density_at_points(
    params: Tuple[np.ndarray, np.ndarray, np.ndarray], y: np.ndarray
) -> np.ndarray:
    """log sum_k pi_k N(y; mu_k, sigma_k^2) evaluated at one y per sample."""
    from scipy.special import logsumexp

    pi, mu, sigma = params
    sigma = np.maximum(sigma, 1e-6)
    yv = np.asarray(y, dtype=float).reshape(-1, 1)
    comp = -0.5 * np.square((yv - mu) / sigma) - np.log(sigma * np.sqrt(2.0 * np.pi))
    return logsumexp(np.log(np.maximum(pi, 1e-300)) + comp, axis=-1)


# ── WQLCP-style arm (recon-loss-scaled + weighted) ─────────────────────


class _ReconAE(nn.Module):
    """Small tabular autoencoder; its per-sample reconstruction error is the
    local uncertainty scale for the WQLCP-style score (vision-native method
    ported to sensor windows — design §11 flags this as fidelity risk)."""

    def __init__(self, input_dim: int, hidden: int = WQLCP_RECON_HIDDEN_DIM,
                 latent: int = WQLCP_RECON_LATENT_DIM):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, latent), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(),
                                 nn.Linear(hidden, input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(self.enc(x))


def _fit_recon(
    model: _ReconAE, X_fit: np.ndarray, device: torch.device,
    epochs: int, seed: int,
) -> _ReconAE:
    """Fit the autoencoder on ``X_fit`` (train features only) and return it."""
    torch.manual_seed(seed)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=WQLCP_RECON_LR)
    xt = torch.FloatTensor(np.asarray(X_fit, dtype=np.float32)).to(device)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xt), batch_size=64, shuffle=True)
    for _ in range(epochs):
        model.train()
        for (xb,) in loader:
            opt.zero_grad()
            loss = torch.mean((model(xb) - xb) ** 2, dim=1).sum()
            loss.backward()
            opt.step()
    return model


def _recon_error(model: _ReconAE, X_eval: np.ndarray, device: torch.device) -> np.ndarray:
    """Per-row reconstruction MSE of a fitted autoencoder on ``X_eval``."""
    model.eval()
    with torch.no_grad():
        xe = torch.FloatTensor(np.asarray(X_eval, dtype=np.float32)).to(device)
        return torch.mean((model(xe) - xe) ** 2, dim=1).cpu().numpy()


def train_test_wqlcp(
    X_train_all: np.ndarray,
    y_train_all: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float = NOMINAL_ALPHA,
    device: Optional[torch.device] = None,
    seed: int = RANDOM_SEED,
    cal_fraction: float = CQR_CALIBRATION_SPLIT,
    density_ratio_model: str = WCP_DENSITY_RATIO_MODEL,
    save_model: bool = False,
    model_dir: Optional[str] = None,
    suffix: str = "",
) -> Tuple[nn.Module, float, Dict[str, np.ndarray], Dict[str, object]]:
    """
    WQLCP-style arm [alijani_wqlcp_2025]: CQR score scaled by a per-sample
    reconstruction-loss uncertainty and calibrated by the weighted conformal
    quantile (loss-scaled + weighted-exchangeability). Raises ``NotImplementedError``
    when ``WQLCP_FEASIBLE`` is False so the study can retain it as a literature
    row instead of a benchmarked arm (audit F13 / design §11 fallback).

    Returns ``(model, q_hat, {"lower","median","upper"}, diagnostics)``.
    """
    from sklearn.model_selection import train_test_split

    if not WQLCP_FEASIBLE:
        raise NotImplementedError(
            "WQLCP-style arm disabled (WQLCP_FEASIBLE=False): retained as a "
            "literature row per design §11.")

    device = device or get_device()
    torch.manual_seed(seed); np.random.seed(seed)
    y_all = np.asarray(y_train_all, dtype=float).ravel()
    y_test = np.asarray(y_test, dtype=float).ravel()

    X_trval, X_cal, y_trval, y_cal = train_test_split(
        X_train_all, y_all, test_size=1.0 - cal_fraction, random_state=seed)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_trval, y_trval, test_size=0.10, random_state=seed)

    model, _, _ = train_qrnn(
        X_tr, y_tr, X_val, y_val, device=device, seed=seed,
        save_model=save_model, model_dir=model_dir, suffix=suffix)

    # reconstruction-loss scale: fit on TRAIN features only (no leakage), then
    # score calibration and test points with the same fitted autoencoder.
    input_dim = X_tr.shape[1]
    ae = _fit_recon(_ReconAE(input_dim), X_tr, device, WQLCP_RECON_EPOCHS, seed)
    r_cal = _recon_error(ae, X_cal, device)
    r_te = _recon_error(ae, X_test, device)

    q_cal = _qrnn_quantiles(model, X_cal, device)
    scores, g = normalized_max_score(y_cal, q_cal[:, 0], q_cal[:, 2], r_cal, eps=WQLCP_SCORE_FLOOR)

    weights, diag = estimate_density_ratio(X_cal, X_test, model=density_ratio_model, seed=seed)
    if diag["collapsed"]:
        q_hat = conformal_quantile(scores, alpha=alpha)
        diag["fell_back"] = True
    else:
        q_hat = weighted_conformal_quantile(scores, weights, alpha=alpha)
        diag["fell_back"] = False

    q_te = _qrnn_quantiles(model, X_test, device)
    y_pred = normalized_predict(q_te[:, 0], q_te[:, 1], q_te[:, 2], r_te, q_hat, eps=WQLCP_SCORE_FLOOR)
    diag["mae"] = float(np.mean(np.abs(y_test - y_pred["median"])))
    logger.info("WQLCP-style: q_hat=%.4f fell_back=%s MAE=%.4f", q_hat, diag["fell_back"], diag["mae"])
    return model, q_hat, y_pred, diag

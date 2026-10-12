"""
Calibrated model-internal / tree baselines and the multi-seed raw QRNN arm.

Implements the remaining fix-M4 requirements (design §4.2/§4.3):

* **raw QRNN, multi-seed** (:func:`raw_qrnn_multiseed`) — the uncalibrated
  quantile baseline, now run over ``SEED_LIST`` (≥5) instead of a single run,
  removing the prior paper's single-run raw-QRNN weakness.
* **C-XGBQ** (:func:`train_test_cxgbq`) — XGBoost quantile regression, then
  calibrated with the SAME split-CP CQR procedure as C-QRNN.
* **C-GPR** (:func:`train_test_cgpr`) — Gaussian Process Regression; its
  analytic Gaussian PIs are treated as the base interval and conformalized the
  same way (fix M4: no baseline is left un-calibrated).

All arms route through :mod:`qrnn_uq_rul.conformal`, so calibration is identical
across the study; only the base predictor differs.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .config import (
    CQR_CALIBRATION_SPLIT,
    N_RESTARTS_OPTIMIZER,
    NOMINAL_ALPHA,
    QUANTILES,
    RANDOM_SEED,
    SEED_LIST,
)
from .conformal import calibrate_cqr_from_quantiles, cqr_predict
from .models import QRNN, train_qrnn
from .utils import get_device

logger = logging.getLogger(__name__)


def _qrnn_quantile_array(models: Sequence[QRNN], X: np.ndarray, device: torch.device) -> np.ndarray:
    """Stack member quantile predictions into ``(M, N, 3)`` (eval mode)."""
    xt = torch.FloatTensor(np.asarray(X, dtype=np.float32)).to(device)
    outs = []
    for m in models:
        m.eval()
        with torch.no_grad():
            outs.append(m(xt).cpu().numpy())
    return np.stack(outs, axis=0)


# ── Raw QRNN (uncalibrated, multi-seed) ────────────────────────────────


def raw_qrnn_multiseed(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seeds: Optional[Sequence[int]] = None,
    device: Optional[torch.device] = None,
    crossing_weight: float = 0.0,
    save_model: bool = False,
    model_dir: Optional[str] = None,
    suffix: str = "",
) -> List[QRNN]:
    """
    Train one QRNN per seed (fix M4: multi-seed, not a single run).

    Args:
        X_train, y_train, X_val, y_val: Training/validation tensors.
        seeds: Seeds; defaults to ``SEED_LIST`` (>= 5 members).
        device: Torch device.
        crossing_weight: Optional MCQRNN non-crossing penalty (fix M7).
        save_model, model_dir, suffix: Persistence passthrough.

    Returns:
        List of trained :class:`QRNN` members.
    """
    device = device or get_device()
    seeds = list(seeds) if seeds is not None else list(SEED_LIST)
    models: List[QRNN] = []
    for i, seed in enumerate(seeds):
        model, _, _ = train_qrnn(
            X_train, y_train, X_val, y_val, device=device, seed=seed,
            save_model=save_model, model_dir=model_dir,
            suffix=f"{suffix}_raw{i}", crossing_weight=crossing_weight)
        models.append(model)
    logger.info("Raw QRNN trained over %d seeds (fix M4)", len(models))
    return models


def raw_qrnn_predict(
    models: Sequence[QRNN], X: np.ndarray, device: Optional[torch.device] = None
) -> Dict[str, np.ndarray]:
    """
    Ensemble-averaged, UNCALIBRATED QRNN quantiles (raw baseline arm).

    ``lower`` / ``median`` / ``upper`` are means across members of the raw
    quantile predictions — NO conformal adjustment is applied (this is the
    "raw, uncalibrated" row of the design method table).
    """
    device = device if device is not None else next(models[0].parameters()).device
    preds = _qrnn_quantile_array(models, X, device)  # (M, N, 3)
    mean = preds.mean(axis=0)
    return {"lower": mean[:, 0], "median": mean[:, 1], "upper": mean[:, 2]}


# ── Calibrated XGBoost quantile regression (C-XGBQ) ─────────────────────


def _fit_xgbq(X: np.ndarray, y: np.ndarray, seed: int):
    import xgboost as xgb

    models = {}
    for q in QUANTILES:
        reg = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=q,
                               random_state=seed, n_estimators=200)
        reg.fit(X, y)
        models[q] = reg
    return models


def _xgbq_predict(models, X: np.ndarray) -> np.ndarray:
    """
    Return ``(N, 3)`` XGBQ quantile predictions in QUANTILES order.

    XGBoost fits each quantile independently, so raw predictions can cross
    (q_hat_0.9 < q_hat_0.1 for some rows). We sort the three columns per sample
    ascending to enforce monotonicity before CQR calibration — otherwise the
    constant-shift interval could come out inverted (lower > upper).
    """
    preds = np.column_stack([models[q].predict(X) for q in QUANTILES])
    return np.sort(preds, axis=1)


def train_test_cxgbq(
    X_train_all: np.ndarray,
    y_train_all: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float = NOMINAL_ALPHA,
    seed: int = RANDOM_SEED,
    cal_fraction: float = CQR_CALIBRATION_SPLIT,
    save_model: bool = False,
    model_dir: Optional[str] = None,
    suffix: str = "",
) -> Tuple[Dict[float, object], float, Dict[str, np.ndarray], Dict[str, float]]:
    """
    C-XGBQ arm: XGBoost quantile regression calibrated with the shared split-CP
    CQR procedure (fix M4 — same calibration as C-QRNN).

    Returns ``(xgbq_models, e_hat, {"lower","median","upper"} on test, {"mae"})``.
    """
    from pathlib import Path

    import joblib
    from sklearn.model_selection import train_test_split

    np.random.seed(seed)
    y_all = np.asarray(y_train_all, dtype=float).ravel()
    y_test = np.asarray(y_test, dtype=float).ravel()

    X_trval, X_cal, y_trval, y_cal = train_test_split(
        X_train_all, y_all, test_size=1.0 - cal_fraction, random_state=seed)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_trval, y_trval, test_size=0.10, random_state=seed)

    models = _fit_xgbq(X_tr, y_tr, seed)

    q_cal = _xgbq_predict(models, X_cal)
    e_hat = calibrate_cqr_from_quantiles(y_cal, q_cal[:, 0], q_cal[:, -1], alpha=alpha)
    q_te = _xgbq_predict(models, X_test)
    y_pred = cqr_predict(q_te[:, 0], q_te[:, 1], q_te[:, 2], e_hat)

    if save_model:
        d = Path(model_dir) if model_dir else Path("./model/xgbq")
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(models, d / f"cxgbq_{suffix}.joblib")

    mae = float(np.mean(np.abs(y_test - y_pred["median"])))
    logger.info("C-XGBQ: e_hat=%.4f MAE=%.4f", e_hat, mae)
    return models, e_hat, y_pred, {"mae": mae}


# ── Calibrated Gaussian Process Regression (C-GPR) ─────────────────────


def _fit_gpr(X: np.ndarray, y: np.ndarray, seed: int):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        RBF,
        ConstantKernel,
        Matern,
        WhiteKernel,
    )

    from .config import NORMALIZE_Y

    kernel = (
        ConstantKernel(constant_value=1.0)
        * (RBF(length_scale=1.0) + Matern(length_scale=1.0, nu=1.5))
        + WhiteKernel(noise_level=0.1)
    )
    gpr = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=N_RESTARTS_OPTIMIZER,
        random_state=seed, normalize_y=NORMALIZE_Y)
    gpr.fit(X, y)
    return gpr


def _gpr_quantiles(gpr, X: np.ndarray) -> np.ndarray:
    """Analytic Gaussian (model-internal) quantiles, columns in QUANTILES order."""
    from scipy.stats import norm

    mean, std = gpr.predict(X, return_std=True)
    std = np.maximum(std, 1e-8)
    return np.column_stack([norm.ppf(q, loc=mean, scale=std) for q in QUANTILES])


def train_test_cgpr(
    X_train_all: np.ndarray,
    y_train_all: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float = NOMINAL_ALPHA,
    seed: int = RANDOM_SEED,
    cal_fraction: float = CQR_CALIBRATION_SPLIT,
    save_model: bool = False,
    model_dir: Optional[str] = None,
    suffix: str = "",
) -> Tuple[object, float, Dict[str, np.ndarray], Dict[str, float]]:
    """
    C-GPR arm: GP regression with its analytic Gaussian PIs conformalized by the
    SAME split-CP CQR procedure (fix M4). Returns ``(gpr, e_hat, intervals, {"mae"})``.
    """
    from pathlib import Path

    import joblib
    from sklearn.model_selection import train_test_split

    np.random.seed(seed)
    y_all = np.asarray(y_train_all, dtype=float).ravel()
    y_test = np.asarray(y_test, dtype=float).ravel()

    X_trval, X_cal, y_trval, y_cal = train_test_split(
        X_train_all, y_all, test_size=1.0 - cal_fraction, random_state=seed)
    # GPR does not need a val split for early stopping; use all train+val for fit
    gpr = _fit_gpr(X_trval, y_trval, seed)

    q_cal = _gpr_quantiles(gpr, X_cal)
    e_hat = calibrate_cqr_from_quantiles(y_cal, q_cal[:, 0], q_cal[:, -1], alpha=alpha)
    q_te = _gpr_quantiles(gpr, X_test)
    y_pred = cqr_predict(q_te[:, 0], q_te[:, 1], q_te[:, 2], e_hat)

    if save_model:
        d = Path(model_dir) if model_dir else Path("./model/gpr")
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(gpr, d / f"cgpr_{suffix}.joblib")

    mae = float(np.mean(np.abs(y_test - y_pred["median"])))
    logger.info("C-GPR: e_hat=%.4f MAE=%.4f", e_hat, mae)
    return gpr, e_hat, y_pred, {"mae": mae}

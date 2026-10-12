"""
Model definitions and training functions for QRNN, GPR, XGBQ, and CQR-QRNN.

Models:
- QRNN: Quantile Regression Neural Network (PyTorch)
- GPR: Gaussian Process Regression (scikit-learn)
- XGBQ: XGBoost Quantile Regression (XGBoost)
- CQR-QRNN: Conformalized Quantile Regression on QRNN
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import xgboost as xgb

from .config import (
    BATCH_SIZE,
    CQR_ALPHA,
    CQR_CALIBRATION_SPLIT,
    EPOCHS,
    LR,
    N_RESTARTS_OPTIMIZER,
    NORMALIZE_Y,
    QUANTILES,
    RANDOM_SEED,
    WEIGHT_DECAY,
)
from .utils import get_device

logger = logging.getLogger(__name__)


# ── PyTorch Dataset ────────────────────────────────────────────────────


class RULDataset(Dataset):
    """PyTorch Dataset for RUL prediction."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ── QRNN Model ─────────────────────────────────────────────────────────


class QRNN(nn.Module):
    """
    Quantile Regression Neural Network.

    Base architecture (Phase 2, unchanged by default): input -> 128 -> 64 -> 32
    -> 3 (quantiles), each hidden layer Linear + BatchNorm + ReLU + Dropout
    (dropouts 0.3 / 0.2 / 0.1). Outputs 3 quantiles: tau in {0.1, 0.5, 0.9}.

    Phase 4 (architecture-influence ablation, task 04 E4): ``hidden_dims`` and
    ``dropouts`` are optional overrides (same length, one dropout per hidden
    layer). Defaults reproduce the Phase-2 base network exactly, so all
    pre-existing behavior and saved state_dicts are unaffected.
    """

    DEFAULT_HIDDEN_DIMS: Tuple[int, ...] = (128, 64, 32)
    DEFAULT_DROPOUTS: Tuple[float, ...] = (0.3, 0.2, 0.1)

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[Tuple[int, ...]] = None,
        dropouts: Optional[Tuple[float, ...]] = None,
    ):
        super().__init__()
        hidden_dims = tuple(hidden_dims) if hidden_dims is not None else self.DEFAULT_HIDDEN_DIMS
        dropouts = tuple(dropouts) if dropouts is not None else self.DEFAULT_DROPOUTS
        if len(dropouts) != len(hidden_dims):
            raise ValueError(
                f"hidden_dims ({len(hidden_dims)}) and dropouts ({len(dropouts)}) "
                "must have the same length (one dropout per hidden layer)"
            )
        layers: List[nn.Module] = []
        prev = input_dim
        for h, d in zip(hidden_dims, dropouts):
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(d),
            ])
            prev = h
        layers.append(nn.Linear(prev, len(QUANTILES)))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def qrnn_loss(
    predictions: torch.Tensor,
    y_batch: torch.Tensor,
    crossing_weight: float = 0.0,
) -> torch.Tensor:
    """
    Calculate the quantile regression (pinball) loss for QRNN, with an optional
    MCQRNN-style non-crossing penalty (fix M7, Cannon 2018).

    Args:
        predictions: Predicted RUL quantiles (N, 3), columns ordered by tau.
        y_batch: True RUL values (N,).
        crossing_weight: λ >= 0 on the hinge penalty
            mean(max(0, q_hat_{tau_j} - q_hat_{tau_{j+1}})). 0 (default) leaves
            the loss identical to the pure pinball loss, so existing baselines
            are unaffected; the non-crossing arms opt in.

    Returns:
        Scalar loss tensor.
    """
    loss = torch.tensor(0.0, device=predictions.device)
    for i, q in enumerate(QUANTILES):
        diff = y_batch.view(-1, 1) - predictions[:, i: i + 1]
        loss = loss + torch.mean(torch.max(q * diff, (q - 1) * diff))

    if crossing_weight > 0 and predictions.shape[1] >= 2:
        # penalize any adjacent pair where a higher-tau quantile falls below a
        # lower-tau one (a crossing)
        gaps = torch.diff(predictions, dim=1)               # (N, cols-1): want >= 0
        violations = torch.clamp(-gaps, min=0.0)            # positive on crossings
        loss = loss + crossing_weight * violations.mean()
    return loss


def train_qrnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: Optional[torch.device] = None,
    save_model: bool = False,
    model_dir: Optional[str] = None,
    suffix: str = "",
    seed: int = RANDOM_SEED,
    crossing_weight: float = 0.0,
    hidden_dims: Optional[Tuple[int, ...]] = None,
    dropouts: Optional[Tuple[float, ...]] = None,
) -> Tuple[QRNN, Optional[str], Dict[str, List[float]]]:
    """
    Train a QRNN model with validation-based early stopping.

    Args:
        X_train: Training features.
        y_train: Training targets.
        X_val: Validation features.
        y_val: Validation targets.
        device: PyTorch device (auto-detect if None).
        save_model: Whether to save the model.
        model_dir: Directory to save the model.
        suffix: Timestamp suffix for model filename.
        seed: Random seed.
        crossing_weight: λ on the MCQRNN non-crossing penalty applied to the
            TRAINING loss only (fix M7). Default 0.0 keeps the baseline pure
            pinball loss; the validation loss is always pure pinball so model
            selection stays comparable across penalty settings.
        hidden_dims: Optional per-layer hidden widths (architecture ablation,
            task 04 E4). Default = base taper (128, 64, 32).
        dropouts: Optional per-layer dropout rates (architecture ablation,
            task 04 E4). Default = base (0.3, 0.2, 0.1).

    Returns:
        Tuple of (trained_model, model_path_or_None, loss_history).
    """
    if device is None:
        device = get_device()

    np.random.seed(seed)
    torch.manual_seed(seed)

    input_dim = X_train.shape[1]
    model = QRNN(input_dim, hidden_dims=hidden_dims, dropouts=dropouts).to(device)

    train_loader = DataLoader(RULDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(RULDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    best_val_loss = float("inf")
    best_model_state: Optional[dict] = None

    train_losses: List[float] = []
    val_losses: List[float] = []

    logger.info("Training QRNN model...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = qrnn_loss(predictions, y_batch, crossing_weight=crossing_weight)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch = X_val_batch.to(device)
                y_val_batch = y_val_batch.to(device)
                val_predictions = model(X_val_batch)
                val_loss += qrnn_loss(val_predictions, y_val_batch).item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            logger.info(
                "  Epoch [%d/%d], Train Loss: %.4f, Val Loss: %.4f",
                epoch + 1, EPOCHS, avg_train_loss, avg_val_loss,
            )

        scheduler.step(val_loss)

    elapsed = time.time() - start_time
    logger.info("Training completed in %.2f seconds.", elapsed)

    assert best_model_state is not None
    model.load_state_dict(best_model_state)

    model_path: Optional[str] = None
    if save_model:
        save_dir = Path(model_dir) if model_dir else Path("./model/qrnn")
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = str(save_dir / f"qrnn_model_{suffix}.pth")
        torch.save(model.state_dict(), model_path)
        logger.info("Model saved to %s", model_path)

    return model, model_path, {"train_losses": train_losses, "val_losses": val_losses}


# ── GPR Model ──────────────────────────────────────────────────────────


def train_test_gpr(
    W_train: pd.DataFrame,
    X_s_train: pd.DataFrame,
    y_train: pd.DataFrame,
    W_test: pd.DataFrame,
    X_s_test: pd.DataFrame,
    y_test: pd.DataFrame,
    save_model: bool = False,
    load_model_path: Optional[str] = None,
    suffix: str = "",
) -> Tuple[GaussianProcessRegressor, Dict[str, np.ndarray], pd.DataFrame]:
    """
    Train and/or test a Gaussian Process Regression model.

    Args:
        W_train, X_s_train, y_train: Training data.
        W_test, X_s_test, y_test: Test data.
        save_model: Whether to save the model.
        load_model_path: Path to load a saved model.
        suffix: Timestamp suffix for model filename.

    Returns:
        Tuple of (gpr_model, predictions_dict, evaluation_df).
    """
    from .metrics import evaluate_point_estimate

    X_train = pd.concat([W_train, X_s_train], axis=1)
    X_test = pd.concat([W_test, X_s_test], axis=1)

    if load_model_path:
        gpr = joblib.load(load_model_path)
        logger.info("Loaded GPR model from %s", load_model_path)
    else:
        kernel = (
            ConstantKernel(constant_value=1.0)
            * (RBF(length_scale=1.0) + Matern(length_scale=1.0, nu=1.5))
            + WhiteKernel(noise_level=0.1)
        )

        logger.info("Training GPR model...")
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=N_RESTARTS_OPTIMIZER,
            random_state=RANDOM_SEED,
            normalize_y=NORMALIZE_Y,
        )

        start_time = time.time()
        gpr.fit(X_train, y_train.values.ravel())
        elapsed = time.time() - start_time
        logger.info("GPR model training completed in %.2f seconds.", elapsed)

        if save_model:
            model_dir = Path("./model/gpr")
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"gpr_model_{suffix}.joblib"
            joblib.dump(gpr, model_path)
            logger.info("Model saved to %s", model_path)

    y_pred_mean, y_pred_std = gpr.predict(X_test, return_std=True)
    y_pred = {
        "lower": norm.ppf(QUANTILES[0], loc=y_pred_mean, scale=y_pred_std),
        "mean": y_pred_mean,
        "upper": norm.ppf(QUANTILES[2], loc=y_pred_mean, scale=y_pred_std),
    }

    evaluation = evaluate_point_estimate(y_test, y_pred["mean"])
    return gpr, y_pred, evaluation


# ── XGBQ Model ─────────────────────────────────────────────────────────


def train_test_xgbq(
    W_train: pd.DataFrame,
    X_s_train: pd.DataFrame,
    y_train: pd.DataFrame,
    W_test: pd.DataFrame,
    X_s_test: pd.DataFrame,
    y_test: pd.DataFrame,
    save_model: bool = False,
    load_model_path: Optional[str] = None,
    suffix: str = "",
    seed: int = RANDOM_SEED,
) -> Tuple[Dict[float, xgb.XGBRegressor], Dict[str, np.ndarray], pd.DataFrame]:
    """
    Train and/or test XGBoost quantile regression models (tau in {0.1, 0.5, 0.9}).

    Args:
        W_train, X_s_train, y_train: Training data.
        W_test, X_s_test, y_test: Test data.
        save_model: Whether to save the models.
        load_model_path: Path to load saved models.
        suffix: Timestamp suffix for model filenames.
        seed: Random seed.

    Returns:
        Tuple of (models_dict, predictions_dict, evaluation_df).
    """
    from .metrics import evaluate_point_estimate

    X_train = pd.concat([W_train, X_s_train], axis=1)
    X_test = pd.concat([W_test, X_s_test], axis=1)

    models: Dict[float, xgb.XGBRegressor] = {}

    if load_model_path is not None:
        for q in QUANTILES:
            model_path = f"{load_model_path}/xgb_quantile_model_{q}.joblib"
            models[q] = joblib.load(model_path)
            logger.info("Loaded XGBQ model for quantile %s from %s", q, model_path)
    else:
        logger.info("Training XGBQ models...")
        start_time = time.time()
        for q in QUANTILES:
            xgb_model = xgb.XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=q,
                random_state=seed,
            )
            xgb_model.fit(
                X_train,
                y_train.values.ravel(),
                eval_set=[(X_test, y_test.values.ravel())],
                verbose=False,
            )
            models[q] = xgb_model
        elapsed = time.time() - start_time
        logger.info("XGBQ training completed in %.2f seconds.", elapsed)

        if save_model:
            model_folder = Path(f"./model/xgb_quantile/xgb_quantile_model_{suffix}")
            model_folder.mkdir(parents=True, exist_ok=True)
            for q in QUANTILES:
                model_path = model_folder / f"xgb_quantile_model_{q}.joblib"
                joblib.dump(models[q], model_path)
                logger.info("Model for quantile %s saved to %s", q, model_path)

    y_pred = {
        "lower": models[QUANTILES[0]].predict(X_test),
        "median": models[QUANTILES[1]].predict(X_test),
        "upper": models[QUANTILES[2]].predict(X_test),
    }

    evaluation = evaluate_point_estimate(y_test, y_pred["median"])
    return models, y_pred, evaluation


# ── CQR (Conformalized Quantile Regression) ────────────────────────────


def compute_nonconformity_scores(
    model: QRNN,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    Compute nonconformity scores on a calibration set.

    Score: E_i = max(q_lower - y_i, y_i - q_upper).
    Samples inside the interval have E_i <= 0.

    Args:
        model: Trained QRNN model (in eval mode).
        X_cal: Calibration features (N, D).
        y_cal: Calibration targets (N,).
        device: PyTorch device.

    Returns:
        Array of nonconformity scores (N,).
    """
    model.eval()
    X_cal_tensor = torch.FloatTensor(X_cal).to(device)

    with torch.no_grad():
        preds = model(X_cal_tensor)
        q_lower = preds[:, 0].cpu().numpy()
        q_upper = preds[:, 2].cpu().numpy()

    return np.maximum(q_lower - y_cal, y_cal - q_upper)


def calibrate_cqr(scores: np.ndarray, alpha: float = CQR_ALPHA) -> float:
    """
    Compute the conformal adjustment E_hat from nonconformity scores.

    E_hat = quantile of scores at level ceil((n+1)*(1-alpha)) / n.

    Args:
        scores: Nonconformity scores from calibration set.
        alpha: Miscoverage rate (default 0.10 → 90% coverage target).

    Returns:
        Conformal adjustment E_hat.
    """
    n = len(scores)
    quantile_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, quantile_level))


def predict_cqr(
    model: QRNN,
    X_test: np.ndarray,
    e_hat: float,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Generate CQR-adjusted prediction intervals: [q_lower - E_hat, q_upper + E_hat].

    Args:
        model: Trained QRNN model (in eval mode).
        X_test: Test features (N, D).
        e_hat: Conformal adjustment from calibration.
        device: PyTorch device.

    Returns:
        Dict with 'lower', 'median', 'upper' keys.
    """
    model.eval()
    X_test_tensor = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        preds = model(X_test_tensor)
        q_lower = preds[:, 0].cpu().numpy()
        q_median = preds[:, 1].cpu().numpy()
        q_upper = preds[:, 2].cpu().numpy()

    return {
        "lower": q_lower - e_hat,
        "median": q_median,
        "upper": q_upper + e_hat,
    }


def train_test_qrnn_cqr(
    W_train: pd.DataFrame,
    X_s_train: pd.DataFrame,
    y_train: pd.DataFrame,
    W_test: pd.DataFrame,
    X_s_test: pd.DataFrame,
    y_test: pd.DataFrame,
    device: Optional[torch.device] = None,
    seed: int = RANDOM_SEED,
    save_model: bool = False,
    model_dir: Optional[str] = None,
    suffix: str = "",
    alpha: float = CQR_ALPHA,
    cal_fraction: float = CQR_CALIBRATION_SPLIT,
) -> Tuple[QRNN, float, Dict[str, np.ndarray], pd.DataFrame, Optional[str], Dict[str, List[float]]]:
    """
    Train QRNN and calibrate via split-conformal CQR, then evaluate on test set.

    Strategy (split conformal prediction):
    1. Split training data into train+val (80%) and calibration (20%).
    2. Further split train+val into train (90%) and val (10%) for early stopping.
    3. Train QRNN on train set with validation-based early stopping.
    4. Compute nonconformity scores on the held-out calibration set.
    5. Calibrate test predictions with E_hat.

    The calibration set is NEVER used during training or validation, ensuring
    exchangeability with the test set.

    Args:
        W_train, X_s_train, y_train: Training data (will be split).
        W_test, X_s_test, y_test: Test data.
        device: PyTorch device.
        seed: Random seed for reproducibility.
        save_model: Whether to save the model.
        model_dir: Directory to save the model.
        suffix: Timestamp suffix for model filename.
        alpha: Miscoverage rate for CQR.
        cal_fraction: Fraction of training data for train+val.

    Returns:
        Tuple of (model, e_hat, predictions_dict, evaluation_df, model_path, loss_history).
    """
    if device is None:
        device = get_device()

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Combine features
    X_all = pd.concat([W_train, X_s_train], axis=1).to_numpy()
    y_all = y_train.to_numpy().ravel()
    X_test = pd.concat([W_test, X_s_test], axis=1).to_numpy()
    y_test_arr = y_test.to_numpy().ravel()

    # Step 1: Split off calibration set
    X_trainval, X_cal, y_trainval, y_cal = train_test_split(
        X_all, y_all, test_size=1.0 - cal_fraction, random_state=seed,
    )

    # Step 2: Split train+val into train (90%) and val (10%)
    X_train, X_val, y_train_split, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.10, random_state=seed,
    )

    logger.info(
        "CQR split: %d train, %d val, %d calibration",
        len(X_train), len(X_val), len(X_cal),
    )

    # Step 3: Train QRNN
    qrnn_model, model_path, loss_history = train_qrnn(
        X_train, y_train_split, X_val, y_val,
        device=device, save_model=save_model, model_dir=model_dir,
        suffix=suffix, seed=seed,
    )

    # Step 4: Compute nonconformity scores on calibration set
    scores = compute_nonconformity_scores(qrnn_model, X_cal, y_cal, device)
    e_hat = calibrate_cqr(scores, alpha=alpha)

    logger.info("CQR conformal adjustment (E_hat): %.4f", e_hat)
    logger.info("Calibration set PICP (raw): %.4f", np.mean(scores <= 0))

    # Step 5: Generate CQR-adjusted predictions on test set
    y_pred = predict_cqr(qrnn_model, X_test, e_hat, device)

    # Evaluate
    from .metrics import evaluate_point_estimate
    evaluation = evaluate_point_estimate(y_test_arr, y_pred["median"])

    # Clear GPU memory
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    return qrnn_model, e_hat, y_pred, evaluation, model_path, loss_history

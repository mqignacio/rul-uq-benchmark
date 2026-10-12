"""
Mixture Density Network (MDN) + the C-MDN *series* ablation arm.

Ported from the project's reference implementation
(``resources/mdn_reference_pilario.py``, ``myMDN``) which is written in
TensorFlow/Keras. This is a device-agnostic PyTorch rewrite that preserves the
reference **architecture** (scalar target, 5-component Gaussian mixture,
``tanh`` MLP backbone ``12 -> 5``) and the **loss** (negative log-likelihood of
the mixture, with the reference's ``1e-6`` sigma floor and density floor). The
loss is unit-tested against a NumPy transcription of the reference to prove
numerical fidelity of the port.

Role in the study (design §4.1): the C-MDN **series** arm is the deliberate
naive combination — MDN quantiles (via numerical CDF inversion) fed into the
SAME conformalized-quantregression max-score + split-CP calibration as C-QRNN,
routed through :mod:`qrnn_uq_rul.conformal` (fix M4: one calibration
procedure). Phase 4 uses it to test whether fusing the density into the
conformal stage beats this bolt-on (H3, the fusion-vs-series ablation).

MDN quantiles are obtained by inverting the mixture CDF, so the three output
quantiles are monotone **by construction** (no quantile crossing for this arm).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .conformal import calibrate_cqr_from_quantiles, cqr_predict
from .config import (
    CQR_CALIBRATION_SPLIT,
    MDN_BATCH_SIZE,
    MDN_CDF_GRID,
    MDN_CDF_RANGE_SIGMAS,
    MDN_DENSITY_FLOOR,
    MDN_EPOCHS,
    MDN_HIDDEN_DIMS,
    MDN_LR,
    MDN_N_COMPONENTS,
    MDN_SIGMA_MIN,
    NOMINAL_ALPHA,
    QUANTILES,
    RANDOM_SEED,
)
from .models import RULDataset
from .utils import get_device

logger = logging.getLogger(__name__)

_SQRT2 = float(np.sqrt(2.0))


# ── MDN model ──────────────────────────────────────────────────────────


class MDN(nn.Module):
    """
    Mixture Density Network with a tanh MLP backbone (reference architecture).

    Emits, per input, a concatenated vector ``[pi, mu, sigma]`` of length
    ``3 * n_components`` where ``pi`` are softmax mixture weights, ``mu`` are
    the component means (linear), and ``sigma`` are softplus component std devs
    (floored at ``MDN_SIGMA_MIN``). The forward output layout matches the
    reference ``myMDN`` so :func:`mdn_loss` is a faithful transcription.
    """

    def __init__(
        self,
        input_dim: int,
        n_components: int = MDN_N_COMPONENTS,
        hidden_dims: Sequence[int] = MDN_HIDDEN_DIMS,
    ):
        super().__init__()
        self.n_components = n_components
        self.input_dim = input_dim

        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        self.backbone = nn.Sequential(*layers)
        # Single head = the three reference Dense(., K) heads concatenated.
        self.head = nn.Linear(prev, 3 * n_components)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.head(self.backbone(x))  # (N, 3K)
        k = self.n_components
        pi_logits, mu, sigma_raw = raw[:, :k], raw[:, k:2 * k], raw[:, 2 * k:]
        pi = torch.softmax(pi_logits, dim=-1)
        sigma = torch.clamp(nn.functional.softplus(sigma_raw), min=MDN_SIGMA_MIN)
        return torch.cat([pi, mu, sigma], dim=-1)

    # -- parameter unpacking ------------------------------------------------

    def unpack(self, y_pred: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split an activated ``(N, 3K)`` output into ``pi, mu, sigma`` ``(N, K)``."""
        k = self.n_components
        return y_pred[:, :k], y_pred[:, k:2 * k], y_pred[:, 2 * k:3 * k]

    @torch.no_grad()
    def predict_params(
        self, X: np.ndarray, device: Optional[torch.device] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return mixture ``(pi, mu, sigma)`` numpy arrays for inputs ``X``.

        Defaults to the device the model's own parameters already live on, so a
        CPU-built model does not get its inputs dispatched to MPS (and vice
        versa). Pass ``device`` explicitly to override.
        """
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        out = self(torch.FloatTensor(np.asarray(X, dtype=np.float32)).to(device))
        pi, mu, sigma = self.unpack(out)
        return pi.cpu().numpy(), mu.cpu().numpy(), sigma.cpu().numpy()


def mdn_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Negative log-likelihood of the Gaussian mixture (reference transcription).

    Mirrors ``myMDN.mdn_loss``: split the activated ``(N, 3K)`` output into
    ``pi, mu, sigma``, floor sigma at ``1e-6``, form
    ``sum_k pi_k * N(y; mu_k, sigma_k)``, floor the density at ``1e-6``, and
    return ``mean(-log density)``.

    Args:
        y_true: Targets, shape ``(N, 1)`` or ``(N,)``.
        y_pred: Activated MDN output, shape ``(N, 3K)``.

    Returns:
        Scalar loss tensor.
    """
    k = y_pred.shape[-1] // 3
    pi = y_pred[:, 0:k]
    mu = y_pred[:, k:2 * k]
    sigma = torch.clamp(y_pred[:, 2 * k:3 * k], min=MDN_SIGMA_MIN)

    y = y_true.reshape(-1, 1)
    exponent = -0.5 * torch.square((y - mu) / sigma)
    coeff = 1.0 / (torch.sqrt(torch.tensor(2.0 * np.pi, device=y_pred.device)) * sigma)
    gaussian = coeff * torch.exp(exponent)
    weighted = torch.sum(pi * gaussian, dim=-1, keepdim=True)
    loss = -torch.log(torch.clamp(weighted, min=MDN_DENSITY_FLOOR))
    return torch.mean(loss)


# ── Training ───────────────────────────────────────────────────────────


def train_mdn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_components: int = MDN_N_COMPONENTS,
    device: Optional[torch.device] = None,
    lr: float = MDN_LR,
    epochs: int = MDN_EPOCHS,
    batch_size: int = MDN_BATCH_SIZE,
    save_model: bool = False,
    model_dir: Optional[str] = None,
    suffix: str = "",
    seed: int = RANDOM_SEED,
) -> Tuple[MDN, Optional[str], Dict[str, List[float]]]:
    """
    Train an MDN with validation-based early stopping (mirrors ``train_qrnn``).

    Args:
        X_train, y_train: Training features/targets.
        X_val, y_val: Validation features/targets (for model selection).
        n_components: Mixture components K.
        device: Torch device (auto-detected if None).
        lr: Adam learning rate (reference default 0.01).
        epochs: Max epochs.
        batch_size: Mini-batch size.
        save_model: Whether to persist the best state dict.
        model_dir: Directory for the saved model.
        suffix: Filename suffix.
        seed: Random seed.

    Returns:
        ``(model, model_path_or_None, {"train_losses": [...], "val_losses": [...]})``.
    """
    device = device or get_device()
    np.random.seed(seed)
    torch.manual_seed(seed)

    input_dim = X_train.shape[1]
    model = MDN(input_dim, n_components=n_components).to(device)
    train_loader = DataLoader(RULDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(RULDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    best_val_loss = float("inf")
    best_state: Optional[dict] = None
    train_losses: List[float] = []
    val_losses: List[float] = []

    logger.info("Training MDN (K=%d)...", n_components)
    start = time.time()
    for epoch in range(epochs):
        model.train()
        tl = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = mdn_loss(yb, model(xb))
            loss.backward()
            optimizer.step()
            tl += loss.item()
        train_losses.append(tl / max(1, len(train_loader)))

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vl += mdn_loss(yb, model(xb)).item()
        val_losses.append(vl / max(1, len(val_loader)))

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        scheduler.step(val_losses[-1])

        if (epoch + 1) % 10 == 0:
            logger.info("  Epoch [%d/%d] train %.4f val %.4f", epoch + 1, epochs,
                        train_losses[-1], val_losses[-1])

    logger.info("MDN training completed in %.2f s", time.time() - start)
    assert best_state is not None
    model.load_state_dict(best_state)

    model_path: Optional[str] = None
    if save_model:
        save_dir = Path(model_dir) if model_dir else Path("./model/mdn")
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = str(save_dir / f"mdn_model_{suffix}.pth")
        torch.save(model.state_dict(), model_path)
        logger.info("MDN saved to %s", model_path)

    return model, model_path, {"train_losses": train_losses, "val_losses": val_losses}


# ── Target standardization for well-conditioned MDN fits ───────────────
#
# The reference tanh backbone (outputs squashed to [-1, 1]) cannot reach raw
# RUL-scale targets (y ~ 50-100) without sigma-collapse: components sit near 0
# with sigma -> MDN_SIGMA_MIN, so the learned density is ~0 at the data. This
# breaks any density-based score (DistCP/HPD) and inflates the C-MDN conformal
# adjustment. We therefore fit the MDN on z-scored targets and map the mixture
# parameters back to data scale; a scalar std means the -log-density scores
# differ from the standardized ones only by a constant (log std) that cancels
# in the conformal quantile, so coverage is unaffected while fit quality is
# restored.


def standardize_y(y: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Return ``(y_standardized, mean, std)`` with a unit floor on ``std``."""
    y = np.asarray(y, dtype=float)
    mean = float(np.mean(y))
    std = float(np.std(y))
    if std <= 0:
        std = 1.0
    return (y - mean) / std, mean, std


def scale_params_to_data(
    params: Tuple[np.ndarray, np.ndarray, np.ndarray], mean: float, std: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map standardized ``(pi, mu, sigma)`` back to data scale ``mu*std+mean``,
    ``sigma*std`` (weights unchanged)."""
    pi, mu, sigma = params
    return pi, mu * std + mean, np.maximum(sigma * std, MDN_SIGMA_MIN)


def fit_mdn_standardized(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_components: int = MDN_N_COMPONENTS,
    device: Optional[torch.device] = None,
    epochs: int = MDN_EPOCHS,
    seed: int = RANDOM_SEED,
    save_model: bool = False,
    model_dir: Optional[str] = None,
    suffix: str = "",
) -> Tuple[MDN, float, float]:
    """Train an MDN on z-scored targets; return ``(model, y_mean, y_std)``."""
    ys_tr, mean, std = standardize_y(y_train)
    ys_val = (np.asarray(y_val, dtype=float) - mean) / std
    model, _, _ = train_mdn(
        X_train, ys_tr, X_val, ys_val, n_components=n_components, device=device,
        epochs=epochs, save_model=save_model, model_dir=model_dir, suffix=suffix,
        seed=seed,
    )
    return model, mean, std


def predict_params_data(
    model: MDN, X: np.ndarray, mean: float, std: float,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mixture params for ``X`` de-standardized to data scale."""
    params = model.predict_params(X, device=device)
    return scale_params_to_data(params, mean, std)



# ── Density + quantile extraction ──────────────────────────────────────


def mdn_log_pdf(
    params: Tuple[np.ndarray, np.ndarray, np.ndarray],
    y: np.ndarray,
) -> np.ndarray:
    """
    Log mixture density ``log sum_k pi_k N(y; mu_k, sigma_k^2)`` at points ``y``.

    Args:
        params: ``(pi, mu, sigma)`` each shape ``(N, K)``.
        y: Evaluation points, shape ``(N,)``.

    Returns:
        Log-density array, shape ``(N,)``.
    """
    from scipy.special import logsumexp

    pi, mu, sigma = params
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    sigma = np.maximum(sigma, MDN_SIGMA_MIN)
    # log of each component density, shape (N, K)
    comp = -0.5 * np.square((y - mu) / sigma) - np.log(sigma * np.sqrt(2.0 * np.pi))
    return logsumexp(np.log(np.maximum(pi, 1e-300)) + comp, axis=-1)


def mdn_cdf(
    params: Tuple[np.ndarray, np.ndarray, np.ndarray],
    y: np.ndarray,
) -> np.ndarray:
    """
    Mixture CDF ``F(y) = sum_k pi_k Phi((y - mu_k)/sigma_k)`` evaluated at ``y``.

    ``y`` is broadcast row-wise against the per-sample components.

    Args:
        params: ``(pi, mu, sigma)`` each shape ``(N, K)``.
        y: Evaluation points, shape ``(N, M)`` (M grid points per sample).

    Returns:
        CDF values, shape ``(N, M)``.
    """
    from scipy.stats import norm

    pi, mu, sigma = params
    sigma = np.maximum(sigma, MDN_SIGMA_MIN)
    # y: (N,M) -> (N,M,1); mu/sigma: (N,K) -> (N,1,K)
    z = (y[..., None] - mu[:, None, :]) / sigma[:, None, :]
    return np.sum(pi[:, None, :] * norm.cdf(z), axis=-1)  # (N,M)


def mdn_quantiles(
    params: Tuple[np.ndarray, np.ndarray, np.ndarray],
    quantile_levels: Sequence[float],
    grid: int = MDN_CDF_GRID,
    range_sigmas: float = MDN_CDF_RANGE_SIGMAS,
    chunk: int = 512,
) -> np.ndarray:
    """
    Invert the mixture CDF to obtain predictive quantiles.

    A per-sample bracket ``[min_k mu_k - R, max_k mu_k + R]`` with
    ``R = range_sigmas * max_k sigma_k`` is gridded into ``grid`` points, the
    CDF is evaluated on the grid (monotone increasing), and each requested
    level is read off by linear interpolation. Monotonicity of the CDF makes the
    returned quantiles monotone in ``tau`` by construction (no crossing).

    Samples are processed in ``chunk``-sized blocks so the ``(B, grid, K)``
    broadcast stays within memory.

    Args:
        params: ``(pi, mu, sigma)`` each shape ``(N, K)``.
        quantile_levels: Ascending levels in (0, 1), e.g. ``[0.1, 0.5, 0.9]``.
        grid: Number of CDF grid points per sample.
        range_sigmas: Bracket half-width in units of the largest component sigma.
        chunk: Max samples evaluated per block.

    Returns:
        Array shape ``(N, len(quantile_levels))`` of quantiles, columns ascending.
    """
    pi, mu, sigma = params
    sigma = np.maximum(sigma, MDN_SIGMA_MIN)
    taus = np.asarray(sorted(quantile_levels), dtype=float)
    n = pi.shape[0]
    out = np.empty((n, taus.size), dtype=float)

    t = np.linspace(0.0, 1.0, grid)[None, :]
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        s = slice(start, stop)
        half = range_sigmas * sigma[s].max(axis=1)
        lo = mu[s].min(axis=1) - half
        hi = mu[s].max(axis=1) + half
        ys = lo[:, None] + (hi - lo)[:, None] * t  # (B, grid)
        cdf = mdn_cdf((pi[s], mu[s], sigma[s]), ys)  # (B, grid), ascending in y

        B = stop - start
        rows = np.arange(B)
        for j, tau in enumerate(taus):
            # per-row insertion point where cdf first reaches tau
            idx = (cdf < tau).sum(axis=1)  # (B,) in [0, grid]
            idx_hi = np.clip(idx, 1, grid - 1)
            idx_lo = idx_hi - 1
            c0 = cdf[rows, idx_lo]
            c1 = cdf[rows, idx_hi]
            y0 = ys[rows, idx_lo]
            y1 = ys[rows, idx_hi]
            denom = c1 - c0
            frac = np.where(denom > 1e-300, (tau - c0) / (denom + 1e-300), 0.0)
            frac = np.clip(frac, 0.0, 1.0)
            q = y0 + frac * (y1 - y0)
            # below/above bracket: clamp to grid ends
            q = np.where(idx == 0, ys[rows, 0], q)
            q = np.where(idx >= grid, ys[rows, grid - 1], q)
            out[s, j] = q
    return out


# ── C-MDN series arm ───────────────────────────────────────────────────


def train_test_mdn_cqr(
    X_train_all: np.ndarray,
    y_train_all: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    quantiles: Sequence[float] = QUANTILES,
    alpha: float = NOMINAL_ALPHA,
    n_components: int = MDN_N_COMPONENTS,
    device: Optional[torch.device] = None,
    seed: int = RANDOM_SEED,
    cal_fraction: float = CQR_CALIBRATION_SPLIT,
    epochs: int = MDN_EPOCHS,
    save_model: bool = False,
    model_dir: Optional[str] = None,
    suffix: str = "",
) -> Tuple[MDN, float, Dict[str, np.ndarray], Dict[str, float]]:
    """
    C-MDN **series** arm: MDN quantiles → CQR max-score → split-CP calibration.

    Uses the SAME conformal procedure as C-QRNN (:mod:`qrnn_uq_rul.conformal`)
    so the comparison isolates the base predictor (density-fusion vs series).

    Split strategy mirrors :func:`models.train_test_qrnn_cqr`: carve a held-out
    calibration set, a further train/val split for early stopping, train the
    MDN, extract quantiles on calibration → CQR scores → E_hat, then build
    calibrated intervals on the test set.

    Args:
        X_train_all, y_train_all: Pool that will be split into train/val/cal.
        X_test, y_test: Held-out test set.
        quantiles: Ascending quantile levels (default [0.1, 0.5, 0.9]).
        alpha: Miscoverage rate for split-CP (fix M1).
        n_components: MDN mixture components.
        device: Torch device.
        seed: Random seed.
        cal_fraction: Fraction of the training pool used for train+val.
        epochs: Max MDN epochs.
        save_model, model_dir, suffix: Persistence options.

    Returns:
        ``(model, e_hat, {"lower","median","upper"} on test, {"mae": ...})``.
    """
    from sklearn.model_selection import train_test_split

    device = device or get_device()
    torch.manual_seed(seed)
    np.random.seed(seed)

    y_train_all = np.asarray(y_train_all, dtype=float).ravel()
    y_test = np.asarray(y_test, dtype=float).ravel()

    X_trval, X_cal, y_trval, y_cal = train_test_split(
        X_train_all, y_train_all, test_size=1.0 - cal_fraction, random_state=seed,
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_trval, y_trval, test_size=0.10, random_state=seed,
    )

    model, y_mean, y_std = fit_mdn_standardized(
        X_tr, y_tr, X_val, y_val, n_components=n_components, device=device,
        epochs=epochs, seed=seed, save_model=save_model, model_dir=model_dir, suffix=suffix,
    )

    qlist = sorted(quantiles)
    # calibration quantiles (data scale) -> CQR E_hat (data-scale units)
    cal_params = predict_params_data(model, X_cal, y_mean, y_std, device=device)
    cal_q = mdn_quantiles(cal_params, qlist)
    e_hat = calibrate_cqr_from_quantiles(y_cal, cal_q[:, 0], cal_q[:, -1], alpha=alpha)

    # test intervals
    test_params = predict_params_data(model, X_test, y_mean, y_std, device=device)
    test_q = mdn_quantiles(test_params, qlist)
    med_idx = qlist.index(0.5) if 0.5 in qlist else len(qlist) // 2
    y_pred = cqr_predict(test_q[:, 0], test_q[:, med_idx], test_q[:, -1], e_hat)

    mae = float(np.mean(np.abs(y_test - y_pred["median"])))
    logger.info("C-MDN series: E_hat=%.4f, test MAE=%.4f", e_hat, mae)
    return model, e_hat, y_pred, {"mae": mae}

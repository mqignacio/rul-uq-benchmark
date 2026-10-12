"""
Deep Evidential Regression (EDL) baseline — Amini, Schwarting, Soleimany & Rus,
"Deep Evidential Regression," NeurIPS 2020 (design §4.2, closes R2.2).

A single network emits the four hyperparameters of a **Normal-Inverse-Gamma
(NIG)** conjugate prior, ``(γ, λ, α, β)``, per input. Marginalising the
Gaussian likelihood over the NIG gives a Student-t predictive

    y | x  ~  t_{2α}( loc = γ ,  scale = sqrt(β (λ + 1) / λ) ),

with predictive variance ``β (λ + 1) / (λ (α - 1))`` for ``α > 1``. This is a
*model-internal* UQ method: its interval is derived from the fitted NIG and is
**not** distribution-free (coverage is model-assumed, design §8), so it is a
reference point that is expected to mis-calibrate under shift. To compare it
fairly (fix M4) we ALSO expose a conformalized variant that feeds the model's
own interval endpoints into the SAME split-CP CQR procedure as every other arm.

Parameterization, the Student-t marginal NLL, and interval construction are
unit-tested against ``torch.distributions.StudentT`` for numerical fidelity.
The annealed evidence (OOD) regularizer follows the reference's form; its exact
coefficients are tagged ``[verify]`` pending the Amini PDF (see
``doc/methodology.md``).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import StudentT
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .conformal import calibrate_cqr_from_quantiles, cqr_predict
from .config import (
    BATCH_SIZE,
    CQR_CALIBRATION_SPLIT,
    EDL_EVIDENCE_ANNEAL_EPOCHS,
    EDL_EVIDENCE_REG_COEF,
    EDL_HNN_HIDDEN_DIMS,
    EDL_NIG_ALPHA_MIN,
    EDL_NIG_NU_MIN,
    EPOCHS,
    LR,
    NOMINAL_ALPHA,
    NOMINAL_COVERAGE,
    RANDOM_SEED,
    WEIGHT_DECAY,
)
from .models import RULDataset
from .utils import get_device

logger = logging.getLogger(__name__)

_BETA_MIN = 1e-8  # β > 0 floor (predictive scale must be positive)


# ── Evidential network ─────────────────────────────────────────────────


class EvidentialMLP(nn.Module):
    """
    MLP emitting the four NIG hyperparameters per input.

    Raw head outputs are transformed to satisfy the NIG support constraints:
    γ (location) unbounded, λ>0, β>0, and α>1 (so the predictive variance is
    finite). The softplus/affine maps mirror the reference DER parameterization.
    """

    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = EDL_HNN_HIDDEN_DIMS):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 4)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        raw = self.head(self.backbone(x))               # (N, 4)
        gamma = raw[:, 0]
        lam = torch.nn.functional.softplus(raw[:, 1]) + EDL_NIG_NU_MIN       # λ > 0
        beta = torch.nn.functional.softplus(raw[:, 2]) + _BETA_MIN           # β > 0
        alpha = 1.0 + torch.nn.functional.softplus(raw[:, 3]) + (EDL_NIG_ALPHA_MIN - 1.0)  # α > 1
        return gamma, lam, alpha, beta

    @property
    def nig(self) -> Dict[str, str]:
        return {"loc": "gamma", "precision_scale": "lambda", "shape": "alpha", "scale": "beta"}


def nig_predictive(gamma, lam, alpha, beta) -> Dict[str, torch.Tensor]:
    """
    Predictive moments + Student-t parameters from NIG hyperparameters.

    Marginalising the Gaussian likelihood over the NIG gives
    ``y|x ~ t_{df}(loc=gamma, scale=sqrt(beta(lam+1)/lam))`` with ``df = 2*alpha``.
    Returns:
        ``mean`` (=γ), ``scale_sq`` (=β(λ+1)/λ; the Student-t squared scale, which
        Amini denote ``v``), ``scale`` (=sqrt(scale_sq)), ``df`` (=2α), and
        ``variance`` — the true marginal variance ``scale_sq * df/(df-2)``
        ``= alpha*beta*(lam+1)/(lam*(alpha-1))`` for ``alpha > 1``.
    """
    scale_sq = beta * (lam + 1.0) / lam
    df = 2.0 * alpha
    variance = scale_sq * df / (df - 2.0)          # == alpha * scale_sq / (alpha - 1)
    return {"mean": gamma, "variance": variance, "scale_sq": scale_sq,
            "scale": torch.sqrt(scale_sq), "df": df}


# ── Loss (DER) ─────────────────────────────────────────────────────────


def edl_marginal_nll(
    y: torch.Tensor, gamma: torch.Tensor, lam: torch.Tensor,
    alpha: torch.Tensor, beta: torch.Tensor,
) -> torch.Tensor:
    """
    Per-sample negative log of the Student-t marginal (identical to Amini's
    closed-form ``negative_log_ev`` for the NIG-Gaussian conjugate model).
    """
    pred = nig_predictive(gamma, lam, alpha, beta)
    st = StudentT(df=pred["df"], loc=pred["mean"], scale=pred["scale"])
    return -st.log_prob(y)


def edl_evidence_regularizer(
    y: torch.Tensor, gamma: torch.Tensor, lam: torch.Tensor, beta: torch.Tensor,
) -> torch.Tensor:
    """
    Amini annealed OOD-evidence penalty. Penalizes large evidence (λ) when the
    residual is large, and rewards evidence overall. Exact coefficients tagged
    ``[verify]`` against the reference implementation's ``reg_loss``.
    """
    resid = torch.abs(y - gamma)
    omega = torch.sqrt(lam * beta)
    return resid * lam + omega


def edl_loss(
    y: torch.Tensor, gamma: torch.Tensor, lam: torch.Tensor,
    alpha: torch.Tensor, beta: torch.Tensor,
    epoch: int = EDL_EVIDENCE_ANNEAL_EPOCHS,
    reg_coef: float = EDL_EVIDENCE_REG_COEF,
    anneal_epochs: int = EDL_EVIDENCE_ANNEAL_EPOCHS,
) -> torch.Tensor:
    """
    Full DER objective = mean marginal NLL + annealed evidence regularizer.

    The regularizer weight ramps linearly from 0 to ``reg_coef`` over
    ``anneal_epochs`` (Amini's annealing schedule), so early training is driven
    by the likelihood before the OOD penalty is applied.
    """
    nll = edl_marginal_nll(y, gamma, lam, alpha, beta).mean()
    k = (min(epoch, anneal_epochs) / max(1, anneal_epochs)) * reg_coef
    reg = edl_evidence_regularizer(y, gamma, lam, beta).mean()
    return nll + k * reg


# ── Training ───────────────────────────────────────────────────────────


def train_edl(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: Optional[torch.device] = None,
    lr: float = LR,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    weight_decay: float = WEIGHT_DECAY,
    reg_coef: float = EDL_EVIDENCE_REG_COEF,
    anneal_epochs: int = EDL_EVIDENCE_ANNEAL_EPOCHS,
    save_model: bool = False,
    model_dir: Optional[str] = None,
    suffix: str = "",
    seed: int = RANDOM_SEED,
) -> Tuple[EvidentialMLP, Optional[str], Dict[str, List[float]]]:
    """Train an EDL model with validation-based early stopping (mirrors train_qrnn)."""
    device = device or get_device()
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = EvidentialMLP(X_train.shape[1]).to(device)
    train_loader = DataLoader(RULDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(RULDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    best_val = float("inf")
    best_state: Optional[dict] = None
    train_losses: List[float] = []
    val_losses: List[float] = []

    logger.info("Training EDL (NIG) model...")
    start = time.time()
    for epoch in range(epochs):
        model.train()
        tl = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            gamma, lam, alpha, beta = model(xb)
            loss = edl_loss(yb, gamma, lam, alpha, beta, epoch=epoch,
                            reg_coef=reg_coef, anneal_epochs=anneal_epochs)
            loss.backward()
            optimizer.step()
            tl += loss.item()
        train_losses.append(tl / max(1, len(train_loader)))

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                gamma, lam, alpha, beta = model(xb)
                vl += edl_marginal_nll(yb, gamma, lam, alpha, beta).mean().item()
        val_losses.append(vl / max(1, len(val_loader)))
        if val_losses[-1] < best_val:
            best_val = val_losses[-1]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        scheduler.step(val_losses[-1])
        if (epoch + 1) % 10 == 0:
            logger.info("  Epoch [%d/%d] train %.4f val %.4f",
                        epoch + 1, epochs, train_losses[-1], val_losses[-1])

    logger.info("EDN training done in %.2f s", time.time() - start)
    assert best_state is not None
    model.load_state_dict(best_state)

    model_path: Optional[str] = None
    if save_model:
        save_dir = Path(model_dir) if model_dir else Path("./model/edl")
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = str(save_dir / f"edl_model_{suffix}.pth")
        torch.save(model.state_dict(), model_path)
        logger.info("EDL saved to %s", model_path)
    return model, model_path, {"train_losses": train_losses, "val_losses": val_losses}


# ── Model-internal prediction interval ─────────────────────────────────


@torch.no_grad()
def edl_predict(
    model: EvidentialMLP, X: np.ndarray, coverage: float = NOMINAL_COVERAGE,
    device: Optional[torch.device] = None,
) -> Dict[str, np.ndarray]:
    """
    Model-internal Student-t predictive interval at the nominal level (fix M1).

    Returns arrays ``mean`` (=γ), ``variance``, and the NIG hyperparameters, plus
    ``lower`` / ``upper`` = γ + t.ppf(·, df=2α) · scale. This is NOT conformally
    valid — it is the model's own interval (design §8). Quantiles use
    ``scipy.stats.t.ppf`` (torch's StudentT exposes no ``icdf``).
    """
    from scipy.stats import t as student_t

    dev = device if device is not None else next(model.parameters()).device
    model.eval()
    xt = torch.FloatTensor(np.asarray(X, dtype=np.float32)).to(dev)
    gamma, lam, alpha, beta = model(xt)
    pred = nig_predictive(gamma, lam, alpha, beta)
    mean_np = pred["mean"].cpu().numpy()
    scale_np = pred["scale"].cpu().numpy()
    df_np = pred["df"].cpu().numpy()
    tail = (1.0 - coverage) / 2.0
    lo = mean_np + scale_np * student_t.ppf(tail, df_np)
    hi = mean_np + scale_np * student_t.ppf(1.0 - tail, df_np)
    return {
        "lower": lo, "median": mean_np, "upper": hi,
        "variance": pred["variance"].cpu().numpy(),
        "gamma": gamma.cpu().numpy(), "lambda": lam.cpu().numpy(),
        "alpha": alpha.cpu().numpy(), "beta": beta.cpu().numpy(),
    }


def train_test_edl(
    X_train_all: np.ndarray,
    y_train_all: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float = NOMINAL_ALPHA,
    device: Optional[torch.device] = None,
    seed: int = RANDOM_SEED,
    cal_fraction: float = CQR_CALIBRATION_SPLIT,
    epochs: int = EPOCHS,
    conformalize: bool = True,
    save_model: bool = False,
    model_dir: Optional[str] = None,
    suffix: str = "",
) -> Tuple[EvidentialMLP, float, Dict[str, np.ndarray], Dict[str, float]]:
    """
    EDL baseline arm.

    Always trains the NIG model and computes its model-internal interval. When
    ``conformalize=True`` the internal endpoints are fed through the SAME split-CP
    CQR procedure as the other arms (fix M4) so calibration is what is compared,
    not the network. Returns ``(model, e_hat, {"lower","median","upper"} on test,
    {"mae", "e_hat"})``; ``e_hat`` is 0.0 when ``conformalize=False``.
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

    model, _, _ = train_edl(
        X_tr, y_tr, X_val, y_val, device=device, epochs=epochs, seed=seed,
        save_model=save_model, model_dir=model_dir, suffix=suffix)

    cal_pred = edl_predict(model, X_cal, device=device)
    test_pred = edl_predict(model, X_test, device=device)

    if conformalize:
        e_hat = calibrate_cqr_from_quantiles(
            y_cal, cal_pred["lower"], cal_pred["upper"], alpha=alpha)
        y_pred = cqr_predict(test_pred["lower"], test_pred["median"], test_pred["upper"], e_hat)
    else:
        e_hat = 0.0
        y_pred = {"lower": test_pred["lower"], "median": test_pred["median"],
                  "upper": test_pred["upper"]}

    mae = float(np.mean(np.abs(y_test - y_pred["median"])))
    logger.info("EDL arm: e_hat=%.4f conformalize=%s MAE=%.4f", e_hat, conformalize, mae)
    return model, e_hat, y_pred, {"mae": mae, "e_hat": e_hat}

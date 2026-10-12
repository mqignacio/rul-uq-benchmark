"""
UACQR-S-style method: uncertainty-aware, epistemic-localized conformalized
quantile regression [Rossellini2024].

This is one of the *benchmarked* distribution-free arms (design §4.1), not a
claimed novelty. A QRNN **ensemble** (one member per seed, fix M4) provides:

* an ensemble-averaged quantile prediction (better point estimate — supports
  prediction rows P8/P9), and
* a **local epistemic scalar** ``ĝ_ep(x)`` = the cross-member disagreement of
  the interval endpoints (where the members disagree, the interval widens).

The nonconformity score is the CQR max-score *normalized* by ``ĝ_ep`` (a locally
adaptive score), calibrated by the SAME split-CP routine as every other arm
(:mod:`qrnn_uq_rul.conformal`, fix M4). Guarantee remains marginal coverage
under exchangeability (design §8) — the epistemic localization targets
*conditional* coverage / sharpness (H2), not a conditional guarantee.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .conformal import conformal_quantile, normalized_max_score, normalized_predict
from .config import (
    CQR_CALIBRATION_SPLIT,
    NOMINAL_ALPHA,
    RANDOM_SEED,
    SEED_LIST,
    UACQR_ENSEMBLE_SIZE,
    UACQR_SCORE_EPS,
)
from .models import QRNN, train_qrnn
from .utils import get_device

logger = logging.getLogger(__name__)


# ── Ensemble ───────────────────────────────────────────────────────────


class QuantileEnsemble:
    """A collection of trained QRNN quantile-regression members."""

    def __init__(self, members: Sequence[QRNN], device: Optional[torch.device] = None):
        self.members = list(members)
        self.device = device or get_device()

    def __len__(self) -> int:
        return len(self.members)

    @torch.no_grad()
    def predict_members(
        self, X: np.ndarray, device: Optional[torch.device] = None
    ) -> np.ndarray:
        """Return per-member quantile predictions, shape ``(M, N, 3)``."""
        dev = device if device is not None else next(self.members[0].parameters()).device
        xt = torch.FloatTensor(np.asarray(X, dtype=np.float32)).to(dev)
        out = []
        for m in self.members:
            m.eval()
            out.append(m(xt).cpu().numpy())
        return np.stack(out, axis=0)

    def predict_ensemble(
        self, X: np.ndarray, device: Optional[torch.device] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ensemble-averaged quantiles and the epistemic disagreement scalar.

        Returns:
            ``(q_mean, g_ep)`` with ``q_mean`` shape ``(N, 3)`` (mean over
            members of each quantile, columns ordered by ``QUANTILES``) and
            ``g_ep`` shape ``(N,)`` — the cross-member std of the lower and
            upper endpoints, averaged.
        """
        preds = self.predict_members(X, device=device)  # (M, N, 3)
        q_mean = preds.mean(axis=0)                       # (N, 3)
        g_ep = epistemic_scale(preds)
        return q_mean, g_ep


def epistemic_scale(member_preds: np.ndarray, eps: float = UACQR_SCORE_EPS) -> np.ndarray:
    """
    Local epistemic scalar ``ĝ_ep(x)`` = disagreement among ensemble members.

    For each sample we take the std across members of the lower endpoint and of
    the upper endpoint and average the two. Zero when all members agree exactly
    (floored at ``eps`` to keep the normalized score finite). This is the
    quantity UACQR-S divides the CQR score by, so intervals widen where the
    ensemble is uncertain [Rossellini2024].

    Args:
        member_preds: ``(M, N, 3)`` per-member quantile predictions.
        eps: Floor applied to the returned scale.

    Returns:
        ``(N,)`` strictly positive epistemic scale.
    """
    preds = np.asarray(member_preds, dtype=float)
    std_lo = preds[:, :, 0].std(axis=0, ddof=0)   # (N,)
    std_hi = preds[:, :, -1].std(axis=0, ddof=0)  # (N,)
    g = 0.5 * (std_lo + std_hi)
    return np.maximum(g, eps)


def fit_quantile_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_members: int = UACQR_ENSEMBLE_SIZE,
    seeds: Optional[Sequence[int]] = None,
    device: Optional[torch.device] = None,
    save_model: bool = False,
    model_dir: Optional[str] = None,
    suffix: str = "",
) -> QuantileEnsemble:
    """
    Train ``n_members`` QRNNs (one per seed) with validation early stopping.

    Each member uses the same architecture as C-QRNN but a distinct seed, so
    their disagreement is an epistemic signal. ``train_qrnn`` governs its own
    epoch budget via the ``EPOCHS`` constant (tests monkeypatch it).

    Args:
        X_train, y_train, X_val, y_val: Training/validation tensors.
        n_members: Number of ensemble members (fix M4: >= 5 for the study).
        seeds: Explicit seeds; defaults to ``SEED_LIST[:n_members]``.
        device: Torch device.
        save_model, model_dir, suffix: Persistence passthrough.

    Returns:
        A :class:`QuantileEnsemble` of the trained members.
    """
    device = device or get_device()
    seeds = list(seeds) if seeds is not None else list(SEED_LIST)
    if n_members > len(seeds):
        raise ValueError(
            f"n_members={n_members} exceeds available seeds ({len(seeds)}); "
            "extend SEED_LIST to keep members distinct (fix M4)."
        )

    members: List[QRNN] = []
    for i, seed in enumerate(seeds[:n_members]):
        model, _, _ = train_qrnn(
            X_train, y_train, X_val, y_val,
            device=device, seed=seed, save_model=save_model,
            model_dir=model_dir, suffix=f"{suffix}_m{i}",
        )
        members.append(model)
        logger.info("UACQR ensemble member %d (seed=%d) trained", i, seed)

    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return QuantileEnsemble(members, device=device)


# ── UACQR-S arm ────────────────────────────────────────────────────────


def train_test_uacqr(
    X_train_all: np.ndarray,
    y_train_all: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float = NOMINAL_ALPHA,
    n_members: int = UACQR_ENSEMBLE_SIZE,
    seeds: Optional[Sequence[int]] = None,
    device: Optional[torch.device] = None,
    seed: int = RANDOM_SEED,
    cal_fraction: float = CQR_CALIBRATION_SPLIT,
    save_model: bool = False,
    model_dir: Optional[str] = None,
    suffix: str = "",
) -> Tuple[QuantileEnsemble, float, Dict[str, np.ndarray], Dict[str, float]]:
    """
    UACQR-S arm: ensemble CQR with an epistemic-normalized score + split-CP.

    Split strategy mirrors the C-QRNN pipeline (calibration carved from the
    training pool, further train/val split for early stopping), so the only
    differences from C-QRNN are (a) the ensemble base predictor and (b) the
    locally-adaptive (epistemic-normalized) nonconformity score.

    Args:
        X_train_all, y_train_all: Pool split into train / val / calibration.
        X_test, y_test: Held-out test set.
        alpha: Miscoverage rate (fix M1).
        n_members: Ensemble size (fix M4: >= 5).
        seeds: Member seeds.
        device: Torch device.
        seed: Split RNG seed.
        cal_fraction: Fraction of pool used for train+val.
        save_model, model_dir, suffix: Persistence passthrough.

    Returns:
        ``(ensemble, q_hat, {"lower","median","upper"} on test,
        {"mae", "mean_epistemic"})``.
    """
    from sklearn.model_selection import train_test_split

    device = device or get_device()
    torch.manual_seed(seed)
    np.random.seed(seed)
    y_all = np.asarray(y_train_all, dtype=float).ravel()
    y_test = np.asarray(y_test, dtype=float).ravel()

    X_trval, X_cal, y_trval, y_cal = train_test_split(
        X_train_all, y_all, test_size=1.0 - cal_fraction, random_state=seed,
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_trval, y_trval, test_size=0.10, random_state=seed,
    )

    ensemble = fit_quantile_ensemble(
        X_tr, y_tr, X_val, y_val, n_members=n_members, seeds=seeds, device=device,
        save_model=save_model, model_dir=model_dir, suffix=suffix,
    )

    # calibration: averaged quantiles + epistemic scale -> normalized scores
    q_cal, g_cal = ensemble.predict_ensemble(X_cal, device=device)
    scores_cal, _ = normalized_max_score(y_cal, q_cal[:, 0], q_cal[:, 2], g_cal, eps=UACQR_SCORE_EPS)
    q_hat = conformal_quantile(scores_cal, alpha=alpha)

    # test: locally-adaptive interval reconstruction
    q_te, g_te = ensemble.predict_ensemble(X_test, device=device)
    y_pred = normalized_predict(q_te[:, 0], q_te[:, 1], q_te[:, 2], g_te, q_hat, eps=UACQR_SCORE_EPS)

    mae = float(np.mean(np.abs(y_test - y_pred["median"])))
    logger.info("UACQR-S: q_hat=%.4f, test MAE=%.4f, mean g_ep=%.4f",
                q_hat, mae, float(np.mean(g_te)))
    return ensemble, q_hat, y_pred, {"mae": mae, "mean_epistemic": float(np.mean(g_te))}

"""
End-to-end integration test: every conformal arm's coverage on one synthetic
*exchangeable* dataset (tasks/03-implement.md acceptance criterion).

This is the single place that runs ALL distribution-free arms (C-QRNN, C-MDN
series, UACQR-S, DistCP/HPD, Weighted-CP CQR, WQLCP-style) plus the calibrated
model-internal baselines (EDL-conformal, C-XGBQ, C-GPR) through the shared split-
CP procedure and asserts each lands near the 0.90 nominal. It is an
implementation-correctness gate, not a benchmark: the arms are compared for
coverage validity, NOT ranked (ranking is the Phase-4 experiment on real data).

Training budgets are kept small; only CPU is used to avoid MPS/xgboost OpenMP
interactions. Marked ``slow`` so it can be deselected during tight dev loops.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import qrnn_uq_rul.models as models_mod
from qrnn_uq_rul.conformal import calibrate_cqr_from_quantiles, cqr_predict
from qrnn_uq_rul.metrics import calculate_picp
from qrnn_uq_rul.models import train_qrnn
from qrnn_uq_rul.models_calibration import train_test_cgpr, train_test_cxgbq
from qrnn_uq_rul.models_edl import train_test_edl
from qrnn_uq_rul.models_mdn import train_test_mdn_cqr
from qrnn_uq_rul.models_uacqr import train_test_uacqr
from qrnn_uq_rul.models_variants import (
    train_test_distcp,
    train_test_weighted_cqr,
    train_test_wqlcp,
)

CPU = torch.device("cpu")
NOMINAL = 0.90
# generous per-arm tolerance: exchangeable split-CP concentrates near 0.90 but a
# small-N MDN/AE fit has extra variance; this gate catches broken calibration,
# not sharpness differences.
TOL_LO, TOL_HI = 0.78, 0.99


@pytest.fixture(autouse=True)
def _fast_epochs(monkeypatch):
    monkeypatch.setattr(models_mod, "EPOCHS", 25, raising=True)


def _hetero(n, seed):
    """Single-law heteroscedastic regression (cal/test share it -> exchangeable)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, 3)).astype(np.float32)
    f = 50 + 3 * x[:, 0] - 2 * x[:, 1]
    s = 1.5 + np.abs(x[:, 2])
    y = (f + s * rng.normal(0, 1, n)).astype(np.float32)
    return x, y


def _cqrnn_numpy(X_tr, y_tr, X_val, y_val, X_cal, y_cal, X_te):
    """C-QRNN via the shared conformal stage (numpy path, mirrors Phase-2 logic)."""
    model, _, _ = train_qrnn(X_tr, y_tr, X_val, y_val, device=CPU, seed=0)
    model.eval()

    def qz(X):
        with torch.no_grad():
            return model(torch.FloatTensor(X).to(CPU)).cpu().numpy()

    qc = qz(X_cal)
    e_hat = calibrate_cqr_from_quantiles(y_cal, qc[:, 0], qc[:, 2], alpha=0.10)
    qt = qz(X_te)
    return cqr_predict(qt[:, 0], qt[:, 1], qt[:, 2], e_hat)


@pytest.fixture(scope="module")
def pools():
    """Big pool for training/calibration + a separate exchangeable test set."""
    Xtr, ytr = _hetero(1500, 1)
    Xte, yte = _hetero(800, 2)
    return (Xtr, ytr), (Xte, yte)


@pytest.mark.slow
class TestAllArmsCoverageExchangeable:
    def _check(self, pred, y_te):
        picp = calculate_picp(y_te, pred["lower"], pred["upper"])
        assert TOL_LO <= picp <= TOL_HI, f"PICP {picp:.3f} outside [{TOL_LO},{TOL_HI}]"
        assert np.all(pred["lower"] <= pred["upper"])

    def test_cqrnn(self, pools):
        (Xtr, ytr), (Xte, yte) = pools
        # disjoint train / val / cal partitions (no overlap -> valid conformal)
        n_val, n_cal = 150, 300
        X_tr, y_tr = Xtr[:-n_val - n_cal], ytr[:-n_val - n_cal]
        X_val, y_val = Xtr[-n_val - n_cal:-n_cal], ytr[-n_val - n_cal:-n_cal]
        X_cal, y_cal = Xtr[-n_cal:], ytr[-n_cal:]
        pred = _cqrnn_numpy(X_tr, y_tr, X_val, y_val, X_cal, y_cal, Xte)
        self._check(pred, yte)

    def test_cmkn_series(self, pools):
        (Xtr, ytr), (Xte, yte) = pools
        _m, _e, pred, _i = train_test_mdn_cqr(Xtr, ytr, Xte, yte, alpha=0.10, epochs=60, device=CPU)
        self._check(pred, yte)

    def test_uacqr(self, pools):
        (Xtr, ytr), (Xte, yte) = pools
        _m, _e, pred, _i = train_test_uacqr(Xtr, ytr, Xte, yte, alpha=0.10, n_members=3,
                                            seeds=[0, 1, 2], device=CPU)
        self._check(pred, yte)

    def test_distcp(self, pools):
        (Xtr, ytr), (Xte, yte) = pools
        _m, _e, pred, _i = train_test_distcp(Xtr, ytr, Xte, yte, alpha=0.10, epochs=60, device=CPU)
        self._check(pred, yte)

    def test_weighted_cqr(self, pools):
        (Xtr, ytr), (Xte, yte) = pools
        _m, _e, pred, _d = train_test_weighted_cqr(Xtr, ytr, Xte, yte, alpha=0.10, device=CPU)
        self._check(pred, yte)

    def test_wqlcp(self, pools):
        (Xtr, ytr), (Xte, yte) = pools
        _m, _e, pred, _d = train_test_wqlcp(Xtr, ytr, Xte, yte, alpha=0.10, device=CPU)
        self._check(pred, yte)

    def test_edl_conformal(self, pools):
        (Xtr, ytr), (Xte, yte) = pools
        _m, _e, pred, _i = train_test_edl(Xtr, ytr, Xte, yte, alpha=0.10, epochs=60,
                                          device=CPU, conformalize=True)
        self._check(pred, yte)

    def test_cxgbq(self, pools):
        (Xtr, ytr), (Xte, yte) = pools
        _m, _e, pred, _i = train_test_cxgbq(Xtr, ytr, Xte, yte, alpha=0.10, seed=0)
        self._check(pred, yte)

    def test_cgpr_small_n(self):
        # GPR is O(n^3); use a smaller exchangeable pair
        Xtr, ytr = _hetero(400, 5)
        Xte, yte = _hetero(300, 6)
        _g, _e, pred, _i = train_test_cgpr(Xtr, ytr, Xte, yte, alpha=0.10, seed=0)
        self._check(pred, yte)

"""
Unit tests for the quantile non-crossing penalty (fix M7) and the calibrated /
multi-seed baselines (fix M4): qrnn_loss crossing term, raw multi-seed QRNN,
C-XGBQ, and C-GPR.

The crossing test checks the *arithmetic* of the penalty exactly and that
training with a large penalty reduces the empirical quantile-crossing frequency
relative to no penalty. The calibration arms are checked for coverage near the
0.90 nominal on exchangeable data and well-formed intervals.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import qrnn_uq_rul.models as models_mod
from qrnn_uq_rul.metrics import calculate_picp, quantile_crossing_frequency
from qrnn_uq_rul.models import QRNN, qrnn_loss, train_qrnn
from qrnn_uq_rul.models_calibration import (
    raw_qrnn_multiseed,
    raw_qrnn_predict,
    train_test_cgpr,
    train_test_cxgbq,
)


@pytest.fixture(autouse=True)
def _fast_epochs(monkeypatch):
    monkeypatch.setattr(models_mod, "EPOCHS", 25, raising=True)


CPU = torch.device("cpu")


def _data(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, 3)).astype(np.float32)
    f = 50 + 3 * x[:, 0] - 2 * x[:, 1]
    s = 1.5 + np.abs(x[:, 2])
    y = (f + s * rng.normal(0, 1, n)).astype(np.float32)
    return x, y


class TestCrossingPenalty:
    def test_penalty_arithmetic_exact(self):
        preds = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])  # row 2 crosses
        y = torch.tensor([1.5, 1.5])
        base = qrnn_loss(preds, y, crossing_weight=0.0)
        pen = qrnn_loss(preds, y, crossing_weight=2.0)
        viol = torch.clamp(-torch.diff(preds, dim=1), min=0.0).mean()
        assert torch.allclose(pen - base, 2.0 * viol, atol=1e-6)
        assert viol > 0

    def test_monotone_input_no_penalty(self):
        preds = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        y = torch.tensor([1.5, 2.5])
        assert qrnn_loss(preds, y, crossing_weight=5.0) == pytest.approx(
            float(qrnn_loss(preds, y, crossing_weight=0.0)), abs=1e-6)

    def test_zero_weight_equals_baseline(self):
        preds = torch.randn(6, 3) * 3
        y = torch.randn(6)
        assert torch.allclose(qrnn_loss(preds, y, crossing_weight=0.0),
                              qrnn_loss(preds, y))

    def test_penalty_reduces_crossings_in_training(self):
        # A wide target spread encourages an unconstrained net to cross; the
        # penalty should lower the crossing frequency at matched epochs/seed.
        x, y = _data(600, 0)
        m_off, _, _ = train_qrnn(x, y, x[:80], y[:80], device=CPU, seed=0, crossing_weight=0.0)
        m_on, _, _ = train_qrnn(x, y, x[:80], y[:80], device=CPU, seed=0, crossing_weight=50.0)

        def crossings(model):
            model.eval()
            with torch.no_grad():
                p = model(torch.FloatTensor(x).to(CPU)).cpu().numpy()
            return quantile_crossing_frequency(p, [0.1, 0.5, 0.9])

        assert crossings(m_on) <= crossings(m_off)


class TestRawQRNNMultiseed:
    def test_at_least_five_distinct_members(self):
        x, y = _data(600, 0)
        models = raw_qrnn_multiseed(x, y, x[:80], y[:80], device=CPU)
        assert len(models) >= 5
        w0 = models[0].network[0].weight.detach()
        w1 = models[1].network[0].weight.detach()
        assert not torch.allclose(w0, w1)  # distinct seeds -> distinct nets

    def test_predict_keys_and_ordering(self):
        x, y = _data(400, 1)
        models = raw_qrnn_multiseed(x, y, x[:60], y[:60], device=CPU, seeds=[0, 1, 2])
        pred = raw_qrnn_predict(models, x[:20])
        assert set(pred) == {"lower", "median", "upper"}
        assert np.all(pred["lower"] <= pred["upper"])  # ensemble mean of ordered cols


class TestCXGBQ:
    def test_coverage_near_nominal(self):
        x_tr, y_tr = _data(1000, 1)
        x_te, y_te = _data(500, 2)  # exchangeable
        _m, e_hat, pred, info = train_test_cxgbq(x_tr, y_tr, x_te, y_te, alpha=0.10, seed=0)
        picp = calculate_picp(y_te, pred["lower"], pred["upper"])
        assert 0.80 <= picp <= 0.97, picp
        assert np.isfinite(e_hat)
        assert np.all(pred["lower"] <= pred["upper"])
        assert info["mae"] >= 0


class TestCGPR:
    def test_coverage_near_nominal(self):
        x_tr, y_tr = _data(400, 1)   # small n: GPR is O(n^3)
        x_te, y_te = _data(300, 2)
        _gpr, e_hat, pred, info = train_test_cgpr(x_tr, y_tr, x_te, y_te, alpha=0.10, seed=0)
        picp = calculate_picp(y_te, pred["lower"], pred["upper"])
        assert 0.70 <= picp <= 0.99, picp
        assert np.isfinite(e_hat)
        assert np.all(pred["lower"] <= pred["upper"])

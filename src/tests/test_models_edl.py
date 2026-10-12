"""
Unit tests for the EDL baseline (qrnn_uq_rul.models_edl) — Amini 2020 DER.

Validates the NIG parameterization (support constraints), predictive moments and
the Student-t interval against ``torch``/``scipy`` references, the DER marginal
NLL, training loss reduction, and the conformalized arm's coverage on synthetic
*exchangeable* data (fix M4: EDL's internal interval is fed through the same
split-CP CQR procedure as the other arms).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.stats import t as student_t
from torch.distributions import StudentT

from qrnn_uq_rul.metrics import calculate_picp
from qrnn_uq_rul.models_edl import (
    EvidentialMLP,
    edl_marginal_nll,
    edl_predict,
    nig_predictive,
    train_edl,
    train_test_edl,
)

CPU = torch.device("cpu")


def _data(n, seed, shift=0.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(shift, 1, (n, 3)).astype(np.float32)
    f = 50 + 3 * x[:, 0] - 2 * x[:, 1]
    s = 1.5 + np.abs(x[:, 2])
    y = (f + s * rng.normal(0, 1, n)).astype(np.float32)
    return x, y


class TestNIGParameterization:
    def test_support_constraints(self):
        torch.manual_seed(0)
        m = EvidentialMLP(4).to(CPU)
        g, lam, alpha, beta = m(torch.randn(50, 4, device=CPU))
        assert (lam > 0).all()
        assert (beta > 0).all()
        assert (alpha > 1).all()  # ensures finite predictive variance

    def test_variance_matches_torch_studentt(self):
        torch.manual_seed(1)
        m = EvidentialMLP(3).to(CPU)
        g, lam, alpha, beta = m(torch.randn(20, 3, device=CPU))
        pred = nig_predictive(g, lam, alpha, beta)
        st = StudentT(df=pred["df"], loc=pred["mean"], scale=pred["scale"])
        assert torch.allclose(pred["variance"], st.variance, atol=1e-4)
        assert torch.allclose(pred["scale_sq"], st.scale ** 2, atol=1e-6)
        assert torch.allclose(pred["mean"], g, atol=1e-9)

    def test_marginal_nll_matches_torch(self):
        torch.manual_seed(2)
        m = EvidentialMLP(3).to(CPU)
        g, lam, alpha, beta = m(torch.randn(15, 3, device=CPU))
        y = torch.randn(15, device=CPU) * 2 + 3
        pred = nig_predictive(g, lam, alpha, beta)
        st = StudentT(df=pred["df"], loc=pred["mean"], scale=pred["scale"])
        assert torch.allclose(edl_marginal_nll(y, g, lam, alpha, beta),
                              -st.log_prob(y), atol=1e-5)


class TestEDLPredictInterval:
    def test_interval_brackets_median(self):
        torch.manual_seed(0)
        m = EvidentialMLP(3).to(CPU).eval()
        x = np.random.randn(30, 3).astype(np.float32)
        ip = edl_predict(m, x, coverage=0.90, device=CPU)
        assert np.all(ip["lower"] <= ip["median"])
        assert np.all(ip["median"] <= ip["upper"])

    def test_interval_equals_scipy_ppf(self):
        torch.manual_seed(3)
        m = EvidentialMLP(3).to(CPU).eval()
        x = np.random.randn(10, 3).astype(np.float32)
        ip = edl_predict(m, x, coverage=0.90, device=CPU)
        with torch.no_grad():
            g, lam, alpha, beta = m(torch.tensor(x, device=CPU))
            pred = nig_predictive(g, lam, alpha, beta)
        mean = pred["mean"].cpu().numpy()
        scale = pred["scale"].cpu().numpy()
        df = pred["df"].cpu().numpy()
        exp_lo = mean + scale * student_t.ppf(0.05, df)
        exp_hi = mean + scale * student_t.ppf(0.95, df)
        assert np.allclose(ip["lower"], exp_lo, atol=1e-4)
        assert np.allclose(ip["upper"], exp_hi, atol=1e-4)

    def test_returns_nig_params(self):
        torch.manual_seed(0)
        m = EvidentialMLP(3).to(CPU).eval()
        ip = edl_predict(m, np.random.randn(6, 3).astype(np.float32), device=CPU)
        for key in {"lower", "median", "upper", "variance", "gamma", "lambda", "alpha", "beta"}:
            assert key in ip


class TestTrainEDL:
    def test_loss_reduces(self):
        x, y = _data(400, 0)
        model, path, hist = train_edl(x, y, x[:60], y[:60], device=CPU, epochs=30)
        assert path is None
        assert len(hist["train_losses"]) == 30
        assert hist["train_losses"][-1] < hist["train_losses"][0]
        assert np.isfinite(hist["train_losses"]).all()


class TestEdlConformalArm:
    def test_conformal_coverage_near_nominal(self):
        x_tr, y_tr = _data(1200, 1)
        x_te, y_te = _data(600, 2)  # exchangeable
        model, e_hat, pred, info = train_test_edl(
            x_tr, y_tr, x_te, y_te, alpha=0.10, epochs=60, device=CPU, conformalize=True)
        picp = calculate_picp(y_te, pred["lower"], pred["upper"])
        assert 0.80 <= picp <= 0.97, picp
        assert np.isfinite(e_hat)
        assert np.all(pred["lower"] <= pred["upper"])
        assert {"mae", "e_hat"} <= set(info)

    def test_nonconformal_returns_zero_ehat(self):
        x_tr, y_tr = _data(600, 1)
        x_te, y_te = _data(300, 2)
        _m, e_hat, pred, _info = train_test_edl(
            x_tr, y_tr, x_te, y_te, alpha=0.10, epochs=40, device=CPU, conformalize=False)
        assert e_hat == 0.0
        assert np.all(pred["lower"] <= pred["upper"])

    def test_determinism_same_seed(self):
        x_tr, y_tr = _data(500, 1)
        x_te, y_te = _data(200, 2)
        a = train_test_edl(x_tr, y_tr, x_te, y_te, epochs=25, device=CPU, seed=7)[2]
        b = train_test_edl(x_tr, y_tr, x_te, y_te, epochs=25, device=CPU, seed=7)[2]
        assert np.allclose(a["lower"], b["lower"], atol=1e-5)

"""
Unit tests for the MDN port + C-MDN series arm (qrnn_uq_rul.models_mdn).

Validates: (1) MDN loss numerical fidelity to a NumPy transcription of Prof.
the reference ``myMDN.mdn_loss``; (2) forward-output invariants (softmax
weights, positive sigma, seed determinism); (3) CDF-inversion quantile
correctness against scipy on a known mixture + monotonicity by construction;
(4) empirical coverage of the full C-MDN series arm on synthetic *exchangeable*
data (design §4.1, fix M4 shared conformal procedure).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.stats import norm

from qrnn_uq_rul.metrics import calculate_picp
from qrnn_uq_rul.models_mdn import (
    MDN,
    mdn_cdf,
    mdn_log_pdf,
    mdn_loss,
    mdn_quantiles,
    train_mdn,
    train_test_mdn_cqr,
)


def _reference_loss(y_true, y_pred):
    """NumPy transcription of myMDN.mdn_loss (the provenance reference)."""
    k = y_pred.shape[-1] // 3
    pi = y_pred[:, 0:k]
    mu = y_pred[:, k:2 * k]
    sigma = np.maximum(y_pred[:, 2 * k:3 * k], 1e-6)
    y = np.asarray(y_true).reshape(-1, 1)
    exponent = -0.5 * ((y - mu) / sigma) ** 2
    coeff = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    gaussian = coeff * np.exp(exponent)
    weighted = np.sum(pi * gaussian, axis=-1, keepdims=True)
    loss = -np.log(np.maximum(weighted, 1e-6))
    return float(np.mean(loss))


class TestMDNLoss:
    def test_matches_reference_transcription(self):
        rng = np.random.default_rng(0)
        n, k = 12, 5
        y_pred = np.concatenate([
            rng.dirichlet(np.ones(k), n), rng.normal(0, 2, (n, k)),
            np.abs(rng.normal(0, 1, (n, k))) + 0.3,
        ], axis=1)
        y_true = rng.normal(0, 2, (n, 1))
        got = float(mdn_loss(torch.tensor(y_true), torch.tensor(y_pred)))
        assert got == pytest.approx(_reference_loss(y_true, y_pred), abs=1e-5)

    def test_accepts_1d_targets(self):
        rng = np.random.default_rng(1)
        n, k = 8, 5
        y_pred = np.concatenate([
            rng.dirichlet(np.ones(k), n), rng.normal(0, 2, (n, k)),
            np.abs(rng.normal(0, 1, (n, k))) + 0.3,
        ], axis=1)
        y1 = rng.normal(0, 2, n)
        got = float(mdn_loss(torch.tensor(y1), torch.tensor(y_pred)))
        assert np.isfinite(got)

    def test_lower_loss_for_correct_density(self):
        # A tight mixture centered on y beats one centered far away.
        y_true = np.zeros((1, 1))
        good = np.concatenate([[[1.0, 0, 0, 0, 0]], [[0.0]*5], [[0.5]*5]], axis=1)
        bad = np.concatenate([[[1.0, 0, 0, 0, 0]], [[50.0]*5], [[0.5]*5]], axis=1)
        assert mdn_loss(torch.tensor(y_true), torch.tensor(good)) < \
            mdn_loss(torch.tensor(y_true), torch.tensor(bad))


class TestMDNForward:
    def test_shape_and_invariants(self):
        torch.manual_seed(0)
        m = MDN(input_dim=4, n_components=5)
        out = m(torch.randn(7, 4))
        assert out.shape == (7, 15)
        pi, mu, sigma = m.unpack(out)
        assert torch.allclose(pi.sum(-1), torch.ones(7), atol=1e-5)  # normalized
        assert (sigma > 0).all()

    def test_seed_determinism(self):
        x = torch.randn(3, 4)
        torch.manual_seed(0); m1 = MDN(4, n_components=5)
        torch.manual_seed(0); m2 = MDN(4, n_components=5)
        assert torch.allclose(m1(x), m2(x))            # same seed -> same init
        torch.manual_seed(1); m3 = MDN(4, n_components=5)
        assert not torch.allclose(m1(x), m3(x))        # different seed -> differs

    def test_predict_params_matches_forward(self):
        torch.manual_seed(0)
        m = MDN(3, n_components=5).eval()
        x = np.random.randn(6, 3).astype(np.float32)
        pi, mu, sigma = m.predict_params(x)
        assert pi.shape == (6, 5) and mu.shape == (6, 5) and sigma.shape == (6, 5)
        assert np.allclose(pi.sum(1), 1.0, atol=1e-5)


class TestMDNQuantiles:
    def test_single_component_matches_scipy(self):
        k = 5
        pi = np.array([[1.0, 0, 0, 0, 0]])
        mu = np.zeros((1, k)); sigma = np.ones((1, k))
        q = mdn_quantiles((pi, mu, sigma), [0.1, 0.5, 0.9], grid=8192)
        exp = np.array([norm.ppf(t) for t in [0.1, 0.5, 0.9]])
        assert np.allclose(q[0], exp, atol=2e-3)

    def test_monotone_by_construction(self):
        rng = np.random.default_rng(2)
        n, k = 20, 5
        pi = rng.dirichlet(np.ones(k), n)
        mu = rng.normal(0, 3, (n, k))
        sigma = np.abs(rng.normal(1, 0.3, (n, k))) + 0.1
        q = mdn_quantiles((pi, mu, sigma), [0.1, 0.5, 0.9])
        assert np.all(np.diff(q, axis=1) >= 0)  # never crosses

    def test_cdf_monotone_and_bounded(self):
        pi = np.array([[0.5, 0.5, 0, 0, 0]])
        mu = np.array([[-2.0, 2.0, 0, 0, 0]]); sigma = np.array([[1.0, 1.0, 1, 1, 1]])
        ys = np.linspace(-10, 10, 50)[None, :]
        cdf = mdn_cdf((pi, mu, sigma), ys)[0]
        assert np.all(np.diff(cdf) >= -1e-9)
        assert cdf[0] >= 0.0 and cdf[-1] <= 1.0 + 1e-9
        assert cdf[0] < 0.02 and cdf[-1] > 0.98

    def test_log_pdf_gaussian(self):
        pi = np.array([[1.0, 0, 0, 0, 0]])
        mu = np.zeros((1, 5)); sigma = np.ones((1, 5))
        lp = mdn_log_pdf((pi, mu, sigma), np.array([0.0]))
        assert lp[0] == pytest.approx(norm.logpdf(0.0), abs=1e-6)


class TestTrainMDN:
    def test_trains_and_reduces_loss(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, (200, 2)).astype(np.float32)
        y = (3 * x[:, 0] - x[:, 1] + rng.normal(0, 0.3, 200)).astype(np.float32)
        model, path, hist = train_mdn(
            x, y, x[:40], y[:40], n_components=3, device=torch.device("cpu"),
            epochs=15, batch_size=32,
        )
        assert path is None
        assert len(hist["train_losses"]) == 15
        assert hist["train_losses"][-1] < hist["train_losses"][0]


class TestCMdnSeriesCoverage:
    """End-to-end: the series arm's conformal intervals hit ~0.90 on exchangeable data."""

    @staticmethod
    def _gen(n, seed):
        rng = np.random.default_rng(seed)
        x = rng.normal(0, 1, (n, 3)).astype(np.float32)
        f = 50 + 3 * x[:, 0] - 2 * x[:, 1]
        s = 1.5 + np.abs(x[:, 2])
        y = (f + s * rng.normal(0, 1, n)).astype(np.float32)
        return x, y

    def test_coverage_near_nominal(self):
        x_tr, y_tr = self._gen(1200, 1)
        x_te, y_te = self._gen(600, 2)  # exchangeable with calibration (same law)
        _model, e_hat, pred, _info = train_test_mdn_cqr(
            x_tr, y_tr, x_te, y_te, alpha=0.10, epochs=60, seed=0,
            device=torch.device("cpu"),
        )
        picp = calculate_picp(y_te, pred["lower"], pred["upper"])
        assert 0.80 <= picp <= 0.97, picp
        assert e_hat > 0
        assert np.all(pred["lower"] <= pred["upper"])  # intervals well-formed

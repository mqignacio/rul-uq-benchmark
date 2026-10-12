"""
Unit tests for the UACQR-S-style arm (qrnn_uq_rul.models_uacqr).

Covers the epistemic disagreement scalar, the QRNN ensemble, the locally
adaptive (normalized) conformal calibration, and an empirical coverage check on
synthetic *exchangeable* data. ``models.EPOCHS`` is monkeypatched low so the
suite stays fast; coverage (not point accuracy) is what a conformal guarantee
must satisfy under a fixed protocol.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import qrnn_uq_rul.models as models_mod
from qrnn_uq_rul.metrics import calculate_picp
from qrnn_uq_rul.models_uacqr import (
    epistemic_scale,
    fit_quantile_ensemble,
    train_test_uacqr,
)


@pytest.fixture(autouse=True)
def _fast_epochs(monkeypatch):
    """Shrink the per-member epoch budget for all tests in this module."""
    monkeypatch.setattr(models_mod, "EPOCHS", 20, raising=True)


def _gen(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, 3)).astype(np.float32)
    f = 50 + 3 * x[:, 0] - 2 * x[:, 1]
    s = 1.5 + np.abs(x[:, 2])
    y = (f + s * rng.normal(0, 1, n)).astype(np.float32)
    return x, y


class TestEpistemicScale:
    def test_positive_and_shaped(self):
        preds = np.random.randn(5, 30, 3) + 50.0
        g = epistemic_scale(preds)
        assert g.shape == (30,)
        assert (g > 0).all()

    def test_identical_members_floor_to_eps(self):
        one = np.random.randn(1, 20, 3)
        preds = np.repeat(one, 4, axis=0)  # all members identical -> zero spread
        g = epistemic_scale(preds, eps=1e-8)
        assert np.allclose(g, 1e-8)

    def test_more_disagreement_larger_scale(self):
        base = np.zeros((4, 10, 3))
        tight = base + np.random.randn(4, 10, 3) * 0.01
        wide = base + np.random.randn(4, 10, 3) * 5.0
        assert epistemic_scale(wide).mean() > epistemic_scale(tight).mean()

    def test_single_member_is_eps(self):
        preds = np.random.randn(1, 8, 3)
        g = epistemic_scale(preds, eps=1e-8)
        assert np.allclose(g, 1e-8)  # std of one member = 0 -> floored


class TestQuantileEnsemble:
    def test_fit_and_predict_shapes(self):
        x, y = _gen(300, 0)
        ens = fit_quantile_ensemble(x, y, x[:60], y[:60], n_members=3,
                                    seeds=[0, 1, 2], device=torch.device("cpu"))
        assert len(ens) == 3
        members = ens.predict_members(x[:10])
        assert members.shape == (3, 10, 3)
        q_mean, g = ens.predict_ensemble(x[:10])
        assert q_mean.shape == (10, 3) and g.shape == (10,)
        assert (g > 0).all()

    def test_n_members_exceeds_seeds_raises(self):
        x, y = _gen(100, 0)
        with pytest.raises(ValueError):
            fit_quantile_ensemble(x, y, x[:30], y[:30], n_members=9,
                                  seeds=[0, 1, 2], device=torch.device("cpu"))

    def test_determinism_same_seeds(self):
        x, y = _gen(200, 3)
        a = fit_quantile_ensemble(x, y, x[:40], y[:40], n_members=2,
                                  seeds=[10, 11], device=torch.device("cpu"))
        b = fit_quantile_ensemble(x, y, x[:40], y[:40], n_members=2,
                                  seeds=[10, 11], device=torch.device("cpu"))
        q1, g1 = a.predict_ensemble(x[:20])
        q2, g2 = b.predict_ensemble(x[:20])
        assert np.allclose(q1, q2, atol=1e-5)
        assert np.allclose(g1, g2, atol=1e-5)

    def test_ensemble_mean_no_worse_than_worst_member(self):
        # Averaging should not be worse than the worst single member on point MAE.
        x, y = _gen(400, 5)
        ens = fit_quantile_ensemble(x, y, x[:80], y[:80], n_members=3,
                                     seeds=[1, 2, 3], device=torch.device("cpu"))
        medians = ens.predict_members(x)[:, :, 1]        # (M, N)
        member_mae = [float(np.mean(np.abs(y - medians[m]))) for m in range(medians.shape[0])]
        worst = max(member_mae)
        mean_mae = float(np.mean(np.abs(y - medians.mean(axis=0))))
        assert mean_mae <= worst + 1e-9  # ensemble mean no worse than the worst member


class TestUacqrCoverage:
    def test_coverage_near_nominal(self):
        x_tr, y_tr = _gen(1200, 1)
        x_te, y_te = _gen(600, 2)  # exchangeable with calibration
        ens, q_hat, pred, info = train_test_uacqr(
            x_tr, y_tr, x_te, y_te, alpha=0.10, n_members=3,
            seeds=[0, 1, 2], device=torch.device("cpu"),
        )
        picp = calculate_picp(y_te, pred["lower"], pred["upper"])
        assert 0.80 <= picp <= 0.97, picp
        assert q_hat > 0
        assert np.all(pred["lower"] <= pred["upper"])
        assert {"mae", "mean_epistemic"} <= set(info)
        assert info["mean_epistemic"] > 0

    def test_returns_three_keys(self):
        x_tr, y_tr = _gen(600, 4)
        x_te, y_te = _gen(300, 5)
        _ens, _q, pred, _info = train_test_uacqr(
            x_tr, y_tr, x_te, y_te, n_members=2, seeds=[0, 1],
            device=torch.device("cpu"),
        )
        assert set(pred) == {"lower", "median", "upper"}

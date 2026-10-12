"""
Unit tests for the remaining distribution-free arms (qrnn_uq_rul.models_variants):
Weighted-CP CQR (fix M2), Distributional-CP / HPD-split, and WQLCP-style.

Focus is on *implementation invariants* (density-ratio diagnostics, level-set
construction, weighted->unweighted reduction, interval validity, and coverage
on exchangeable data). The weighted-vs-split coverage *win* under shift is an
empirical hypothesis (H1) tested in Phase 4, not asserted here as a fixed result.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import qrnn_uq_rul.models as models_mod
from qrnn_uq_rul.metrics import calculate_picp
from qrnn_uq_rul.models_variants import (
    _density_level_set_interval,
    estimate_density_ratio,
    train_test_distcp,
    train_test_weighted_cqr,
    train_test_wqlcp,
)


@pytest.fixture(autouse=True)
def _fast_epochs(monkeypatch):
    monkeypatch.setattr(models_mod, "EPOCHS", 20, raising=True)


def _data(n, seed, shift=0.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(shift, 1, (n, 3)).astype(np.float32)
    f = 50 + 3 * x[:, 0] - 2 * x[:, 1]
    s = 1.5 + np.abs(x[:, 2])
    y = (f + s * rng.normal(0, 1, n)).astype(np.float32)
    return x, y


CPU = torch.device("cpu")


# ── Density-ratio estimator ────────────────────────────────────────────


class TestDensityRatio:
    def test_no_shift_near_uniform(self):
        rng = np.random.default_rng(0)
        w, d = estimate_density_ratio(rng.normal(0, 1, (600, 3)),
                                      rng.normal(0, 1, (600, 3)), seed=0)
        assert w.shape == (600,)
        assert not d["collapsed"]
        assert d["ess_fraction"] > 0.5
        assert (w > 0).all()

    def test_shift_detected(self):
        rng = np.random.default_rng(1)
        _w, d = estimate_density_ratio(rng.normal(0, 1, (500, 3)),
                                       rng.normal(3, 1, (500, 3)), seed=0)
        # severe shift -> weights degenerate (collapsed) or clip activity
        assert d["collapsed"] or d["max_weight"] < 1.0

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError):
            estimate_density_ratio(np.zeros((5, 2)), np.zeros((5, 2)), model="mlp")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            estimate_density_ratio(np.zeros((0, 2)), np.zeros((5, 2)))

    def test_gbm_variant_runs(self):
        rng = np.random.default_rng(2)
        w, d = estimate_density_ratio(rng.normal(0, 1, (300, 3)),
                                      rng.normal(0.5, 1, (300, 3)), model="gbm", seed=0)
        assert w.shape == (300,) and "ess" in d


# ── Level-set interval + multimodal detector ───────────────────────────


class TestLevelSet:
    def test_unimodal_contiguous(self):
        pi = np.array([[1.0, 0, 0, 0, 0]])
        mu = np.array([[0.0, 0, 0, 0, 0]])
        sg = np.array([[1.0, 1, 1, 1, 1]])
        lo, hi, mm = _density_level_set_interval((pi, mu, sg), q_hat=1.5, grid=4096)
        assert lo[0] < 0 < hi[0] and mm == 0.0

    def test_bimodal_flagged_disconnected(self):
        pi = np.array([[0.5, 0.5, 0, 0, 0]])
        mu = np.array([[-5.0, 5.0, 0, 0, 0]])
        sg = np.array([[0.5, 0.5, 1, 1, 1]])
        lo, hi, mm = _density_level_set_interval((pi, mu, sg), q_hat=2.5, grid=4096)
        # enclosing interval spans both peaks, but the superlevel set is disjoint
        assert lo[0] < -5 and hi[0] > 5 and mm == 1.0

    def test_larger_qhat_widens_interval(self):
        pi = np.array([[1.0, 0, 0, 0, 0]])
        mu = np.array([[0.0, 0, 0, 0, 0]])
        sg = np.array([[1.0, 1, 1, 1, 1]])
        lo1, hi1, _ = _density_level_set_interval((pi, mu, sg), q_hat=1.0, grid=4096)
        lo2, hi2, _ = _density_level_set_interval((pi, mu, sg), q_hat=2.5, grid=4096)
        # larger q_hat -> lower density level e^{-q_hat} -> more points qualify -> wider
        assert (hi2[0] - lo2[0]) >= (hi1[0] - lo1[0])


# ── Weighted-CP CQR arm ────────────────────────────────────────────────


class TestWeightedCqr:
    def test_runs_and_intervals_valid(self):
        x_tr, y_tr = _data(1000, 1)
        x_te, y_te = _data(500, 2, shift=1.0)
        _m, q_hat, pred, diag = train_test_weighted_cqr(
            x_tr, y_tr, x_te, y_te, alpha=0.10, device=CPU)
        assert q_hat > 0
        assert np.all(pred["lower"] <= pred["upper"])
        assert set(diag) >= {"ess", "ess_fraction", "collapsed", "fell_back", "mae"}

    def test_no_shift_reduces_like_split(self):
        # exchangeable test -> weights ~ uniform -> not collapsed
        x_tr, y_tr = _data(1000, 1)
        x_te, y_te = _data(500, 2)
        _m, _q, pred, diag = train_test_weighted_cqr(
            x_tr, y_tr, x_te, y_te, alpha=0.10, device=CPU)
        picp = calculate_picp(y_te, pred["lower"], pred["upper"])
        assert 0.80 <= picp <= 0.97, picp


# ── Distributional-CP / HPD arm ────────────────────────────────────────


class TestDistcp:
    def test_coverage_near_nominal(self):
        x_tr, y_tr = _data(1200, 1)
        x_te, y_te = _data(600, 2)  # exchangeable
        _m, q_hat, pred, info = train_test_distcp(
            x_tr, y_tr, x_te, y_te, alpha=0.10, epochs=60, device=CPU)
        picp = calculate_picp(y_te, pred["lower"], pred["upper"])
        assert 0.80 <= picp <= 0.99, picp
        assert q_hat > 0
        assert np.all(pred["lower"] <= pred["upper"])
        assert 0.0 <= info["multimodal_frac"] <= 1.0


# ── WQLCP-style arm ────────────────────────────────────────────────────


class TestWqlcp:
    def test_coverage_near_nominal(self):
        x_tr, y_tr = _data(1200, 1)
        x_te, y_te = _data(600, 2)
        _m, q_hat, pred, diag = train_test_wqlcp(
            x_tr, y_tr, x_te, y_te, alpha=0.10, device=CPU)
        picp = calculate_picp(y_te, pred["lower"], pred["upper"])
        assert 0.80 <= picp <= 0.97, picp
        assert q_hat > 0
        assert np.all(pred["lower"] <= pred["upper"])
        assert diag["mae"] >= 0

    def test_disabled_raises(self, monkeypatch):
        import qrnn_uq_rul.models_variants as mv
        monkeypatch.setattr(mv, "WQLCP_FEASIBLE", False, raising=True)
        x_tr, y_tr = _data(200, 1)
        with pytest.raises(NotImplementedError):
            train_test_wqlcp(x_tr, y_tr, x_tr[:50], y_tr[:50], device=CPU)

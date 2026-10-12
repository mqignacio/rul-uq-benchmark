"""
Unit tests for the shared conformal stage (qrnn_uq_rul.conformal).

Validates split-CP calibration, CQR scores/intervals, the normalized (locally
adaptive) score path, the weighted-conformal quantile (fix M2), and the ESS
weight diagnostic. Includes an empirical coverage check on synthetic
*exchangeable* data (must land near the 0.90 nominal, fix M1).
"""
from __future__ import annotations

import numpy as np
import pytest

from qrnn_uq_rul import conformal as C
from qrnn_uq_rul.config import NOMINAL_ALPHA


class TestConformalQuantile:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            C.conformal_quantile(np.array([]), 0.10)

    def test_order_statistic_of_uniform(self):
        # scores 1..100, alpha=0.10 -> level ceil(101*0.9)/100 = 91/100 -> 91st smallest
        scores = np.arange(1, 101, dtype=float)
        q = C.conformal_quantile(scores, alpha=0.10)
        assert q == pytest.approx(91.0)

    def test_weighted_reduces_to_unweighted(self):
        rng = np.random.default_rng(0)
        s = rng.normal(size=500)
        assert C.conformal_quantile(s, 0.10) == pytest.approx(
            C.weighted_conformal_quantile(s, np.ones_like(s), 0.10)
        )

    def test_default_alpha_is_nominal(self):
        rng = np.random.default_rng(1)
        s = rng.normal(size=300)
        assert C.conformal_quantile(s) == pytest.approx(C.conformal_quantile(s, NOMINAL_ALPHA))


class TestCQR:
    def test_score_sign_inside_vs_outside(self):
        y = np.array([5.0, 8.0, 2.0])
        lo = np.array([4.0, 4.0, 4.0])
        hi = np.array([6.0, 9.0, 6.0])
        s = C.cqr_max_score(y, lo, hi)
        assert s[0] <= 0 and s[1] <= 0  # both inside interval
        assert s[2] == pytest.approx(2.0)  # y=2 below lo=4 -> lo - y = 2

    def test_predict_shifts_bounds_not_median(self):
        p = C.cqr_predict(np.array([1.0]), np.array([2.0]), np.array([3.0]), 0.5)
        assert p["lower"][0] == pytest.approx(0.5)
        assert p["upper"][0] == pytest.approx(3.5)
        assert p["median"][0] == pytest.approx(2.0)

    def test_calibrate_returns_threshold(self):
        rng = np.random.default_rng(2)
        y = rng.normal(size=400)
        lo = y - 1.0
        hi = y + 1.0
        e_hat = C.calibrate_cqr_from_quantiles(y, lo, hi, alpha=0.10)
        assert np.isfinite(e_hat)


class TestNormalized:
    def test_unit_scale_equals_cqr(self):
        rng = np.random.default_rng(3)
        y = rng.normal(size=100)
        lo, hi = y - 1.0, y + 1.0
        s_norm, g = C.normalized_max_score(y, lo, hi, np.ones(100))
        assert np.allclose(s_norm, C.cqr_max_score(y, lo, hi))
        assert np.allclose(g, 1.0)

    def test_scale_floor_applied(self):
        y = np.array([0.0]); lo = np.array([-1.0]); hi = np.array([1.0])
        _, g = C.normalized_max_score(y, lo, hi, np.array([0.0]), eps=1e-3)
        assert g[0] == pytest.approx(1e-3)


class TestWeightedQuantile:
    def test_errors_on_length_mismatch(self):
        with pytest.raises(ValueError):
            C.weighted_conformal_quantile(np.arange(5.0), np.ones(4), 0.1)

    def test_errors_on_negative_weight(self):
        with pytest.raises(ValueError):
            C.weighted_conformal_quantile(np.arange(5.0), -np.ones(5), 0.1)

    def test_degenerate_weights_fallback(self):
        # all-zero weights -> warn + fall back to unweighted
        s = np.arange(1, 101, dtype=float)
        assert C.weighted_conformal_quantile(s, np.zeros(100), 0.10) == pytest.approx(
            C.conformal_quantile(s, 0.10)
        )

    def test_ess(self):
        assert C.effective_sample_size(np.ones(50)) == pytest.approx(50.0)
        w = np.zeros(50); w[0] = 3.0
        assert C.effective_sample_size(w) == pytest.approx(1.0)
        assert C.effective_sample_size(np.zeros(5)) == 0.0


class TestCoverageExchangeable:
    """Split-CP must achieve ~nominal coverage when calibration and test are exchangeable."""

    @staticmethod
    def _synth(n, seed):
        """Predictor yhat + unit-variance residual; cal and test share the law."""
        rng = np.random.default_rng(seed)
        yhat = rng.normal(0, 3, n)
        y = yhat + rng.normal(0, 1, n)
        return y, yhat

    @pytest.mark.parametrize("seed", [7, 8, 9])
    def test_split_cp_hits_nominal(self, seed):
        y, yhat = self._synth(4000, seed)
        lo, hi = yhat - 1.0, yhat + 1.0
        # calibrate on the first half, evaluate on the exchangeable second half
        e_hat = C.calibrate_cqr_from_quantiles(y[:2000], lo[:2000], hi[:2000], alpha=0.10)
        pred = C.cqr_predict(lo[2000:], yhat[2000:], hi[2000:], e_hat)
        picp = float(np.mean((y[2000:] >= pred["lower"]) & (y[2000:] <= pred["upper"])))
        assert 0.86 <= picp <= 0.94, picp

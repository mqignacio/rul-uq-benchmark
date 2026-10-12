"""
Unit tests for Phase-3 metric and statistical extensions.

Covers: Winkler score (A5), RUL asymmetric score (M7), quantile-crossing
frequency (M7), TOST equivalence (M5), mixed-effects model (M5/A6), and the
Winkler wiring into evaluate_uq_metrics. All sanity bounds are asserted
(PICP in [0,1], Winkler >= 0, crossing in [0,1]).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qrnn_uq_rul import metrics as M
from qrnn_uq_rul.config import ALPHA, CQR_ALPHA, NOMINAL_ALPHA, NOMINAL_COVERAGE


# ── M1: single nominal level ───────────────────────────────────────────


class TestNominalLevel:
    def test_coverage_and_miscoverage_are_complementary(self):
        assert NOMINAL_COVERAGE == pytest.approx(1.0 - NOMINAL_ALPHA)
        assert CQR_ALPHA == pytest.approx(NOMINAL_ALPHA)

    def test_rs_reference_equals_coverage(self):
        # ALPHA is the coverage reference for RS; must be 0.90 (was 0.95)
        assert ALPHA == pytest.approx(0.90)


# ── Winkler score (A5) ─────────────────────────────────────────────────


class TestWinkler:
    def test_covered_sample_equals_width(self):
        # y inside [L, U] -> score == width, no penalty
        assert M.winkler_score([5.0], [4.0], [6.0], alpha=0.10) == pytest.approx(2.0)

    def test_miss_below_adds_penalty(self):
        # y=3, [4,6]: width 2 + (2*0.1/0.9)*(4-3) = 2 + 0.2222...
        expected = 2.0 + (2 * 0.1 / 0.9) * 1.0
        assert M.winkler_score([3.0], [4.0], [6.0], alpha=0.10) == pytest.approx(expected)

    def test_miss_above_adds_penalty(self):
        expected = 2.0 + (2 * 0.1 / 0.9) * 2.0  # y=8 is 2 above U=6
        assert M.winkler_score([8.0], [4.0], [6.0], alpha=0.10) == pytest.approx(expected)

    def test_non_negative_and_vectorized(self):
        rng = np.random.default_rng(0)
        y = rng.normal(50, 10, 200)
        lo = y - 5
        hi = y + 5  # always covers -> score == width == 10
        val = M.winkler_score(y, lo, hi, alpha=0.10)
        assert val >= 0.0
        assert val == pytest.approx(10.0, abs=1e-9)

    def test_default_alpha_is_miscoverage(self):
        # default alpha = 1 - ALPHA = 0.10
        y, lo, hi = [10.0], [8.0], [12.0]  # covered -> width 4 regardless
        assert M.winkler_score(y, lo, hi) == pytest.approx(4.0)


class TestEvaluateUQMetricsWinklerWiring:
    """Regression: evaluate_uq_metrics must pass the MISOVERAGE rate (0.10)
    to winkler_score, not the RS coverage reference (0.90). Passing the
    coverage reference inflates the miss penalty 81x (18 vs 0.222) -- caught
    by the Phase-4 E5 smoke run 2026-09-23 (stored sweep Winkler inflated).
    """

    def test_winkler_matches_direct_miscoverage_call(self):
        rng = np.random.default_rng(0)
        y = rng.normal(50, 10, 100)
        lo, hi = y - 4.0, y + 4.0  # a meaningful share of misses
        out = M.evaluate_uq_metrics(pd.Series(y), {"lower": lo, "upper": hi})
        wink = float(out.loc[out["Metric"] == "Winkler", "Value"].iloc[0])
        assert wink == pytest.approx(M.winkler_score(y, lo, hi))

    def test_miss_penalty_uses_0p222_not_18(self):
        out = M.evaluate_uq_metrics(
            pd.Series([3.0]), {"lower": [4.0], "upper": [6.0]})
        wink = float(out.loc[out["Metric"] == "Winkler", "Value"].iloc[0])
        assert wink == pytest.approx(2.0 + (2 * 0.1 / 0.9) * 1.0)
        assert wink == pytest.approx(2.2222222, abs=1e-6)


# ── RUL asymmetric score (M7) ──────────────────────────────────────────


class TestRULAsymmetric:
    def test_perfect_prediction_zero(self):
        assert M.rul_asymmetric_score([50.0], [50.0]) == pytest.approx(0.0)

    def test_over_prediction_penalized_more_than_under(self):
        # equal-magnitude error; late-life params a_late<a_early effect on asymmetry
        y = [50.0]
        over = M.rul_asymmetric_score(y, [60.0], a_late=13.0, a_early=10.0)   # predicted long
        under = M.rul_asymmetric_score(y, [40.0], a_late=13.0, a_early=10.0)  # predicted short
        assert over > 0 and under > 0
        # both are exp(10/·)-1; a_late=13 -> smaller exponent than a_early=10
        assert over == pytest.approx(np.exp(10 / 13) - 1)
        assert under == pytest.approx(np.exp(10 / 10) - 1)
        assert under > over  # early-scale (10) steeper than late-scale (13)

    def test_non_negative(self):
        rng = np.random.default_rng(1)
        y = rng.uniform(1, 100, 300)
        p = y + rng.normal(0, 5, 300)
        assert M.rul_asymmetric_score(y, p) >= 0.0


# ── Quantile crossing frequency (M7) ───────────────────────────────────


class TestCrossing:
    def test_monotone_zero(self):
        preds = np.array([[1.0, 2.0, 3.0], [1.0, 2.5, 4.0]])
        assert M.quantile_crossing_frequency(preds, [0.1, 0.5, 0.9]) == 0.0

    def test_fraction_of_crossings(self):
        preds = np.array([[1.0, 2.0, 3.0], [1.0, 3.0, 2.0]])  # row 2 crosses
        assert M.quantile_crossing_frequency(preds, [0.1, 0.5, 0.9]) == pytest.approx(0.5)

    def test_all_crossing(self):
        preds = np.array([[3.0, 2.0, 1.0]])
        assert M.quantile_crossing_frequency(preds, [0.1, 0.5, 0.9]) == pytest.approx(1.0)

    def test_single_column_zero(self):
        preds = np.array([[5.0], [7.0]])
        assert M.quantile_crossing_frequency(preds) == 0.0

    def test_unsorted_levels_raise(self):
        preds = np.array([[1.0, 2.0, 3.0]])
        with pytest.raises(AssertionError):
            M.quantile_crossing_frequency(preds, [0.9, 0.5, 0.1])


# ── evaluate_uq_metrics wiring (Winkler included) ──────────────────────


class TestEvaluateUQWiring:
    def test_dataframe_includes_winkler(self):
        rng = np.random.default_rng(2)
        y = rng.uniform(10, 100, 100)
        y_pred = {"lower": y - 5, "upper": y + 5}
        df = M.evaluate_uq_metrics(y, y_pred)
        assert set(df["Metric"]) == {"PICP", "NMPIW", "CWC", "RS", "Winkler"}
        # all covered -> PICP==1, Winkler==width==10
        vals = dict(zip(df["Metric"], df["Value"]))
        assert vals["PICP"] == pytest.approx(1.0)
        assert vals["Winkler"] == pytest.approx(10.0)
        assert 0.0 <= vals["PICP"] <= 1.0


# ── TOST equivalence (M5) ──────────────────────────────────────────────


class TestTOST:
    def _paired(self, offset=0.0, n=40, scale=0.01, seed=0):
        rng = np.random.default_rng(seed)
        idx = list(range(n))
        a = pd.DataFrame({"PICP": rng.normal(0.9, scale, n)}, index=idx)
        b = pd.DataFrame({"PICP": rng.normal(0.9, scale, n) + offset}, index=idx)
        return a, b

    def test_identical_is_equivalent(self):
        rng = np.random.default_rng(3)
        a = pd.DataFrame({"PICP": rng.normal(0.9, 0.01, 40)})
        r = M.tost_equivalence(a, a.copy(), "PICP", margin=0.03)
        assert r["equivalent"] is True

    def test_large_offset_not_equivalent(self):
        a, b = self._paired(offset=0.20)
        r = M.tost_equivalence(a, b, "PICP", margin=0.03)
        assert r["equivalent"] is False

    def test_small_offset_within_margin_equivalent(self):
        a, b = self._paired(offset=0.002, scale=0.005, n=60)
        r = M.tost_equivalence(a, b, "PICP", margin=0.03)
        assert r["equivalent"] is True

    def test_default_margin_from_config(self):
        from qrnn_uq_rul.config import TOST_EQUIV_MARGIN
        rng = np.random.default_rng(4)
        a = pd.DataFrame({"PICP": rng.normal(0.9, 0.01, 30)})
        r = M.tost_equivalence(a, a.copy(), "PICP")
        assert r["margin"] == pytest.approx(TOST_EQUIV_MARGIN)

    def test_keys_present(self):
        rng = np.random.default_rng(5)
        a = pd.DataFrame({"PICP": rng.normal(0.9, 0.01, 30)})
        r = M.tost_equivalence(a, a.copy(), "PICP")
        for k in ("mean_diff", "t_lower", "p_lower", "t_upper", "p_upper", "equivalent", "margin"):
            assert k in r


# ── Mixed-effects model (M5/A6) ────────────────────────────────────────


@pytest.mark.filterwarnings("ignore::UserWarning")  # statsmodels near-zero-variance-convergence warnings
class TestMixedEffects:
    @staticmethod
    def _frame(method_shift=-0.05, n_units=15, n_seeds=4, seed=1):
        rng = np.random.default_rng(seed)
        rows = []
        for mth, base in [("A", 0.90), ("B", 0.90 + method_shift)]:
            for un in range(n_units):
                for sd in range(n_seeds):
                    val = base + rng.normal(0, 0.015) + 0.004 * sd
                    rows.append({"value": val, "method": mth, "unit": un, "seed": sd})
        return pd.DataFrame(rows)

    def test_recovers_method_effect(self):
        res = M.mixed_effects_model(self._frame())
        assert res["converged"] is True
        coef = float(res["fixed_effects"]["C(method)[T.B]"])
        assert coef == pytest.approx(-0.05, abs=0.02)

    def test_returns_expected_keys(self):
        res = M.mixed_effects_model(self._frame())
        for k in ("fixed_effects", "p_values", "random_effect_variances",
                  "converged", "aic", "bic", "model_summary"):
            assert k in res
        assert "vc_unit_seed" in res["random_effect_variances"]

    def test_no_method_difference_near_zero_effect(self):
        res = M.mixed_effects_model(self._frame(method_shift=0.0))
        coef = float(res["fixed_effects"]["C(method)[T.B]"])
        assert abs(coef) < 0.02

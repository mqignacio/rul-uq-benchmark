"""
Unit tests for preprocessing pipeline including group-aware splits.

Tests use synthetic fixtures under tests/fixtures/ — no real data required.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qrnn_uq_rul.preprocess import (
    balance_train_dataset,
    create_intercycle_sliding_window,
    get_dataset_name,
    group_aware_unit_split,
    load_cmapss_subset,
    load_ncmapss_dataset,
    preprocess_cmapss,
    preprocess_ds,
    standardize_features,
    transform_intracycle_features,
    transform_piecewise_rul,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
NCFIXTURE = FIXTURES_DIR / "synthetic_ncmapss.h5"
CMAPSS_FIXTURE_DIR = FIXTURES_DIR / "cmapss"


# ── N-CMAPSS Loader Tests ──────────────────────────────────────────────


class TestLoadNCmapss:
    """Test loading N-CMAPSS HDF5 files."""

    def test_load_returns_all_keys(self):
        ds = load_ncmapss_dataset(str(NCFIXTURE), verbose=False)
        expected_keys = {"A_train", "W_train", "X_s_train", "y_train",
                         "A_test", "W_test", "X_s_test", "y_test"}
        assert set(ds.keys()) == expected_keys

    def test_shapes_consistent(self):
        ds = load_ncmapss_dataset(str(NCFIXTURE), verbose=False)
        n_train = len(ds["A_train"])
        assert len(ds["W_train"]) == n_train
        assert len(ds["X_s_train"]) == n_train
        assert len(ds["y_train"]) == n_train

    def test_unit_column_integer(self):
        ds = load_ncmapss_dataset(str(NCFIXTURE), verbose=False)
        assert ds["A_train"]["unit"].dtype.kind in ("i", "u")

    def test_get_dataset_name(self):
        assert get_dataset_name("/path/to/N-CMAPSS_DS03-012.h5") == "DS03"
        assert get_dataset_name("N-CMAPSS_DS08a-009.h5") == "DS08a"


# ── Piecewise RUL Tests ────────────────────────────────────────────────


class TestPiecewiseRUL:
    """Test piecewise RUL transformation."""

    def test_output_shape_preserved(self):
        ds = load_ncmapss_dataset(str(NCFIXTURE), verbose=False)
        A_out, y_out = transform_piecewise_rul(ds["A_train"], ds["y_train"])
        # Output is per-timestep (same row count as input, grouped by unit)
        assert len(y_out) == len(ds["A_train"])
        assert "RUL" in y_out.columns

    def test_rul_non_negative(self):
        ds = load_ncmapss_dataset(str(NCFIXTURE), verbose=False)
        _, y_out = transform_piecewise_rul(ds["A_train"], ds["y_train"])
        assert (y_out["RUL"] >= 0).all()


# ── Intra-cycle Features Tests ─────────────────────────────────────────


class TestIntracycleFeatures:
    """Test intra-cycle feature aggregation."""

    def test_reduces_to_per_cycle(self):
        ds = load_ncmapss_dataset(str(NCFIXTURE), verbose=False)
        A, y = transform_piecewise_rul(ds["A_train"], ds["y_train"])
        A_new, W_new, X_s_new, y_new = transform_intracycle_features(
            A, ds["W_train"], ds["X_s_train"], y
        )
        # Output should have one row per (unit, cycle)
        assert len(A_new) == len(W_new) == len(X_s_new) == len(y_new)
        # No duplicate unit+cycle pairs
        assert not A_new.duplicated(subset=["unit", "cycle"]).any()


# ── Group-Aware Split Tests ────────────────────────────────────────────


class TestGroupAwareSplit:
    """Test group-aware unit-level splitting (M3 fix)."""

    @pytest.fixture
    def per_cycle_data(self):
        """Create synthetic per-cycle data with known units."""
        units = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        cycles_per_unit = 5
        rows = []
        for u in units:
            for c in range(cycles_per_unit):
                rows.append({"unit": u, "cycle": c})
        A = pd.DataFrame(rows)
        W = pd.DataFrame(np.random.randn(len(A), 3), columns=["w1", "w2", "w3"])
        X_s = pd.DataFrame(np.random.randn(len(A), 4), columns=["s1", "s2", "s3", "s4"])
        y = pd.DataFrame({"RUL": np.random.randint(10, 100, len(A))})
        return A, W, X_s, y

    def test_no_unit_overlap(self, per_cycle_data):
        """Critical: no engine unit appears in BOTH train and test."""
        A, W, X_s, y = per_cycle_data
        (A_tr, W_tr, Xs_tr, y_tr), (A_te, W_te, Xs_te, y_te) = group_aware_unit_split(
            A, W, X_s, y, train_fraction=0.8, seed=42
        )
        train_units = set(A_tr["unit"].unique())
        test_units = set(A_te["unit"].unique())
        assert train_units.isdisjoint(test_units), \
            f"Unit overlap detected: {train_units & test_units}"

    def test_all_units_assigned(self, per_cycle_data):
        """Every unit must appear in exactly one partition."""
        A, W, X_s, y = per_cycle_data
        all_units = set(A["unit"].unique())
        (A_tr, *_), (A_te, *_) = group_aware_unit_split(A, W, X_s, y, train_fraction=0.8, seed=42)
        assigned = set(A_tr["unit"].unique()) | set(A_te["unit"].unique())
        assert assigned == all_units

    def test_train_fraction(self, per_cycle_data):
        """Approximately correct fraction of units in training."""
        A, W, X_s, y = per_cycle_data
        n_units = A["unit"].nunique()
        (A_tr, *_), (A_te, *_) = group_aware_unit_split(A, W, X_s, y, train_fraction=0.8, seed=42)
        n_train_units = A_tr["unit"].nunique()
        # With 10 units and 0.8 fraction: expect 8 train units
        assert n_train_units == int(np.floor(n_units * 0.8))

    def test_deterministic_with_seed(self, per_cycle_data):
        """Same seed produces same split."""
        A, W, X_s, y = per_cycle_data
        r1 = group_aware_unit_split(A, W, X_s, y, seed=42)
        r2 = group_aware_unit_split(A, W, X_s, y, seed=42)
        pd.testing.assert_frame_equal(r1[0][0], r2[0][0])
        pd.testing.assert_frame_equal(r1[1][0], r2[1][0])


# ── Sliding Window Tests ───────────────────────────────────────────────


class TestSlidingWindow:
    """Test inter-cycle sliding window creation."""

    @pytest.fixture
    def simple_cycle_data(self):
        """One unit, 10 cycles, 2 features each."""
        A = pd.DataFrame({"unit": [1] * 10, "cycle": range(10)})
        W = pd.DataFrame({"w1": np.arange(10, dtype=float), "w2": np.arange(10, dtype=float) * 2})
        X_s = pd.DataFrame({"s1": np.arange(10, dtype=float) ** 2})
        y = pd.DataFrame({"RUL": np.arange(10, 0, -1, dtype=float)})
        return A, W, X_s, y

    def test_output_length(self, simple_cycle_data):
        """With window=3, stride=1, 10 cycles → 8 samples."""
        A, W, X_s, y = simple_cycle_data
        A_new, W_new, X_s_new, y_new = create_intercycle_sliding_window(
            A, W, X_s, y, window_length=3, stride=1
        )
        assert len(A_new) == 8
        assert len(W_new) == 8
        assert len(X_s_new) == 8
        assert len(y_new) == 8

    def test_feature_expansion(self, simple_cycle_data):
        """W has 2 cols × window 3 = 6 output columns."""
        A, W, X_s, y = simple_cycle_data
        _, W_new, _, _ = create_intercycle_sliding_window(A, W, X_s, y, window_length=3)
        assert W_new.shape[1] == 2 * 3  # 2 features × 3 time steps

    def test_no_cross_unit_windows(self):
        """Windows must NOT span across different units."""
        A = pd.DataFrame({"unit": [1]*5 + [2]*5, "cycle": list(range(5))*2})
        W = pd.DataFrame(np.random.randn(10, 2), columns=["w1", "w2"])
        X_s = pd.DataFrame(np.random.randn(10, 1), columns=["s1"])
        y = pd.DataFrame({"RUL": np.arange(10, 0, -1, dtype=float)})

        A_new, W_new, X_s_new, y_new = create_intercycle_sliding_window(
            A, W, X_s, y, window_length=3, stride=1
        )
        # Each unit contributes 5-3+1=3 windows → total 6
        assert len(A_new) == 6
        # First 3 from unit 1, last 3 from unit 2
        assert (A_new["unit"].values[:3] == 1).all()
        assert (A_new["unit"].values[3:] == 2).all()


# ── Balancing Tests ────────────────────────────────────────────────────


class TestBalancing:
    """Test dataset balancing (must only apply to train partition)."""

    def test_balance_produces_uniform_bins(self):
        """After balancing, each RUL bin should have ~equal size."""
        n = 200
        A = pd.DataFrame({"unit": np.repeat(range(10), 20), "cycle": tile_range(20, 10)})
        W = pd.DataFrame({"w1": np.random.randn(n)})
        X_s = pd.DataFrame({"s1": np.random.randn(n)})
        y = pd.DataFrame({"RUL": np.linspace(0, 200, n)})

        A_b, W_b, Xs_b, y_b = balance_train_dataset(A, W, X_s, y, seed=42)
        # All bins should have similar sizes
        bins = pd.cut(y_b["RUL"], bins=5, labels=False)
        counts = bins.value_counts()
        # Max/min ratio should be close to 1
        assert counts.max() / counts.min() <= 1.5

    def test_balance_does_not_modify_input(self):
        """Original DataFrames should not be mutated."""
        n = 100
        A = pd.DataFrame({"unit": np.repeat(range(10), 10), "cycle": list(range(10))*10})
        W = pd.DataFrame({"w1": np.random.randn(n)})
        X_s = pd.DataFrame({"s1": np.random.randn(n)})
        y = pd.DataFrame({"RUL": np.linspace(0, 100, n)})

        orig_len = len(A)
        balance_train_dataset(A, W, X_s, y, seed=42)
        assert len(A) == orig_len


def tile_range(val, times):
    """Helper: [0,1,...,val-1] repeated `times` times."""
    return list(range(val)) * times


# ── Standardization Tests ──────────────────────────────────────────────


class TestStandardization:
    """Test MinMax feature scaling."""

    def test_train_scaled_to_0_1(self):
        W_tr = pd.DataFrame({"w1": [0, 5, 10]})
        Xs_tr = pd.DataFrame({"s1": [-1, 0, 1]})
        W_te = pd.DataFrame({"w1": [2, 8]})
        Xs_te = pd.DataFrame({"s1": [0.5, -0.5]})

        W_tr_s, Xs_tr_s, W_te_s, Xs_te_s = standardize_features(W_tr, Xs_tr, W_te, Xs_te)
        assert W_tr_s["w1"].min() == pytest.approx(0.0)
        assert W_tr_s["w1"].max() == pytest.approx(1.0)

    def test_scaler_fit_on_train_only(self):
        """Test values can exceed [0,1] if outside train range."""
        W_tr = pd.DataFrame({"w1": [0, 1, 2]})
        Xs_tr = pd.DataFrame({"s1": [0, 1, 2]})
        W_te = pd.DataFrame({"w1": [100]})  # Far outside train range
        Xs_te = pd.DataFrame({"s1": [0]})

        W_tr_s, Xs_tr_s, W_te_s, Xs_te_s = standardize_features(W_tr, Xs_tr, W_te, Xs_te)
        # Test value 100 should map well above 1.0
        assert W_te_s["w1"].iloc[0] > 1.0


# ── Full Pipeline Integration Test (Legacy Mode) ───────────────────────


class TestPreprocessDSLegacy:
    """Integration test: full pipeline in legacy mode on synthetic fixture."""

    def test_legacy_mode_runs(self):
        result = preprocess_ds(str(NCFIXTURE), split_mode="legacy", seed=42)
        A_train, W_train, X_s_train, y_train, A_test, W_test, X_s_test, y_test = result
        assert len(A_train) > 0
        assert len(A_test) > 0
        assert "unit" in A_train.columns
        assert "RUL" in y_train.columns

    def test_group_aware_mode_runs(self):
        result = preprocess_ds(str(NCFIXTURE), split_mode="group_aware", seed=42)
        A_train, W_train, X_s_train, y_train, A_test, W_test, X_s_test, y_test = result
        assert len(A_train) > 0
        assert len(A_test) > 0
        # Critical: no unit overlap
        train_units = set(A_train["unit"].unique())
        test_units = set(A_test["unit"].unique())
        assert train_units.isdisjoint(test_units)


# ── C-MAPSS Loader Tests ───────────────────────────────────────────────


class TestCmapssLoader:
    """Test C-MAPSS text file loading using synthetic fixtures."""

    def test_load_train(self):
        A, X, y, _ = load_cmapss_subset(CMAPSS_FIXTURE_DIR, subset="FD001", max_rul=125)
        assert len(A) > 0
        assert "unit" in A.columns
        assert "cycle" in A.columns
        assert X.shape[0] == len(A)
        assert (y["RUL"] >= 0).all()
        assert (y["RUL"] <= 125).all()

    def test_load_with_test(self):
        A_tr, X_tr, y_tr, test_data = load_cmapss_subset(
            CMAPSS_FIXTURE_DIR, subset="FD001", max_rul=125
        )
        A_te, X_te, y_te = test_data
        assert A_te is not None
        assert len(A_te) > 0
        assert X_te.shape[0] == len(A_te)

    def test_preprocess_cmapss_group_aware(self):
        """Full C-MAPSS pipeline with group-aware split."""
        result = preprocess_cmapss(
            CMAPSS_FIXTURE_DIR, subset="FD001",
            split_mode="group_aware", seed=42, window_length=3
        )
        A_train, W_train, X_s_train, y_train, A_test, W_test, X_s_test, y_test = result
        assert len(A_train) > 0
        # No unit overlap
        train_units = set(A_train["unit"].unique())
        test_units = set(A_test["unit"].unique())
        assert train_units.isdisjoint(test_units)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_cmapss_subset("/nonexistent/path", subset="FD999")


# ── Real-data integration (skipped if data absent) ─────────────────────


class TestRealData:
    """Integration tests against real data — skipped if unavailable."""

    REAL_NC = Path(__file__).resolve().parent.parent.parent / \
        "resources/old-files/src/data/raw/N-CMAPSS_DS01-005.h5"
    REAL_CMAPSS = Path(__file__).resolve().parent.parent.parent / \
        "resources/cmapss/train_FD001.txt"

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parent.parent.parent /
             "resources/old-files/src/data/raw/N-CMAPSS_DS01-005.h5").exists(),
        reason="Real N-CMAPSS data not available",
    )
    def test_real_ncmapss_load(self):
        path = Path(__file__).resolve().parent.parent.parent / \
            "resources/old-files/src/data/raw/N-CMAPSS_DS01-005.h5"
        ds = load_ncmapss_dataset(str(path), verbose=False)
        assert len(ds["A_train"]) > 1000

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parent.parent.parent /
             "resources/cmapss/train_FD001.txt").exists(),
        reason="Real C-MAPSS data not downloaded",
    )
    def test_real_cmapss_load(self):
        path = Path(__file__).resolve().parent.parent.parent / "resources/cmapss"
        A, X, y, test_data = load_cmapss_subset(str(path), subset="FD001")
        assert len(A) > 1000
        assert test_data[0] is not None

"""
Generate synthetic test fixtures for unit tests.

Creates:
- tests/fixtures/synthetic_ncmapss.h5: Minimal N-CMAPSS HDF5 with correct schema
- tests/fixtures/cmapss/train_FD001.txt: Tiny C-MAPSS training file (synthetic)
- tests/fixtures/cmapss/test_FD001.txt: Tiny C-MAPSS test file
- tests/fixtures/cmapss/RUL_FD001.txt: RUL ground truth for test units
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def generate_synthetic_ncmapss(n_units_train: int = 6, n_units_test: int = 3,
                                n_cycles_per_unit: int = 20, n_timesteps: int = 50):
    """
    Generate a minimal N-CMAPSS-like HDF5 fixture.

    Schema matches real N-CMAPSS:
    - A: [unit, cycle, Fc, hs]
    - W: [alt, T24, ...] (operating conditions)
    - X_s: sensor channels
    - Y: RUL
    """
    import h5py

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    outpath = FIXTURES_DIR / "synthetic_ncmapss.h5"

    n_W_features = 5  # alt, ThreRat, Nc_f, Nc_c, Bypass
    n_Xs_features = 8  # T24, T30, T40, T50, Ps30, Phi, N18, N50

    def make_partition(unit_ids):
        rows_A, rows_W, rows_Xs, rows_Y = [], [], [], []
        for unit in unit_ids:
            degradation_start = np.random.randint(5, n_cycles_per_unit - 2)
            for cycle in range(n_cycles_per_unit):
                hs = 1.0 if cycle < degradation_start else max(0.0, 1.0 - (cycle - degradation_start) * 0.05)
                rul = max(0, (n_cycles_per_unit - cycle) * 10)
                for t in range(n_timesteps):
                    rows_A.append([unit, cycle, 1, hs])
                    rows_W.append(np.random.randn(n_W_features) * 0.1 + np.arange(n_W_features))
                    rows_Xs.append(np.random.randn(n_Xs_features) * 0.5 + hs * np.arange(n_Xs_features))
                    rows_Y.append([rul])
        return (
            np.array(rows_A, dtype=np.float64),
            np.array(rows_W, dtype=np.float64),
            np.array(rows_Xs, dtype=np.float64),
            np.array(rows_Y, dtype=np.float64),
        )

    np.random.seed(42)
    train_units = list(range(1, n_units_train + 1))
    test_units = list(range(101, 101 + n_units_test))

    A_dev, W_dev, Xs_dev, Y_dev = make_partition(train_units)
    A_test, W_test, Xs_test, Y_test = make_partition(test_units)

    A_var = ["unit", "cycle", "Fc", "hs"]
    W_var = ["alt", "ThreRat", "Nc_f", "Nc_c", "Bypass"][:n_W_features]
    Xs_var = ["T24", "T30", "T40", "T50", "Ps30", "Phi", "N18", "N50"][:n_Xs_features]

    with h5py.File(outpath, "w") as f:
        f.create_dataset("A_var", data=np.array(A_var, dtype="S20"))
        f.create_dataset("W_var", data=np.array(W_var, dtype="S20"))
        f.create_dataset("X_s_var", data=np.array(Xs_var, dtype="S20"))
        f.create_dataset("A_dev", data=A_dev)
        f.create_dataset("W_dev", data=W_dev)
        f.create_dataset("X_s_dev", data=Xs_dev)
        f.create_dataset("Y_dev", data=Y_dev)
        f.create_dataset("A_test", data=A_test)
        f.create_dataset("W_test", data=W_test)
        f.create_dataset("X_s_test", data=Xs_test)
        f.create_dataset("Y_test", data=Y_test)

    print(f"Created {outpath} (train: {A_dev.shape[0]} rows, test: {A_test.shape[0]} rows)")


def generate_synthetic_cmapss(n_train_units: int = 5, n_test_units: int = 3,
                               max_cycles_train: int = 30, max_cycles_test: int = 20):
    """
    Generate tiny C-MAPSS-format text fixtures.

    C-MAPSS format: space-separated columns per time step:
    unit_id, time_cycle, op1, op2, op3, s1..s26 (some constant)
    Total: 1 (unit) + 1 (time) + 26 (features) = 28 columns typically.
    We use 28 cols (3 op settings + 23 sensors + 2 id cols).
    """
    np.random.seed(123)
    cmapss_dir = FIXTURES_DIR / "cmapss"
    cmapss_dir.mkdir(parents=True, exist_ok=True)

    n_features = 26  # total feature columns after unit+cycle

    # Training data
    train_rows = []
    for unit in range(1, n_train_units + 1):
        n_cycles = np.random.randint(max_cycles_train - 5, max_cycles_train + 1)
        for cycle in range(1, n_cycles + 1):
            # Constant op columns (op1, op2, op31 in real data)
            row = [unit, cycle, 0.003, 0.04, 575.0]
            # Varying features (degradation trend + noise)
            degradation = cycle / n_cycles
            features = np.random.randn(n_features - 3) * 0.5 + degradation * np.arange(n_features - 3) * 0.1
            row.extend(features.tolist())
            train_rows.append(row)

    train_df = pd.DataFrame(train_rows)
    train_df.to_csv(cmapss_dir / "train_FD001.txt", sep=" ", header=False, index=False)

    # Test data (truncated sequences, no failure at end)
    test_rows = []
    true_ruls = []
    for unit in range(1, n_test_units + 1):
        n_cycles = np.random.randint(max_cycles_test - 5, max_cycles_test)
        true_rul = np.random.randint(20, 100)
        true_ruls.append(true_rul)
        for cycle in range(1, n_cycles + 1):
            row = [unit, cycle, 0.003, 0.04, 575.0]
            degradation = cycle / (n_cycles + true_rul)
            features = np.random.randn(n_features - 3) * 0.5 + degradation * np.arange(n_features - 3) * 0.1
            row.extend(features.tolist())
            test_rows.append(row)

    test_df = pd.DataFrame(test_rows)
    test_df.to_csv(cmapss_dir / "test_FD001.txt", sep=" ", header=False, index=False)

    # RUL file (one value per test unit)
    pd.DataFrame(true_ruls).to_csv(cmapss_dir / "RUL_FD001.txt", sep=" ", header=False, index=False)

    print(f"Created C-MAPSS fixtures in {cmapss_dir}")
    print(f"  train_FD001.txt: {len(train_rows)} rows, {n_train_units} units")
    print(f"  test_FD001.txt: {len(test_rows)} rows, {n_test_units} units")
    print(f"  RUL_FD001.txt: {true_ruls}")


if __name__ == "__main__":
    generate_synthetic_ncmapss()
    generate_synthetic_cmapss()

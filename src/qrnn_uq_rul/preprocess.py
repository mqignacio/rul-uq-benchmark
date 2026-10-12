"""
Data preprocessing pipeline for N-CMAPSS and C-MAPSS aero-engine RUL datasets.

Pipeline steps (N-CMAPSS):
1. Load raw HDF5 data
2. Piecewise RUL transformation
3. Intra-cycle feature engineering
4. Group-aware split by engine unit (BEFORE sliding windows) — or legacy random split
5. Inter-cycle sliding window aggregation
6. Training dataset balancing (train partition only, AFTER split)
7. Feature standardization

Split modes:
- "legacy": original behavior (sliding windows on full dataset, then random split)
- "group_aware": split by engine unit BEFORE sliding windows (closes reviewer M3 leakage)

All steps support lazy-compute caching to avoid re-processing.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

import h5py
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler

from .config import (
    CMAPSS_RAW_DIR,
    FILENAMES,
    PREPROCESSED_DIR,
    RAW_DATA_DIR,
    STRIDE,
    WINDOW_LENGTH,
)

logger = logging.getLogger(__name__)

SplitMode = Literal["legacy", "group_aware"]


# ── Step 0: Load raw data ──────────────────────────────────────────────


def load_ncmapss_dataset(dataset_path: str, verbose: bool = True) -> dict:
    """
    Load an N-CMAPSS dataset from HDF5 format.

    Args:
        dataset_path: Path to the .h5 file.
        verbose: Print dataset info if True.

    Returns:
        Dictionary with keys: A_train, W_train, X_s_train, y_train,
        A_test, W_test, X_s_test, y_test.
    """
    with h5py.File(dataset_path, "r") as f:
        A_var = list(np.array(f["A_var"][:], dtype="U20"))
        W_var = list(np.array(f["W_var"][:], dtype="U20"))
        X_s_var = list(np.array(f["X_s_var"][:], dtype="U20"))

        A_train = pd.DataFrame(np.array(f["A_dev"][:]), columns=A_var)
        W_train = pd.DataFrame(np.array(f["W_dev"][:]), columns=W_var)
        X_s_train = pd.DataFrame(np.array(f["X_s_dev"][:]), columns=X_s_var)
        y_train = pd.DataFrame(np.array(f["Y_dev"][:]), columns=["RUL"])

        A_train[["unit", "cycle", "Fc"]] = A_train[["unit", "cycle", "Fc"]].astype(int)

        A_test = pd.DataFrame(np.array(f["A_test"][:]), columns=A_var)
        W_test = pd.DataFrame(np.array(f["W_test"][:]), columns=W_var)
        X_s_test = pd.DataFrame(np.array(f["X_s_test"][:]), columns=X_s_var)
        y_test = pd.DataFrame(np.array(f["Y_test"][:]), columns=["RUL"])

        A_test[["unit", "cycle", "Fc"]] = A_test[["unit", "cycle", "Fc"]].astype(int)

    ds = {
        "A_train": A_train,
        "W_train": W_train,
        "X_s_train": X_s_train,
        "y_train": y_train,
        "A_test": A_test,
        "W_test": W_test,
        "X_s_test": X_s_test,
        "y_test": y_test,
    }

    if verbose:
        ds_name = get_dataset_name(dataset_path)
        logger.info("Dataset: %s", ds_name)
        for key, df in ds.items():
            logger.info("  %s: %s", key, df.shape)

    return ds


def get_dataset_name(dataset_path: str) -> str:
    """Extract dataset name (e.g., 'DS03') from file path."""
    fname = Path(dataset_path).stem  # e.g., N-CMAPSS_DS03-012
    for part in fname.split("_"):
        if part.startswith("DS"):
            # Strip trailing -NNN suffix (e.g. "DS03-012" → "DS03", "DS08a-009" → "DS08a")
            return part.split("-")[0]
    return fname


# ── Step 1: Piecewise RUL ──────────────────────────────────────────────


def transform_piecewise_rul(
    A: pd.DataFrame, y: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert RUL to piecewise degradation using health state (hs).

    Before degradation starts, RUL is clamped to the value at the
    degradation-start cycle, creating a piecewise-linear target.

    Args:
        A: Auxiliary data containing engine unit, cycle, and health state (hs).
        y: Target data containing Remaining Useful Life (RUL).

    Returns:
        Tuple of (transformed_A, transformed_y).
    """
    merged = A.merge(y, left_index=True, right_index=True)
    transformed = []

    for unit, group in merged.groupby("unit"):
        degradation_start_cycle = group[group["hs"] < 1]["cycle"].min()
        early_rul = group[group["cycle"] == degradation_start_cycle]["RUL"].values[0]

        group = group.copy()
        group["RUL"] = group.apply(
            lambda row: early_rul if row["cycle"] < degradation_start_cycle else row["RUL"],
            axis=1,
        )
        group = group.drop(columns=["Fc", "hs"])
        transformed.append(group)

    transformed_df = pd.concat(transformed)
    return transformed_df.drop(columns=["RUL"]), transformed_df[["RUL"]]


# ── Step 2: Intra-cycle features ───────────────────────────────────────


def transform_intracycle_features(
    A_df: pd.DataFrame,
    W_df: pd.DataFrame,
    X_s_df: pd.DataFrame,
    y_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Transform intra-cycle sensor readings to per-cycle aggregated features.

    For each flight cycle:
    - Smooth altitude with Savitzky-Golay filter
    - Detect climb (alt_diff > 0) and cruise (|alt_diff| <= 0.1) phases
    - Aggregate W features: climb max, cruise mean
    - Aggregate X_s features: climb max, cruise mean
    - Add flight_hour (timesteps / 3600)

    Args:
        A_df: Auxiliary data containing unit and cycle information.
        W_df: Operating conditions (scenario descriptors).
        X_s_df: Sensor measurements.
        y_df: Remaining Useful Life (RUL) data.

    Returns:
        Tuple of (A_new, W_new, X_s_new, y_new).
    """
    A = A_df.copy()
    W = W_df.copy()
    X_s = X_s_df.copy()
    y = y_df.copy()

    y = pd.merge(A[["unit", "cycle"]], y, left_index=True, right_index=True)

    aggregated_data = []

    for unit in A["unit"].unique():
        unit_mask = A["unit"] == unit

        for cycle in A[unit_mask]["cycle"].unique():
            cycle_mask = (A["unit"] == unit) & (A["cycle"] == cycle)

            cycle_W = W[cycle_mask]
            cycle_X_s = X_s[cycle_mask]

            if len(cycle_W) >= 5:
                alt_smooth = savgol_filter(cycle_W["alt"].values, min(5, len(cycle_W) | 1), 2)
            else:
                alt_smooth = cycle_W["alt"].values

            alt_diff = np.zeros(len(alt_smooth))
            alt_diff[1:] = np.diff(alt_smooth)

            climb_mask = alt_diff > 0
            cruise_mask = np.abs(alt_diff) <= 0.1

            features: dict = {}

            if climb_mask.any():
                for col in W.columns:
                    features[f"{col}_climb_max"] = cycle_W[col][climb_mask].max()

            if cruise_mask.any():
                for col in W.columns:
                    features[f"{col}_cruise_mean"] = cycle_W[col][cruise_mask].mean()

            if climb_mask.any():
                for col in X_s.columns:
                    features[f"{col}_climb_max"] = cycle_X_s[col][climb_mask].max()

            if cruise_mask.any():
                for col in X_s.columns:
                    features[f"{col}_cruise_mean"] = cycle_X_s[col][cruise_mask].mean()

            features["flight_hour"] = len(cycle_W) / 3600
            features["unit"] = unit
            features["cycle"] = cycle

            aggregated_data.append(features)

    result_df = pd.DataFrame(aggregated_data).fillna(0)

    W_cols = [
        col for col in result_df.columns
        if any(w in col for w in W.columns) or col == "flight_hour"
    ]
    W_cols = [col for col in W_cols if "T24" not in col]
    X_s_cols = [
        col for col in result_df.columns
        if any(x in col for x in X_s.columns)
    ]

    A_new = result_df[["unit", "cycle"]]
    W_new = result_df[W_cols]
    X_s_new = result_df[X_s_cols]
    y_new = y.groupby(["unit", "cycle"])["RUL"].first().reset_index()
    y_new = y_new.drop(columns=["unit", "cycle"])

    return A_new, W_new, X_s_new, y_new


# ── Step 3: Group-aware split ──────────────────────────────────────────


def group_aware_unit_split(
    A_per_cycle: pd.DataFrame,
    W_per_cycle: pd.DataFrame,
    X_s_per_cycle: pd.DataFrame,
    y_per_cycle: pd.DataFrame,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> tuple[tuple, tuple]:
    """
    Split per-cycle data by engine unit into train and test partitions.

    This ensures NO engine unit appears in both train and test, closing
    the sliding-window leakage identified by reviewer M3.

    Args:
        A_per_cycle: Auxiliary data with 'unit' column (per-cycle granularity).
        W_per_cycle: Operating conditions (per-cycle).
        X_s_per_cycle: Sensor features (per-cycle).
        y_per_cycle: RUL targets (per-cycle).
        train_fraction: Fraction of units assigned to training.
        seed: Random seed for unit assignment.

    Returns:
        Tuple of ((A_tr, W_tr, Xs_tr, y_tr), (A_te, W_te, Xs_te, y_te)).
    """
    rng = np.random.default_rng(seed)
    units = A_per_cycle["unit"].unique()
    rng.shuffle(units)
    n_train = max(1, int(np.floor(len(units) * train_fraction)))
    train_units = set(units[:n_train])

    train_mask = A_per_cycle["unit"].isin(train_units)
    test_mask = ~train_mask

    return (
        (A_per_cycle[train_mask], W_per_cycle[train_mask],
         X_s_per_cycle[train_mask], y_per_cycle[train_mask]),
        (A_per_cycle[test_mask], W_per_cycle[test_mask],
         X_s_per_cycle[test_mask], y_per_cycle[test_mask]),
    )


# ── Step 4: Inter-cycle sliding window ─────────────────────────────────


def create_intercycle_sliding_window(
    A_df: pd.DataFrame,
    W_df: pd.DataFrame,
    X_s_df: pd.DataFrame,
    y_df: pd.DataFrame,
    window_length: int = WINDOW_LENGTH,
    stride: int = STRIDE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create an inter-cycle sliding window for the given datasets.

    Each sample is a flattened concatenation of `window_length` consecutive
    cycles. The RUL target is from the last cycle in the window.

    Args:
        A_df: Auxiliary data containing unit and cycle information.
        W_df: Operating conditions.
        X_s_df: Sensor measurements.
        y_df: Remaining Useful Life (RUL) data.
        window_length: Length of the sliding window (default=3).
        stride: Stride for the sliding window (default=1).

    Returns:
        Tuple of (A_new, W_new, X_s_new, y_new).
    """
    A_new, W_new, X_s_new, y_new = [], [], [], []

    for unit in A_df["unit"].unique():
        unit_mask = A_df["unit"] == unit

        A_unit = A_df[unit_mask].reset_index(drop=True)
        W_unit = W_df[unit_mask].reset_index(drop=True)
        X_s_unit = X_s_df[unit_mask].reset_index(drop=True)
        y_unit = y_df[unit_mask].reset_index(drop=True)

        for start in range(0, len(A_unit) - window_length + 1, stride):
            end = start + window_length

            A_new.append(A_unit.iloc[end - 1].values)
            W_new.append(W_unit.iloc[start:end].values.flatten())
            X_s_new.append(X_s_unit.iloc[start:end].values.flatten())
            y_new.append(y_unit.iloc[end - 1].values)

    A_new = pd.DataFrame(A_new, columns=A_df.columns)
    W_new = pd.DataFrame(
        W_new,
        columns=[f"{col}_t{i}" for i in range(window_length) for col in W_df.columns],
    )
    X_s_new = pd.DataFrame(
        X_s_new,
        columns=[f"{col}_t{i}" for i in range(window_length) for col in X_s_df.columns],
    )
    y_new = pd.DataFrame(y_new, columns=y_df.columns)

    return A_new, W_new, X_s_new, y_new


# ── Step 5: Balancing ──────────────────────────────────────────────────


def balance_train_dataset(
    A_df: pd.DataFrame,
    W_df: pd.DataFrame,
    X_s_df: pd.DataFrame,
    y_df: pd.DataFrame,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Balance the train dataset by oversampling/undersampling RUL bins.

    Splits RUL into 5 equal-width bins, then samples each bin to the
    median bin size (oversample with replacement if below, undersample
    without replacement if above).

    IMPORTANT: This function operates ONLY on the training partition.
    The calibration/test sets must never be balanced (preserves exchangeability).

    Args:
        A_df: Auxiliary data (train only).
        W_df: Operating conditions (train only).
        X_s_df: Sensor measurements (train only).
        y_df: RUL targets (train only).
        seed: Random seed for sampling reproducibility.

    Returns:
        Tuple of (A_new, W_new, X_s_new, y_new) — balanced training data.
    """
    train_data = pd.concat([A_df.reset_index(drop=True), W_df.reset_index(drop=True),
                            X_s_df.reset_index(drop=True), y_df.reset_index(drop=True)], axis=1)

    bins = np.linspace(0, y_df["RUL"].max(), 5)
    train_data["RUL_bin"] = pd.cut(
        train_data["RUL"], bins=bins, labels=False, include_lowest=True
    )

    target_size = int(train_data.groupby("RUL_bin").size().median())

    balanced_data = []
    for bin_label in sorted(train_data["RUL_bin"].unique()):
        bin_data = train_data[train_data["RUL_bin"] == bin_label]

        if bin_data.empty:
            continue

        if len(bin_data) > target_size:
            sampled_data = bin_data.sample(n=target_size, replace=False, random_state=seed)
        else:
            sampled_data = bin_data.sample(n=target_size, replace=True, random_state=seed)

        balanced_data.append(sampled_data)

    balanced_data = pd.concat(balanced_data, axis=0).reset_index(drop=True)

    A_new = balanced_data[A_df.columns]
    W_new = balanced_data[W_df.columns]
    X_s_new = balanced_data[X_s_df.columns]
    y_new = balanced_data[y_df.columns]

    return A_new, W_new, X_s_new, y_new


# ── Step 6: Standardization ────────────────────────────────────────────


def standardize_features(
    W_train: pd.DataFrame,
    X_s_train: pd.DataFrame,
    W_test: pd.DataFrame,
    X_s_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Standardize W and X_s features using Min-Max scaling (fit on train only).

    Args:
        W_train: Operating conditions for training set.
        X_s_train: Sensor measurements for training set.
        W_test: Operating conditions for test set.
        X_s_test: Sensor measurements for test set.

    Returns:
        Tuple of (W_train_scaled, X_s_train_scaled, W_test_scaled, X_s_test_scaled).
    """
    features_train = pd.concat([W_train, X_s_train], axis=1)
    features_test = pd.concat([W_test, X_s_test], axis=1)

    scaler = MinMaxScaler()
    features_train_scaled = scaler.fit_transform(features_train)
    features_test_scaled = scaler.transform(features_test)

    W_train_scaled = pd.DataFrame(
        features_train_scaled[:, : W_train.shape[1]], columns=W_train.columns
    )
    X_s_train_scaled = pd.DataFrame(
        features_train_scaled[:, W_train.shape[1]:], columns=X_s_train.columns
    )
    W_test_scaled = pd.DataFrame(
        features_test_scaled[:, : W_test.shape[1]], columns=W_test.columns
    )
    X_s_test_scaled = pd.DataFrame(
        features_test_scaled[:, W_test.shape[1]:], columns=X_s_test.columns
    )

    return W_train_scaled, X_s_train_scaled, W_test_scaled, X_s_test_scaled


# ── Full pipeline (single DS subset) ──────────────────────────────────


def preprocess_ds(
    dataset_path: str,
    split_mode: SplitMode = "legacy",
    seed: int = 42,
) -> tuple:
    """
    Run the full preprocessing pipeline on a single N-CMAPSS subset.

    In "legacy" mode:
      1. Piecewise RUL → 2. Intra-cycle → 3. Sliding window → 4. Balance → 5. Standardize
      (Train/test split is inherent in N-CMAPSS dev/test structure)

    In "group_aware" mode:
      1. Piecewise RUL → 2. Intra-cycle → 3. Unit split → 4. Sliding window
      → 5. Balance (train only) → 6. Standardize

    Args:
        dataset_path: Path to the .h5 file.
        split_mode: "legacy" or "group_aware".
        seed: Random seed for group-aware split.

    Returns:
        Tuple of (A_train, W_train, X_s_train, y_train,
                   A_test, W_test, X_s_test, y_test).
    """
    ds = load_ncmapss_dataset(dataset_path, verbose=False)

    # Step 1: Piecewise RUL
    A_train, y_train = transform_piecewise_rul(ds["A_train"], ds["y_train"])
    A_test, y_test = transform_piecewise_rul(ds["A_test"], ds["y_test"])

    # Step 2: Intra-cycle features
    A_train, W_train, X_s_train, y_train = transform_intracycle_features(
        A_train, ds["W_train"], ds["X_s_train"], y_train
    )
    A_test, W_test, X_s_test, y_test = transform_intracycle_features(
        A_test, ds["W_test"], ds["X_s_test"], y_test
    )

    if split_mode == "legacy":
        # Legacy: apply sliding window to each partition independently
        # (train and test are already separated by N-CMAPSS structure)
        A_train, W_train, X_s_train, y_train = create_intercycle_sliding_window(
            A_train, W_train, X_s_train, y_train
        )
        A_test, W_test, X_s_test, y_test = create_intercycle_sliding_window(
            A_test, W_test, X_s_test, y_test
        )
        # Balance train
        A_train, W_train, X_s_train, y_train = balance_train_dataset(
            A_train, W_train, X_s_train, y_train, seed=seed
        )
    else:
        # Group-aware: combine all cycles, split by unit, THEN sliding window
        # Merge train+test per-cycle data for unified unit-level split
        all_A = pd.concat([A_train, A_test], ignore_index=True)
        all_W = pd.concat([W_train, W_test], ignore_index=True)
        all_X_s = pd.concat([X_s_train, X_s_test], ignore_index=True)
        all_y = pd.concat([y_train, y_test], ignore_index=True)

        (A_tr, W_tr, Xs_tr, y_tr), (A_te, W_te, Xs_te, y_te) = group_aware_unit_split(
            all_A, all_W, all_X_s, all_y, train_fraction=0.8, seed=seed
        )

        # Apply sliding window AFTER split (no cross-partition window overlap)
        A_tr, W_tr, Xs_tr, y_tr = create_intercycle_sliding_window(A_tr, W_tr, Xs_tr, y_tr)
        A_te, W_te, Xs_te, y_te = create_intercycle_sliding_window(A_te, W_te, Xs_te, y_te)

        # Balance train only
        A_tr, W_tr, Xs_tr, y_tr = balance_train_dataset(A_tr, W_tr, Xs_tr, y_tr, seed=seed)

        A_train, W_train, X_s_train, y_train = A_tr, W_tr, Xs_tr, y_tr
        A_test, W_test, X_s_test, y_test = A_te, W_te, Xs_te, y_te

    # Standardize (fit on train, transform test)
    W_train, X_s_train, W_test, X_s_test = standardize_features(
        W_train, X_s_train, W_test, X_s_test
    )

    return A_train, W_train, X_s_train, y_train, A_test, W_test, X_s_test, y_test


# ── Multi-dataset assembly ─────────────────────────────────────────────


def preprocess_save_ncmapss(
    ncmapss_path: Optional[str] = None,
    filenames: Optional[dict] = None,
    split_mode: SplitMode = "legacy",
    seed: int = 42,
) -> None:
    """
    Preprocess all N-CMAPSS datasets and save concatenated results to CSV.

    Uses lazy-compute: skips if output files already exist.

    Args:
        ncmapss_path: Directory containing .h5 files.
        filenames: Dict of {ds_name: filename} mappings.
        split_mode: "legacy" or "group_aware".
        seed: Random seed.
    """
    if ncmapss_path is None:
        ncmapss_path = str(RAW_DATA_DIR)
    if filenames is None:
        filenames = FILENAMES

    # Check if already processed
    output_files = [
        PREPROCESSED_DIR / f"preprocessed_{key}.csv"
        for key in ["A_train", "W_train", "X_s_train", "y_train",
                     "A_test", "W_test", "X_s_test", "y_test"]
    ]
    if all(f.exists() for f in output_files):
        logger.info("Preprocessed data already exists. Skipping.")
        return

    datasets = {
        key: pd.DataFrame()
        for key in ["A_train", "W_train", "X_s_train", "y_train",
                     "A_test", "W_test", "X_s_test", "y_test"]
    }

    for ds_name, filename in filenames.items():
        logger.info("Preprocessing N-CMAPSS dataset: %s (mode=%s)", ds_name, split_mode)
        result = preprocess_ds(
            str(Path(ncmapss_path) / filename), split_mode=split_mode, seed=seed
        )
        A_train, W_train, X_s_train, y_train, A_test, W_test, X_s_test, y_test = result

        # Prefix unit numbers with dataset name
        A_train["unit"] = A_train["unit"].apply(lambda x: f"{ds_name}_U{x}")
        A_test["unit"] = A_test["unit"].apply(lambda x: f"{ds_name}_U{x}")

        for key, val in zip(
            ["A_train", "W_train", "X_s_train", "y_train",
             "A_test", "W_test", "X_s_test", "y_test"],
            [A_train, W_train, X_s_train, y_train, A_test, W_test, X_s_test, y_test],
        ):
            datasets[key] = pd.concat([datasets[key], val], ignore_index=True)

    # Save
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for key, df in datasets.items():
        output_path = PREPROCESSED_DIR / f"preprocessed_{key}.csv"
        df.to_csv(output_path, index=False)
        logger.info("Saved %s to %s", key, output_path)


def load_preprocessed_from_csv() -> tuple:
    """
    Load preprocessed datasets from CSV files.

    Returns:
        Tuple of (A_train, W_train, X_s_train, y_train,
                   A_test, W_test, X_s_test, y_test).
    """
    file_paths = {
        key: PREPROCESSED_DIR / f"preprocessed_{key}.csv"
        for key in ["A_train", "W_train", "X_s_train", "y_train",
                     "A_test", "W_test", "X_s_test", "y_test"]
    }

    return tuple(pd.read_csv(file_paths[k]) for k in
                 ["A_train", "W_train", "X_s_train", "y_train",
                  "A_test", "W_test", "X_s_test", "y_test"])


# ── C-MAPSS Loader ─────────────────────────────────────────────────────


def load_cmapss_subset(
    data_dir: str | Path,
    subset: str = "FD001",
    max_rul: int = 125,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load a C-MAPSS subset (FD001-FD004) from raw text files.

    C-MAPSS file format (space-separated, no header):
    - train_FDxxx.txt: unit_id, time_cycle, 26 operational/sensor settings, 14 sensor outputs
    - test_FDxxx.txt: same format (shorter sequences, no failure label)
    - RUL_FDxxx.txt: true remaining useful life per test unit (one line per unit)

    Args:
        data_dir: Directory containing the txt files.
        subset: One of FD001, FD002, FD003, FD004.
        max_rul: Maximum RUL value (piecewise linear cap, default 125).

    Returns:
        Tuple of (A_train, X_train, y_train, A_test_combined) where:
        - A_train: DataFrame with 'unit', 'cycle' columns
        - X_train: Feature DataFrame (operational + sensor)
        - y_train: DataFrame with 'RUL' column
        - A_test_combined: (A_test, X_test, y_test) packed together
    """
    data_dir = Path(data_dir)

    # Column names for C-MAPSS: unit, time_cycle, 26 op conditions + 14 sensors = 40 total cols
    # Actually: id, t, then 26 columns (3 op settings + 23 sensor channels after dropping constants)
    n_cols_expected = None  # determined from first read

    train_path = data_dir / f"train_{subset}.txt"
    test_path = data_dir / f"test_{subset}.txt"
    rul_path = data_dir / f"RUL_{subset}.txt"

    if not train_path.exists():
        raise FileNotFoundError(f"C-MAPSS train file not found: {train_path}")

    # Read training data
    train_raw = pd.read_csv(train_path, sep=r"\s+", header=None)
    n_cols = train_raw.shape[1]

    # Assign column names: unit, cycle, then feature columns
    # Real C-MAPSS: id, time_cycle_of_life, then 26 columns (op settings + sensors)
    # We name them generically: unit, cycle, feat_0, feat_1, ...
    cols = ["unit", "cycle"] + [f"feat_{i}" for i in range(n_cols - 2)]
    train_raw.columns = cols

    # Drop constant columns (op1, op2, op31 typically constant in C-MAPSS)
    const_cols = [c for c in train_raw.columns if train_raw[c].nunique() <= 1
                  and c not in ("unit", "cycle")]
    feature_cols = [c for c in train_raw.columns if c not in ("unit", "cycle", *const_cols)]

    A_train = train_raw[["unit", "cycle"]].copy()
    X_train = train_raw[feature_cols].copy()

    # Compute RUL: for training, RUL = (max_cycle_for_unit - current_cycle), capped at max_rul
    unit_max_cycle = train_raw.groupby("unit")["cycle"].transform("max")
    rul_values = (unit_max_cycle - train_raw["cycle"]).clip(upper=max_rul)
    y_train = pd.DataFrame({"RUL": rul_values})

    # Read test data
    if test_path.exists() and rul_path.exists():
        test_raw = pd.read_csv(test_path, sep=r"\s+", header=None)
        test_raw.columns = cols[:test_raw.shape[1]]

        test_true_rul = pd.read_csv(rul_path, sep=r"\s+", header=None)[0].values

        A_test = test_raw[["unit", "cycle"]].copy()
        X_test = test_raw[feature_cols].copy()

        # Test RUL: from last cycle of each unit, subtract current cycle
        # Each test unit has one true RUL (at its final cycle)
        unit_last_cycle = test_raw.groupby("unit")["cycle"].transform("max")
        test_rul = pd.Series(0.0, index=test_raw.index)
        for u_idx, unit_id in enumerate(sorted(test_raw["unit"].unique())):
            mask = test_raw["unit"] == unit_id
            last_cycle = test_raw.loc[mask, "cycle"].max()
            # True RUL at the LAST cycle; earlier cycles get RUL + (last_cycle - cycle)
            offset = last_cycle - test_raw.loc[mask, "cycle"]
            test_rul[mask] = test_true_rul[u_idx] + offset.values

        y_test = pd.DataFrame({"RUL": test_rul.clip(upper=max_rul)})

        return A_train, X_train[feature_cols], y_train, (A_test, X_test[feature_cols], y_test)
    else:
        return A_train, X_train[feature_cols], y_train, (None, None, None)


def preprocess_cmapss(
    data_dir: str | Path | None = None,
    subset: str = "FD001",
    window_length: int = WINDOW_LENGTH,
    stride: int = STRIDE,
    split_mode: SplitMode = "group_aware",
    seed: int = 42,
    max_rul: int = 125,
) -> tuple:
    """
    Full preprocessing pipeline for a C-MAPSS subset.

    Steps:
    1. Load raw data (already per-cycle, no intra-cycle aggregation needed)
    2. Group-aware split by unit (or legacy random split)
    3. Sliding window
    4. Balance train only
    5. Standardize

    Args:
        data_dir: Directory with C-MAPSS txt files. Defaults to config.CMAPSS_RAW_DIR.
        subset: FD001-FD004.
        window_length: Sliding window length.
        stride: Sliding window stride.
        split_mode: "legacy" or "group_aware".
        seed: Random seed.
        max_rul: RUL cap for piecewise linear.

    Returns:
        Tuple of (A_train, W_train, X_s_train, y_train,
                   A_test, W_test, X_s_test, y_test).
        Note: For C-MAPSS, W (operating conditions) and X_s (sensors) are combined
        into a single feature set; we put them in X_s and leave W as empty DataFrames.
    """
    if data_dir is None:
        data_dir = CMAPSS_RAW_DIR

    A_train_raw, X_train_raw, y_train_raw, test_data = load_cmapss_subset(
        data_dir, subset=subset, max_rul=max_rul
    )

    if test_data[0] is None:
        raise FileNotFoundError(
            f"C-MAPSS test/RUL files not found for {subset} in {data_dir}"
        )

    A_test_raw, X_test_raw, y_test_raw = test_data

    # C-MAPSS has no separate W/X_s distinction at this level
    # Use empty W, put all features in X_s for API compatibility
    W_train_empty = pd.DataFrame(index=A_train_raw.index)
    W_test_empty = pd.DataFrame(index=A_test_raw.index)

    if split_mode == "group_aware":
        # Combine train+test, split by unit
        all_A = pd.concat([A_train_raw, A_test_raw], ignore_index=True)
        all_X = pd.concat([X_train_raw, X_test_raw], ignore_index=True)
        all_y = pd.concat([y_train_raw, y_test_raw], ignore_index=True)
        all_W = pd.DataFrame(index=all_A.index)

        (A_tr, W_tr, Xs_tr, y_tr), (A_te, W_te, Xs_te, y_te) = group_aware_unit_split(
            all_A, all_W, all_X, all_y, train_fraction=0.8, seed=seed
        )

        # Sliding window AFTER split
        A_tr, W_tr, Xs_tr, y_tr = create_intercycle_sliding_window(
            A_tr, W_tr, Xs_tr, y_tr, window_length=window_length, stride=stride
        )
        A_te, W_te, Xs_te, y_te = create_intercycle_sliding_window(
            A_te, W_te, Xs_te, y_te, window_length=window_length, stride=stride
        )

        # Balance train only
        A_tr, W_tr, Xs_tr, y_tr = balance_train_dataset(A_tr, W_tr, Xs_tr, y_tr, seed=seed)

    else:
        # Legacy: sliding window first, then use train/test as-is
        A_tr, W_tr, Xs_tr, y_tr = create_intercycle_sliding_window(
            A_train_raw, W_train_empty, X_train_raw, y_train_raw,
            window_length=window_length, stride=stride
        )
        A_te, W_te, Xs_te, y_te = create_intercycle_sliding_window(
            A_test_raw, W_test_empty, X_test_raw, y_test_raw,
            window_length=window_length, stride=stride
        )
        A_tr, W_tr, Xs_tr, y_tr = balance_train_dataset(A_tr, W_tr, Xs_tr, y_tr, seed=seed)

    # Standardize
    W_tr, Xs_tr, W_te, Xs_te = standardize_features(W_tr, Xs_tr, W_te, Xs_te)

    return A_tr, W_tr, Xs_tr, y_tr, A_te, W_te, Xs_te, y_te

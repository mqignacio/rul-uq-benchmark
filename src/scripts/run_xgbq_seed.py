"""
Standalone XGBQ runner — avoids MPS+XGBoost segfault on macOS.

Run as a subprocess: python run_xgbq_seed.py <seed>
Outputs JSON metrics to stdout. Does NOT import torch.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Paths to old preprocessed data
OLD_PREPROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / \
    "resources/old-files/src/data/preprocessed"


def calculate_picp(y_true, lower, upper):
    in_pi = np.logical_and(y_true >= lower, y_true <= upper)
    return float(np.mean(in_pi))


def calculate_nmpiw(y_true, lower, upper):
    width = upper - lower
    return float(np.mean(width) / np.mean(np.abs(y_true)))


def calculate_cwc(y_true, lower, upper, tau=50, kappa=0.95):
    picp = calculate_picp(y_true, lower, upper)
    nmpiw = calculate_nmpiw(y_true, lower, upper)
    penalty = max(0, kappa - picp)
    return float(nmpiw * (1 + tau * penalty))


def calculate_rs(y_true, lower, upper, alpha=0.95):
    picp = calculate_picp(y_true, lower, upper)
    return float(abs(picp - alpha))


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    # Optional: data_dir as second arg (defaults to old preprocessed)
    data_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else OLD_PREPROCESSED_DIR

    # Load data
    W_train = pd.read_csv(data_dir / "preprocessed_W_train.csv")
    X_s_train = pd.read_csv(data_dir / "preprocessed_X_s_train.csv")
    y_train = pd.read_csv(data_dir / "preprocessed_y_train.csv")
    W_test = pd.read_csv(data_dir / "preprocessed_W_test.csv")
    X_s_test = pd.read_csv(data_dir / "preprocessed_X_s_test.csv")
    y_test = pd.read_csv(data_dir / "preprocessed_y_test.csv")

    X_tr = pd.concat([W_train, X_s_train], axis=1)
    X_te = pd.concat([W_test, X_s_test], axis=1)
    y_tr = y_train["RUL"].values
    y_te = y_test["RUL"].values

    quantiles = [0.1, 0.5, 0.9]
    preds = {}

    for q in quantiles:
        model = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=q,
            random_state=seed,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        preds[q] = model.predict(X_te)

    lower = preds[0.1]
    median = preds[0.5]
    upper = preds[0.9]

    results = {
        "seed": seed,
        "PICP": calculate_picp(y_te, lower, upper),
        "NMPIW": calculate_nmpiw(y_te, lower, upper),
        "CWC": calculate_cwc(y_te, lower, upper),
        "RS": calculate_rs(y_te, lower, upper),
    }

    print(json.dumps(results))


if __name__ == "__main__":
    main()

"""
Two-stage baseline reproduction script.

Stage 1 (port validation): Load the OLD preprocessed data (legacy split), run
QRNN/CQR-QRNN/GPR/XGBQ with the ported code, verify results within tolerance.

Stage 2 (leakage correction): Re-preprocess with group-aware split, re-run,
report delta vs. Stage 1 as the leakage correction.

Usage:
    cd src && uv run python scripts/run_reproduction.py --stage 1
    cd src && uv run python scripts/run_reproduction.py --stage 2
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qrnn_uq_rul.config import RESULTS_DIR, SEED_LIST
from qrnn_uq_rul.metrics import evaluate_uq_metrics
from qrnn_uq_rul.models import train_test_gpr, train_test_qrnn_cqr, train_test_xgbq
from qrnn_uq_rul.utils import get_device, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger("reproduction")

# Old preprocessed data location (legacy split, already computed by prior team)
OLD_PREPROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / \
    "resources/old-files/src/data/preprocessed"


def load_old_preprocessed() -> tuple:
    """Load the old preprocessed CSVs directly (legacy split data)."""
    keys = ["A_train", "W_train", "X_s_train", "y_train",
            "A_test", "W_test", "X_s_test", "y_test"]
    dfs = []
    for k in keys:
        path = OLD_PREPROCESSED_DIR / f"preprocessed_{k}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Old preprocessed file missing: {path}")
        dfs.append(pd.read_csv(path))
    return tuple(dfs)


def evaluate_predictions(y_test_arr: np.ndarray, y_pred: dict) -> dict:
    """Compute UQ metrics from predictions."""
    uq = evaluate_uq_metrics(y_test_arr, y_pred)
    return dict(zip(uq["Metric"], uq["Value"]))


def run_stage1():
    """
    Stage 1: Port validation using OLD preprocessed data (legacy split).

    Tolerances:
    - C-QRNN PICP: 0.7913 ± 0.0149 (within mean ± 1σ across 5 seeds)
    - XGBQ PICP: 0.6113 ± 0.0207
    - GPR PICP: 0.8294 ± 0.005 (single deterministic run)
    """
    logger.info("=" * 70)
    logger.info("STAGE 1: PORT VALIDATION (legacy split, old preprocessed data)")
    logger.info("=" * 70)

    device = get_device()
    A_train, W_train, X_s_train, y_train, A_test, W_test, X_s_test, y_test = load_old_preprocessed()

    logger.info("Data loaded: train=%d samples, test=%d samples", len(A_train), len(A_test))
    logger.info("Features: W=%d, X_s=%d", W_train.shape[1], X_s_train.shape[1])

    y_test_arr = y_test["RUL"].values.ravel()
    results = {"cqr_qrnn": [], "xgbq": [], "gpr": None}

    # --- CQR-QRNN (5 seeds) ---
    logger.info("\n--- CQR-QRNN (5 seeds) ---")
    for seed in SEED_LIST:
        set_seed(seed)
        t0 = time.time()
        _, e_hat, y_pred, _, _, _ = train_test_qrnn_cqr(
            W_train, X_s_train, y_train,
            W_test, X_s_test, y_test,
            device=device, seed=seed, save_model=False,
        )
        metrics = evaluate_predictions(y_test_arr, y_pred)
        metrics["e_hat"] = e_hat
        metrics["seed"] = seed
        metrics["time_s"] = time.time() - t0
        results["cqr_qrnn"].append(metrics)
        logger.info("  Seed %d: PICP=%.4f, NMPIW=%.4f, E_hat=%.4f (%.1fs)",
                    seed, metrics["PICP"], metrics["NMPIW"], e_hat, metrics["time_s"])

    # --- XGBQ (5 seeds) — run in subprocess to avoid MPS+XGBoost segfault ---
    logger.info("\n--- XGBQ (5 seeds, subprocess) ---")
    import subprocess
    xgbq_script = Path(__file__).resolve().parent / "run_xgbq_seed.py"
    for seed in SEED_LIST:
        t0 = time.time()
        result = subprocess.run(
            [sys.executable, str(xgbq_script), str(seed)],
            capture_output=True, text=True, check=True,
        )
        metrics = json.loads(result.stdout)
        metrics["time_s"] = time.time() - t0
        results["xgbq"].append(metrics)
        logger.info("  Seed %d: PICP=%.4f, NMPIW=%.4f (%.1fs)",
                    seed, metrics["PICP"], metrics["NMPIW"], metrics["time_s"])

    # --- GPR (1 run, deterministic) ---
    logger.info("\n--- GPR (deterministic) ---")
    t0 = time.time()
    _, y_pred, _ = train_test_gpr(
        W_train, X_s_train, y_train,
        W_test, X_s_test, y_test,
        save_model=False,
    )
    gpr_metrics = evaluate_predictions(y_test_arr, y_pred)
    gpr_metrics["time_s"] = time.time() - t0
    results["gpr"] = gpr_metrics
    logger.info("  PICP=%.4f, NMPIW=%.4f (%.1fs)",
                gpr_metrics["PICP"], gpr_metrics["NMPIW"], gpr_metrics["time_s"])

    # Save raw results
    out_path = RESULTS_DIR / "stage1_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert numpy types for JSON serialization
    serializable = json.loads(json.dumps(results, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))
    out_path.write_text(json.dumps(serializable, indent=2))
    logger.info("Stage 1 results saved to %s", out_path)

    return results


def run_stage2():
    """
    Stage 2: Leakage correction using group-aware split data.

    Loads the NEW preprocessed CSVs (from src/data/preprocessed/) which were
    generated with split_mode='group_aware'. Runs same methods as Stage 1.
    """
    logger.info("=" * 70)
    logger.info("STAGE 2: LEAKAGE CORRECTION (group-aware split)")
    logger.info("=" * 70)

    device = get_device()

    # Load new group-aware preprocessed data
    from qrnn_uq_rul.preprocess import load_preprocessed_from_csv
    A_train, W_train, X_s_train, y_train, A_test, W_test, X_s_test, y_test = (
        load_preprocessed_from_csv()
    )

    logger.info("Data loaded: train=%d samples (%d units), test=%d samples (%d units)",
                len(A_train), A_train['unit'].nunique(),
                len(A_test), A_test['unit'].nunique())
    # Verify no leakage
    overlap = set(A_train['unit']) & set(A_test['unit'])
    assert len(overlap) == 0, f"Unit overlap detected: {overlap}"
    logger.info("Features: W=%d, X_s=%d", W_train.shape[1], X_s_train.shape[1])

    y_test_arr = y_test["RUL"].values.ravel()
    results = {"cqr_qrnn": [], "xgbq": [], "gpr": None}

    # --- CQR-QRNN (5 seeds) ---
    logger.info("\n--- CQR-QRNN (5 seeds) ---")
    for seed in SEED_LIST:
        set_seed(seed)
        t0 = time.time()
        _, e_hat, y_pred, _, _, _ = train_test_qrnn_cqr(
            W_train, X_s_train, y_train,
            W_test, X_s_test, y_test,
            device=device, seed=seed, save_model=False,
        )
        metrics = evaluate_predictions(y_test_arr, y_pred)
        metrics["e_hat"] = e_hat
        metrics["seed"] = seed
        metrics["time_s"] = time.time() - t0
        results["cqr_qrnn"].append(metrics)
        logger.info("  Seed %d: PICP=%.4f, NMPIW=%.4f, E_hat=%.4f (%.1fs)",
                    seed, metrics["PICP"], metrics["NMPIW"], e_hat, metrics["time_s"])

    # --- XGBQ (5 seeds, subprocess) ---
    logger.info("\n--- XGBQ (5 seeds, subprocess) ---")
    import subprocess
    xgbq_script = Path(__file__).resolve().parent / "run_xgbq_seed.py"
    from qrnn_uq_rul.config import PREPROCESSED_DIR
    for seed in SEED_LIST:
        t0 = time.time()
        result = subprocess.run(
            [sys.executable, str(xgbq_script), str(seed), str(PREPROCESSED_DIR)],
            capture_output=True, text=True, check=True,
        )
        metrics = json.loads(result.stdout)
        metrics["time_s"] = time.time() - t0
        results["xgbq"].append(metrics)
        logger.info("  Seed %d: PICP=%.4f, NMPIW=%.4f (%.1fs)",
                    seed, metrics["PICP"], metrics["NMPIW"], metrics["time_s"])

    # --- GPR (1 run, deterministic) ---
    logger.info("\n--- GPR (deterministic) ---")
    t0 = time.time()
    _, y_pred, _ = train_test_gpr(
        W_train, X_s_train, y_train,
        W_test, X_s_test, y_test,
        save_model=False,
    )
    gpr_metrics = evaluate_predictions(y_test_arr, y_pred)
    gpr_metrics["time_s"] = time.time() - t0
    results["gpr"] = gpr_metrics
    logger.info("  PICP=%.4f, NMPIW=%.4f (%.1fs)",
                gpr_metrics["PICP"], gpr_metrics["NMPIW"], gpr_metrics["time_s"])

    # Save raw results
    out_path = RESULTS_DIR / "stage2_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = json.loads(json.dumps(results, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))
    out_path.write_text(json.dumps(serializable, indent=2))
    logger.info("Stage 2 results saved to %s", out_path)

    return results


def check_tolerances(results: dict) -> dict:
    """Check Stage 1 results against prior tolerances."""
    checks = {}

    # C-QRNN: mean PICP within ±1σ of 0.7913±0.0149
    cqr_picps = [r["PICP"] for r in results["cqr_qrnn"]]
    cqr_mean = np.mean(cqr_picps)
    cqr_std = np.std(cqr_picps, ddof=1)
    # Tolerance: reported mean ± reported σ
    target_mean, target_std = 0.7913, 0.0149
    in_band = abs(cqr_mean - target_mean) <= (target_std + cqr_std)
    checks["C-QRNN PICP"] = {
        "value": round(cqr_mean, 4),
        "std": round(cqr_std, 4),
        "target": f"{target_mean}±{target_std}",
        "pass": in_band,
    }

    # XGBQ: mean PICP within ±1σ of 0.6113±0.0207
    xgbq_picps = [r["PICP"] for r in results["xgbq"]]
    xgbq_mean = np.mean(xgbq_picps)
    xgbq_std = np.std(xgbq_picps, ddof=1)
    target_mean_x, target_std_x = 0.6113, 0.0207
    in_band_x = abs(xgbq_mean - target_mean_x) <= (target_std_x + xgbq_std)
    checks["XGBQ PICP"] = {
        "value": round(xgbq_mean, 4),
        "std": round(xgbq_std, 4),
        "target": f"{target_mean_x}±{target_std_x}",
        "pass": in_band_x,
    }

    # GPR: single run within ±0.005 of 0.8294
    gpr_picp = results["gpr"]["PICP"]
    gpr_pass = abs(gpr_picp - 0.8294) <= 0.005
    checks["GPR PICP"] = {
        "value": round(gpr_picp, 4),
        "target": "0.8294 ± 0.005",
        "pass": gpr_pass,
    }

    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-stage baseline reproduction")
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True,
                        help="1=port validation (legacy), 2=leakage correction (group-aware)")
    args = parser.parse_args()

    if args.stage == 1:
        results = run_stage1()
        checks = check_tolerances(results)
        logger.info("\n" + "=" * 70)
        logger.info("TOLERANCE CHECKS:")
        for name, info in checks.items():
            status = "PASS" if info["pass"] else "FAIL"
            logger.info("  %s: %.4f (target %s) [%s]", name, info["value"], info["target"], status)
        all_pass = all(c["pass"] for c in checks.values())
        logger.info("Overall: %s", "ALL PASS" if all_pass else "SOME FAILED")
        if not all_pass:
            sys.exit(1)

    elif args.stage == 2:
        results = run_stage2()
        # Compare with Stage 1
        stage1_path = RESULTS_DIR / "stage1_results.json"
        if stage1_path.exists():
            stage1 = json.loads(stage1_path.read_text())
            logger.info("\n" + "=" * 70)
            logger.info("LEAKAGE CORRECTION DELTA (Stage 2 − Stage 1):")
            # C-QRNN
            s1_cqr = np.mean([r["PICP"] for r in stage1["cqr_qrnn"]])
            s2_cqr = np.mean([r["PICP"] for r in results["cqr_qrnn"]])
            logger.info("  C-QRNN PICP: %.4f → %.4f (Δ=%.4f)", s1_cqr, s2_cqr, s2_cqr - s1_cqr)
            # XGBQ
            s1_xgbq = np.mean([r["PICP"] for r in stage1["xgbq"]])
            s2_xgbq = np.mean([r["PICP"] for r in results["xgbq"]])
            logger.info("  XGBQ PICP:   %.4f → %.4f (Δ=%.4f)", s1_xgbq, s2_xgbq, s2_xgbq - s1_xgbq)
            # GPR
            s1_gpr = stage1["gpr"]["PICP"]
            s2_gpr = results["gpr"]["PICP"]
            logger.info("  GPR PICP:    %.4f → %.4f (Δ=%.4f)", s1_gpr, s2_gpr, s2_gpr - s1_gpr)

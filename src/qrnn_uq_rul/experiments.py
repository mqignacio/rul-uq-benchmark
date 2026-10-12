"""
Multi-seed experiment runner and statistical analysis for baseline reproduction.

Provides:
- run_multi_seed_experiment: Run QRNN (raw + CQR) and XGBQ across multiple seeds.
- run_gpr_once: Run GPR deterministically (single run).
- generate_results_report: Markdown summary of all results.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from .config import (
    RESULTS_DIR,
    SEED_LIST,
)
from .metrics import (
    evaluate_efficiency_metrics,
    evaluate_uq_metrics,
)
from .models import (
    QRNN,
    train_qrnn,
    train_test_gpr,
    train_test_qrnn_cqr,
    train_test_xgbq,
)
from .preprocess import load_preprocessed_from_csv
from .utils import get_device, set_seed

logger = logging.getLogger(__name__)


def run_single_seed_qrnn(
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    W_test: pd.DataFrame,
    X_s_test: pd.DataFrame,
    y_test: pd.DataFrame,
    device: torch.device,
    suffix: str = "",
) -> Dict:
    """Train raw QRNN for a single seed. Returns metrics dict."""
    set_seed(seed)

    qrnn_model, _, _ = train_qrnn(
        X_train, y_train, X_val, y_val,
        device=device, save_model=False, suffix=f"seed{seed}_{suffix}", seed=seed,
    )

    qrnn_model.eval()
    X_test = pd.concat([W_test, X_s_test], axis=1).to_numpy()
    X_test_tensor = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        preds = qrnn_model(X_test_tensor)
        y_pred = {
            "lower": preds[:, 0].cpu().numpy(),
            "median": preds[:, 1].cpu().numpy(),
            "upper": preds[:, 2].cpu().numpy(),
        }

    uq = evaluate_uq_metrics(y_test.values.ravel(), y_pred)
    uq_dict = dict(zip(uq["Metric"], uq["Value"]))

    return {"seed": seed, **uq_dict, "y_pred": y_pred}


def run_single_seed_qrnn_cqr(
    seed: int,
    W_train: pd.DataFrame,
    X_s_train: pd.DataFrame,
    y_train: pd.DataFrame,
    W_test: pd.DataFrame,
    X_s_test: pd.DataFrame,
    y_test: pd.DataFrame,
    device: torch.device,
    suffix: str = "",
) -> Dict:
    """Train CQR-QRNN for a single seed. Returns metrics dict."""
    set_seed(seed)

    qrnn_model, e_hat, y_pred, point_eval, _, _ = train_test_qrnn_cqr(
        W_train, X_s_train, y_train,
        W_test, X_s_test, y_test,
        device=device, seed=seed, save_model=False,
        suffix=f"seed{seed}_{suffix}",
    )

    uq = evaluate_uq_metrics(y_test.values.ravel(), y_pred)
    uq_dict = dict(zip(uq["Metric"], uq["Value"]))

    return {"seed": seed, **uq_dict, "e_hat": e_hat, "y_pred": y_pred}


def run_single_seed_xgbq(
    seed: int,
    W_train: pd.DataFrame,
    X_s_train: pd.DataFrame,
    y_train: pd.DataFrame,
    W_test: pd.DataFrame,
    X_s_test: pd.DataFrame,
    y_test: pd.DataFrame,
    suffix: str = "",
) -> Dict:
    """Train XGBQ for a single seed. Returns metrics dict."""
    set_seed(seed)

    models, y_pred, point_eval = train_test_xgbq(
        W_train, X_s_train, y_train,
        W_test, X_s_test, y_test,
        save_model=False, suffix=f"seed{seed}_{suffix}", seed=seed,
    )

    uq = evaluate_uq_metrics(y_test.values.ravel(), y_pred)
    uq_dict = dict(zip(uq["Metric"], uq["Value"]))

    return {"seed": seed, **uq_dict, "y_pred": y_pred}


def run_multi_seed_experiment(
    seeds: Optional[List[int]] = None,
    suffix: str = "",
) -> Dict[str, pd.DataFrame]:
    """
    Run multi-seed experiments for all stochastic methods (QRNN, CQR-QRNN, XGBQ).

    GPR is deterministic — run separately via run_gpr_once().

    Args:
        seeds: List of random seeds. Defaults to config.SEED_LIST.
        suffix: Suffix for output filenames.

    Returns:
        Dict with keys 'qrnn', 'qrnn_cqr', 'xgbq', each a DataFrame with per-seed metrics.
    """
    if seeds is None:
        seeds = SEED_LIST

    device = get_device()

    logger.info("Loading preprocessed data...")
    A_train, W_train, X_s_train, y_train, A_test, W_test, X_s_test, y_test = (
        load_preprocessed_from_csv()
    )

    X_train = pd.concat([W_train, X_s_train], axis=1).to_numpy()
    y_train_arr = y_train.to_numpy().ravel()
    # Use test partition as validation for raw QRNN (legacy behavior)
    X_val = pd.concat([W_test, X_s_test], axis=1).to_numpy()
    y_val_arr = y_test.to_numpy().ravel()

    results: Dict[str, list] = {"qrnn": [], "qrnn_cqr": [], "xgbq": []}

    for seed in seeds:
        logger.info("=" * 60)
        logger.info("Seed %d", seed)
        logger.info("=" * 60)

        # QRNN (raw)
        logger.info("--- QRNN (raw) seed=%d ---", seed)
        qrnn_res = run_single_seed_qrnn(
            seed, X_train, y_train_arr, X_val, y_val_arr,
            W_test, X_s_test, y_test, device, suffix=suffix,
        )
        results["qrnn"].append({k: v for k, v in qrnn_res.items() if k != "y_pred"})
        logger.info(
            "  PICP=%.4f, NMPIW=%.4f, CWC=%.4f, RS=%.4f",
            qrnn_res["PICP"], qrnn_res["NMPIW"], qrnn_res["CWC"], qrnn_res["RS"],
        )

        # CQR-QRNN
        logger.info("--- CQR-QRNN seed=%d ---", seed)
        cqr_res = run_single_seed_qrnn_cqr(
            seed, W_train, X_s_train, y_train,
            W_test, X_s_test, y_test, device, suffix=suffix,
        )
        results["qrnn_cqr"].append({k: v for k, v in cqr_res.items() if k != "y_pred"})
        logger.info(
            "  PICP=%.4f, NMPIW=%.4f, CWC=%.4f, RS=%.4f, E_hat=%.4f",
            cqr_res["PICP"], cqr_res["NMPIW"], cqr_res["CWC"], cqr_res["RS"], cqr_res["e_hat"],
        )

        # XGBQ
        logger.info("--- XGBQ seed=%d ---", seed)
        xgbq_res = run_single_seed_xgbq(
            seed, W_train, X_s_train, y_train,
            W_test, X_s_test, y_test, suffix=suffix,
        )
        results["xgbq"].append({k: v for k, v in xgbq_res.items() if k != "y_pred"})
        logger.info(
            "  PICP=%.4f, NMPIW=%.4f, CWC=%.4f, RS=%.4f",
            xgbq_res["PICP"], xgbq_res["NMPIW"], xgbq_res["CWC"], xgbq_res["RS"],
        )

    # Convert to DataFrames
    for key in results:
        results[key] = pd.DataFrame(results[key])

    # Save per-seed results
    per_seed_dir = RESULTS_DIR / "eval" / "per_seed"
    per_seed_dir.mkdir(parents=True, exist_ok=True)

    for method, df in results.items():
        outpath = per_seed_dir / f"per_seed_{method}_{suffix}.csv"
        df.to_csv(outpath, index=False)
        logger.info("Saved %s per-seed results to %s", method, outpath)

    return results


def run_gpr_once(suffix: str = "") -> Dict:
    """
    Run GPR once (deterministic). Returns metrics dict.

    Args:
        suffix: Suffix for output filenames.

    Returns:
        Dict with GPR UQ metrics and efficiency.
    """
    logger.info("--- GPR (single run, deterministic) ---")
    A_train, W_train, X_s_train, y_train, A_test, W_test, X_s_test, y_test = (
        load_preprocessed_from_csv()
    )

    gpr_model, y_pred, point_eval = train_test_gpr(
        W_train, X_s_train, y_train,
        W_test, X_s_test, y_test,
        save_model=False, suffix=f"gpr_{suffix}",
    )

    uq = evaluate_uq_metrics(y_test.values.ravel(), y_pred)
    uq_dict = dict(zip(uq["Metric"], uq["Value"]))

    device = get_device()
    X_test_combined = pd.concat([W_test, X_s_test], axis=1)
    eff = evaluate_efficiency_metrics(gpr_model, X_test_combined)
    eff_dict = dict(zip(eff["Metric"], eff["Value"]))

    result = {**uq_dict, **eff_dict}
    logger.info(
        "  PICP=%.4f, NMPIW=%.4f, CWC=%.4f, RS=%.4f",
        uq_dict["PICP"], uq_dict["NMPIW"], uq_dict["CWC"], uq_dict["RS"],
    )
    return result


def generate_results_report(
    multi_seed_results: Dict[str, pd.DataFrame],
    gpr_result: Optional[Dict] = None,
    suffix: str = "",
) -> str:
    """
    Generate a Markdown results report from multi-seed experiment results.

    Args:
        multi_seed_results: Dict from run_multi_seed_experiment.
        gpr_result: Dict from run_gpr_once.
        suffix: Suffix for output filenames.

    Returns:
        Markdown string of the report.
    """
    lines: List[str] = []
    lines.append("# Multi-Seed Experiment Results\n")
    lines.append(f"**Seeds:** {SEED_LIST}")
    lines.append(f"**Dataset:** N-CMAPSS DS01-DS07 + DS08a + DS08c\n")

    lines.append("## UQ Metrics (Mean ± Std)\n")
    lines.append("| Method | PICP | NMPIW | CWC | RS |")
    lines.append("|---|---|---|---|---|")

    for method_name, df in multi_seed_results.items():
        picp = f"{df['PICP'].mean():.4f} ± {df['PICP'].std(ddof=1):.4f}"
        nmpiw = f"{df['NMPIW'].mean():.4f} ± {df['NMPIW'].std(ddof=1):.4f}"
        cwc = f"{df['CWC'].mean():.4f} ± {df['CWC'].std(ddof=1):.4f}"
        rs = f"{df['RS'].mean():.4f} ± {df['RS'].std(ddof=1):.4f}"
        lines.append(f"| {method_name} | {picp} | {nmpiw} | {cwc} | {rs} |")

    if gpr_result:
        lines.append(
            f"| GPR (deterministic) | {gpr_result['PICP']:.4f} | "
            f"{gpr_result['NMPIW']:.4f} | {gpr_result['CWC']:.4f} | "
            f"{gpr_result['RS']:.4f} |"
        )

    report_text = "\n".join(lines)

    report_path = RESULTS_DIR / "eval" / f"results_report_{suffix}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text)
    logger.info("Saved results report to %s", report_path)

    return report_text

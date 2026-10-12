#!/usr/bin/env python
"""
Phase-4 experiment runner (tasks/04-experiments.md, execution spec E1).

Runs the Path-B comparative method set (design §4) on N-CMAPSS (DS01-DS07 +
DS08a/DS08c) and C-MAPSS (FD001-FD004), multi-seed, with incremental
checkpointing and a planned-vs-done run matrix.

Design notes
------------
* Two incompatible arm APIs are reconciled here: the Phase-2 C-QRNN arm takes
  DataFrames (``models.train_test_qrnn_cqr``); every Phase-3 arm takes raw
  NumPy X and splits internally. Features are assembled uniformly as
  ``X = concat([W, X_s], axis=1)`` (C-MAPSS has empty W).
* Group-aware (engine-unit) splits use ``SPLIT_SEED = 42`` (fix M3) via
  ``preprocess_ds`` / ``preprocess_cmapss`` with ``split_mode="group_aware"``.
* C-XGBQ runs in a separate process by default: on macOS, PyTorch and XGBoost
  each bundle libomp and double-initializing the OpenMP pool in one process
  can segfault (Phase-2 finding; coding skill §2.2). The worker re-enters this
  script via ``--worker-xgbq``.
* C-GPR is deterministic: one manifest row per subset (``seed=None``,
  ``deterministic=True``). Training sets larger than ``--gpr-max-samples``
  (default 5248, the Phase-2 measured budget) are subsampled with a seeded RNG
  (exact GP training is O(n^3); see doc/data_provenance.md).
* raw QRNN multi-seed is one manifest row per subset: the arm's output is the
  ensemble mean over ``SEED_LIST``; members are trained incrementally with a
  per-member side checkpoint so a crash never loses completed members.
* Every cell's per-sample arrays (y_true, lower/median/upper, unit ids,
  RUL-stage labels) are stored as a compressed .npz under ``results/pred/``
  (gitignored); ``experiment_results.json`` carries scalars only.
* The architecture ablation (E4) reuses the C-QRNN building blocks with
  ``QRNN(hidden_dims=..., dropouts=...)`` overrides. Grid note: the E4 spec's
  width grid {8,16,32} referenced a "base 16" default; the shipped QRNN base
  is the 128/64/32 taper (verified in models.py per the spec's own
  instruction), so the grid brackets the actual base: width {32,64,128},
  depth {1,2,3}, dropout {0.0,0.2,0.5}.

Usage
-----
    uv run python scripts/run_phase4_experiments.py --dry-run
    uv run python scripts/run_phase4_experiments.py            # full sweep
    uv run python scripts/run_phase4_experiments.py --resume   # continue
    uv run python scripts/run_phase4_experiments.py --datasets ncmapss \
        --subsets ds02 --methods c_qrnn
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve()
SRC_ROOT = SCRIPT.parent.parent  # .../uq_rul_aeroengine-project/src
RESULTS_DIR = SRC_ROOT / "results"
sys.path.insert(0, str(SRC_ROOT))

# Static per-cell wall-clock estimates (seconds) for --dry-run budgeting.
# Phase-2 measured: QRNN ~30 s (MPS), C-XGBQ ~3 s, GPR ~46 min at n=5248.
EST_S: Dict[str, float] = {
    "c_qrnn": 30, "c_mdn": 60, "uacqr": 150, "distcp": 60,
    "weighted_cqr": 45, "wqlcp": 90, "edl": 60, "cgpr": 2760,
    "cxgbq": 60, "raw_qrnn": 200, "arch_ablation": 30,
}

ALL_METHODS = ["c_qrnn", "c_mdn", "uacqr", "distcp", "weighted_cqr",
               "wqlcp", "edl", "cgpr", "cxgbq", "raw_qrnn"]
# Methods with a single manifest row per subset (seed=None).
SINGLE_ROW_METHODS = {"cgpr", "raw_qrnn"}

SPLIT_SEED = 42  # fix M3: group-aware split seed (config RANDOM_SEED)

# E4 architecture ablation: (config id, factor, value, hidden_dims, dropouts).
# Base QRNN = 128/64/32 taper, dropouts 0.3/0.2/0.1 (models.py).
ABLATION_SUBSETS: List[Tuple[str, str]] = [("ncmapss", "DS02"), ("cmapss", "FD001")]
ABLATION_SEED = 42
ABLATION_CONFIGS: List[Dict[str, Any]] = [
    {"id": "width_32", "factor": "width", "value": 32,
     "hidden_dims": (32, 32, 32), "dropouts": (0.3, 0.2, 0.1)},
    {"id": "width_64", "factor": "width", "value": 64,
     "hidden_dims": (64, 64, 64), "dropouts": (0.3, 0.2, 0.1)},
    {"id": "width_128", "factor": "width", "value": 128,
     "hidden_dims": (128, 128, 128), "dropouts": (0.3, 0.2, 0.1)},
    {"id": "depth_1", "factor": "depth", "value": 1,
     "hidden_dims": (32,), "dropouts": (0.3,)},
    {"id": "depth_2", "factor": "depth", "value": 2,
     "hidden_dims": (64, 32), "dropouts": (0.3, 0.2)},
    {"id": "depth_3", "factor": "depth", "value": 3,
     "hidden_dims": (128, 64, 32), "dropouts": (0.3, 0.2, 0.1)},
    {"id": "drop_0.0", "factor": "dropout", "value": 0.0,
     "hidden_dims": (128, 64, 32), "dropouts": (0.0, 0.0, 0.0)},
    {"id": "drop_0.2", "factor": "dropout", "value": 0.2,
     "hidden_dims": (128, 64, 32), "dropouts": (0.2, 0.2, 0.2)},
    {"id": "drop_0.5", "factor": "dropout", "value": 0.5,
     "hidden_dims": (128, 64, 32), "dropouts": (0.5, 0.5, 0.5)},
]

log = logging.getLogger("phase4")

# ── Generic helpers ────────────────────────────────────────────────────


def py(o: Any) -> Any:
    """Recursively convert numpy/pandas scalars and arrays to JSON-safe types."""
    if isinstance(o, dict):
        return {k: py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [py(v) for v in o]
    if isinstance(o, np.ndarray):
        return py(o.tolist())
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v
    if isinstance(o, (np.integer, int)) and not isinstance(o, bool):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return o


def atomic_json_write(path: Path, obj: Any) -> None:
    """Write JSON atomically (tmp file + os.replace) so a crash never corrupts the record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, allow_nan=False)
    os.replace(tmp, path)


def setup_logging(log_path: Path, verbose: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(level)
    root.addHandler(sh)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    root.addHandler(fh)
    # quiet noisy third-party loggers
    for name in ("matplotlib", "sklearn", "xgboost", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)


# ── Run matrix manifest (E1a) ──────────────────────────────────────────


def job_id(job: Dict[str, Any]) -> str:
    s = f"{job['dataset']}|{job['subset']}|{job['method']}|{job['seed']}"
    if "config" in job:
        s += f"|{job['config']['id']}"
    return s


def build_manifest(datasets: List[str], subsets: Dict[str, List[str]],
                   methods: List[str], seeds: List[int],
                   gpr_max_samples: int, do_ablation: bool) -> Dict[str, Any]:
    jobs: List[Dict[str, Any]] = []
    for ds in datasets:
        for sub in subsets[ds]:
            for m in methods:
                if m in SINGLE_ROW_METHODS:
                    job: Dict[str, Any] = {"dataset": ds, "subset": sub, "method": m, "seed": None}
                    if m == "cgpr":
                        job["deterministic"] = True
                    if m == "raw_qrnn":
                        job["ensemble_seeds"] = list(seeds)
                    jobs.append(job)
                else:
                    for s in seeds:
                        jobs.append({"dataset": ds, "subset": sub, "method": m, "seed": int(s)})
    if do_ablation:
        for ds, sub in ABLATION_SUBSETS:
            if ds not in datasets:
                continue
            for cfg in ABLATION_CONFIGS:
                jobs.append({"dataset": ds, "subset": sub, "method": "arch_ablation",
                             "seed": ABLATION_SEED,
                             "config": {"id": cfg["id"], "factor": cfg["factor"], "value": cfg["value"]}})
    return {
        "schema_version": 1,
        "split_seed": SPLIT_SEED,
        "seeds": list(seeds),
        "gpr_max_samples": gpr_max_samples,
        "jobs": jobs,
    }


def index_cells(results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Map job_id -> cell record for every cell currently in the results dict."""
    idx: Dict[str, Dict[str, Any]] = {}
    for ds in ("ncmapss", "cmapss"):
        for _sub, methods in results.get(ds, {}).items():
            for _m, cells in methods.items():
                for c in cells:
                    idx[job_id(c)] = c
    for _sub, cfgs in results.get("arch_ablation", {}).items():
        for _cfg, cells in cfgs.items():
            for c in cells:
                idx[job_id(c)] = c
    return idx

# ── Data loading & feature assembly ────────────────────────────────────


def load_subset(dataset: str, subset: str, cache_dir: Path) -> Dict[str, Any]:
    """Load + preprocess one subset (group-aware split, SPLIT_SEED), with a
    disk cache so resumes and the XGBQ subprocess worker reuse the same data."""
    from qrnn_uq_rul.config import CMAPSS_RAW_DIR, FILENAMES, RAW_DATA_DIR
    from qrnn_uq_rul.preprocess import preprocess_cmapss, preprocess_ds

    cache = cache_dir / f"{dataset}_{subset}_split{SPLIT_SEED}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            d = pickle.load(f)
        log.debug("cache hit: %s", cache.name)
        return d

    t0 = time.time()
    if dataset == "ncmapss":
        (A_tr, W_tr, Xs_tr, y_tr,
         A_te, W_te, Xs_te, y_te) = preprocess_ds(
            str(RAW_DATA_DIR / FILENAMES[subset]), split_mode="group_aware", seed=SPLIT_SEED)
    elif dataset == "cmapss":
        (A_tr, W_tr, Xs_tr, y_tr,
         A_te, W_te, Xs_te, y_te) = preprocess_cmapss(
            data_dir=CMAPSS_RAW_DIR, subset=subset, split_mode="group_aware", seed=SPLIT_SEED)
    else:
        raise ValueError(f"unknown dataset {dataset!r}")

    # Tag unit ids with the subset (as preprocess_save_ncmapss does: "DSxx_U##")
    # so per-subset/per-unit coverage is derivable from the unit column alone.
    A_tr = A_tr.copy()
    A_te = A_te.copy()
    A_tr["unit"] = [f"{subset}_{u}" for u in A_tr["unit"]]
    A_te["unit"] = [f"{subset}_{u}" for u in A_te["unit"]]

    d = {"A_train": A_tr, "W_train": W_tr, "X_s_train": Xs_tr, "y_train": y_tr,
         "A_test": A_te, "W_test": W_te, "X_s_test": Xs_te, "y_test": y_te,
         "dataset": dataset, "subset": subset, "split_seed": SPLIT_SEED}
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(d, f)
    os.replace(tmp, cache)
    log.info("preprocessed %s/%s in %.1f s (n_train=%d n_test=%d, %d train units / %d test units)",
             dataset, subset, time.time() - t0, len(y_tr), len(y_te),
             A_tr["unit"].nunique(), A_te["unit"].nunique())
    return d


def assemble_features(d: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unified X = concat([W, X_s]) for both datasets (C-MAPSS has empty W)."""
    X_train = pd.concat([d["W_train"], d["X_s_train"]], axis=1).to_numpy(dtype=np.float64)
    y_train = d["y_train"]["RUL"].to_numpy(dtype=np.float64).ravel()
    X_test = pd.concat([d["W_test"], d["X_s_test"]], axis=1).to_numpy(dtype=np.float64)
    y_test = d["y_test"]["RUL"].to_numpy(dtype=np.float64).ravel()
    return X_train, y_train, X_test, y_test


def internal_split_counts(n_pool: int, cal_fraction: float = 0.80) -> Tuple[int, int, int]:
    """Mirror the arms' internal sklearn splits (same formulas) to record
    n_train/n_val/n_cal without re-running them: 80/20 carve of cal, then 90/10."""
    n_cal = int(np.ceil((1.0 - cal_fraction) * n_pool))
    n_trval = n_pool - n_cal
    n_val = int(np.ceil(0.10 * n_trval))
    return n_trval - n_val, n_val, n_cal


def internal_split(X_all: np.ndarray, y_all: np.ndarray, seed: int,
                   cal_fraction: float = 0.80):
    """The exact split convention every arm uses (needed by raw-QRNN and the
    architecture ablation, which build their own train/val/cal pools)."""
    from sklearn.model_selection import train_test_split
    X_trval, X_cal, y_trval, y_cal = train_test_split(
        X_all, y_all, test_size=1.0 - cal_fraction, random_state=seed)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_trval, y_trval, test_size=0.10, random_state=seed)
    return X_tr, y_tr, X_val, y_val, X_cal, y_cal


def rul_stage(y: np.ndarray) -> np.ndarray:
    """RUL-stage labels (early/mid/late thirds of the RUL range).

    RUL counts down to failure, so the LOW-RUL tercile is the LATE-life stage
    and the HIGH-RUL tercile is the EARLY-life stage. Returns int labels
    0=early, 1=mid, 2=late.
    """
    lo, hi = float(np.min(y)), float(np.max(y))
    if hi <= lo:
        return np.zeros(len(y), dtype=np.int64)
    t1 = lo + (hi - lo) / 3.0
    t2 = lo + 2.0 * (hi - lo) / 3.0
    return np.where(y < t1, 2, np.where(y < t2, 1, 0)).astype(np.int64)


STAGE_NAMES = {0: "early", 1: "mid", 2: "late"}

# ── Metrics, efficiency, artifact storage ──────────────────────────────


def compute_cell_metrics(y_test: np.ndarray, y_pred: Dict[str, np.ndarray],
                         A_test: pd.DataFrame,
                         raw_quantiles: Optional[np.ndarray] = None,
                         crossing_note: Optional[str] = None) -> Dict[str, Any]:
    """All per-cell metrics via qrnn_uq_rul.metrics (E3: never recompute by hand)."""
    from qrnn_uq_rul.metrics import (
        calculate_picp, compute_per_unit_metrics, evaluate_point_estimate,
        evaluate_uq_metrics, quantile_crossing_frequency, rul_asymmetric_score,
    )
    y_test = np.asarray(y_test).ravel()
    lower, median, upper = (np.asarray(y_pred[k]).ravel() for k in ("lower", "median", "upper"))

    uq = dict(zip(evaluate_uq_metrics(y_test, y_pred)["Metric"],
                  evaluate_uq_metrics(y_test, y_pred)["Value"]))
    pt = dict(zip(evaluate_point_estimate(y_test, median)["Metric"],
                  evaluate_point_estimate(y_test, median)["Value"]))

    if raw_quantiles is not None:
        crossing = quantile_crossing_frequency(raw_quantiles)
        note = None
    else:
        crossing = 0.0
        note = crossing_note or "monotone by construction"

    # per-engine-unit coverage (M7/A1)
    per_unit_df = compute_per_unit_metrics(A_test, y_test, y_pred)
    per_unit = {
        "units": list(per_unit_df.index),
        "picp": [float(v) for v in per_unit_df["PICP"]],
        "n": [int(v) for v in per_unit_df["n_samples"]],
        "mean_picp": float(np.mean(per_unit_df["PICP"])) if len(per_unit_df) else None,
    }

    # per-RUL-stage coverage (early/mid/late thirds of the RUL range)
    stage = rul_stage(y_test)
    per_stage: Dict[str, Any] = {}
    for code in (0, 1, 2):
        mask = stage == code
        if mask.sum() == 0:
            continue
        per_stage[STAGE_NAMES[code]] = {
            "PICP": float(calculate_picp(y_test[mask], lower[mask], upper[mask])),
            "n": int(mask.sum()),
        }

    return {
        "PICP": float(uq["PICP"]), "NMPIW": float(uq["NMPIW"]),
        "CWC": float(uq["CWC"]), "RS": float(uq["RS"]),
        "Winkler": float(uq["Winkler"]),
        "MAE": float(pt["MAE"]), "RMSE": float(pt["RMSE"]), "R2": float(pt["R²"]),
        "RUL_score": float(rul_asymmetric_score(y_test, median)),
        "crossing_freq": crossing, "crossing_note": note,
        "per_unit": per_unit, "per_stage": per_stage,
    }


def torch_quantiles(model, X: np.ndarray, device) -> np.ndarray:
    """Raw (N, 3) QRNN quantile predictions in QUANTILES order, for the
    crossing-frequency metric (E3: pass the raw quantile matrix, not the
    conformalized interval)."""
    import torch
    model.eval()
    with torch.no_grad():
        out = model(torch.FloatTensor(np.asarray(X, dtype=np.float32)).to(device))
    return out.cpu().numpy()


def measure_efficiency(model: Any, method: str, X_test: np.ndarray,
                       device) -> Dict[str, float]:
    """Footprint (MB) and per-sample inference latency (ms).

    Latency protocol: 10 iterations over a batch of min(1000, n_test) rows,
    averaged per sample — dataset-size independent (documented in
    results_of_record.md).
    """
    import torch
    from qrnn_uq_rul.config import N_SAMPLES
    from qrnn_uq_rul.metrics import get_model_size

    n_batch = min(1000, len(X_test))
    Xb = np.asarray(X_test[:n_batch], dtype=np.float32)

    def one_pass() -> None:
        if method in ("c_qrnn", "c_mdn", "distcp", "edl") or method == "arch_ablation":
            model.eval()
            with torch.no_grad():
                model(torch.FloatTensor(Xb).to(device))
        elif method in ("uacqr", "weighted_cqr", "wqlcp"):
            # weighted_cqr/wqlcp return a single QRNN; uacqr returns an ensemble
            members = getattr(model, "members", [model])
            for m in members:
                m.eval()
                with torch.no_grad():
                    m(torch.FloatTensor(Xb).to(device))
        elif method == "raw_qrnn":
            for m in model:
                m.eval()
                with torch.no_grad():
                    m(torch.FloatTensor(Xb).to(device))
        elif method == "cxgbq":
            for q in (0.1, 0.5, 0.9):
                model[q].predict(Xb)
        elif method == "cgpr":
            model.predict(Xb)

    t0 = time.perf_counter()
    for _ in range(N_SAMPLES):
        one_pass()
    dt = time.perf_counter() - t0
    latency = dt / (N_SAMPLES * n_batch) * 1000.0

    if method == "cxgbq":
        footprint = get_model_size(model, quantile=True)
    elif method == "cgpr":
        footprint = get_model_size(model)
    else:
        members = getattr(model, "members", model) if method in ("uacqr",) else (
            list(model) if method == "raw_qrnn" else [model])
        footprint = sum(get_model_size(m, pytorch=True) for m in members)

    return {"footprint_mb": float(footprint), "latency_ms_per_sample": float(latency)}


def save_pred_npz(path: Path, y_test: np.ndarray, y_pred: Dict[str, np.ndarray],
                  A_test: pd.DataFrame, stage: np.ndarray,
                  extra: Optional[Dict[str, np.ndarray]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "y_true": np.asarray(y_test).ravel(),
        "lower": np.asarray(y_pred["lower"]).ravel(),
        "median": np.asarray(y_pred["median"]).ravel(),
        "upper": np.asarray(y_pred["upper"]).ravel(),
        "units": A_test["unit"].astype(str).to_numpy(dtype=str),
        "stage": stage,
    }
    if extra:
        payload.update(extra)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:  # file handle: np.savez would otherwise append ".npz"
        np.savez_compressed(f, **payload)
    os.replace(tmp, path)

# ── Arm dispatch (E2 method table) ─────────────────────────────────────


def run_arm(method: str, d: Dict[str, Any], X_all: np.ndarray, y_all: np.ndarray,
            X_test: np.ndarray, y_test: np.ndarray, seed: int, args, device
            ) -> Tuple[Any, Any, Dict[str, np.ndarray], Optional[np.ndarray], Dict[str, Any], Optional[str], Optional[Dict[str, Any]]]:
    """Run one method arm. Returns (model, q_hat, y_pred, raw_quantiles,
    diagnostics, crossing_note, precomputed)."""
    from qrnn_uq_rul import models
    from qrnn_uq_rul import models_calibration, models_edl, models_mdn, models_uacqr
    from qrnn_uq_rul import models_variants
    from qrnn_uq_rul.metrics import calculate_picp, calculate_nmpiw, winkler_score

    if method == "c_qrnn":
        model, e_hat, y_pred, _eval, _path, _hist = models.train_test_qrnn_cqr(
            d["W_train"], d["X_s_train"], d["y_train"],
            d["W_test"], d["X_s_test"], d["y_test"],
            device=device, seed=seed)
        raw_q = torch_quantiles(model, X_test, device)
        return (model, float(e_hat), y_pred, raw_q, {}, None, None)

    if method == "c_mdn":
        model, e_hat, y_pred, diag = models_mdn.train_test_mdn_cqr(
            X_all, y_all, X_test, y_test, device=device, seed=seed)
        return (model, float(e_hat), y_pred, None,
                {"mae": diag["mae"]},
                "MDN CDF-inversion quantiles are monotone in tau by construction", None)

    if method == "uacqr":
        ensemble, q_hat, y_pred, diag = models_uacqr.train_test_uacqr(
            X_all, y_all, X_test, y_test, device=device, seed=seed)
        raw_q, _g = ensemble.predict_ensemble(X_test, device=device)
        return (ensemble, float(q_hat), y_pred, raw_q,
                {"mae": diag["mae"], "mean_epistemic": float(diag["mean_epistemic"])}, None, None)

    if method == "distcp":
        model, q_hat, y_pred, diag = models_variants.train_test_distcp(
            X_all, y_all, X_test, y_test, device=device, seed=seed)
        return (model, float(q_hat), y_pred, None,
                {"mae": float(diag["mae"]), "disconnected_frac": float(diag["multimodal_frac"])},
                "MDN density level-set; enclosing interval, no quantile crossing by construction",
                None)

    if method == "weighted_cqr":
        model, q_hat, y_pred, diag = models_variants.train_test_weighted_cqr(
            X_all, y_all, X_test, y_test, device=device, seed=seed)
        raw_q = torch_quantiles(model, X_test, device)
        return (model, float(q_hat), y_pred, raw_q, py(dict(diag)), None, None)

    if method == "wqlcp":
        model, q_hat, y_pred, diag = models_variants.train_test_wqlcp(
            X_all, y_all, X_test, y_test, device=device, seed=seed)
        raw_q = torch_quantiles(model, X_test, device)
        return (model, float(q_hat), y_pred, raw_q, py(dict(diag)), None, None)

    if method == "edl":
        model, e_hat, y_pred, diag = models_edl.train_test_edl(
            X_all, y_all, X_test, y_test, device=device, seed=seed, conformalize=True)
        # model-internal (uncalibrated) diagnostic for the manuscript
        internal = models_edl.edl_predict(model, X_test, device=device)
        internal_diag = {
            "PICP": float(calculate_picp(y_test, internal["lower"], internal["upper"])),
            "NMPIW": float(calculate_nmpiw(y_test, internal["lower"], internal["upper"])),
            "Winkler": float(winkler_score(y_test, internal["lower"], internal["upper"])),
            "MAE": float(np.mean(np.abs(y_test - internal["median"]))),
        }
        return (model, float(e_hat), y_pred, None,
                {"mae": float(diag["mae"]), "e_hat": float(diag["e_hat"]), "internal": internal_diag},
                "Student-t quantiles are monotone by construction", None)

    if method == "cgpr":
        n_pool = len(X_all)
        n_fit = n_pool
        X_fit, y_fit = X_all, y_all
        if n_pool > args.gpr_max_samples:
            rng = np.random.default_rng(seed)
            idx = rng.choice(n_pool, size=args.gpr_max_samples, replace=False)
            X_fit, y_fit = X_all[idx], y_all[idx]
            n_fit = int(idx.size)
        model, e_hat, y_pred, diag = models_calibration.train_test_cgpr(
            X_fit, y_fit, X_test, y_test, seed=seed)
        return (model, float(e_hat), y_pred, None,
                {"mae": float(diag["mae"]), "n_gpr_train": n_fit,
                 "n_gpr_capped": bool(n_pool > args.gpr_max_samples)},
                "Gaussian posterior quantiles are monotone by construction", None)

    if method == "cxgbq":
        if args.xgbq_subprocess == "always":
            return _run_cxgbq_subprocess(d, X_all, y_all, X_test, y_test, seed, args)
        models_dict, e_hat, y_pred, diag = models_calibration.train_test_cxgbq(
            X_all, y_all, X_test, y_test, seed=seed)
        raw_q = np.column_stack([models_dict[q].predict(X_test) for q in (0.1, 0.5, 0.9)])
        for q in (0.1, 0.5, 0.9):
            models_dict[q].set_params(nthread=-1)
        return (models_dict, float(e_hat), y_pred, raw_q,
                {"mae": float(diag["mae"])}, None, None)

    raise ValueError(f"unknown method {method!r}")


def _run_cxgbq_subprocess(d: Dict[str, Any], X_all: np.ndarray, y_all: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray, seed: int,
                          args) -> Tuple[Any, Any, Dict[str, np.ndarray], Optional[np.ndarray], Dict[str, Any], Optional[str], Optional[Dict[str, Any]]]:
    """C-XGBQ in a child process (libomp isolation, coding skill §2.2).

    The worker trains, calibrates, and computes ALL cell scalars (metrics,
    per-unit/per-stage coverage, crossing, efficiency, timings); the parent
    merges the payload and stores the npz artifact.
    """
    tag = f"{d['dataset']}_{d['subset']}_cxgbq_s{seed}"
    out_json = args.workdir / f"{tag}.json"
    cmd = [sys.executable, str(SCRIPT), "--worker-xgbq",
           str(args.cache_dir / f"{d['dataset']}_{d['subset']}_split{SPLIT_SEED}.pkl"),
           str(seed), str(out_json)]
    log.info("C-XGBQ subprocess (seed=%d)", seed)
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if p.returncode != 0:
        raise RuntimeError(f"cxgbq worker exited {p.returncode}: {p.stderr.strip()[-800:]}")
    with open(out_json) as f:
        payload = json.load(f)
    y_pred = {k: np.asarray(v) for k, v in payload.pop("y_pred").items()}
    raw_q = np.asarray(payload.pop("raw_quantiles")) if payload.get("raw_quantiles") is not None else None
    payload.pop("raw_quantiles", None)
    precomputed = payload  # metrics, efficiency, time_s, n_* counts
    return None, float(precomputed["q_hat"]), y_pred, raw_q, {}, None, precomputed


def run_raw_qrnn(d: Dict[str, Any], X_all: np.ndarray, y_all: np.ndarray,
                 X_test: np.ndarray, y_test: np.ndarray, seeds: List[int],
                 args, device) -> Tuple[List[Any], Dict[str, np.ndarray], Dict[str, Any]]:
    """Raw QRNN multi-seed (fix M4): one QRNN per seed, ensemble-mean raw
    quantiles, NO conformal step. Members are checkpointed to side files so a
    crash never loses completed members. The train/val split is shared across
    members (SPLIT_SEED), so only the initialization seeds differ."""
    import torch
    from qrnn_uq_rul.models import train_qrnn
    from qrnn_uq_rul.models_calibration import raw_qrnn_predict

    X_tr, y_tr, X_val, y_val, _X_cal, _y_cal = internal_split(X_all, y_all, seed=SPLIT_SEED)
    member_dir = args.cache_dir / "rawq_ens" / f"{d['dataset']}_{d['subset']}"
    member_dir.mkdir(parents=True, exist_ok=True)

    # collect/refresh member raw quantiles; state_dicts are checkpointed too
    # so a resume reloads members without retraining (coding skill §9).
    from qrnn_uq_rul.models import QRNN
    models = []
    for s in seeds:
        mf = member_dir / f"member_s{s}.npz"
        sf = member_dir / f"member_s{s}.pth"
        if mf.exists() and sf.exists():
            with np.load(mf) as z:
                q = np.stack([z["lower"], z["median"], z["upper"]], axis=1)
                input_dim = int(z["input_dim"])
            model = QRNN(input_dim).to(device)
            model.load_state_dict(torch.load(sf, map_location=device))
            model.eval()
            models.append(model)
            log.info("raw_qrnn member seed=%d loaded from cache", s)
            continue
        t0 = time.time()
        model, _, _ = train_qrnn(X_tr, y_tr, X_val, y_val, device=device, seed=s)
        q = torch_quantiles(model, X_test, device)
        torch.save(model.state_dict(), sf)
        np.savez_compressed(mf, lower=q[:, 0], median=q[:, 1], upper=q[:, 2],
                            input_dim=np.int64(X_tr.shape[1]))
        models.append(model)
        log.info("raw_qrnn member seed=%d trained in %.1f s", s, time.time() - t0)
        if device.type == "mps":
            torch.mps.empty_cache()

    y_pred = raw_qrnn_predict(models, X_test, device=device)
    raw_q = np.column_stack([y_pred[k] for k in ("lower", "median", "upper")])
    return models, y_pred, {"n_members": len(seeds), "member_seeds": list(seeds)}


def run_arch_ablation(cfg: Dict[str, Any], X_all: np.ndarray, y_all: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray, device) -> Tuple[Any, float, Dict[str, np.ndarray], np.ndarray, Dict[str, Any]]:
    """E4: C-QRNN pipeline with an alternative QRNN architecture (one factor
    at a time over the base config), using the same public building blocks
    and split convention as the c_qrnn arm."""
    from qrnn_uq_rul import models
    X_tr, y_tr, X_val, y_val, X_cal, y_cal = internal_split(X_all, y_all, seed=SPLIT_SEED)
    model, _, _ = models.train_qrnn(
        X_tr, y_tr, X_val, y_val, device=device, seed=ABLATION_SEED,
        hidden_dims=tuple(cfg["hidden_dims"]), dropouts=tuple(cfg["dropouts"]))
    scores = models.compute_nonconformity_scores(model, X_cal, y_cal, device)
    e_hat = models.calibrate_cqr(scores)
    raw_q = torch_quantiles(model, X_test, device)
    y_pred = models.predict_cqr(model, X_test, e_hat, device)
    return model, float(e_hat), y_pred, raw_q, {"config_id": cfg["id"]}

# ── Cell construction ──────────────────────────────────────────────────


def build_cell(job: Dict[str, Any], d: Dict[str, Any], X_test: np.ndarray,
               y_test: np.ndarray, device, args, arm_result, time_s: float,
               n_train: int, n_val: Optional[int], n_cal: Optional[int]) -> Dict[str, Any]:
    """Assemble the JSON-safe cell record (spec E1 field list)."""
    (model, q_hat, y_pred, raw_q, diagnostics, crossing_note, precomputed) = arm_result
    A_test = d["A_test"]

    if precomputed is not None:
        # cxgbq-subprocess: worker already computed every scalar
        m = {k: precomputed[k] for k in ("PICP", "NMPIW", "CWC", "RS", "Winkler",
                                         "MAE", "RMSE", "R2", "RUL_score",
                                         "crossing_freq", "crossing_note",
                                         "per_unit", "per_stage")}
        efficiency = precomputed["efficiency"]
        diagnostics = {**diagnostics, "worker": True}
    else:
        m = compute_cell_metrics(y_test, y_pred, A_test, raw_q, crossing_note)
        try:
            efficiency = measure_efficiency(model, job["method"], X_test, device)
        except Exception as e:  # efficiency is secondary; never fail the cell for it
            log.warning("efficiency measurement failed for %s: %s", job_id(job), e)
            efficiency = {"footprint_mb": None, "latency_ms_per_sample": None}

    finite = all(np.isfinite(y_pred[k]).all() for k in ("lower", "median", "upper"))
    cell: Dict[str, Any] = {
        "dataset": job["dataset"], "subset": job["subset"], "method": job["method"],
        "seed": job.get("seed"), "status": "OK" if finite else "FAIL",
        "reason": None if finite else "non-finite prediction intervals",
        "q_hat": q_hat,
        **m,
        "diagnostics": py(diagnostics),
        "efficiency": py(efficiency),
        "time_s": round(float(time_s), 2),
        "n_train": n_train, "n_val": n_val, "n_cal": n_cal,
        "n_test": int(len(y_test)),
        "n_units_train": int(d["A_train"]["unit"].nunique()),
        "n_units_test": int(A_test["unit"].nunique()),
    }
    if job.get("method") == "arch_ablation":
        cell["config"] = job["config"]
    if not finite:
        for k in ("PICP", "NMPIW", "CWC", "RS", "Winkler", "MAE", "RMSE", "R2", "RUL_score", "crossing_freq"):
            cell[k] = None
        return cell

    pred_name = (f"{job['dataset']}_{job['subset']}_{job['method']}"
                 + (f"_{job['config']['id']}" if job.get("method") == "arch_ablation" else "")
                 + (f"_s{job['seed']}" if job.get("seed") is not None
                    else ("_ens" if job["method"] == "raw_qrnn" else "_det"))
                 + ".npz")
    pred_path = args.pred_dir / pred_name
    save_pred_npz(pred_path, y_test, y_pred, A_test, rul_stage(y_test))
    cell["pred_path"] = str(pred_path.relative_to(args.results_dir))
    return py(cell)


# ── XGBQ subprocess worker ─────────────────────────────────────────────


def worker_xgbq(cache_pkl: str, seed: int, out_json: str) -> None:
    """Child-process entry (libomp isolation). Sets OMP limits BEFORE any
    xgboost/torch import; trains C-XGBQ, computes every cell scalar, writes
    a JSON payload the parent merges."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    t0 = time.time()
    with open(cache_pkl, "rb") as f:
        d = pickle.load(f)
    X_all, y_all, X_test, y_test = assemble_features(d)
    from qrnn_uq_rul.config import CQR_CALIBRATION_SPLIT
    from qrnn_uq_rul.models_calibration import train_test_cxgbq
    models_dict, e_hat, y_pred, diag = train_test_cxgbq(
        X_all, y_all, X_test, y_test, seed=seed)
    raw_q = np.column_stack([models_dict[q].predict(X_test) for q in (0.1, 0.5, 0.9)])
    for q in (0.1, 0.5, 0.9):
        models_dict[q].set_params(nthread=-1)  # latency measured at full threads
    n_train, n_val, n_cal = internal_split_counts(len(X_all), CQR_CALIBRATION_SPLIT)
    m = compute_cell_metrics(y_test, y_pred, d["A_test"], raw_q, None)
    efficiency = measure_efficiency(models_dict, "cxgbq", X_test, None)
    atomic_json_write(Path(out_json), py({
        "q_hat": float(e_hat),
        "y_pred": {k: y_pred[k] for k in ("lower", "median", "upper")},
        "raw_quantiles": raw_q,
        **m,
        "efficiency": efficiency,
        "time_s": round(time.time() - t0, 2),
        "n_train": n_train, "n_val": n_val, "n_cal": n_cal,
        "n_test": int(len(y_test)),
        "n_units_train": int(d["A_train"]["unit"].nunique()),
        "n_units_test": int(d["A_test"]["unit"].nunique()),
        "mae": float(diag["mae"]),
    }))
    print(json.dumps({"ok": True, "seed": seed, "time_s": round(time.time() - t0, 2)}))

# ── Dry-run budget table ───────────────────────────────────────────────


def print_dry_run(manifest: Dict[str, Any]) -> None:
    from collections import Counter, defaultdict
    by_method: Counter = Counter()
    by_ds_sub: Counter = Counter()
    est_total = 0.0
    for j in manifest["jobs"]:
        by_method[j["method"]] += 1
        by_ds_sub[(j["dataset"], j["subset"])] += 1
        est_total += EST_S[j["method"]]
    lines = [
        "DRY RUN — planned job matrix (no training will start)",
        f"  total jobs: {len(manifest['jobs'])}   seeds: {manifest['seeds']}"
        f"   split_seed: {manifest['split_seed']}   gpr_max_samples: {manifest['gpr_max_samples']}",
        "",
        f"  {'dataset/subset':<18}{'cells':>7}{'est (h)':>10}",
    ]
    est_by_ds_sub: Dict[Tuple[str, str], float] = defaultdict(float)
    for j in manifest["jobs"]:
        est_by_ds_sub[(j["dataset"], j["subset"])] += EST_S[j["method"]]
    for (ds, sub), n in sorted(by_ds_sub.items()):
        lines.append(f"  {ds + '/' + sub:<18}{n:>7}{est_by_ds_sub[(ds, sub)] / 3600:>10.1f}")
    lines += ["", f"  {'method':<18}{'cells':>7}{'est (h)':>10}"]
    for m, n in sorted(by_method.items()):
        lines.append(f"  {m:<18}{n:>7}{n * EST_S[m] / 3600:>10.1f}")
    lines += ["", f"  GRAND TOTAL (estimate, GPR at worst-case cap): {est_total / 3600:.1f} h"]
    lines.append("  (estimates are static Phase-2/3 measurements; actuals land in results_of_record.md)")
    log.info("\n".join(lines))


# ── Main sweep ─────────────────────────────────────────────────────────


def run_sweep(args, manifest: Dict[str, Any], results: Dict[str, Any],
              existing: Dict[str, Dict[str, Any]], device) -> Dict[str, Any]:
    import torch
    from qrnn_uq_rul.config import CQR_CALIBRATION_SPLIT, SEED_LIST

    missing = [j for j in manifest["jobs"]
               if _needs_run(j, existing, args.skip_failed)]
    log.info("%d of %d manifest jobs need running (resume=%s)",
             len(missing), len(manifest["jobs"]), args.resume)
    if not missing:
        return results

    data: Dict[Tuple[str, str], Dict[str, Any]] = {}
    done = 0
    for job in missing:
        done += 1
        key = (job["dataset"], job["subset"])
        tag = job_id(job)
        if key not in data:
            d = load_subset(job["dataset"], job["subset"], args.cache_dir)
            X_all, y_all, X_test, y_test = assemble_features(d)
            data[key] = (d, X_all, y_all, X_test, y_test)
        d, X_all, y_all, X_test, y_test = data[key]

        log.info("[%d/%d] %s/%s/%s seed=%s starting…", done, len(missing),
                 job["dataset"], job["subset"], job["method"], job.get("seed"))
        t0 = time.time()
        try:
            if job["method"] == "raw_qrnn":
                models_list, y_pred, diag = run_raw_qrnn(
                    d, X_all, y_all, X_test, y_test,
                    job.get("ensemble_seeds") or list(SEED_LIST), args, device)
                n_train, n_val, _ = internal_split_counts(len(X_all), CQR_CALIBRATION_SPLIT)
                cell = build_cell(job, d, X_test, y_test, device, args,
                                  (models_list, None, y_pred,
                                   np.column_stack([y_pred[k] for k in ("lower", "median", "upper")]),
                                   diag, "ensemble-mean of member raw quantiles", None),
                                  time.time() - t0, n_train, n_val, None)
            elif job["method"] == "arch_ablation":
                cfg = next(c for c in ABLATION_CONFIGS if c["id"] == job["config"]["id"])
                model, e_hat, y_pred, raw_q, diag = run_arch_ablation(
                    cfg, X_all, y_all, X_test, y_test, device)
                n_train, n_val, n_cal = internal_split_counts(len(X_all), CQR_CALIBRATION_SPLIT)
                cell = build_cell(job, d, X_test, y_test, device, args,
                                  (model, e_hat, y_pred, raw_q, diag, None, None),
                                  time.time() - t0, n_train, n_val, n_cal)
            else:
                # cgpr is deterministic (E2): pin its split/subsample/optimizer
                # RNG to SPLIT_SEED (job seed is None for single-row methods)
                seed = job["seed"] if job["seed"] is not None else SPLIT_SEED
                arm = run_arm(job["method"], d, X_all, y_all, X_test, y_test,
                              seed, args, device)
                pre = arm[6]
                if pre is not None:  # cxgbq subprocess: worker owns the counts
                    n_train, n_val, n_cal = pre["n_train"], pre["n_val"], pre["n_cal"]
                    time_s = pre["time_s"]
                else:
                    n_train, n_val, n_cal = internal_split_counts(len(X_all), CQR_CALIBRATION_SPLIT)
                    if job["method"] == "cgpr":
                        n_train = min(n_train, args.gpr_max_samples) if len(X_all) > args.gpr_max_samples else n_train
                        n_val = None  # GPR uses all train+val for the fit
                    time_s = time.time() - t0
                cell = build_cell(job, d, X_test, y_test, device, args, arm,
                                  time_s, n_train, n_val, n_cal)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            log.error("cell %s FAILED: %s", tag, e)
            tb = traceback.format_exc(limit=5).splitlines()
            cell = {"dataset": job["dataset"], "subset": job["subset"],
                    "method": job["method"], "seed": job.get("seed"),
                    "status": "FAIL", "reason": f"{type(e).__name__}: {e}",
                    "traceback_tail": tb[-5:], "time_s": round(time.time() - t0, 2)}
            if job.get("method") == "arch_ablation":
                cell["config"] = job["config"]

        _place_cell(results, job, cell)
        atomic_json_write(args.out, results)
        if cell["status"] == "OK":
            log.info("[%d/%d] %s OK  PICP=%.4f Winkler=%.3f (%.1f s)",
                     done, len(missing), tag, cell["PICP"], cell["Winkler"], cell["time_s"])
        else:
            log.info("[%d/%d] %s FAIL: %s", done, len(missing), tag, cell["reason"])
        if job["method"] in ("c_qrnn", "c_mdn", "uacqr", "distcp", "weighted_cqr",
                             "wqlcp", "edl", "raw_qrnn", "arch_ablation") and device.type == "mps":
            torch.mps.empty_cache()
    return results


def _needs_run(job: Dict[str, Any], existing: Dict[str, Dict[str, Any]],
               skip_failed: bool) -> Optional[Dict[str, Any]]:
    """Return the job if it still needs running (absent, or FAIL not skipped)."""
    c = existing.get(job_id(job))
    if c is None:
        return job
    if c.get("status") != "OK" and not skip_failed:
        return job
    return None


def _place_cell(results: Dict[str, Any], job: Dict[str, Any], cell: Dict[str, Any]) -> None:
    if job["method"] == "arch_ablation":
        node = results.setdefault("arch_ablation", {}).setdefault(
            job["subset"], {}).setdefault(job["config"]["id"], [])
    else:
        node = results.setdefault(job["dataset"], {}).setdefault(
            job["subset"], {}).setdefault(job["method"], [])
    node[:] = [c for c in node if c.get("seed") != cell.get("seed")] + [cell]


# ── CLI ────────────────────────────────────────────────────────────────


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Phase-4 multi-seed multi-dataset experiment runner")
    ap.add_argument("--datasets", choices=["ncmapss", "cmapss", "both"], default="both")
    ap.add_argument("--subsets", default="", help="csv, e.g. 'ds01,ds02,fd001' (default: all)")
    ap.add_argument("--seeds", default="", help="csv (default: config SEED_LIST)")
    ap.add_argument("--methods", default="all", help="all or csv of method names")
    ap.add_argument("--arch-ablation", action="store_true", help="include the E4 architecture sweep")
    ap.add_argument("--out", default=str(RESULTS_DIR / "experiment_results.json"))
    ap.add_argument("--matrix", default=str(RESULTS_DIR / "run_matrix.json"))
    ap.add_argument("--cache-dir", default=str(RESULTS_DIR / "cache"))
    ap.add_argument("--pred-dir", default=str(RESULTS_DIR / "pred"))
    ap.add_argument("--workdir", default=str(RESULTS_DIR / "cache" / "work"))
    ap.add_argument("--log", default=str(RESULTS_DIR / "logs" / "phase4_run.log"))
    ap.add_argument("--resume", action="store_true", help="skip cells already OK (default: run everything in the manifest)")
    ap.add_argument("--skip-failed", action="store_true", help="also skip cells recorded as FAIL")
    ap.add_argument("--dry-run", action="store_true", help="write the manifest, print the budget table, exit")
    ap.add_argument("--xgbq-subprocess", choices=["always", "never"], default="always")
    ap.add_argument("--gpr-max-samples", type=int, default=5248,
                    help="GPR training-sample cap (Phase-2 measured budget at 5248 ≈ 46 min)")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    ap.add_argument("--verbose", action="store_true")
    # hidden worker mode (libomp isolation for C-XGBQ)
    ap.add_argument("--worker-xgbq", nargs=3, metavar=("CACHE_PKL", "SEED", "OUT_JSON"),
                    help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.worker_xgbq:
        cache_pkl, seed, out_json = args.worker_xgbq
        worker_xgbq(cache_pkl, int(seed), out_json)
        return 0

    setup_logging(Path(args.log), args.verbose)

    from qrnn_uq_rul.config import CMAPSS_SUBSETS, FILENAMES, SEED_LIST
    import qrnn_uq_rul.models as _models  # noqa: F401  (device init before sweep)

    if args.device == "auto":
        from qrnn_uq_rul.utils import get_device
        device = get_device()
    else:
        import torch
        device = torch.device(args.device)

    # dataset/subset selection
    datasets = ["ncmapss", "cmapss"] if args.datasets == "both" else [args.datasets]
    want = {s.upper() for s in args.subsets.split(",") if s.strip()}
    subsets: Dict[str, List[str]] = {}
    for ds in datasets:
        all_subs = list(FILENAMES) if ds == "ncmapss" else list(CMAPSS_SUBSETS)
        subsets[ds] = [s for s in all_subs if not want or s in want]
        if not subsets[ds]:
            raise SystemExit(f"no valid subsets for {ds} in {sorted(want)}")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()] or list(SEED_LIST)
    if len(seeds) < 5:
        raise SystemExit(f"need >= 5 seeds (acceptance criterion), got {len(seeds)}")

    if args.methods == "all":
        methods = list(ALL_METHODS)
    else:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        bad = [m for m in methods if m not in ALL_METHODS]
        if bad:
            raise SystemExit(f"unknown methods: {bad}")

    args.out = Path(args.out); args.matrix = Path(args.matrix)
    args.cache_dir = Path(args.cache_dir); args.pred_dir = Path(args.pred_dir)
    args.workdir = Path(args.workdir); args.results_dir = RESULTS_DIR
    for pth in (args.out.parent, args.cache_dir, args.pred_dir, args.workdir):
        pth.mkdir(parents=True, exist_ok=True)

    # manifest (E1a): rewritten on every invocation
    manifest = build_manifest(datasets, subsets, methods, seeds,
                              args.gpr_max_samples, args.arch_ablation)
    atomic_json_write(args.matrix, manifest)
    log.info("run matrix written: %s (%d jobs)", args.matrix, len(manifest["jobs"]))

    if args.dry_run:
        print_dry_run(manifest)
        return 0

    # results container
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        with open(args.out) as f:
            results = json.load(f)
        log.info("loaded existing results: %s", args.out)
    else:
        try:
            import subprocess as _sp
            git = _sp.run(["git", "-C", str(SRC_ROOT), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
        except Exception:
            git = "unknown"
        results = {"meta": {
            "schema_version": 1,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "split_seed": SPLIT_SEED, "seeds": seeds,
            "nominal_alpha": 0.10, "cal_fraction": 0.80,
            "gpr_max_samples": args.gpr_max_samples, "device": str(device),
            "git_commit": git, "python": platform.python_version(),
            "host": platform.node(),
            "ablation_note": ("E4 grid adapted to the shipped QRNN base "
                              "(128/64/32 taper, dropouts 0.3/0.2/0.1): width {32,64,128}, "
                              "depth {1,2,3}, dropout {0.0,0.2,0.5}. The spec's {8,16,32} "
                              "grid assumed a base-16 default; models.py shows 128/64/32."),
        }, "ncmapss": {}, "cmapss": {}}
    results["meta"]["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    existing = index_cells(results)
    try:
        results = run_sweep(args, manifest, results, existing, device)
    except KeyboardInterrupt:
        results["meta"]["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        atomic_json_write(args.out, results)
        log.error("interrupted — checkpoint saved. Resume with: --resume")
        return 130

    results["meta"]["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    atomic_json_write(args.out, results)

    # planned-vs-done summary (E1a)
    existing = index_cells(results)
    n_ok = sum(1 for j in manifest["jobs"] if existing.get(job_id(j), {}).get("status") == "OK")
    n_fail = sum(1 for j in manifest["jobs"] if existing.get(job_id(j), {}).get("status") == "FAIL")
    n_missing = len(manifest["jobs"]) - n_ok - n_fail
    log.info("DONE: %d OK, %d FAIL, %d missing of %d planned jobs",
             n_ok, n_fail, n_missing, len(manifest["jobs"]))
    return 0 if n_missing == 0 else 2


if __name__ == "__main__":
    sys.exit(main())

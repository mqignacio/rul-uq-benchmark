#!/usr/bin/env python3
"""Phase 4 analysis toolkit: E5 verification, E6 tables, E7 prediction cross-check.

Reads the Phase-4 sweep outputs (results/experiment_results.json,
results/pred/*.npz, results/run_matrix.json) and emits:

  tables       aggregate / per-seed / per-subset / per-unit / per-stage /
               crossing / shift / ablation tables (markdown)
  hypotheses   H1-H4 test results with every statistic shown (design sec.7)
  prediction   P1-P11 observed best + HIT/MISS (tie-aware P2/P6) + hit rate
               with binomial CI; --fill-xlsx writes the XLSX working copy
               from a human-reviewed JSON spec (no text is written to the
               adviser-facing tracker without review)
  verify       E5 numerical verification: planned==done manifest diff,
               FAIL-cell inventory, and recompute of every interval metric
               from the raw npz artifacts vs. the stored JSON values

Population definitions (documented; used consistently in all outputs):
  SHIFTED    = N-CMAPSS pooled. The documented train->test distribution shift:
               group-aware unit split with the dataset's built-in dev/test
               partition and per-DS operating profiles (data_provenance
               sec.2.2, design sec.8: "N-CMAPSS train->test shift violates
               exchangeability").
  TIE-REGIME = both datasets pooled; per-unit, seed-averaged. The regime in
               which the split-CP construction makes the four unweighted
               split-CP arms statistically indistinguishable (P2/P6).
  PRIMARY    = N-CMAPSS (9 subsets) for the single-winner rows P3-P11;
               C-MAPSS is reported as a secondary validation view.
  Pooled PICP/RS = sample-weighted across cells (sum covered / sum n_test) --
               exact linear pooling for coverage metrics.
  Seed aggregates = unweighted mean +/- std (ddof=1) across seeds, matching
               the prior results-of-record house style.
  Per-unit pairing: unit strings carry the subset prefix (e.g. "DS01_2",
               "FD001_17") so they are unique across subsets; per-unit values
               are seed-averaged before cross-method pairing.
  Records are selected by unique key (dataset, subset, method, seed,
               config-id) -- never by list position (append/replacement
               checkpointing makes list order non-contractual).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from qrnn_uq_rul import config as cfg  # noqa: E402
from qrnn_uq_rul import metrics as M  # noqa: E402

RESULTS_DIR = SRC_ROOT / "results"
PRED_DIR = RESULTS_DIR / "pred"
NOMINAL = 0.90  # 1 - alpha, alpha = 0.10 (config)
TOST_MARGIN = cfg.TOST_EQUIV_MARGIN  # 0.03 (design H1 / F12)

METHODS = ["c_qrnn", "c_mdn", "uacqr", "distcp", "weighted_cqr", "wqlcp",
           "edl", "cgpr", "cxgbq", "raw_qrnn"]
SPLIT_CP_4 = ["c_qrnn", "c_mdn", "uacqr", "distcp"]
WEIGHTED_FAMILY = ["weighted_cqr", "wqlcp"]
CALIBRATED_BASELINES = ["cxgbq", "cgpr"]

METHOD_LABEL = {
    "c_qrnn": "C-QRNN", "c_mdn": "C-MDN series", "uacqr": "UACQR-S-style",
    "distcp": "DistCP/HPD-split", "weighted_cqr": "Weighted-CP CQR",
    "wqlcp": "WQLCP-style", "edl": "EDL", "cgpr": "C-GPR",
    "cxgbq": "C-XGBQ", "raw_qrnn": "Raw QRNN",
}

# cell-level interval metrics (lower is better unless noted)
METRIC_KEYS = ["PICP", "RS", "NMPIW", "CWC", "Winkler", "MAE", "RMSE",
               "R2", "RUL_score", "crossing_freq"]
EFF_KEYS = ["footprint_mb", "latency_ms_per_sample"]
LOWER_BETTER = {"RS", "NMPIW", "CWC", "Winkler", "MAE", "RMSE", "RUL_score",
                "crossing_freq", "footprint_mb", "latency_ms_per_sample"}
DATASETS = ("ncmapss", "cmapss")


# ── Loading & indexing ─────────────────────────────────────────────────────

def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def iter_cells(results: dict):
    """Yield (dataset, subset, method, seed, cell) over both datasets plus the
    top-level arch_ablation section (nested as arch_ablation[subset][cfg_id]).
    Ablation cells carry their own dataset/method fields; those take precedence
    so the manifest diff and Winkler patch cover them."""
    for ds in DATASETS:
        for subset, methods in results.get(ds, {}).items():
            for method, cells in methods.items():
                for cell in cells:
                    yield ds, subset, method, cell.get("seed"), cell
    for subset, cfgs in results.get("arch_ablation", {}).items():
        for cfg_id, cells in cfgs.items():
            for cell in cells:
                yield cell.get("dataset", "ncmapss"), subset, \
                    cell.get("method", "arch_ablation"), cell.get("seed"), cell


def ok_cells(results: dict, dataset=None, method=None, subset=None):
    for ds, sub, m, seed, cell in iter_cells(results):
        if dataset is not None and ds != dataset:
            continue
        if method is not None and m != method:
            continue
        if subset is not None and sub != subset:
            continue
        if cell.get("status") == "OK":
            yield ds, sub, m, seed, cell


def pooled_coverage(cells, key: str = "PICP") -> float:
    """Sample-weighted pool of a coverage metric across cells (exact)."""
    num = sum(c[key] * c["n_test"] for c in cells)
    den = sum(c["n_test"] for c in cells)
    return num / den if den else float("nan")


def cell_mean(cells, key: str) -> tuple[float, float, int]:
    """Unweighted mean, std (ddof=1), n over cells (seed-level aggregate)."""
    vals = [c[key] for c in cells if c.get(key) is not None]
    if not vals:
        return float("nan"), float("nan"), 0
    return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0), len(vals)


def fmt(v: float, nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v:.{nd}f}"


def mpm(v: float, s: float, n: int, nd: int = 4) -> str:
    """mean +/- std (house style); bare value for single-row methods."""
    if n == 0:
        return "-"
    if n == 1 or s == 0.0:
        return fmt(v, nd)
    return f"{v:.{nd}f} \u00b1 {s:.{nd}f}"


# ── Table builders (E6) ────────────────────────────────────────────────────

def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def t_aggregate(results: dict) -> str:
    """Per method x dataset: mean +/- std across seeds (house style)."""
    lines = ["## Aggregate (mean +/- std across seeds, per method x dataset)", ""]
    for ds in DATASETS:
        lines += [f"### {ds}", ""]
        headers = ["Method"] + METRIC_KEYS + EFF_KEYS
        rows = []
        for m in METHODS:
            cells = [c for d, s, mm, sd, c in ok_cells(results, dataset=ds, method=m)]
            if not cells:
                rows.append([METHOD_LABEL[m]] + ["-" for _ in headers[1:]])
                continue
            row = [METHOD_LABEL[m]]
            for k in METRIC_KEYS:
                v, s, n = cell_mean(cells, k)
                row.append(mpm(v, s, n))
            for ek in EFF_KEYS:
                vals = [c["efficiency"][ek] for c in cells
                        if isinstance(c.get("efficiency"), dict)]
                v, s, n = (float(np.mean(vals)),
                           float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                           len(vals)) if vals else (float("nan"), 0.0, 0)
                row.append(mpm(v, s, n, nd=3))
            rows.append(row)
        lines.append(_md_table(headers, rows))
        lines.append("")
    lines.append("Note: `cgpr` (deterministic GPR) and `raw_qrnn` (5-member "
                 "ensemble) are single rows per subset, so their stds are "
                 "over the 9/4 subsets, not seeds.")
    return "\n".join(lines)


def t_per_seed(results: dict) -> str:
    lines = ["## Per-seed appendix", ""]
    for ds in DATASETS:
        for m in METHODS:
            cells = sorted((c for d, s, mm, sd, c in ok_cells(results, dataset=ds, method=m)),
                           key=lambda c: (c.get("seed") is None, c.get("seed")))
            if not cells:
                continue
            lines += [f"### {ds} / {METHOD_LABEL[m]}", ""]
            headers = ["seed"] + METRIC_KEYS
            rows = [[str(c.get("seed"))] + [fmt(c.get(k)) for k in METRIC_KEYS]
                    for c in cells]
            lines += [_md_table(headers, rows), ""]
    return "\n".join(lines)


def t_per_subset(results: dict) -> str:
    """Per subset x method: seed-mean metrics + |PICP-0.90| (shift view)."""
    lines = ["## Per-subset coverage (seed-mean)", ""]
    for ds in DATASETS:
        lines += [f"### {ds}", ""]
        subsets = sorted({s for d, s, mm, sd, c in ok_cells(results, dataset=ds)})
        headers = ["subset"] + [METHOD_LABEL[m] for m in METHODS]
        sub_rows, dev_rows = [], []
        for sub in subsets:
            row_s, row_d = [sub], []
            for m in METHODS:
                cells = [c for d, s, mm, sd, c in ok_cells(results, dataset=ds,
                                                           method=m, subset=sub)]
                if not cells:
                    row_s.append("-"); row_d.append("-")
                    continue
                v, s, n = cell_mean(cells, "PICP")
                row_s.append(fmt(v))
                row_d.append(fmt(abs(v - NOMINAL), nd=3))
            sub_rows.append(row_s)
            dev_rows.append([sub] + row_d)
        lines += ["PICP (seed-mean):", _md_table(headers, sub_rows), "",
                  "|PICP - 0.90| (seed-mean):", _md_table(headers, dev_rows), ""]
    return "\n".join(lines)


def t_per_unit(results: dict) -> str:
    """Per-engine-unit coverage: unit-level PICP, seed-averaged, per method."""
    lines = ["## Per-engine-unit coverage (seed-averaged PICP per unit)", ""]
    for ds in DATASETS:
        lines += [f"### {ds}", ""]
        # unit -> {method -> [per-unit PICP across seeds]}
        units: dict[str, dict[str, list[float]]] = {}
        unit_n: dict[str, int] = {}
        for d, sub, m, sd, c in ok_cells(results, dataset=ds):
            pu = c.get("per_unit") or {}
            for u, p, n in zip(pu.get("units", []), pu.get("picp", []), pu.get("n", [])):
                units.setdefault(u, {}).setdefault(m, []).append(p)
                unit_n[u] = n
        if not units:
            lines.append("- (no data)")
            continue
        headers = ["unit", "n_samples"] + [METHOD_LABEL[m] for m in METHODS]
        rows = []
        for u in sorted(units):
            row = [u, str(unit_n[u])]
            for m in METHODS:
                vals = units[u].get(m, [])
                row.append(fmt(np.mean(vals)) if vals else "-")
            rows.append(row)
        lines += [_md_table(headers, rows), ""]
    return "\n".join(lines)


def t_per_stage(results: dict) -> str:
    """Per-RUL-stage coverage: n-weighted PICP across cells (seeds x subsets)."""
    lines = ["## Per-RUL-stage coverage (n-weighted PICP)", ""]
    for ds in DATASETS:
        lines += [f"### {ds}", ""]
        headers = ["stage"] + [METHOD_LABEL[m] for m in METHODS]
        rows = []
        for stage in ("early", "mid", "late"):
            row = [stage]
            for m in METHODS:
                cells = [c for d, s, mm, sd, c in ok_cells(results, dataset=ds, method=m)]
                num = den = 0.0
                for c in cells:
                    st = (c.get("per_stage") or {}).get(stage)
                    if st:
                        num += st["PICP"] * st["n"]; den += st["n"]
                row.append(fmt(num / den) if den else "-")
            rows.append(row)
        lines += [_md_table(headers, rows), ""]
    return "\n".join(lines)


def t_crossing(results: dict) -> str:
    lines = ["## Quantile-crossing frequency (seed-mean)", ""]
    headers = ["dataset"] + [METHOD_LABEL[m] for m in METHODS]
    rows = []
    for ds in DATASETS:
        row = [ds]
        for m in METHODS:
            cells = [c for d, s, mm, sd, c in ok_cells(results, dataset=ds, method=m)]
            v, s, n = cell_mean(cells, "crossing_freq")
            row.append(mpm(v, s, n, nd=3))
        rows.append(row)
    lines += [_md_table(headers, rows), "",
              "Note: for MDN/EDL analytic-quantile arms the crossing is recorded "
              "with a construction note (see `crossing_note` in the JSON)."]
    return "\n".join(lines)


def t_shift(results: dict) -> str:
    """Conditional coverage under the documented train->test shift (M2/A2).

    Per subset x method: seed-mean PICP and |PICP-0.90|, plus weighted-CP
    diagnostics (ESS fraction, clip rate, collapse/fallback) so the report can
    say where each method deteriorates and why.
    """
    lines = ["## Conditional coverage under documented train->test shift", "",
             "SHIFTED population = N-CMAPSS (group-aware split; per-DS operating "
             "profiles). Weighted-CP diagnostics from the arm's own weight "
             "estimator (Tibshirani et al. 2020; estimated weights -> "
             "approximate guarantee, design sec.8 / F7).", ""]
    lines += ["### N-CMAPSS: per-subset |PICP-0.90| (seed-mean) -- smaller is better", ""]
    subsets = sorted({s for d, s, mm, sd, c in ok_cells(results, dataset="ncmapss")})
    headers = ["subset"] + [METHOD_LABEL[m] for m in METHODS]
    rows = []
    for sub in subsets:
        row = [sub]
        for m in METHODS:
            cells = [c for d, s, mm, sd, c in ok_cells(results, dataset="ncmapss",
                                                       method=m, subset=sub)]
            if not cells:
                row.append("-"); continue
            v, s, n = cell_mean(cells, "PICP")
            row.append(fmt(abs(v - NOMINAL), nd=3))
        rows.append(row)
    lines += [_md_table(headers, rows), ""]
    # weighted diagnostics
    lines += ["### Weighted-CP diagnostics (seed-mean per subset)", ""]
    headers = ["subset"] + [f"{METHOD_LABEL[m]}: ESS/n | clip | collapsed/fallback"
                            for m in WEIGHTED_FAMILY]
    rows = []
    for sub in subsets:
        row = [sub]
        for m in WEIGHTED_FAMILY:
            cells = [c for d, s, mm, sd, c in ok_cells(results, dataset="ncmapss",
                                                       method=m, subset=sub)]
            diags = [c.get("diagnostics") or {} for c in cells
                     if isinstance(c.get("diagnostics"), dict)
                     and "ess_fraction" in c.get("diagnostics", {})]
            if not diags:
                row.append("-"); continue
            ess = float(np.mean([d["ess_fraction"] for d in diags]))
            clip = float(np.mean([d.get("clip_rate", 0.0) for d in diags]))
            nf = int(sum(1 for d in diags if d.get("collapsed") or d.get("fell_back")))
            row.append(f"{ess:.3f} | {clip:.3f} | {nf}/{len(diags)}")
        rows.append(row)
    lines += [_md_table(headers, rows), ""]
    # pooled coverage of weighted vs plain split-CP on the shifted set
    lines += ["### Pooled (sample-weighted) N-CMAPSS coverage: shift-validity gap", ""]
    headers = ["method", "pooled PICP", "|PICP-0.90|"]
    rows = []
    for m in METHODS:
        cells = [c for d, s, mm, sd, c in ok_cells(results, dataset="ncmapss", method=m)]
        if cells:
            p = pooled_coverage(cells)
            rows.append([METHOD_LABEL[m], fmt(p), fmt(abs(p - NOMINAL), nd=3)])
    lines += [_md_table(headers, rows), ""]
    return "\n".join(lines)


def t_ablation(results: dict) -> str:
    """E4 architecture-influence ablation (QRNN family, seed 42)."""
    aa = results.get("arch_ablation", {})
    lines = ["## Architecture-influence ablation (QRNN family, seed 42, one factor at a time)", ""]
    for sub, cfgs in aa.items():
        lines += [f"### {sub}", ""]
        headers = ["config", "factor", "value", "PICP", "NMPIW", "Winkler", "MAE", "RS"]
        rows = []
        for cid in sorted(cfgs, key=lambda c: (cfgs[c][0]["config"]["factor"],
                                               cfgs[c][0]["config"]["value"])):
            cells = [c for c in cfgs[cid] if c.get("status") == "OK"]
            if not cells:
                rows.append([cid, "-", "-", "-", "-", "-", "-", "-"])
                continue
            c = max(cells, key=lambda c: c.get("time_s", 0))  # latest re-run
            cfgd = c["config"]
            rows.append([cid, cfgd["factor"], str(cfgd["value"]), fmt(c["PICP"]),
                         fmt(c["NMPIW"]), fmt(c["Winkler"]), fmt(c["MAE"]), fmt(c["RS"])])
        lines += [_md_table(headers, rows), ""]
    return "\n".join(lines)


# ── Per-unit frames & statistics (E5/E6/E7) ────────────────────────────────

def unit_metric_frame(results: dict, dataset: str, metric: str = "PICP",
                      methods: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Per-method DataFrame indexed by unit: seed-averaged per-unit metric.

    metric: 'PICP' (stored per-unit) or 'RS' (derived: |PICP - 0.90|).
    Returns {method: DataFrame[unit -> value]}.
    """
    methods = methods or METHODS
    acc: dict[str, dict[str, list[float]]] = {m: {} for m in methods}
    for d, sub, m, sd, c in ok_cells(results, dataset=dataset):
        if m not in acc:
            continue
        pu = c.get("per_unit") or {}
        for u, p in zip(pu.get("units", []), pu.get("picp", [])):
            acc[m].setdefault(u, []).append(p)
    out = {}
    for m in methods:
        rows = {u: (abs(np.mean(v) - NOMINAL) if metric == "RS" else np.mean(v))
                for u, v in acc[m].items()}
        out[m] = pd.DataFrame({"value": list(rows.values())},
                              index=pd.Index(list(rows.keys()), name="unit"))
    return out


def load_npz(cell: dict) -> np.ndarray:
    # allow_pickle: 9 legacy DS01 smoke cells (written by an early runner
    # revision) store `units` as a pickled object array of strings. The files
    # are locally generated sweep artifacts; no untrusted content is loaded.
    z = np.load(PRED_DIR.parent / cell["pred_path"], allow_pickle=True)
    return z


def unit_winkler_frame(results: dict, dataset: str,
                       methods: list[str]) -> dict[str, pd.DataFrame]:
    """Per-method per-unit Winkler (seed-averaged), recomputed from npz."""
    acc: dict[str, dict[str, list[float]]] = {m: {} for m in methods}
    for d, sub, m, sd, c in ok_cells(results, dataset=dataset, method=None):
        if m not in acc:
            continue
        z = load_npz(c)
        yt, lo, up, units = z["y_true"], z["lower"], z["upper"], z["units"]
        for u in np.unique(units):
            mask = units == u
            w = M.winkler_score(yt[mask], lo[mask], up[mask])
            acc[m].setdefault(u, []).append(float(w))
    out = {}
    for m in methods:
        rows = {u: float(np.mean(v)) for u, v in acc[m].items()}
        out[m] = pd.DataFrame({"value": list(rows.values())},
                              index=pd.Index(list(rows.keys()), name="unit"))
    return out


def paired_bootstrap_ci(diff_by_unit: pd.Series, n_boot: int = 2000,
                        conf: float = 0.95, seed: int = 42) -> tuple[float, float, float]:
    """Cluster (unit) bootstrap CI of the mean paired difference."""
    d = diff_by_unit.dropna().to_numpy()
    rng = np.random.default_rng(seed)
    n = len(d)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = d[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(means, [100 * (1 - conf) / 2, 100 * (1 + conf) / 2])
    return float(d.mean()), float(lo), float(hi)


def tost_power_mde(n_units: int, sd_diff: float, margin: float,
                   alpha: float = 0.05, power: float = 0.80) -> float:
    """Minimum detectable effect for TOST at the given power (F12 power check).

    MDE = (z_{1-alpha} + z_{power}) * sd_diff / sqrt(n)  (two one-sided tests,
    each at level alpha; standard TOST sample-size inversion).
    """
    from scipy.stats import norm
    if n_units < 2 or sd_diff == 0:
        return float("nan")
    return float((norm.ppf(1 - alpha) + norm.ppf(power)) * sd_diff / np.sqrt(n_units))


def four_arm_tie(frame: dict[str, pd.DataFrame], metric: str = "value",
                 margin: float = TOST_MARGIN) -> dict:
    """All-pairs TOST among the four split-CP arms (P2/P6 tie test).

    Returns per-pair TOST results + overall tie verdict + power (F12) info.
    """
    arms = [m for m in SPLIT_CP_4 if not frame[m].empty]
    pairs = []
    sd_diffs = []
    n_units = None
    for a, b in itertools.combinations(arms, 2):
        common = frame[a].index.intersection(frame[b].index)
        fa = frame[a].reindex(common)[metric]
        fb = frame[b].reindex(common)[metric]
        r = M.tost_equivalence(
            fa.rename_axis("unit").to_frame("PICP"),
            fb.rename_axis("unit").to_frame("PICP"),
            metric_name="PICP", margin=margin)
        d = (fa - fb).to_numpy()
        sd_diffs.append(float(np.std(d, ddof=1)) if len(d) > 1 else 0.0)
        n_units = len(d)
        pairs.append((a, b, r))
    tie = all(r["equivalent"] for _, _, r in pairs) and len(arms) == len(SPLIT_CP_4)
    mde = [tost_power_mde(n_units or 0, sd, margin) for sd in sd_diffs]
    return {"n_units": n_units, "pairs": pairs, "tie": bool(tie),
            "mde_80": mde, "margin": margin,
            "powered": all(not np.isnan(x) and x <= margin for x in mde)}


def paired_compare(frame_a: pd.DataFrame, frame_b: pd.DataFrame,
                   metric: str = "value") -> dict:
    """Paired t-test + Cohen's d + cluster-bootstrap CI on per-unit values."""
    common = frame_a.index.intersection(frame_b.index)
    a = frame_a.reindex(common)[metric]
    b = frame_b.reindex(common)[metric]
    t, p, sig = M.paired_test_per_unit(a.to_frame("PICP"), b.to_frame("PICP"),
                                       metric_name="PICP")
    d = M.cohens_d_per_unit(a.to_frame("PICP"), b.to_frame("PICP"), metric_name="PICP")
    mean_diff, ci_lo, ci_hi = paired_bootstrap_ci((a - b).rename("diff"))
    return {"n_units": len(common), "t": t, "p": p, "significant": sig,
            "cohens_d": d, "mean_diff": mean_diff,
            "ci95": (ci_lo, ci_hi)}


# ── Hypothesis tests (design sec.7) ────────────────────────────────────────

def unit_winkler_long(results: dict, dataset: str, methods: list[str]) -> pd.DataFrame:
    """Long table (method, unit, seed, winkler) recomputed from npz."""
    rows = []
    for d, sub, m, sd, c in ok_cells(results, dataset=dataset):
        if m not in methods:
            continue
        z = load_npz(c)
        yt, lo, up, units = z["y_true"], z["lower"], z["upper"], z["units"]
        for u in np.unique(units):
            mask = units == u
            rows.append({"method": m, "unit": str(u), "seed": sd,
                         "value": float(M.winkler_score(yt[mask], lo[mask], up[mask]))})
    return pd.DataFrame(rows)


def h1_shift_validity(results: dict) -> tuple[str, dict]:
    """H1: Weighted-CP holds >= 0.87 coverage on the shifted set where
    plain split-CP under-covers; confirm = PICP >= 0.87 AND the paired
    difference CI excludes 0 (design sec.7)."""
    stats: dict = {"arm": "weighted_cqr", "ref": "c_qrnn", "population": "ncmapss"}
    cells_w = [c for d, s, m, sd, c in ok_cells(results, dataset="ncmapss",
                                                method="weighted_cqr")]
    cells_r = [c for d, s, m, sd, c in ok_cells(results, dataset="ncmapss",
                                                method="c_qrnn")]
    pw, pr = pooled_coverage(cells_w), pooled_coverage(cells_r)
    stats["weighted_pooled_picp"] = pw
    stats["splitcp_pooled_picp"] = pr
    fw = unit_metric_frame(results, "ncmapss", "PICP",
                           ["weighted_cqr", "c_qrnn"])
    comp = paired_compare(fw["weighted_cqr"], fw["c_qrnn"])
    d = (fw["weighted_cqr"] - fw["c_qrnn"])["value"]
    sd_d = float(np.std(d.to_numpy(), ddof=1))
    stats.update(comp)
    stats["sd_diff"] = sd_d
    stats["mde_80"] = tost_power_mde(comp["n_units"], sd_d, TOST_MARGIN)
    confirm = (pw >= 0.87) and (comp["p"] < cfg.SIGNIFICANCE_ALPHA) \
        and (comp["mean_diff"] > 0) and (comp["ci95"][0] > 0)
    refute = comp["p"] >= cfg.SIGNIFICANCE_ALPHA
    stats["verdict"] = "CONFIRMED" if confirm else ("REFUTED (no significant gap)"
                                                    if refute else "INCONCLUSIVE")
    lines = [
        "## H1 -- shift validity (Weighted-CP vs plain split-CP on N-CMAPSS)",
        "",
        f"- Weighted-CP CQR pooled PICP (sample-weighted, 5 seeds x 9 subsets): **{pw:.4f}**",
        f"- C-QRNN pooled PICP: **{pr:.4f}**",
        f"- Paired per-unit (seed-averaged) diff, n={comp['n_units']} units: "
        f"mean={comp['mean_diff']:+.4f}, 95% CI [{comp['ci95'][0]:+.4f}, "
        f"{comp['ci95'][1]:+.4f}], t={comp['t']:.3f}, p={comp['p']:.4g}, "
        f"Cohen's d={comp['cohens_d']:.3f}",
        f"- TOST note: H1's 'CI excludes the split-CP level' is the paired-diff CI test above; TOST equivalence of the two levels is reported by the P2/P6 tie machinery",
        f"- F12 power check: sd(diff)={sd_d:.4f}, MDE@80%={stats['mde_80']:.4f} "
        f"({'powered' if stats['mde_80'] <= TOST_MARGIN else 'UNDERPOWERED for ' + str(TOST_MARGIN) + ' margin'})",
        "",
        f"Criterion (design sec.7): confirm if Weighted-CP PICP >= 0.87 and the "
        f"paired-difference CI excludes the split-CP level.  **Verdict: {stats['verdict']}**",
    ]
    return "\n".join(lines), stats


def h2_local_adaptivity(results: dict) -> tuple[str, dict]:
    """H2: >=1 of {DistCP, UACQR-S, Weighted-CP} beats C-QRNN on Winkler with
    paired-test significance (design sec.7)."""
    challengers = ["distcp", "uacqr", "weighted_cqr"]
    stats: dict = {}
    lines = ["## H2 -- local adaptivity (Winkler vs C-QRNN)", ""]
    any_confirm = False
    for ds in DATASETS:
        lines.append(f"### {ds} (per-unit Winkler, seed-averaged; from npz)")
        lines.append("")
        try:
            frames = unit_winkler_frame(results, ds, ["c_qrnn"] + challengers)
        except Exception as exc:  # npz missing for some cell
            lines.append(f"- skipped: {exc}")
            continue
        rows = []
        for ch in challengers:
            comp = paired_compare(frames[ch], frames["c_qrnn"])
            confirms = comp["p"] < cfg.SIGNIFICANCE_ALPHA and comp["mean_diff"] < 0
            any_confirm = any_confirm or confirms
            stats[f"{ds}:{ch}"] = comp
            rows.append([METHOD_LABEL[ch], fmt(frames[ch]["value"].mean()),
                         fmt(frames["c_qrnn"]["value"].mean()),
                         f"{comp['mean_diff']:+.2f}", fmt(comp["ci95"][0], 2),
                         fmt(comp["ci95"][1], 2), f"{comp['p']:.4g}",
                         f"{comp['cohens_d']:.2f}", "YES" if confirms else "no"])
        lines.append(_md_table(["challenger", "Winkler (chall.)", "Winkler (C-QRNN)",
                                "mean diff", "CI lo", "CI hi", "p", "d", "beats?"], rows))
        lines.append("")
    # mixed-effects (seed nested in unit) on the shifted population
    lines.append("### Mixed-effects model (value ~ C(method) + (1|unit) + (1|unit:seed))")
    try:
        long_df = unit_winkler_long(results, "ncmapss", ["c_qrnn"] + challengers)
        me = M.mixed_effects_model(long_df, "value", "method", "unit", "seed")
        lines.append("")
        lines.append(f"- converged={me['converged']}, aic={me['aic']:.1f}, bic={me['bic']:.1f}")
        fe = me["fixed_effects"]; pv = me["p_values"]
        lines.append(_md_table(["method", "fixed effect", "p"],
                               [[m_, fmt(fe.get(m_)), fmt(pv.get(m_))]
                                for m_ in ["c_qrnn"] + challengers if m_ in fe.index]))
        stats["mixed_effects"] = {"converged": me["converged"],
                                  "aic": float(me["aic"]), "bic": float(me["bic"])}
    except Exception as exc:
        lines.append(f"- mixed-effects skipped: {exc}")
    stats["verdict"] = "CONFIRMED" if any_confirm else "NOT CONFIRMED (no significant Winkler beat)"
    lines += ["", f"**Verdict: {stats['verdict']}**"]
    return "\n".join(lines), stats


def h3_fusion_vs_series(results: dict) -> tuple[str, dict]:
    """H3: C-MDN series NOT significantly better than C-QRNN on Winkler
    (confirm); refuted if significantly better (design sec.7 / F8)."""
    stats: dict = {}
    lines = ["## H3 -- fusion-vs-series (C-MDN series vs C-QRNN on Winkler)", ""]
    rows = []
    refuted = False
    for ds in DATASETS:
        try:
            frames = unit_winkler_frame(results, ds, ["c_mdn", "c_qrnn"])
        except Exception as exc:
            lines.append(f"- {ds} skipped: {exc}")
            continue
        comp = paired_compare(frames["c_mdn"], frames["c_qrnn"])
        refuted = refuted or (comp["p"] < cfg.SIGNIFICANCE_ALPHA and comp["mean_diff"] < 0)
        stats[ds] = comp
        rows.append([ds, fmt(frames["c_mdn"]["value"].mean()),
                     fmt(frames["c_qrnn"]["value"].mean()),
                     f"{comp['mean_diff']:+.2f}",
                     f"[{comp['ci95'][0]:+.2f}, {comp['ci95'][1]:+.2f}]",
                     f"{comp['p']:.4g}", f"{comp['cohens_d']:.2f}"])
    lines.append(_md_table(["dataset", "Winkler (C-MDN)", "Winkler (C-QRNN)",
                            "mean diff", "95% CI", "p", "d"], rows))
    stats["verdict"] = ("REFUTED (C-MDN series significantly better)"
                        if refuted else "CONFIRMED (not significantly better)")
    lines += ["",
              "Criterion (design sec.7): confirm = C-MDN-series Winkler not "
              f"significantly better than C-QRNN.  **Verdict: {stats['verdict']}**",
              "",
              "Note: a refutation does not reopen the novelty question (F8); it "
              "triggers a study-framing discussion at T2."]
    return "\n".join(lines), stats


def _best_by_metric(results: dict, dataset: str) -> pd.DataFrame:
    """Per-method value per metric on the population + the best method."""
    recs = []
    for m in METHODS:
        cells = [c for d, s, mm, sd, c in ok_cells(results, dataset=dataset, method=m)]
        if not cells:
            continue
        row = {"method": m}
        for k in METRIC_KEYS:
            v, s, n = cell_mean(cells, k)
            row[k] = v
        for ek in EFF_KEYS:
            vals = [c["efficiency"][ek] for c in cells
                    if isinstance(c.get("efficiency"), dict)]
            row[ek] = float(np.mean(vals)) if vals else float("nan")
        row["picp_dev"] = abs(pooled_coverage(cells) - NOMINAL)
        recs.append(row)
    df = pd.DataFrame(recs).set_index("method")
    best = {}
    for k in list(METRIC_KEYS) + EFF_KEYS:
        col = "picp_dev" if k == "PICP" else k
        if k == "PICP":
            best[k] = df["picp_dev"].idxmin()
        elif k == "R2":
            best[k] = df[k].idxmax()
        else:
            best[k] = df[k].idxmin()
    df["best_method"] = best
    return df


def h4_baselines(results: dict) -> tuple[str, dict]:
    """H4: post-calibration, C-QRNN is not the sole best method on >= 1 metric
    (confirm); refuted if C-QRNN dominates all calibrated baselines (design sec.7)."""
    stats: dict = {}
    lines = ["## H4 -- calibrated baselines (N-CMAPSS primary population)", ""]
    df = _best_by_metric(results, "ncmapss")
    metrics_shown = ["PICP", "RS", "NMPIW", "CWC", "Winkler", "MAE", "RMSE",
                     "RUL_score", "crossing_freq", "footprint_mb", "latency_ms_per_sample"]
    rows = []
    baseline_best = []
    for k in metrics_shown:
        col = "picp_dev" if k == "PICP" else k
        b = df[col].idxmin() if k != "R2" else df["R2"].idxmax()
        cqrnn = df.loc["c_qrnn", col]
        bestv = df.loc[b, col]
        is_base = b in CALIBRATED_BASELINES
        baseline_best.append(k if is_base else None)
        rows.append([k, METHOD_LABEL[b], fmt(bestv), fmt(cqrnn),
                     "YES" if is_base else "no"])
    lines.append(_md_table(["metric", "best (N-CMAPSS)", "best value",
                            "C-QRNN value", "best is calibrated baseline?"], rows))
    metrics_won = [k for k in baseline_best if k]
    sole_best = all(k is None for k in baseline_best)
    stats["metrics_where_baseline_best"] = metrics_won
    stats["verdict"] = ("CONFIRMED (calibrated baselines best on: "
                        + ", ".join(metrics_won) + ")") if metrics_won else \
        "REFUTED (C-QRNN sole best on all shown metrics)"
    lines += ["", f"**Verdict: {stats['verdict']}**"]
    return "\n".join(lines), stats


# ── Prediction-table cross-check (E7) ──────────────────────────────────────

PREDICTED = {
    # FROZEN pre-registration (doc/prediction_table.md, approved 2026-09-08).
    # Scoring populations: P1 SHIFTED (N-CMAPSS pooled); P2/P6 TIE-REGIME (both
    # pooled, per-unit seed-averaged) + P6b SHIFTED; P3-P11 PRIMARY (N-CMAPSS),
    # C-MAPSS reported secondarily.
    "P1": {"label": "PICP calibration, shifted test sets", "predicted": ["weighted_cqr"]},
    "P2": {"label": "PICP calibration, stationary test sets",
           "predicted": list(SPLIT_CP_4), "tie_aware": True},
    "P3": {"label": "Winkler score", "predicted": ["uacqr"]},
    "P4": {"label": "NMPIW", "predicted": ["uacqr"]},
    "P5": {"label": "CWC", "predicted": ["uacqr"]},
    "P6": {"label": "Reliability score",
           "predicted": list(SPLIT_CP_4) + ["weighted_cqr"], "tie_aware": True},
    "P7": {"label": "Conditional coverage (per-stage / per-subset)", "predicted": ["uacqr"]},
    "P8": {"label": "Point metrics (MAE / RMSE)", "predicted": ["uacqr"]},
    "P9": {"label": "RUL asymmetric score", "predicted": ["uacqr"]},
    "P10": {"label": "Quantile-crossing frequency", "predicted": ["cgpr"]},
    "P11": {"label": "Footprint + latency", "predicted": ["cxgbq"]},
}


def _p1(results):
    dev = {}
    for m in METHODS:
        cells = [c for d, s, mm, sd, c in ok_cells(results, dataset="ncmapss", method=m)]
        if cells:
            dev[m] = abs(pooled_coverage(cells) - NOMINAL)
    winners = [m for m, v in dev.items() if v <= min(dev.values()) + 1e-12]
    return ("N-CMAPSS pooled |PICP-0.90|", dev, winners,
            {m: f"{dev[m]:.4f}" for m in sorted(dev, key=dev.get)})


def _tie_row(results, metric: str):
    frame = unit_metric_frame(results, "ncmapss", metric)
    frame2 = unit_metric_frame(results, "cmapss", metric)
    both = {m: pd.concat([frame[m], frame2[m]]) for m in SPLIT_CP_4}
    tie = four_arm_tie(both)
    return both, tie


def _single_winner(results, key: str, invert: bool = False):
    """Mean across cells per method on N-CMAPSS (primary); C-MAPSS secondary."""
    def pop(ds):
        out = {}
        for m in METHODS:
            cells = [c for d, s, mm, sd, c in ok_cells(results, dataset=ds, method=m)]
            if cells:
                v, s, n = cell_mean(cells, key)
                out[m] = v
        return out
    prim, sec = pop("ncmapss"), pop("cmapss")
    if not prim:
        return None, None, {}, {}

    def _win(d):
        if not d:
            return []
        t = min(d.values()) if not invert else max(d.values())
        tol = 1e-12
        if invert:
            return [m for m, v in d.items() if v >= t - tol]
        return [m for m, v in d.items() if v <= t + tol]

    target = min(prim.values()) if not invert else max(prim.values())
    winners = _win(prim)
    sec_winners = _win(sec)
    return target, winners, prim, {**sec, "__winners__": sec_winners}


def _p7_condcov(results):
    """Mean abs deviation from 0.90 over per-subset + per-stage groups (cells)."""
    out = {}
    for m in METHODS:
        cells = [c for d, s, mm, sd, c in ok_cells(results, method=m)]
        devs = []
        for c in cells:
            d_sub = abs(c["PICP"] - NOMINAL)
            stages = c.get("per_stage") or {}
            num = den = 0.0
            for st in ("early", "mid", "late"):
                if st in stages:
                    num += abs(stages[st]["PICP"] - NOMINAL) * stages[st]["n"]
                    den += stages[st]["n"]
            d_stage = num / den if den else d_sub
            devs.append(0.5 * d_sub + 0.5 * d_stage)
        if devs:
            out[m] = float(np.mean(devs))
    winners = [m for m, v in out.items() if v <= min(out.values()) + 1e-12]
    return winners, out


def _p11_efficiency(results):
    foot, lat = {}, {}
    for m in METHODS:
        for ds in DATASETS:
            cells = [c for d, s, mm, sd, c in ok_cells(results, dataset=ds, method=m)]
            fv = [c["efficiency"]["footprint_mb"] for c in cells
                  if isinstance(c.get("efficiency"), dict)]
            lv = [c["efficiency"]["latency_ms_per_sample"] for c in cells
                  if isinstance(c.get("efficiency"), dict)]
            if fv:
                foot.setdefault(m, []).append(float(np.mean(fv)))
                lat.setdefault(m, []).append(float(np.mean(lv)))
    foot_m = {m: float(np.mean(v)) for m, v in foot.items()}
    lat_m = {m: float(np.mean(v)) for m, v in lat.items()}

    def rank(d):
        order = sorted(d, key=d.get)
        r = {}
        for i, m in enumerate(order):
            r[m] = i + 1
        return r
    rf, rl = rank(foot_m), rank(lat_m)
    rs = {m: rf[m] + rl[m] for m in foot_m}
    winners = [m for m, v in rs.items() if v <= min(rs.values()) + 1e-12]
    return winners, {"ranksum": rs, "footprint_mb": foot_m, "latency_ms": lat_m}


def build_prediction_report(results: dict) -> tuple[str, dict]:
    """P1-P11 observed best + HIT/MISS (tie-aware P2/P6) + hit rate + CI."""
    from scipy.stats import binomtest
    rows, details = [], {}
    tie_picp, tie_picp_res = _tie_row(results, "PICP")
    tie_rs, tie_rs_res = _tie_row(results, "RS")

    def add(pid, observed, winners, predicted, status, evidence, detail):
        rows.append([pid, PREDICTED[pid]["label"],
                     ", ".join(METHOD_LABEL[m] for m in predicted),
                     observed, status, evidence])
        details[pid] = {"observed": observed, "status": status,
                        "winners": winners, "evidence": evidence, **detail}

    # P1 -- shifted |PICP-0.90|
    obs, dev, winners, _ = _p1(results)
    hit = any(w in PREDICTED["P1"]["predicted"] for w in winners)
    add("P1", f"{', '.join(METHOD_LABEL[m] for m in winners)} "
              f"(|dev|={dev[winners[0]]:.4f})", winners, PREDICTED["P1"]["predicted"],
        "HIT" if hit else "MISS", f"N-CMAPSS pooled; per-method |dev|: "
        + ", ".join(f"{METHOD_LABEL[m]} {v:.3f}" for m, v in
                    sorted(dev.items(), key=lambda kv: kv[1])[:4]), {"dev": dev})

    # P2 -- four-arm TOST tie (TIE-REGIME)
    tie = tie_picp_res["tie"]
    add("P2", "tie (4 split-CP arms, TOST eq.)" if tie else
        "tie BROKEN: " + ", ".join(METHOD_LABEL[m] for m in SPLIT_CP_4),
        SPLIT_CP_4 if tie else [], PREDICTED["P2"]["predicted"],
        "HIT" if tie else "MISS",
        f"per-unit PICP seed-averaged, n={tie_picp_res['n_units']} units, "
        f"margin {TOST_MARGIN}; all 6 pairs equivalent={tie}; "
        f"powered={tie_picp_res['powered']} (F12)",
        {"tie": tie_picp_res})

    # P3-P5, P9 -- single winner, lower better
    for pid, key in (("P3", "Winkler"), ("P4", "NMPIW"), ("P5", "CWC"),
                     ("P9", "RUL_score")):
        target, winners, prim, sec = _single_winner(results, key)
        hit = any(w in PREDICTED[pid]["predicted"] for w in winners)
        add(pid, ", ".join(METHOD_LABEL[m] for m in winners), winners,
            PREDICTED[pid]["predicted"], "HIT" if hit else "MISS",
            f"N-CMAPSS cell-mean {key}={target:.4f}; C-MAPSS winner: "
            + ", ".join(METHOD_LABEL[m] for m in sec.get("__winners__", [])),
            {"primary": prim, "secondary_winners": sec.get("__winners__", [])})

    # P6 -- RS: (a) four-arm tie (TIE-REGIME) AND (b) Weighted-CP best on SHIFTED
    a = tie_rs_res["tie"]
    dev = {}
    for m in METHODS:
        cells = [c for d, s, mm, sd, c in ok_cells(results, dataset="ncmapss", method=m)]
        if cells:
            dev[m] = pooled_coverage(cells, "RS")
    w_best = [m for m, v in dev.items() if v <= min(dev.values()) + 1e-12]
    b = "weighted_cqr" in w_best
    hit = a and b
    add("P6", f"tie={a}; shifted best: " +
        ", ".join(METHOD_LABEL[m] for m in w_best),
        (SPLIT_CP_4 if a else []) + (w_best if b else []),
        PREDICTED["P6"]["predicted"], "HIT" if hit else "MISS",
        f"(a) 4-arm RS TOST tie (n={tie_rs_res['n_units']} units)={a}; "
        f"(b) shifted (N-CMAPSS) pooled RS best={w_best} "
        f"({'incl.' if b else 'excl.'} Weighted-CP)",
        {"tie_rs": tie_rs_res, "shifted_rs": dev})

    # P7 -- conditional coverage
    winners, out = _p7_condcov(results)
    hit = any(w in PREDICTED["P7"]["predicted"] for w in winners)
    add("P7", ", ".join(METHOD_LABEL[m] for m in winners), winners,
        PREDICTED["P7"]["predicted"], "HIT" if hit else "MISS",
        "both-dataset mean |dev| over per-subset + per-stage groups; "
        + ", ".join(f"{METHOD_LABEL[m]} {v:.3f}" for m, v in
                    sorted(out.items(), key=lambda kv: kv[1])[:4]), {"dev": out})

    # P8 -- MAE and RMSE both
    t_mae, w_mae, prim_mae, sec_mae = _single_winner(results, "MAE")
    t_rmse, w_rmse, prim_rmse, sec_rmse = _single_winner(results, "RMSE")
    both = [m for m in w_mae if m in w_rmse]
    hit = any(m in PREDICTED["P8"]["predicted"] for m in both)
    add("P8", f"MAE: {', '.join(METHOD_LABEL[m] for m in w_mae)}; RMSE: "
              f"{', '.join(METHOD_LABEL[m] for m in w_rmse)}", both,
        PREDICTED["P8"]["predicted"], "HIT" if hit else "MISS",
        "HIT requires the predicted method best on BOTH MAE and RMSE",
        {"mae": w_mae, "rmse": w_rmse})

    # P10 -- crossing
    t_c, w_c, prim_c, sec_c = _single_winner(results, "crossing_freq")
    hit = any(w in PREDICTED["P10"]["predicted"] for w in w_c)
    add("P10", ", ".join(METHOD_LABEL[m] for m in w_c), w_c,
        PREDICTED["P10"]["predicted"], "HIT" if hit else "MISS",
        f"cell-mean crossing_freq; zeros: " +
        ", ".join(METHOD_LABEL[m] for m in prim_c
                  if prim_c[m] is not None and prim_c[m] == 0.0),
        {"primary": prim_c})

    # P11 -- efficiency rank-sum
    winners, eff = _p11_efficiency(results)
    hit = any(w in PREDICTED["P11"]["predicted"] for w in winners)
    add("P11", ", ".join(METHOD_LABEL[m] for m in winners), winners,
        PREDICTED["P11"]["predicted"], "HIT" if hit else "MISS",
        f"rank-sum over footprint+latency (both datasets): " +
        ", ".join(f"{METHOD_LABEL[m]} {v}" for m, v in
                  sorted(eff['ranksum'].items(), key=lambda kv: kv[1])[:4]),
        {"efficiency": eff})

    hits = sum(1 for r in rows if r[4] == "HIT")
    n = len(rows)
    bt = binomtest(hits, n)
    ci = bt.proportion_ci()
    lines = ["## Prediction-table cross-check (P1-P11, tie-aware P2/P6)", ""]
    lines.append(_md_table(["#", "metric", "predicted best", "observed best",
                            "status", "evidence"], rows))
    lines += ["",
              f"**Hit rate: {hits}/{n} = {hits / n:.3f}, "
              f"95% CI [{ci.low:.3f}, {ci.high:.3f}]** (exact binomial)", ""]
    misses = [r for r in rows if r[4] == "MISS"]
    if misses:
        lines += ["### Misses (why-paragraphs to be completed at doc assembly)", ""]
        for r in misses:
            lines.append(f"- **{r[0]}** ({r[1]}): predicted "
                         f"{r[2]}, observed {r[3]}. Why: *[fill at E6 assembly]*")
    return "\n".join(lines), {"rows": rows, "details": details, "hits": hits,
                              "n": n, "ci": (float(ci.low), float(ci.high))}


def fill_xlsx(xlsx_path: Path, spec_path: Path) -> str:
    """Write the human-reviewed spec into the XLSX working copy.

    Spec JSON: {"P1": {"status": "HIT"|"MISS", "observed": str,
                       "why": str (MISS only), "date": "YYYY-MM-DD"}, ...}
    Never edits columns A-D (frozen prediction columns); writes E/F/G/H only.
    """
    import openpyxl
    spec = json.loads(Path(spec_path).read_text())
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb["Predictions"]
    for pid, vals in spec.items():
        r = 1 + int(pid[1:])
        assert ws.cell(row=r, column=1).value == pid, f"row {r} is not {pid}"
        ws.cell(row=r, column=5).value = vals["status"]
        ws.cell(row=r, column=6).value = vals["observed"]
        if vals.get("why"):
            ws.cell(row=r, column=7).value = vals["why"]
        if vals.get("date"):
            ws.cell(row=r, column=8).value = vals["date"]
    wb.save(xlsx_path)
    # verification pass
    wb2 = openpyxl.load_workbook(xlsx_path)
    ws2 = wb2["Predictions"]
    lines = [f"Wrote {len(spec)} rows to {xlsx_path}; verification pass:"]
    pending = 0
    for r in range(2, 13):
        pid = ws2.cell(row=r, column=1).value
        status = ws2.cell(row=r, column=5).value
        if status == "PENDING":
            pending += 1
        lines.append(f"  {pid}: status={status} observed="
                     f"{str(ws2.cell(row=r, column=6).value)[:60]!r} "
                     f"date={ws2.cell(row=r, column=8).value}")
    lines.append(f"PENDING remaining: {pending}")
    return "\n".join(lines)


# ── E5 numerical verification ──────────────────────────────────────────────

def verify_manifest(results: dict, matrix_path: Path) -> tuple[str, dict]:
    """Planned==done diff of run_matrix.json vs experiment_results.json (E8 AC#2)."""
    matrix = json.loads(Path(matrix_path).read_text())

    def done_key(ds, sub, m, seed, cfg_id=None):
        return (ds, sub, m, str(seed) if seed is not None else None, cfg_id)

    done = set()
    for d, s, m, sd, c in iter_cells(results):
        cfg_id = c["config"]["id"] if m == "arch_ablation" else None
        done.add(done_key(d, s, m, sd, cfg_id))
    missing, extra = [], []
    for job in matrix["jobs"]:
        k = done_key(job["dataset"], job["subset"], job["method"],
                     job.get("seed"), job.get("config", {}).get("id"))
        if k not in done:
            missing.append(job)
    n_jobs = len(matrix["jobs"])
    lines = ["## E5 verification -- planned vs done (run_matrix.json)", "",
             f"- manifest jobs: {n_jobs}",
             f"- done cells (unique keys): {len(done)}",
             f"- missing: {len(missing)}", ""]
    if missing:
        lines += [f"- MISSING: {json.dumps(missing[:20])}",
                  "  (resume: `uv run python scripts/run_phase4_experiments.py --resume`)"]
    stats = {"n_jobs": n_jobs, "n_done": len(done), "n_missing": len(missing)}
    return "\n".join(lines), stats


def verify_recompute(results: dict, tolerance: dict | None = None) -> tuple[str, dict]:
    """Recompute every interval metric from the raw npz vs stored JSON (E5).

    Tolerances: PICP/RS exact (coverage counts) to 1e-9; NMPIW/CWC/Winkler to
    5e-3 abs (npz stores float32 intervals); MAE/RMSE/R2/RUL to 5e-3.
    """
    tol = tolerance or {"PICP": 1e-9, "RS": 1e-9, "NMPIW": 5e-3, "CWC": 5e-3,
                        "Winkler": 5e-3, "MAE": 5e-3, "RMSE": 5e-3,
                        "R2": 5e-3, "RUL_score": 5e-3}
    diffs = {k: [] for k in tol}
    n_checked = n_fail = n_nopred = 0
    fails = []
    for d, s, m, sd, c in iter_cells(results):
        if c.get("status") != "OK" or not c.get("pred_path"):
            continue
        z = load_npz(c)
        yt, lo, up = z["y_true"], z["lower"], z["upper"]
        ref = {
            "PICP": M.calculate_picp(yt, lo, up),
            "RS": abs(M.calculate_picp(yt, lo, up) - NOMINAL),
            "NMPIW": M.calculate_nmpiw(yt, lo, up),
            "CWC": M.calculate_cwc(yt, lo, up),
            "Winkler": M.winkler_score(yt, lo, up),
            "MAE": float(np.mean(np.abs(yt - z["median"]))),
            "RMSE": float(np.sqrt(np.mean((yt - z["median"]) ** 2))),
        }
        n_checked += 1
        for k, rec in ref.items():
            stored = c.get(k)
            if stored is None:
                continue
            delta = abs(rec - stored)
            diffs[k].append(delta)
            if delta > tol[k]:
                n_fail += 1
                fails.append((d, s, m, sd, k, stored, rec, delta))
    lines = ["## E5 verification -- recompute from raw npz", "",
             f"- cells checked: {n_checked}",
             f"- metric-cell mismatches beyond tolerance: {n_fail}", ""]
    headers = ["metric", "n", "max |delta|", "mean |delta|", "tolerance"]
    rows = []
    for k, ds_ in diffs.items():
        if ds_:
            rows.append([k, str(len(ds_)), fmt(max(ds_), 6),
                         fmt(float(np.mean(ds_)), 6), str(tol[k])])
    lines.append(_md_table(headers, rows))
    if fails:
        lines += ["", "### Failing cells", ""]
        for f in fails[:40]:
            lines.append(f"- {f[0]}/{f[1]}/{f[2]} s{f[3]} {f[4]}: "
                         f"stored={f[5]:.6f} recomputed={f[6]:.6f} d={f[7]:.2e}")
    stats = {"n_checked": n_checked, "n_fail": n_fail,
             "max_delta": {k: (max(v) if v else None) for k, v in diffs.items()}}
    return "\n".join(lines), stats


def verify_fail_inventory(results: dict) -> tuple[str, dict]:
    lines = ["## E5 verification -- FAIL cell inventory", ""]
    fails = [(d, s, m, sd, c.get("reason"))
             for d, s, m, sd, c in iter_cells(results)
             if c.get("status") != "OK"]
    if not fails:
        lines.append("- none (all completed cells OK)")
    else:
        lines.append(_md_table(["dataset", "subset", "method", "seed", "reason"],
                               [[a, b, c, str(dd), str(e)[:120]]
                                for a, b, c, dd, e in fails]))
    return "\n".join(lines), {"n_fail_cells": len(fails), "fails": fails}


def patch_winkler(results: dict, results_path: Path) -> tuple[str, dict]:
    """Recompute the Winkler field of every OK cell from its npz and rewrite
    the JSON in place (atomic tmp+rename).

    Required once: the running sweep process (launched 2026-09-23 11:48) has
    the pre-fix evaluate_uq_metrics in memory and stored Winkler with the
    coverage reference (0.90) as the miscoverage rate -- penalty coefficient
    18 instead of 0.222, inflating every stored Winkler. The fix landed in
    metrics.py (test: TestEvaluateUQMetricsWinklerWiring); this pass repairs
    the already-written cells without retraining. Run AFTER the sweep
    completes (never while the runner is checkpointing), then re-run
    `verify` -- expect 0 mismatches.
    """
    n_fixed = n_skipped = 0
    for d, s, m, sd, c in iter_cells(results):
        if c.get("status") != "OK" or not c.get("pred_path"):
            n_skipped += 1
            continue
        z = load_npz(c)
        rec = M.winkler_score(z["y_true"], z["lower"], z["upper"])
        if abs(rec - c.get("Winkler", float("nan"))) > 1e-9:
            c["Winkler"] = rec
            n_fixed += 1
    results["meta"]["updated"] = __import__("time").strftime("%Y-%m-%dT%H:%M:%S")
    results["meta"]["winkler_patch"] = ("2026-09-23: Winkler field recomputed "
                                        "from npz with the miscoverage-rate "
                                        "convention (2*0.1/0.9 penalty); see "
                                        "tasks/04-experiments.md session log")
    tmp = results_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=1)
    tmp.replace(results_path)
    lines = ["## Winkler patch (post-sweep)", "",
             f"- cells fixed: {n_fixed}, skipped (FAIL/no pred): {n_skipped}",
             "- meta.winkler_patch note written; file rewritten atomically"]
    return "\n".join(lines), {"n_fixed": n_fixed, "n_skipped": n_skipped}


# ── CLI ────────────────────────────────────────────────────────────────────

def cmd_tables(results, args) -> int:
    parts = {"aggregate": t_aggregate, "per_seed": t_per_seed,
             "per_subset": t_per_subset, "per_unit": t_per_unit,
             "per_stage": t_per_stage, "crossing": t_crossing,
             "shift": t_shift, "ablation": t_ablation}
    want = [args.which] if args.which != "all" else list(parts)
    out = "\n\n".join(parts[w](results) for w in want if w in parts)
    print(out)
    if args.out_dir:
        p = Path(args.out_dir); p.mkdir(parents=True, exist_ok=True)
        (p / "tables.md").write_text(out + "\n")
        print(f"[wrote {p / 'tables.md'}]", file=sys.stderr)
    return 0


def cmd_hypotheses(results, args) -> int:
    out, stats = [], {}
    for name, fn in (("h1", h1_shift_validity), ("h2", h2_local_adaptivity),
                     ("h3", h3_fusion_vs_series), ("h4", h4_baselines)):
        md, st = fn(results)
        out.append(md)
        stats[name] = st
    text = "\n\n".join(out)
    print(text)
    if args.out_dir:
        p = Path(args.out_dir); p.mkdir(parents=True, exist_ok=True)
        (p / "hypotheses.md").write_text(text + "\n")
        (p / "hypotheses_stats.json").write_text(json.dumps(
            {k: {kk: (vv if not isinstance(vv, (np.floating, np.integer)) else float(vv))
                 for kk, vv in v.items() if kk != "pairs"}
             for k, v in stats.items()}, indent=1, default=str))
        print(f"[wrote {p / 'hypotheses.md'}]", file=sys.stderr)
    return 0


def cmd_prediction(results, args) -> int:
    md, st = build_prediction_report(results)
    print(md)
    if args.out_dir:
        p = Path(args.out_dir); p.mkdir(parents=True, exist_ok=True)
        (p / "prediction.md").write_text(md + "\n")
        det = {pid: {k: v for k, v in d.items() if k != "tie" and k != "tie_rs"}
               for pid, d in st["details"].items()}
        (p / "prediction_details.json").write_text(json.dumps(
            {**{k: st[k] for k in ("hits", "n", "ci")}, "details": det},
            indent=1, default=str))
        print(f"[wrote {p / 'prediction.md'}]", file=sys.stderr)
    if args.fill_xlsx:
        print(fill_xlsx(Path(args.fill_xlsx), Path(args.fill_spec)))
    return 0


def cmd_verify(results, args, matrix_path: Path) -> int:
    if args.patch_winkler:
        md, st = patch_winkler(results, args.results)
        print(md)
        return 0
    out = []
    out.append(verify_fail_inventory(results)[0])
    out.append(verify_manifest(results, matrix_path)[0])
    out.append(verify_recompute(results)[0])
    text = "\n\n".join(out)
    print(text)
    if args.out_dir:
        p = Path(args.out_dir); p.mkdir(parents=True, exist_ok=True)
        (p / "verify.md").write_text(text + "\n")
        print(f"[wrote {p / 'verify.md'}]", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("subcommand",
                    choices=["tables", "hypotheses", "prediction", "verify"])
    ap.add_argument("--results", type=Path,
                    default=RESULTS_DIR / "experiment_results.json")
    ap.add_argument("--matrix", type=Path,
                    default=RESULTS_DIR / "run_matrix.json")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="also write markdown to <out-dir>/")
    ap.add_argument("--which", default="all",
                    choices=["all", "aggregate", "per_seed", "per_subset",
                             "per_unit", "per_stage", "crossing", "shift",
                             "ablation"])
    ap.add_argument("--fill-xlsx", type=Path, default=None,
                    help="XLSX working copy to fill (prediction subcommand)")
    ap.add_argument("--fill-spec", type=Path, default=None,
                    help="reviewed JSON spec for --fill-xlsx")
    ap.add_argument("--patch-winkler", action="store_true",
                    help="verify: rewrite the Winkler field of every OK cell "
                         "from its npz (post-sweep repair; see patch_winkler)")
    args = ap.parse_args(argv)

    results = load_results(args.results)
    if args.subcommand == "tables":
        return cmd_tables(results, args)
    if args.subcommand == "hypotheses":
        return cmd_hypotheses(results, args)
    if args.subcommand == "prediction":
        if args.fill_xlsx and not args.fill_spec:
            ap.error("--fill-xlsx requires --fill-spec")
        return cmd_prediction(results, args)
    return cmd_verify(results, args, args.matrix)


if __name__ == "__main__":
    raise SystemExit(main())

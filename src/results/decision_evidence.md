# Decision-Gate Evidence Pack — Phase 5 input

One-page summary of how the Phase-4 comparative method set (564-job sweep, 5 seeds ×
{N-CMAPSS 9 subsets, C-MAPSS 4 subsets}; numbers in `results_of_record.md`, mechanisms in
`analysis_report.md`) resolves the design §7 hypotheses H1–H4, plus the pre-registered
prediction cross-check. Framing: best-method-first (adviser directive, T1/T1.1).

## Verdict summary

| # | Hypothesis | Verdict | Decisive evidence |
|---|---|---|---|
| H1 | Shift validity of Weighted-CP | **CAUTION** (toolkit: INCONCLUSIVE) | Weighted-CP pooled PICP 0.7769 vs 0.87 bar; significant but small shift edge over split-CP (+0.36 pts, p = 0.033); weight estimator collapses on DS05–DS07 |
| H2 | Local-adaptivity benefit over plain CQR | **PASS** (CONFIRMED) | UACQR-S beats C-QRNN per-unit Winkler on C-MAPSS: −0.43, CI [−0.57, −0.30], p = 6.8e-9 (n = 142 units) |
| H3 | Fusion-vs-series (C-MDN series vs C-QRNN) | **PASS** (CONFIRMED, not better) | C-MDN − C-QRNN Winkler: N-CMAPSS +1.31 [−0.75, +3.59] p = 0.257; C-MAPSS +3.19 [+2.79, +3.61] p = 1.8e-31 |
| H4 | Calibrated baselines close the gap | **PASS** (CONFIRMED) | C-GPR best reliability score on N-CMAPSS (0.1154 vs C-QRNN 0.1492) and best Winkler among pooled-PICP ≥ 0.80 arms (38.76); C-XGBQ uniform C-MAPSS calibration (pooled 0.937, all subsets |dev| ≤ 0.041) |

## (a) H1 — shift validity of Weighted-CP — CAUTION

Criterion (design §7): confirm if Weighted-CP pooled PICP ≥ 0.87 **and** the paired
per-unit difference CI excludes the split-CP level.

- Weighted-CP CQR pooled (sample-weighted) PICP on the shifted N-CMAPSS population: **0.7769**;
  plain split-CP (C-QRNN): **0.7737** (5 seeds × 9 subsets each).
- Paired per-unit (seed-averaged, n = 20 units): mean diff **+0.0036**, 95% CI
  **[+0.0008, +0.0067]**, t = 2.305, p = 0.0326, Cohen's d = 0.017; F12 power check powered
  (MDE @80% = 0.0038).

The directional shift benefit is real and significant but small, and the absolute 0.87 bar is
not met — hence CAUTION rather than PASS. Mechanism (toolkit diagnostics): the weight
estimator's effective sample size collapses exactly where the train→test shift is strongest —
ESS/n = 0.091 / 0.063 / 0.068 on DS05 / DS06 / DS07 with fallback-to-uniform in 4/5, 5/5 and
5/5 seeds — so Weighted-CP reverts to plain split-CP precisely on the subsets that drag the
pooled figure down (the QRNN cluster sits at 0.49–0.57 seed-mean PICP on DS07/DS08a). On the
healthy shifted subsets (ESS/n ≥ 0.41) the weighted-quantile variants post the family's
tightest deviations (0.005 on DS02, 0.018 on DS08c) and lead the C-QRNN family on DS01,
DS02, DS04 and DS08c. The best shifted-population coverage belongs to the adaptive arms:
EDL 0.847 and DistCP/HPD 0.846 pooled.

## (b) H2 — local-adaptivity benefit over plain CQR — PASS

Criterion: ≥1 of {DistCP, UACQR-S, Weighted-CP} beats C-QRNN on per-unit Winkler (paired test).

| challenger | dataset | Winkler diff vs C-QRNN | 95% CI | p | beats? |
|---|---|---|---|---|---|
| UACQR-S-style | C-MAPSS (n = 142) | **−0.43** | [−0.57, −0.30] | 6.8e-9 | **YES** |
| UACQR-S-style | N-CMAPSS (n = 20) | +0.80 | [−0.05, +1.72] | 0.109 | no (n.s.) |
| Weighted-CP CQR | C-MAPSS (n = 142) | +0.26 | [+0.13, +0.39] | 1.9e-4 | no |
| Weighted-CP CQR | N-CMAPSS (n = 20) | +0.24 | [+0.06, +0.43] | 0.027 | no |
| DistCP/HPD-split | C-MAPSS (n = 142) | +46.83 | [+45.48, +48.29] | 4.4e-109 | no (intervals wider) |
| DistCP/HPD-split | N-CMAPSS (n = 20) | +12.58 | [+8.93, +16.41] | 2.0e-6 | no (intervals wider) |

UACQR-S-style confirms local adaptivity on the stationary C-MAPSS population and is
non-inferior on the shifted one; its unit-level edge over C-QRNN is typical on the hard units
(FD002_196: 0.597 vs 0.603; FD001_66: 0.595 vs 0.601), while its pooled Winkler sits 0.43
below C-QRNN's.

## (c) H3 — fusion-vs-series (C-MDN series vs C-QRNN) — PASS

Criterion: confirm = C-MDN-series Winkler **not** significantly better than C-QRNN.

- N-CMAPSS (n = 20 units): +1.31, 95% CI [−0.75, +3.59], p = 0.257, d = 0.09 — not significant.
- C-MAPSS (n = 142 units): +3.19, 95% CI [+2.79, +3.61], p = 1.8e-31, d = 0.73 — C-MDN series
  is significantly *worse* (C-QRNN fusion holds its ground).

The fusion architecture (C-QRNN) is therefore the defensible series-vs-fusion choice for the
study's framing: the single-series MDN arm gains nothing and pays a small sharpness cost on
C-MAPSS. (Per design F8, a refutation would not reopen the novelty question; the observed
result is the confirmatory one.)

## (d) H4 — calibrated baselines close the gap — PASS

Toolkit criterion: a calibrated baseline (C-GPR / C-XGBQ) is best on ≥1 primary metric.

- N-CMAPSS best-per-metric (seed-mean): reliability score **C-GPR 0.1154** (C-QRNN 0.1492);
  CWC Weighted-CP 4.7245; MAE/RMSE/RUL-score UACQR-S (8.40 / 11.0 / 2.33); coverage-adequate
  sharpness: **C-GPR best Winkler 38.76 and best NMPIW 1.179 among arms with pooled PICP ≥ 0.80**
  (DistCP 39.08 / 1.196 next).
- C-MAPSS: per-subset calibration is spread across arms — C-MDN best on FD001 (0.014) and
  FD003 (0.019), WQLCP on FD002 (0.022), UACQR-S on FD004 (0.008) — while DistCP holds the
  best pooled coverage (0.975) and C-XGBQ the most uniform profile (0.031–0.041 on all four
  subsets).
- The baselines do not dominate — the comparative set keeps the point-estimate lead (UACQR-S)
  and the shifted-coverage lead (DistCP/EDL) — but they close the gap on reliability and
  coverage-adequate sharpness, which is exactly the H4 question.

## Prediction cross-check (P1–P11, frozen 2026-09-08)

**Hit rate: 3/11 = 0.273, exact binomial 95% CI [0.060, 0.610]** (`scipy.stats.binomtest`).
Tie-aware P2/P6: HIT requires all six TOST pairs among the named split-CP arms equivalent at
margin 0.03 (F12-powered). Tracker: `doc/emails/20260907-prediction_table.xlsx` (pre-registered
snapshot in `doc/emails/build/`); fill spec `results/prediction_fill_spec.json`.

| # | metric | predicted | observed best | status |
|---|---|---|---|---|
| P1 | PICP calibration, shifted | Weighted-CP | EDL (|dev| 0.0534) | MISS |
| P2 | PICP calibration, stationary (tie of 4 arms) | split-CP four-way tie | tie BROKEN (all 6 TOST pairs rejected, n = 162) | MISS |
| P3 | Winkler | UACQR-S | Raw QRNN (14.21, at pooled PICP 0.499) | MISS |
| P4 | NMPIW | UACQR-S | Raw QRNN (0.410, at pooled PICP 0.499) | MISS |
| P5 | CWC | UACQR-S | Weighted-CP CQR (4.7245 vs 5.0156) | MISS |
| P6 | Reliability score (tie of 5 arms) | five-way tie | tie=False; EDL best shifted RS (0.126) | MISS |
| P7 | Conditional coverage | UACQR-S | EDL (mean |dev| 0.107 vs 0.131) | MISS |
| P8 | Point metrics (MAE+RMSE) | UACQR-S | UACQR-S (both) | **HIT** |
| P9 | RUL asymmetric score | UACQR-S | UACQR-S (2.3342) | **HIT** |
| P10 | Quantile-crossing frequency | C-GPR | C-GPR (0.0000, in 6-arm zero-crossing tie) | **HIT** |
| P11 | Footprint + latency | C-XGBQ | DistCP + EDL (rank-sum 4 tie; C-XGBQ 16) | MISS |

Miss pattern (for the T2 narrative): the three hits are all in the point-estimate / efficiency
corner where the priors were grounded; the misses cluster in (i) coverage under shift, where
the ensemble/epistemic arms (EDL, DistCP) beat every QRNN-calibrated prior, and (ii) the two
tie predictions, which the unit-level TOST machinery correctly broke. P3/P4 are sharpness wins
by Raw QRNN only via narrow uncalibrated intervals (pooled PICP 0.499, far short of the 0.90
nominal) — the coverage-adequate sharpness leader is C-GPR.

## Best-method summary (T2 framing)

- **UACQR-S-style** — the study's best point-estimate arm: best MAE (8.40), RMSE (10.98) and
  RUL score (2.33) on N-CMAPSS; the H2 local-adaptivity confirmation (C-MAPSS Winkler −0.43,
  p = 6.8e-9); best C-MAPSS Winkler among the calibrated arms at pooled PICP ≥ 0.80 (54.68);
  zero crossings.
- **DistCP/HPD-split** — the coverage champion: best pooled coverage on stationary C-MAPSS
  (0.975) and tied-best on shifted N-CMAPSS (0.846 vs EDL 0.847), strongest on the hardest
  units, smallest footprint (0.006 MB, tied) and top efficiency rank-sum (4).
- **EDL** — the shifted-population coverage and conditional-coverage leader (pooled PICP
  0.847; mean |dev| 0.107; best mid-stage coverage 0.951; fastest inference 0.0018 ms).
- **C-GPR** — the calibrated-baseline reference: best reliability score (0.1154) and best
  coverage-adequate Winkler (38.76) on N-CMAPSS.
- **Weighted-CP CQR** — the significant shift-robustness result (H1): tightest intervals of
  the QRNN family on N-CMAPSS (CWC 4.72, best of the family) with the weight-collapse
  mechanism documented as the scope boundary.

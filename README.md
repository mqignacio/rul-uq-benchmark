# Uncertainty Quantification for Aero-Engine Remaining Useful Life

A comparative benchmark of distribution-free and model-internal uncertainty methods for
aero-engine remaining useful life (RUL) prediction.

## What this repository is

This repository holds the experiment code and the verified result records for a comparative
study. The study runs 10 method arms on two public benchmarks under one protocol. A companion
report explains the design, the results, and the cross-check of a pre-registered prediction
table. The report is in this repository at `doc/study_report.docx`.

The manuscript is in preparation for IEEE Transactions on Reliability. The submission target
is 04 December 2026.

## The question

Uncertainty methods for RUL come in two families. Distribution-free methods add a calibration
step after training. Conformal prediction is such a step. It reads held-out data and sets the
interval width [2]. Model-internal methods build the uncertainty estimate inside the model.
Evidential deep learning (EDL) [3] and Gaussian process regression (GPR) are examples.

No published paper compares the two families under one protocol on aero-engine RUL data.
CRULP benchmarks conformal methods only [4]. Basora et al. benchmark model-internal methods
only [5]. This repository closes that gap.

## The method arms

Every arm runs at one nominal level of 90 percent. The splits are by engine unit. Every
baseline is calibrated.

| Method | Base predictor | Uncertainty mechanism | Family |
|---|---|---|---|
| C-QRNN (prior method) | Quantile regression network | Conformalized quantile regression, split conformal | Distribution-free |
| C-MDN (series ablation) | Mixture density network, 5 Gaussian components [1] | CQR-style score on MDN quantiles, split conformal | Distribution-free. Tests fusion versus series. |
| UACQR-S style | QRNN ensemble | CQR score scaled by the spread of the ensemble [6] | Distribution-free |
| Distributional CP / HPD-split | MDN conditional density | Density level-set score [7] | Distribution-free |
| Weighted-CP CQR | QRNN | CQR score with likelihood-ratio weighted quantile [8] | Distribution-free. Built for shift. |
| WQLCP style | QRNN plus reconstruction-loss model | Loss-scaled score with weighted quantile [9] | Distribution-free. Built for shift. |
| EDL | Normal-Inverse-Gamma head [3] | Evidential posterior | Model-internal |
| GPR | Gaussian process, composite kernel | Posterior quantiles, calibrated | Model-internal |
| C-XGBQ | XGBoost quantile regression | Same conformal calibration as C-QRNN | Calibrated baseline |
| Raw QRNN | QRNN, multiple seeds | None. Uncalibrated reference point. | Baseline |

## The data

The study uses two public benchmarks from the NASA Prognostics Center of Excellence (PCoE)
turbofan engine degradation repository.

- N-CMAPSS is the primary benchmark. It has 9 sub-datasets, DS01 to DS08c. Its test data
  differ from its training data in a documented way. The study calls it the shifted
  population.
- C-MAPSS FD001 to FD004 is the second benchmark. Its test units come from the same
  distribution as its training units. The study calls it the stationary population.

The raw data is not in this repository. Download the files from the NASA PCoE repository.
Then place them in the layout below. The file names must match `src/qrnn_uq_rul/config.py`.

    src/data/raw/        the N-CMAPSS HDF5 files (for example N-CMAPSS_DS01-005.h5)
    resources/cmapss/    the C-MAPSS train, test, and RUL text files (FD001 to FD004)

Two sub-datasets are excluded. DS08b is absent from the archive that was used. DS08d is
excluded because the HDF5 file fails an integrity check in some public mirrors. Nine
sub-datasets remain: DS01 to DS07, DS08a, and DS08c.

## Splits and evaluation

- Engine units are split into train, validation, calibration, and test groups before any
  window preprocessing. No unit appears in two groups.
- The preprocessing is fit on training units only.
- Each method runs on 5 seeds. A seed is a fixed random starting point for training.
- GPR is deterministic. One row per sub-dataset is recorded. Training sets larger than 5248
  samples are subsampled with a seeded generator, because exact Gaussian process training
  costs O(n^3).
- All uncertainty metrics are measured at the 90 percent level. PICP is the share of test
  points where the true RUL falls inside the interval. Higher is better. NMPIW is the
  average interval width. Lower is better. CWC and the Winkler score judge width and
  coverage misses together. Lower is better. Quantile crossing counts the cases where the
  lower band lands above the upper band. Lower is better. Footprint and latency are model
  size and inference time. Lower is better.
- The study also reports point accuracy (MAE, RMSE, and the asymmetric RUL score), coverage
  per sub-dataset, coverage per engine unit, and coverage per degradation stage.
- Unit-level comparisons use paired tests, equivalence tests (TOST), and a mixed-effects
  model with random effects for unit and seed.

## How to run

1. Install `uv` (https://docs.astral.sh/uv/).
2. Open a terminal in the `src/` folder.
3. Install the environment: `uv sync`
4. Run the fast test set: `uv run pytest -m "not slow"`
   The full test set is `uv run pytest`. Two tests skip when the raw data is absent.
5. Download the two benchmarks and place them as described in "The data" above.
6. Reproduce the prior C-QRNN baseline: `uv run python scripts/run_reproduction.py`
7. Run the comparative sweep: `uv run python scripts/run_phase4_experiments.py --resume`
   The `--resume` flag skips cells that are already complete. The full sweep is 564 cells
   and takes about 20 hours on a Mac mini M4 Pro.
8. Rebuild the tables and the checks:
   `uv run python scripts/analyze_phase4.py verify`
   `uv run python scripts/analyze_phase4.py tables`
   `uv run python scripts/analyze_phase4.py hypotheses`
   `uv run python scripts/analyze_phase4.py prediction`

## Results

The four pre-stated hypotheses resolved as follows.

- H1, shift validity: CAUTION. Weighted conformal reached 77.7 percent pooled coverage on
  the shifted N-CMAPSS population. Plain split conformal reached 77.4 percent. The gap is
  small (0.36 percentage points, p = 0.033). Both miss the 87 percent bar that was
  pre-stated. The weight estimate collapses on the hardest shifted sub-datasets (DS05 to
  DS07).
- H2, local adaptivity: CONFIRMED. The UACQR-S style arm beats plain CQR on per-unit Winkler
  score on C-MAPSS. The difference is 0.43 over 142 engine units (95 percent CI 0.30 to
  0.57, p < 1e-8).
- H3, fusion versus series: CONFIRMED. The series MDN arm does not beat the C-QRNN pipeline
  on either dataset. On C-MAPSS it is worse by 3.19 Winkler points (p < 1e-30).
- H4, calibrated baselines: CONFIRMED. Calibrated GPR posts the best reliability score on
  N-CMAPSS (0.115 against 0.149 for C-QRNN) and the best Winkler score among arms with
  adequate coverage (38.8).

The prediction cross-check scored 3 hits out of 11 rows (exact 95 percent CI 6 percent to
61 percent). The hits are the point-estimate rows.

The best method depends on the use case. UACQR-S style is the best point estimator.
Distributional CP and HPD-split give the best coverage. EDL leads shifted and conditional
coverage. Calibrated GPR is the reliability reference. Weighted CP keeps the tightest
intervals of the QRNN family under shift.

The full records are in `src/results/`:

| File | Content |
|---|---|
| `results_of_record.md` | Every reported number, with the verification log |
| `analysis_report.md` | Per-subset, per-unit, per-stage, crossing, shift, and ablation analysis |
| `decision_evidence.md` | The hypothesis verdicts and the prediction cross-check |
| `experiment_results.json` | The raw result store (one entry per experiment cell) |
| `run_matrix.json` | The planned-versus-done run matrix |

## Reproducibility notes

- All reported numbers were recomputed from the raw experiment outputs. The verification
  pass covers 564 cells and reports zero mismatches.
- The split seed, the training seeds, and the GPR subsample seed are fixed in the code.
- The experiments ran on a Mac mini M4 Pro with 24 GB of memory.
- The per-cell prediction arrays, the training cache, and the logs are not in this
  repository. They are large and machine-local. Re-run the sweep to rebuild them.

## Repository layout

    src/qrnn_uq_rul/     the method arms, the conformal stage, the metrics, and the pipeline
    src/scripts/         the sweep runner, the analysis toolkit, and the reproduction scripts
    src/tests/           the unit tests and the small synthetic fixtures
    src/results/         the verified result records (see the table above)
    doc/study_report.docx  the companion report (design, results, prediction cross-check)

## Citation

The manuscript is in preparation. Until it is published, cite this repository:

    M. Q. Ignacio, "Uncertainty Quantification for Aero-Engine Remaining Useful Life:
    A Comparative Benchmark," 2026. https://github.com/mqignacio/rul-uq-benchmark

## License

MIT. See `LICENSE`.

## References

- [1] C. M. Bishop, "Mixture Density Networks," 1994.
- [2] J. Lei, M. G'Sell, A. Rinaldo, R. J. Tibshirani, and L. Wasserman, "Distribution-Free
  Predictive Inference For Regression," arXiv:1604.04173, 2017.
- [3] A. Amini, K. Schwarting, A. Soleimany, and D. Rus, "Deep Evidential Regression,"
  NeurIPS, 2020.
- [4] S. Piao, R. Huang, and F. Tsung, "CRULP: Reliable RUL estimation inspired by conformal
  prediction," IEEE Trans. Instrum. Meas., vol. 74, pp. 1-11, 2024.
- [5] L. Basora, A. Viens, M. A. Chao, and X. Olive, "A benchmark on uncertainty
  quantification for deep learning prognostics," Reliab. Eng. Syst. Saf., vol. 253,
  p. 110513, 2025.
- [6] R. Rossellini, R. F. Barber, and R. Willett, "Integrating Uncertainty Awareness into
  Conformalized Quantile Regression," PMLR, pp. 1540-1548, 2024.
- [7] R. Izbicki, G. Shimizu, and R. B. Stern, "Efficient Conformal Regions in High
  Dimensions," J. Mach. Learn. Res., vol. 23, pp. 1-32, 2022.
- [8] R. J. Tibshirani, R. F. Barber, E. J. Candes, and A. Ramdas, "Conformal Prediction
  Under Covariate Shift," arXiv:1904.06019, 2020.
- [9] S. Alijani and H. Najjaran, "WQLCP: Weighted Adaptive Conformal Prediction for Robust
  Uncertainty Quantification Under Distribution Shifts," arXiv:2505.19587, 2025.

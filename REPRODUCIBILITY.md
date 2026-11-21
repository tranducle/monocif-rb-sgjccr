# Reproducibility Guide

## 1. Software and Hardware Environment

The archived 300 five-seed benchmark runs completed in `84.6` minutes (`5,073.3` seconds total wall-clock time) on a single workstation equipped with an NVIDIA GeForce RTX 3090 GPU (`24` GB VRAM), Python `3.12.10`, PyTorch `2.6.0` (CUDA `12.4`), and XGBoost `3.2.0`.

Install exact pinned library versions in a clean virtual environment:

```bash
python -m venv .venv
pip install -r requirements.txt
```

## 2. Fast Verification Without External Data (Under 15 Seconds)

Run the standalone verification script and automated test suite:

```bash
python reproduce_smoke.py
pytest tests/
```

This checks:
1. Exact `0/1` arbitrary-precision rational proofs (`experiments/verify_math_proofs.py` and `experiments/verify_score_gate.py`).
2. Seeded (`seed = 42`) simplex-monotone conservation (`MonoV = 0.000%`, `SimpV = 0.000%`), exact AIPCW telescoping (`< 1e-12`), and Rao-Blackwellized target variance dominance via `src/monocif_core.py`.
3. Numerical consistency of all five archived benchmark JSON files in `results/` and all eight figure CSV tables in `results/figures/`.

## 3. Regenerating Synthetic and Clinical Benchmarks

All multi-seed runs use fixed seeds `[42, 102, 202, 302, 402]`, `B = 500` paired bootstrap resamples for clinical confidence intervals, `B_MC = 200` Monte Carlo draws for calibration operating characteristics, and a pre-specified minimum meaningful effect (`MME = 0.0050`, plus `0.0020` parity bound on `S3-PH`).

### Synthetic Phases (No External Data Needed)

```bash
python experiments/run_benchmarks.py -p E1
python experiments/run_benchmarks.py -p E2
```

- Phase `E1` (`~14.6` min on RTX 3090): `S1` 16-estimator matched-backbone ablation, `S1-2x2` outcome-versus-censoring misspecification, `S3-PH` proportional-hazards control, and `S4-InfCens` shared-frailty sweep (`results/simulation_suite_results.json`).
- Phase `E2` (`~2.8` min on RTX 3090): `S2A` and `S2B` ten-arm recalibration curves across `n_cal` in `{250, 500, 1000, 2500}`, two-by-two lifecycle factorial, and `S3-NullShift` operating characteristics (`results/mechanism_ablation_results.json`).

### Clinical and Longitudinal Governance Phases (Requires Local `data/raw/` Exports)

After preparing local CSV exports under `data/raw/` per `DATA.md`:

```bash
python experiments/run_benchmarks.py -p E3
python experiments/run_benchmarks.py -p E4
python experiments/run_benchmarks.py -p register
```

- Phase `E3` (`~59.5` min on RTX 3090): Rotterdam Breast, Mayo Clinic MGUS2, SEER Prostate, SEER Breast 6-year and 10-year prospective cohorts, 16-subgroup demographic equity audit, and elderly competing-risk DCA (`results/clinical_cohorts_results.json`).
- Phase `E4` (`~7.6` min on RTX 3090): SEER 2018 to 2022 pandemic shock-rebound surveillance, `S5-Lag` 0/1/2-year reporting-lag audit, AJCC grade harmonization and landmarking ablations, and non-SEER temporal recalibration (`results/recalibration_and_governance_results.json` and `results/reporting_lag_results.json`).

### Exporting Figure CSV Tables

To regenerate all eight CSV tables in `results/figures/` directly from `results/*.json`:

```bash
python scripts/export_figure_tables.py
```

## 4. File Integrity Verification

Every distributed file in this repository is checksummed in `RELEASE_SHA256.txt`, and every aggregated summary file under `results/` is hash-bound in `DATA_MANIFEST.json`.

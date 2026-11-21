# MonoCIF-RB and Hier SG-JCCR

Research code, exact rational verification probes, and archived benchmark summaries for:

> **Censoring-Ancillary Competing-Risk Prognosis and Shock-Gated Hierarchical Cross-Cause Maintenance for Population Cancer Registries**
> Evelyn Vo and Tran Duc Le, University of Wisconsin-Stout

## Overview

Population cancer registries combine heavy competing non-cancer mortality in older adults, staggered right-censoring, periodic staging or grading revisions, and external mortality shocks. Under conditional independent right-censoring, censoring survival probabilities factor out of the discrete cause-specific hazard likelihood. This repository implements a four-stage lifecycle on one shared interval softmax hazard-logit representation:

1. **Registry curation safeguards**: historical American Joint Committee on Cancer (AJCC) tumor grade harmonization, diagnosis-time landmarking (`t = 0`), retention of `No/Unknown` therapy records, zero-month survival mapping (`T = 0 -> 1`), and administrative release-date truncation.
2. **MonoCIF-RB training (`src/monocif_core.py`)**: monotone Rao-Blackwellized cumulative incidence estimation using out-of-fold conditional completion targets instead of inverse-censoring weights, with a bounded quadratic variant (`MonoCIF-RB-Brier`) for small rare-event cohorts and a boosted residual head (`MonoCIF-RB (GBDT-Res)`) for tabular registries.
3. **Censoring-free martingale surveillance**: directional interval count screening (`Z_k`) paired with a multi-parameter Rao score test (`Q_k`) over intercept, prognostic slope, and centered log-time drift.
4. **Hier SG-JCCR recalibration**: Bayesian Information Criterion (BIC) selection across `Tier 0 (Hold)`, `Tier 1` cause-specific intercept shifts, and `Tier 2` shared intercept, slope, and log-time updates while maintaining zero monotonicity and simplex violations.

## Quick Verification

Install pinned dependencies and run the single-command verification entry point:

```bash
pip install -r requirements.txt
python reproduce_smoke.py
pytest tests/
```

The smoke script verifies exact `0/1` rational proofs for Theorems 1 to 2 and Propositions 1 to 2, checks seeded simplex conservation and conditional-completion variance reduction in `src/monocif_core.py`, and audits all archived benchmark summaries in `results/`.

## Repository Structure

| Path | Description |
| :- | :- |
| `src/monocif_core.py` | Discrete-time competing-risk estimators, martingale drift tests, hierarchical recalibrators, Aalen-Johansen ICI, IPCW IBS, and competing-risk DCA metrics. |
| `experiments/run_benchmarks.py` | Five-seed simulation, clinical cohort, subgroup equity, and longitudinal reporting-lag benchmark runner. |
| `experiments/verify_math_proofs.py` | Arbitrary-precision (`fractions.Fraction`) verification of Theorem 1, Proposition 1, Theorem 2, and Proposition 2. |
| `experiments/verify_score_gate.py` | Exact rational martingale covariance check and Monte Carlo power comparison of `Q_k` versus `Z_k`. |
| `scripts/export_figure_tables.py` | Exports aggregated CSV figure tables into `results/figures/` from `results/*.json`. |
| `tests/test_reproducibility.py` | Automated unit and reproducibility tests. |
| `results/` | Aggregated five-seed JSON benchmark summaries and figure CSV tables (no individual-level patient records; hash-pinned in `DATA_MANIFEST.json`). |
| `DATA.md` | Official dataset acquisition instructions, cohort schemas, and SEER Data Use Agreement restrictions. |
| `REPRODUCIBILITY.md` | Hardware specifications, fixed random seeds, and step-by-step regeneration commands. |

## Archived Benchmark Summary

| Manuscript Anchor | Headline Metric | Source Script | Archived Summary |
| :- | :- | :- | :- |
| Appendix A (Theorems 1 to 2, Propositions 1 to 2) | Exact `0/1` rational discrepancy | `experiments/verify_math_proofs.py`, `experiments/verify_score_gate.py` | Verified live by `reproduce_smoke.py` |
| Table 3 & Figure 2 (S1, S3-PH, S4-InfCens) | S1 `MonoCIF-RB (NN)` MAE `0.04695`, `0.0%` violations | `experiments/run_benchmarks.py` | `results/simulation_suite_results.json`, `results/figures/s1_causal_ablation.csv` |
| Table 4 & Figure 3a (S2A, S2B, Factorial, S3-NullShift) | S2B `Hier SG-JCCR` MAE `0.04781`, exact-null FWER `5.0%/5.5%` | `experiments/run_benchmarks.py` | `results/mechanism_ablation_results.json`, `results/figures/s2b_recalibration_sample_scaling.csv` |
| Table 5 & Figure 3b (Four Clinical Cohorts) | Rotterdam ICI `0.02334`; SEER 6y IBS `0.16385` | `experiments/run_benchmarks.py` | `results/clinical_cohorts_results.json`, `results/figures/multicohort_clinical_calibration.csv` |
| Table 7 & Figure 4 (16 Subgroups & Elderly DCA) | Age 85+ ICI `0.01423`; Non-Hispanic Black ratio `2.18x` to `4.24x` | `experiments/run_benchmarks.py` | `results/clinical_cohorts_results.json`, `results/figures/seer_subgroup_calibration_audit.csv` |
| Table 8 & Figure 5 (SEER 2018 to 2022 & S5-Lag) | 1-year-lag 2022 Age 75+ `E/O_2 = 1.268` (`+26.8%`) | `experiments/run_benchmarks.py` | `results/recalibration_and_governance_results.json`, `results/reporting_lag_results.json` |

## Citation

If you use this software or benchmark suite, cite the accompanying manuscript:

```bibtex
@article{vo2026monocif,
  author  = {Vo, Evelyn and Le, Tran Duc},
  title   = {Censoring-Ancillary Competing-Risk Prognosis and Shock-Gated Hierarchical Cross-Cause Maintenance for Population Cancer Registries},
  journal = {Journal of Biomedical Informatics},
  year    = {2026},
  note    = {Submitted manuscript; code repository: https://github.com/tranducle/monocif-rb-sgjccr}
}
```

## License

Source code, verification probes, and aggregated summary tables are released under the MIT License (`LICENSE`). Individual-level clinical registry datasets remain governed by their upstream providers (`DATA.md`).

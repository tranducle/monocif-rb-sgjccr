# Data Access, Schemas, and Governance

This repository does **not** redistribute individual-level clinical or cancer registry records. Every file under `results/` contains only aggregated summary metrics, multi-seed averages, or synthetic simulation outputs, verified by SHA-256 checksums in `DATA_MANIFEST.json`.

## 1. Synthetic Competing-Risk Benchmarks

Seven controlled simulation regimes (`S1`, `S1-2x2`, `S2A`, `S2B`, `S3-PH`, `S3-NullShift`, and `S4-InfCens`) are generated deterministically inside `experiments/run_benchmarks.py` using fixed random seeds `[42, 102, 202, 302, 402]` with no external data files required.

## 2. US NCI SEER Breast Cancer Cohort (Restricted Access)

- **Source**: Surveillance, Epidemiology, and End Results (SEER) Program, SEER*Stat Database: Incidence, SEER 17 Registries, November 2024 Submission (2000 to 2022 diagnoses; administrative follow-up cutoff December 31, 2023; released April 2025).
- **Official portal**: https://seer.cancer.gov/data/
- **Documentation**: https://seer.cancer.gov/data-software/documentation/seerstat/nov2024/
- **Access restriction**: Individual-level case listings cannot be redistributed under the standard US National Cancer Institute SEER Research Data Use Agreement (DUA). Authorized investigators may request SEER Research Data access directly from NCI and export breast cancer case listings (diagnosis years 2010 to 2022) to `data/raw/seer_breast_cancer.csv` (ignored by Git via `.gitignore`).
- **Cohort retention**: `827,869` retained cases after excluding `3,470` missing-survival records and `3,425` unknown-cause deaths. Primary predictors exclude diagnosis year, coding-era flags, and post-diagnosis treatment columns (`t = 0` landmarking), while retaining `No/Unknown` therapy rows and mapping zero-month survival records safely to interval `1`.
- **Required SEER*Stat CSV columns**: `Survival months`, `SEER cause-specific death classification`, `Vital status recode (study cutoff used)`, `Year of diagnosis`, `Age recode with <1 year olds and 90+`, `Grade Recode (thru 2017)`, `Derived Summary Grade 2018 (2018+)`, `Tumor Size Over Time Recode (1988+)`, `Regional nodes positive (1988+)`, `Histologic Type ICD-O-3`, `Breast Subtype (2010+)`, `Combined Summary Stage (2004+)`, `Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)`, `Sex`, `Laterality`, `ER Status Recode Breast Cancer (1990+)`, `PR Status Recode Breast Cancer (1990+)`, `Derived HER2 Recode (2010+)`, `RX Summ-Surg Prim Site (1998+)`, `Radiation recode`, and `Chemotherapy recode (yes, no/unk)`.

## 3. Public Clinical Competing-Risk Cohorts (CRAN R Packages)

Three public oncology cohorts are distributed upstream through CRAN and placed locally in `data/raw/` before running Phase `E3` and Phase `E4`:

| Cohort | Retained `N` | Upstream Package | Export Command (R) | Target Local Path |
| :- | :- | :- | :- | :- |
| Rotterdam Breast Cancer | `2,982` | R `survival` (`rotterdam`) | `write.csv(survival::rotterdam, "data/raw/rotterdam.csv", row.names = FALSE)` | `data/raw/rotterdam.csv` |
| Mayo Clinic MGUS2 | `1,384` | R `survival` (`mgus2`) | `write.csv(survival::mgus2, "data/raw/mgus2.csv", row.names = FALSE)` | `data/raw/mgus2.csv` |
| SEER Prostate (`prostateSurvival`) | `14,294` | R `asaur` (`prostateSurvival`) | `write.csv(asaur::prostateSurvival, "data/raw/prostateSurvival.csv", row.names = FALSE)` | `data/raw/prostateSurvival.csv` |

## 4. Authorized Aggregated Artifacts in `results/`

All five multi-seed JSON benchmark summaries and eight aggregated CSV figure tables shipped in `results/` are explicitly cataloged with SHA-256 hashes, MIT licensing, and provenance in `DATA_MANIFEST.json`.

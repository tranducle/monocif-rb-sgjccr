"""
Exports aggregated CSV figure tables directly from results/*.json into results/figures/
with deterministic SHA-256 checksums.
"""
from __future__ import annotations

import csv
import hashlib
import json
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RES_DIR = REPO_ROOT / "results"
FIG_DATA = RES_DIR / "figures"
FIG_DATA.mkdir(parents=True, exist_ok=True)

e1 = json.loads((RES_DIR / "simulation_suite_results.json").read_text(encoding="utf-8"))
e2 = json.loads((RES_DIR / "mechanism_ablation_results.json").read_text(encoding="utf-8"))
e3 = json.loads((RES_DIR / "clinical_cohorts_results.json").read_text(encoding="utf-8"))
e4 = json.loads((RES_DIR / "recalibration_and_governance_results.json").read_text(encoding="utf-8"))
lag_res = json.loads((RES_DIR / "reporting_lag_results.json").read_text(encoding="utf-8"))

manifest = {}

def write_csv(name: str, headers: list[str], rows: list[list], source_json: str) -> None:
    p = FIG_DATA / name
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(headers)
        w.writerows(rows)
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    manifest[name] = {
        "path": f"results/figures/{name}",
        "sha256": h,
        "source_artifact": source_json,
        "rows": len(rows),
    }

# 1. Fig 2a: S1 Stepwise Causal MAE Decomposition (10 key matched-backbone arms + Oracle ref)
s1_arms = [
    ("MonoCIF-RB (NN)", "MonoCIF-RB", "proposed"),
    ("Discrete-Hazard-NN", "Disc-Haz-NN", "baseline1"),
    ("MonoCIF-RB-Brier (NN)", "RB-Brier", "ablation"),
    ("MonoCIF-CRNLL (CovIPCW)", "Mono-CovIPCW", "baseline2"),
    ("MonoCIF-CRNLL (KM)", "Mono-KM", "baseline2"),
    ("PerHorizon-BCE", "PerHoriz-BCE", "failure"),
    ("CRNLL-KM", "CRNLL-KM", "failure"),
    ("DR-MonoCIF-Brier", "DR-Brier", "failure"),
    ("CS-XGBoost (PH)", "CS-XGB (PH)", "baseline3"),
    ("Cutoff-Status XGB", "Cutoff-XGB", "failure"),
]
rows_2a = []
for idx, (full_name, short_lbl, role) in enumerate(s1_arms, start=1):
    d = e1["S1_matched_backbone"][full_name]
    rows_2a.append([
        idx, short_lbl, role,
        f"{d['MAE']['mean']:.5f}", f"{d['MAE']['sd']:.5f}",
        f"{d['MonoV']['mean'] * 100.0:.1f}", f"{d['SimpV']['mean'] * 100.0:.1f}",
    ])
write_csv(
    "s1_causal_ablation.csv",
    ["idx", "label", "role", "mae_mean", "mae_sd", "monov_pct", "simpv_pct"],
    rows_2a,
    "results/simulation_suite_results.json",
)

# 2. Fig 2b: S4-InfCens Shared-Frailty Sweep (gamma = 0.0, 0.5, 1.0, 1.5)
rows_2b = []
for gam in (0.0, 0.5, 1.0, 1.5):
    gk = f"gamma_{gam:.1f}"
    gd = e1["S4_InfCens"][gk]
    rows_2b.append([
        f"{gam:.1f}",
        f"{gd['MonoCIF-RB (NN)']['MAE']['mean']:.5f}", f"{gd['MonoCIF-RB (NN)']['MAE']['sd']:.5f}",
        f"{gd['Discrete-Hazard-NN']['MAE']['mean']:.5f}", f"{gd['Discrete-Hazard-NN']['MAE']['sd']:.5f}",
        f"{gd['MonoCIF-CRNLL (CovIPCW)']['MAE']['mean']:.5f}", f"{gd['MonoCIF-CRNLL (CovIPCW)']['MAE']['sd']:.5f}",
        f"{gd['DR-MonoCIF-Brier']['MAE']['mean']:.5f}", f"{gd['DR-MonoCIF-Brier']['MAE']['sd']:.5f}",
    ])
write_csv(
    "s4_shared_frailty_sensitivity.csv",
    ["gamma", "rb_mean", "rb_sd", "haz_mean", "haz_sd", "cov_mean", "cov_sd", "dr_mean", "dr_sd"],
    rows_2b,
    "results/simulation_suite_results.json",
)

# 3. Fig 3a: SEER 2018-2022 Yearly Competing Mortality (6m & 12m F2 % + Z12, Q12)
rows_3a = []
for yr in ("2018", "2019", "2020", "2021", "2022"):
    d = lag_res["SEER_2018_2022_yearly"][yr]
    rows_3a.append([
        yr, d["N"],
        f"{d['obs_6m_F2'] * 100.0:.4f}", f"{d['obs_12m_F2'] * 100.0:.4f}",
        f"{d['EO_12m_F2']:.4f}", f"{d['Z_12m_census']['Z2']:.2f}", f"{d['Z_12m_census']['Q2']:.1f}",
    ])
write_csv(
    "seer_yearly_competing_mortality.csv",
    ["year", "N", "obs_6m_f2_pct", "obs_12m_f2_pct", "eo12_f2", "z12_f2", "q12_f2"],
    rows_3a,
    "results/reporting_lag_results.json",
)

# 4. Fig 3b: S5-Lag 0/1/2-Year Reporting-Lag Cause-2 E/O across 2019-2022 (overall + Age 75+ for lag 1y)
rows_3b = []
for yr in ("2019", "2020", "2021", "2022"):
    l0 = lag_res["S5_Lag"]["lag_0y"]["by_year"][yr]
    l1 = lag_res["S5_Lag"]["lag_1y"]["by_year"][yr]
    l2 = lag_res["S5_Lag"]["lag_2y"]["by_year"][yr]
    frz = lag_res["SEER_2018_2022_yearly"][yr]["EO_12m_F2"]
    rows_3b.append([
        yr,
        f"{l0['EO2']:.4f}", f"{l1['EO2']:.4f}", f"{l2['EO2']:.4f}",
        f"{frz:.4f}", f"{l1['EO2_75p']:.4f}",
    ])
write_csv(
    "reporting_lag_calibration_ratio.csv",
    ["year", "lag0_eo2", "lag1_eo2", "lag2_eo2", "frozen_eo2", "lag1_eo2_75p"],
    rows_3b,
    "results/reporting_lag_results.json",
)

# 5. Fig 4a: SEER Breast 2016-2017 Subgroup ICI Audit (selected representative Age, Race, Subtype, Stage strata)
sg_order = [
    ("Age_<50", "Age <50"),
    ("Age_65_74", "Age 65-74"),
    ("Age_75_84", "Age 75-84"),
    ("Age_85+", "Age 85+"),
    ("Race_NHW", "NH White"),
    ("Race_Hispanic", "Hispanic"),
    ("Race_NHAPI", "NH API"),
    ("Race_NHB", "NH Black"),
    ("Subtype_HRp_HER2n", "HR+/HER2-"),
    ("Subtype_TNBC", "TNBC"),
    ("Stage_Localized", "Localized"),
    ("Stage_Regional", "Regional"),
]
rows_4a = []
for idx, (k, lbl) in enumerate(sg_order, start=1):
    sd = e3["SEER_Breast_Subgroup_Audit"][k]
    rows_4a.append([
        idx, lbl, sd["N"], f"{sd['competing_share'] * 100.0:.1f}",
        f"{sd['MonoCIF-RB (GBDT-Res)']['ICI'] * 1000.0:.3f}",
        f"{sd['MonoCIF-RB (NN)']['ICI'] * 1000.0:.3f}",
        f"{sd['CS-XGBoost (PH)']['ICI'] * 1000.0:.3f}",
        f"{sd['Cutoff-Status XGB']['ICI'] * 1000.0:.3f}",
    ])
write_csv(
    "seer_subgroup_calibration_audit.csv",
    ["idx", "label", "N", "comp_pct", "rb_gbdt_mici", "rb_nn_mici", "csxgb_mici", "cutxgb_mici"],
    rows_4a,
    "results/clinical_cohorts_results.json",
)

# 6. Fig 4b: SEER Breast 6y Age 75+ 5-Year Competing-Risk Oncology DCA (pt = 0.05..0.25)
b6 = e3["SEER_Breast_Prospective_6y"]["paired_bootstrap_500"]
rows_4b = []
for pt_int in (5, 10, 15, 20, 25):
    k = f"75p_NB@{pt_int:02d}"
    rows_4b.append([
        f"{pt_int / 100.0:.2f}",
        f"{b6['MonoCIF-RB (GBDT-Res)'][k]:.5f}",
        f"{b6['MonoCIF-RB (NN)'][k]:.5f}",
        f"{b6['CS-XGBoost (PH)'][k]:.5f}",
        f"{b6['Fine-Gray'][k]:.5f}",
        f"{b6['DeepHit'][k]:.5f}",
    ])
write_csv(
    "seer_elderly_oncology_dca.csv",
    ["pt", "rb_gbdt", "rb_nn", "csxgb", "finegray", "deephit"],
    rows_4b,
    "results/clinical_cohorts_results.json",
)

# 7. Fig 5a: S2B Complex Shift Recalibration Curve across n_cal = 250, 500, 1000, 2500
rows_5a = []
for xidx, n_cal in enumerate((250, 500, 1000, 2500), start=1):
    d = e2["S2B_by_ncal"][f"n_{n_cal}"]
    rows_5a.append([
        xidx, n_cal,
        f"{d['9. Ours: Hier SG-JCCR']['MAE']['mean']:.5f}", f"{d['9. Ours: Hier SG-JCCR']['MAE']['sd']:.5f}",
        f"{d['6. Indep-Logit (Hazard)']['MAE']['mean']:.5f}",
        f"{d['7. Booth Sep-Baseline (SIM 23)']['MAE']['mean']:.5f}",
        f"{d['5. Indep-Logit (CIF)']['MAE']['mean']:.5f}",
        f"{d['2. AJ-Recal (AISTATS 26)']['MAE']['mean']:.5f}",
        f"{d['9b. Ablation: 1-df Z-Gate SG-JCCR']['MAE']['mean']:.5f}",
        f"{d['1. Frozen']['MAE']['mean']:.5f}",
    ])
write_csv(
    "s2b_recalibration_sample_scaling.csv",
    ["xidx", "n_cal", "hier_mean", "hier_sd", "hazindep", "booth", "cifindep", "ajrecal", "z1df", "frozen"],
    rows_5a,
    "results/mechanism_ablation_results.json",
)

# 8. Fig 5b: Four Clinical Cohorts 5-Seed ICI (scaled x1000 -> milli-ICI for clean reading)
cohorts_5b = [
    ("Rotterdam_Breast", "Rotterdam"),
    ("Mayo_MGUS2", "MGUS2"),
    ("SEER_Prostate_Boundary", "SEER Prost."),
    ("SEER_Breast_Prospective_6y", "SEER Br. 6y"),
]
rows_5b = []
for idx, (ck, clbl) in enumerate(cohorts_5b, start=1):
    ps = e3[ck]["per_seed_summary"]
    rows_5b.append([
        idx, clbl,
        f"{ps['MonoCIF-RB (NN)']['ICI']['mean'] * 1000.0:.3f}", f"{ps['MonoCIF-RB (NN)']['ICI']['sd'] * 1000.0:.3f}",
        f"{ps['MonoCIF-RB-Brier (NN)']['ICI']['mean'] * 1000.0:.3f}", f"{ps['MonoCIF-RB-Brier (NN)']['ICI']['sd'] * 1000.0:.3f}",
        f"{ps['MonoCIF-RB (GBDT-Res)']['ICI']['mean'] * 1000.0:.3f}", f"{ps['MonoCIF-RB (GBDT-Res)']['ICI']['sd'] * 1000.0:.3f}",
        f"{ps['Discrete-Hazard-NN']['ICI']['mean'] * 1000.0:.3f}", f"{ps['Discrete-Hazard-NN']['ICI']['sd'] * 1000.0:.3f}",
        f"{ps['CS-XGBoost (PH)']['ICI']['mean'] * 1000.0:.3f}", f"{ps['CS-XGBoost (PH)']['ICI']['sd'] * 1000.0:.3f}",
        f"{ps['Cutoff-Status XGB']['ICI']['mean'] * 1000.0:.3f}", f"{ps['Cutoff-Status XGB']['ICI']['sd'] * 1000.0:.3f}",
    ])
write_csv(
    "multicohort_clinical_calibration.csv",
    ["idx", "cohort", "rb_nn", "rb_nn_sd", "rb_brier", "rb_brier_sd", "rb_gbdt", "rb_gbdt_sd", "haz_nn", "haz_nn_sd", "csxgb", "csxgb_sd", "cutxgb", "cutxgb_sd"],
    rows_5b,
    "results/clinical_cohorts_results.json",
)

(FIG_DATA / "figure_tables_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"Exported {len(manifest)} audited CSVs to {FIG_DATA}")
for k, v in manifest.items():
    print(f"  {k:32s} rows={v['rows']:2d} sha256={v['sha256'][:16]}...")

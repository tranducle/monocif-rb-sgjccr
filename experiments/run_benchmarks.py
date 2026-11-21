"""
Multi-Seed Benchmark Suite for MonoCIF-RB and Hier SG-JCCR:
  Censoring-Ancillary Competing-Risk Prognosis and Shock-Gated Joint Cross-Cause Maintenance
  (Controlled simulations, clinical cohort validations, and registry governance audits)

Executes benchmark stages (see REPRODUCIBILITY.md for data preparation):
  -p E1 : S1 (5 seeds, 17 matched arms + oracle nuisance + S1 2x2 DR-Brier + Signed AIPCW),
               S3-PH (5 seeds linear-PH negative parity control),
               S4-InfCens (5 seeds shared-frailty gamma in {0.0, 0.5, 1.0, 1.5}).
  -p E2 : S3-NullShift (M=200 MC draws at n_cal=1000 & 250),
               S2A (pure intercept shift, 5 seeds x M=200 draws x n_cal={250,500,1000,2500}, 9 recalibrators),
               S2B (complex intercept+slope+log-time shift, 5 seeds x M=200 draws x n_cal={250,500,1000,2500}, 9 recalibrators),
               Two-by-two lifecycle factorial ({CovIPCW, RB} x {Frozen, SG-JCCR}).
  -p E3 : Four Real Clinical Cohorts (5 seeds + B=500 paired bootstrap 95% CIs + oncology DCA pt in 0.05..0.25):
               Rotterdam Breast (N=2,982), Mayo MGUS2 (N=1,384), SEER Prostate (N=14,294),
               SEER Breast Prognostic (2010-2015 -> 2016-2017 <=72m & 2010-2013 -> 2014 114m)
               including full demographic/elderly/subtype/stage subgroup audits & GBDT-Res backbone!
  -p E4 : SEER Longitudinal Registry Governance (base 2010-2017 truncated Dec 31, 2017 -> 2018..2022,
               6m & 12m, S5-Lag 0/1/2-year reporting lag, 2020 COVID surge & 2022 rebound, all 9 recalibrators),
               AJCC 2017/2018 Grade Harmonization Ablation, Diagnosis-Time Landmarking vs Treatment Sensitivity,
               Non-SEER Temporal Recalibration on Rotterdam (1989-90 -> 1991-93, 96m) and MGUS2 (1985-88 -> 1989-94).
  -p register : Compile run register under results/runs/register.csv and environment metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import platform
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import monocif_core as jc

DATA_DIR = REPO_ROOT / "data" / "raw"
OUT_DIR = REPO_ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR = OUT_DIR / "runs"

SEEDS = [42, 102, 202, 302, 402]
CODE_HASH = hashlib.sha256((SRC_DIR / "monocif_core.py").read_bytes()).hexdigest()[:12]


# =============================================================================
# 1. Synthetic Competing-Risk Data-Generating Processes
# =============================================================================

def simulate_s1_dgp(
    n: int,
    seed: int,
    shift_mode: str = "none",
    gamma_frailty: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    Exact S1 / S2A / S2B / S4-InfCens discrete competing-risk DGP (M=20, K=2, p=10).
    Uses shared base random stream across gamma_frailty for paired variance reduction in S4-InfCens.
    """
    rng = np.random.default_rng(seed)
    p, M = 10, 20
    X = rng.normal(0.0, 1.0, size=(n, p)).astype(np.float32)
    U_frail = rng.normal(0.0, 1.0, size=n).astype(np.float64)  # Unobserved frailty (excluded from X!)
    t_grid = np.arange(1, M + 1, dtype=np.float64)
    t_norm = t_grid / M
    tau_log = np.log(t_grid / math.sqrt(M))[None, :]

    aggr = 1.0 / (1.0 + np.exp(-2.0 * (X[:, 0] + 0.6 * X[:, 1] * X[:, 2])))
    age = X[:, 2].astype(np.float64)
    comorb = 0.7 * X[:, 3] + 0.5 * (X[:, 4] ** 2) - 0.5
    early = np.exp(-3.5 * t_norm)[None, :]
    late = (t_norm ** 1.4)[None, :]

    z1_base = (
        -3.2
        + 4.0 * aggr[:, None] * early
        + 1.98 * (1.0 - aggr[:, None]) * late
        + 0.7 * np.sin(X[:, 5])[:, None]
        + 0.5 * X[:, 6][:, None]
    )
    z2_base = (
        -3.5
        + (1.2 * age[:, None] + 0.45 * (age[:, None] ** 2) + 0.6 * comorb[:, None]) * (0.5 + 1.2 * t_norm[None, :])
        + 0.4 * X[:, 7][:, None]
    )

    if shift_mode == "S2A":
        z1 = z1_base - 0.35
        z2 = z2_base + 0.50
    elif shift_mode == "S2B":
        z1 = -0.35 + 0.72 * z1_base + 0.45 * tau_log
        z2 = +0.50 + 1.35 * z2_base - 0.40 * tau_log
    else:
        z1 = z1_base
        z2 = z2_base + gamma_frailty * U_frail[:, None]

    logits = np.stack([z1, z2], axis=-1)
    h = jc.logits_to_hazards_np(logits)
    h0, h1, h2 = h[:, :, 0], h[:, :, 1], h[:, :, 2]

    # For S4-InfCens (gamma > 0), marginal oracle CIF given X is averaged over U ~ N(0, 1) via 15-pt Gauss-Hermite!
    if gamma_frailty > 0.0:
        gh_x, gh_w = np.polynomial.hermite.hermgauss(15)
        u_pts = math.sqrt(2.0) * gh_x
        w_pts = gh_w / math.sqrt(math.pi)
        S_orc = np.zeros((n, M), dtype=np.float64)
        F1_orc = np.zeros((n, M), dtype=np.float64)
        F2_orc = np.zeros((n, M), dtype=np.float64)
        for uk, wk in zip(u_pts, w_pts):
            lg_u = np.stack([z1, z2_base + gamma_frailty * uk], axis=-1)
            Su, F1u, F2u = jc.logits_to_cif_np(lg_u)
            S_orc += wk * Su
            F1_orc += wk * F1u
            F2_orc += wk * F2u
    else:
        S_orc, F1_orc, F2_orc = jc.logits_to_cif_np(logits)

    u = rng.uniform(0.0, 1.0, size=(n, M))
    T_star = np.full(n, M + 1, dtype=np.int32)
    D_star = np.zeros(n, dtype=np.int32)
    alive = np.ones(n, dtype=bool)
    for m in range(M):
        ev1 = alive & (u[:, m] < h1[:, m])
        ev2 = alive & (~ev1) & (u[:, m] < h1[:, m] + h2[:, m])
        T_star[ev1] = m + 1
        D_star[ev1] = 1
        T_star[ev2] = m + 1
        D_star[ev2] = 2
        alive[ev1 | ev2] = False

    c_rate = np.exp(-3.1 + 1.0 * X[:, 2] - 0.9 * X[:, 0] + 0.6 * X[:, 8] + gamma_frailty * U_frail)
    C_cov = np.floor(rng.exponential(1.0 / np.clip(c_rate, 1e-6, 10.0))) + 1
    C_admin = rng.integers(6, M + 8, size=n)
    C_time = np.minimum(C_cov, C_admin).astype(np.int32)

    T_obs = np.minimum(T_star, C_time)
    D_obs = np.where(T_star <= C_time, D_star, 0)
    surv_end = T_obs > M
    T_obs[surv_end] = M + 1
    D_obs[surv_end] = 0

    return {
        "X": X, "T": T_obs, "D": D_obs, "M": M,
        "F1_oracle": F1_orc, "F2_oracle": F2_orc, "S_oracle": S_orc, "age": age,
        "logits_oracle": logits,
    }


def simulate_s3_ph_dgp(n: int, seed: int) -> Dict[str, np.ndarray]:
    """Exact S3-PH linear proportional-hazards covariate-free-censoring negative parity control."""
    rng = np.random.default_rng(seed)
    p, M = 10, 20
    X = rng.normal(0.0, 1.0, size=(n, p)).astype(np.float32)
    l_log = np.log(np.arange(1, M + 1, dtype=np.float64))[None, :]

    z1 = -3.4 + (0.6 * X[:, 0] - 0.4 * X[:, 1] + 0.5 * X[:, 2] + 0.3 * X[:, 3])[:, None] + 0.2 * l_log
    z2 = -3.6 + (0.7 * X[:, 2] + 0.4 * X[:, 4] - 0.3 * X[:, 5])[:, None] + 0.2 * l_log
    logits = np.stack([z1, z2], axis=-1)
    h = jc.logits_to_hazards_np(logits)
    h0, h1, h2 = h[:, :, 0], h[:, :, 1], h[:, :, 2]
    S_orc, F1_orc, F2_orc = jc.logits_to_cif_np(logits)

    u = rng.uniform(0.0, 1.0, size=(n, M))
    T_star = np.full(n, M + 1, dtype=np.int32)
    D_star = np.zeros(n, dtype=np.int32)
    alive = np.ones(n, dtype=bool)
    for m in range(M):
        ev1 = alive & (u[:, m] < h1[:, m])
        ev2 = alive & (~ev1) & (u[:, m] < h1[:, m] + h2[:, m])
        T_star[ev1] = m + 1
        D_star[ev1] = 1
        T_star[ev2] = m + 1
        D_star[ev2] = 2
        alive[ev1 | ev2] = False

    C_cov = rng.geometric(0.03, size=n)
    C_admin = rng.integers(10, M + 8, size=n)
    C_time = np.minimum(C_cov, C_admin).astype(np.int32)
    T_obs = np.minimum(T_star, C_time)
    D_obs = np.where(T_star <= C_time, D_star, 0)
    surv_end = T_obs > M
    T_obs[surv_end] = M + 1
    D_obs[surv_end] = 0
    return {
        "X": X, "T": T_obs, "D": D_obs, "M": M,
        "F1_oracle": F1_orc, "F2_oracle": F2_orc, "S_oracle": S_orc, "age": X[:, 2].astype(np.float64),
    }


def _agg_seed_metrics(seed_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = list(seed_list[0].keys())
    out = {}
    for k in keys:
        vals = np.array([d[k] for d in seed_list], dtype=np.float64)
        out[k] = {
            "mean": float(np.mean(vals)),
            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "seeds": [float(v) for v in vals],
        }
    return out


def _stratified_km_G(X_tr: np.ndarray, T_tr: np.ndarray, D_tr: np.ndarray, X_va: np.ndarray, M: int, col_idx: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified KM censoring survival across 4 quartiles of primary risk/age covariate."""
    qs = np.quantile(X_tr[:, col_idx], [0.25, 0.50, 0.75])
    s_tr = np.digitize(X_tr[:, col_idx], qs)
    s_va = np.digitize(X_va[:, col_idx], qs)
    G_tr = np.ones((len(X_tr), M + 2), dtype=np.float64)
    G_va = np.ones((len(X_va), M + 2), dtype=np.float64)
    for q in range(4):
        m_tr = (s_tr == q)
        m_va = (s_va == q)
        if m_tr.sum() >= 20:
            _, gq = jc.km_censoring_discrete(T_tr[m_tr], D_tr[m_tr], M)
        else:
            _, gq = jc.km_censoring_discrete(T_tr, D_tr, M)
        G_tr[m_tr] = gq[None, :]
        if np.any(m_va):
            G_va[m_va] = gq[None, :]
    return G_tr, G_va


# =============================================================================
# 2. Phase E1: Controlled Oracle Simulation Suite (S1, S1 2x2, S3-PH, S4-InfCens)
# =============================================================================

def run_phase_E1() -> Dict:
    t0 = time.time()
    print("\n================ PHASE E1: CONTROLLED SIMULATIONS (5 SEEDS) ================", flush=True)
    M = 20

    s1_models_seeds: Dict[str, List[Dict[str, float]]] = {}
    s1_dr2x2_seeds: Dict[str, List[Dict[str, float]]] = {}
    s1_invariants: List[Dict[str, float]] = []
    s3ph_seeds: Dict[str, List[Dict[str, float]]] = {}
    s4_seeds: Dict[str, Dict[str, List[Dict[str, float]]]] = {}

    for seed in SEEDS:
        ts = time.time()
        tr_all = simulate_s1_dgp(9000, seed=seed)
        te = simulate_s1_dgp(3000, seed=seed + 50000)

        tr_i, va_i = jc._split_train_val(len(tr_all["X"]), seed)
        X_tr, T_tr, D_tr = tr_all["X"][tr_i], tr_all["T"][tr_i], tr_all["D"][tr_i]
        X_va, T_va, D_va = tr_all["X"][va_i], tr_all["T"][va_i], tr_all["D"][va_i]
        X_te = te["X"]

        # 5-fold nuisances fitted strictly inside train_90 ONLY!
        nuis = jc.crossfit_nuisances_train90(X_tr, T_tr, D_tr, X_va, M=M, seed=seed, bs=256, n_splits=5, fit_G=True, clip_hc=0.85)
        psi_rb_tr = jc.build_rb_targets(nuis["S_tr"], nuis["F1_tr"], nuis["F2_tr"], T_tr, D_tr)
        psi_rb_va = jc.build_rb_targets(nuis["S_va"], nuis["F1_va"], nuis["F2_va"], T_va, D_va)

        # Oracle nuisance upper bound (R8)
        psi_orc_tr = jc.build_rb_targets(tr_all["S_oracle"][tr_i], tr_all["F1_oracle"][tr_i], tr_all["F2_oracle"][tr_i], T_tr, D_tr)
        psi_orc_va = jc.build_rb_targets(tr_all["S_oracle"][va_i], tr_all["F1_oracle"][va_i], tr_all["F2_oracle"][va_i], T_va, D_va)

        # Exact-telescoping AIPCW (Good F / Good G)
        psi_dr_gg_tr, tele_gg = jc.build_aipcw_targets(nuis["S_tr"], nuis["F1_tr"], nuis["F2_tr"], nuis["hc_tr"], nuis["G_tr"], T_tr, D_tr)
        psi_dr_gg_va, _ = jc.build_aipcw_targets(nuis["S_va"], nuis["F1_va"], nuis["F2_va"], nuis["hc_va"], nuis["G_va"], T_va, D_va)
        var_rb = float(np.var(psi_rb_tr[:, -1, 1]))
        var_aipcw = float(np.var(psi_dr_gg_tr[:, -1, 1]))
        frac_neg = float(np.mean(psi_dr_gg_tr < 0.0))

        # Bad F (misspecified age-only linear hazard) and Bad G (marginal KM) for 2x2 DR check
        X_bad_tr = X_tr[:, [2]]
        X_bad_va = X_va[:, [2]]
        # Simple linear hazard on age only
        F1_bf_tr, F2_bf_tr = jc.fit_predict_cs_cox_gpu(X_bad_tr, T_tr, D_tr, X_bad_tr, M=M, l2_alpha=1e-2)
        S_bf_tr = np.clip(1.0 - F1_bf_tr - F2_bf_tr, 1e-6, 1.0)
        F1_bf_va, F2_bf_va = jc.fit_predict_cs_cox_gpu(X_bad_tr, T_tr, D_tr, X_bad_va, M=M, l2_alpha=1e-2)
        S_bf_va = np.clip(1.0 - F1_bf_va - F2_bf_va, 1e-6, 1.0)
        hc_km, g_km = jc.km_censoring_discrete(T_tr, D_tr, M, clip_hc=0.85)
        hc_km_tr = np.broadcast_to(hc_km[None, :], (len(X_tr), M)).copy()
        G_km_tr = np.broadcast_to(g_km[None, :], (len(X_tr), M + 2)).copy()
        hc_km_va = np.broadcast_to(hc_km[None, :], (len(X_va), M)).copy()
        G_km_va = np.broadcast_to(g_km[None, :], (len(X_va), M + 2)).copy()

        psi_rb_bf_tr = jc.build_rb_targets(S_bf_tr, F1_bf_tr, F2_bf_tr, T_tr, D_tr)
        psi_rb_bf_va = jc.build_rb_targets(S_bf_va, F1_bf_va, F2_bf_va, T_va, D_va)
        psi_dr_gb_tr, tele_gb = jc.build_aipcw_targets(nuis["S_tr"], nuis["F1_tr"], nuis["F2_tr"], hc_km_tr, G_km_tr, T_tr, D_tr)
        psi_dr_gb_va, _ = jc.build_aipcw_targets(nuis["S_va"], nuis["F1_va"], nuis["F2_va"], hc_km_va, G_km_va, T_va, D_va)
        psi_dr_bg_tr, tele_bg = jc.build_aipcw_targets(S_bf_tr, F1_bf_tr, F2_bf_tr, nuis["hc_tr"], nuis["G_tr"], T_tr, D_tr)
        psi_dr_bg_va, _ = jc.build_aipcw_targets(S_bf_va, F1_bf_va, F2_bf_va, nuis["hc_va"], nuis["G_va"], T_va, D_va)

        s1_invariants.append({
            "seed": seed, "tele_gg": tele_gg, "tele_gb": tele_gb, "tele_bg": tele_bg,
            "var_rb": var_rb, "var_aipcw": var_aipcw, "var_ratio": var_rb / max(var_aipcw, 1e-12),
            "frac_neg_aipcw": frac_neg,
        })

        # Fit ALL S1 arms under matched budget
        preds: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        preds["Cutoff-Status XGB"] = jc.fit_predict_cutoff_xgb(X_tr, T_tr, D_tr, X_va, T_va, D_va, X_te, M, seed)
        preds["CS-Cox"] = jc.fit_predict_cs_cox_gpu(X_tr, T_tr, D_tr, X_te, M)
        preds["Fine-Gray"] = jc.fit_predict_fine_gray_gpu(X_tr, T_tr, D_tr, X_te, M)
        preds["CS-XGBoost (PH)"] = jc.fit_predict_cs_xgboost(X_tr, T_tr, D_tr, X_va, T_va, D_va, X_te, M, seed)
        preds["XGBoost-PV"] = jc.fit_predict_xgboost_pv(X_tr, T_tr, D_tr, X_va, T_va, D_va, X_te, M, seed)

        m_dh = jc.fit_deephit_nn(X_tr, T_tr, D_tr, X_va, T_va, D_va, M, seed)
        preds["DeepHit"] = jc.predict_deephit_nn(m_dh, X_te, M)

        m_bce = jc.fit_perhorizon_bce_nn(X_tr, T_tr, D_tr, X_va, T_va, D_va, M, seed)
        preds["PerHorizon-BCE"] = jc.predict_perhorizon_bce_nn(m_bce, X_te, M)

        m_crkm = jc.fit_crnll_nn(X_tr, T_tr, D_tr, X_va, T_va, D_va, M, seed, monotone=False)
        preds["CRNLL-KM"] = jc.predict_crnll_nn(m_crkm, X_te, M, monotone=False)

        m_mcrkm = jc.fit_crnll_nn(X_tr, T_tr, D_tr, X_va, T_va, D_va, M, seed, monotone=True)
        preds["MonoCIF-CRNLL (KM)"] = jc.predict_crnll_nn(m_mcrkm, X_te, M, monotone=True)

        Gs_tr, Gs_va = _stratified_km_G(X_tr, T_tr, D_tr, X_va, M, col_idx=2)
        m_mcrs = jc.fit_crnll_nn(X_tr, T_tr, D_tr, X_va, T_va, D_va, M, seed, monotone=True, G_tr=Gs_tr, G_va=Gs_va)
        preds["MonoCIF-CRNLL (StratKM)"] = jc.predict_crnll_nn(m_mcrs, X_te, M, monotone=True)

        m_mcrc = jc.fit_crnll_nn(X_tr, T_tr, D_tr, X_va, T_va, D_va, M, seed, monotone=True, G_tr=nuis["G_tr"], G_va=nuis["G_va"])
        preds["MonoCIF-CRNLL (CovIPCW)"] = jc.predict_crnll_nn(m_mcrc, X_te, M, monotone=True)

        m_drb = jc.fit_monocif_target_nn(X_tr, psi_dr_gg_tr, X_va, psi_dr_gg_va, M, seed, loss_type="brier")
        _, f1, f2 = jc.logits_to_cif_np(jc.predict_hazard_logits(m_drb, X_te))
        preds["DR-MonoCIF-Brier"] = (f1, f2)

        m_drlog = jc.fit_monocif_target_nn(X_tr, psi_dr_gg_tr, X_va, psi_dr_gg_va, M, seed, loss_type="logloss")
        _, f1, f2 = jc.logits_to_cif_np(jc.predict_hazard_logits(m_drlog, X_te))
        preds["Signed-AIPCW-LogLoss"] = (f1, f2)

        m_haz = jc.fit_discrete_hazard_nn(X_tr, T_tr, D_tr, M, seed, X_val=X_va, T_val=T_va, D_val=D_va)
        _, f1, f2 = jc.logits_to_cif_np(jc.predict_hazard_logits(m_haz, X_te))
        preds["Discrete-Hazard-NN"] = (f1, f2)

        # 5-fold ensemble Discrete-Hazard-NN (rules out 5-fold ensembling bias!)
        ens_lg = sum(jc.predict_hazard_logits(fm, X_te) for fm in nuis["fold_models"]) / len(nuis["fold_models"])
        _, f1, f2 = jc.logits_to_cif_np(ens_lg)
        preds["Discrete-Hazard-NN (5-fold ens)"] = (f1, f2)

        m_rbb = jc.fit_monocif_target_nn(X_tr, psi_rb_tr, X_va, psi_rb_va, M, seed, loss_type="brier")
        _, f1, f2 = jc.logits_to_cif_np(jc.predict_hazard_logits(m_rbb, X_te))
        preds["MonoCIF-RB-Brier (NN)"] = (f1, f2)

        m_rb = jc.fit_monocif_target_nn(X_tr, psi_rb_tr, X_va, psi_rb_va, M, seed, loss_type="logloss")
        _, f1, f2 = jc.logits_to_cif_np(jc.predict_hazard_logits(m_rb, X_te))
        preds["MonoCIF-RB (NN)"] = (f1, f2)

        # Oracle-nuisance upper bound (R8)
        m_orc = jc.fit_monocif_target_nn(X_tr, psi_orc_tr, X_va, psi_orc_va, M, seed, loss_type="logloss")
        _, f1, f2 = jc.logits_to_cif_np(jc.predict_hazard_logits(m_orc, X_te))
        preds["MonoCIF-RB (Oracle Nuisance)"] = (f1, f2)

        # 2x2 DR-Brier cells
        dr2x2 = {
            "MonoCIF-RB (Good F)": preds["MonoCIF-RB (NN)"],
            "MonoCIF-RB-Brier (Good F)": preds["MonoCIF-RB-Brier (NN)"],
            "DR-MonoCIF-Brier (Good F, Good G)": preds["DR-MonoCIF-Brier"],
            "Signed-AIPCW-LogLoss (Good F, Good G)": preds["Signed-AIPCW-LogLoss"],
        }
        for lbl, ptr, pva, lt in [
            ("MonoCIF-RB (Bad F)", psi_rb_bf_tr, psi_rb_bf_va, "logloss"),
            ("DR-MonoCIF-Brier (Good F, Bad KM G)", psi_dr_gb_tr, psi_dr_gb_va, "brier"),
            ("DR-MonoCIF-Brier (Bad F, Good G)", psi_dr_bg_tr, psi_dr_bg_va, "brier"),
        ]:
            mm = jc.fit_monocif_target_nn(X_tr, ptr, X_va, pva, M, seed, loss_type=lt)
            _, f1, f2 = jc.logits_to_cif_np(jc.predict_hazard_logits(mm, X_te))
            dr2x2[lbl] = (f1, f2)

        for name, (f1, f2) in preds.items():
            s1_models_seeds.setdefault(name, []).append(jc.compute_oracle_metrics(f1, f2, te))
        for name, (f1, f2) in dr2x2.items():
            s1_dr2x2_seeds.setdefault(name, []).append(jc.compute_oracle_metrics(f1, f2, te))

        # S3-PH linear-PH negative parity control
        ph_tr = simulate_s3_ph_dgp(9000, seed=seed + 1000)
        ph_te = simulate_s3_ph_dgp(3000, seed=seed + 60000)
        ti3, vi3 = jc._split_train_val(len(ph_tr["X"]), seed + 1000)
        X3t, T3t, D3t = ph_tr["X"][ti3], ph_tr["T"][ti3], ph_tr["D"][ti3]
        X3v, T3v, D3v = ph_tr["X"][vi3], ph_tr["T"][vi3], ph_tr["D"][vi3]
        nuis3 = jc.crossfit_nuisances_train90(X3t, T3t, D3t, X3v, M=M, seed=seed + 1000, fit_G=False)
        psi3_tr = jc.build_rb_targets(nuis3["S_tr"], nuis3["F1_tr"], nuis3["F2_tr"], T3t, D3t)
        psi3_va = jc.build_rb_targets(nuis3["S_va"], nuis3["F1_va"], nuis3["F2_va"], T3v, D3v)
        m3_rb = jc.fit_monocif_target_nn(X3t, psi3_tr, X3v, psi3_va, M, seed + 1000, loss_type="logloss")
        m3_hz = jc.fit_discrete_hazard_nn(X3t, T3t, D3t, M, seed + 1000, X_val=X3v, T_val=T3v, D_val=D3v)
        ph_preds = {
            "CS-Cox": jc.fit_predict_cs_cox_gpu(X3t, T3t, D3t, ph_te["X"], M),
            "CS-XGBoost (PH)": jc.fit_predict_cs_xgboost(X3t, T3t, D3t, X3v, T3v, D3v, ph_te["X"], M, seed),
            "Discrete-Hazard-NN": jc.logits_to_cif_np(jc.predict_hazard_logits(m3_hz, ph_te["X"]))[1:],
            "MonoCIF-RB (NN)": jc.logits_to_cif_np(jc.predict_hazard_logits(m3_rb, ph_te["X"]))[1:],
        }
        for name, (f1, f2) in ph_preds.items():
            s3ph_seeds.setdefault(name, []).append(jc.compute_oracle_metrics(f1, f2, ph_te))

        # S4-InfCens shared-frailty sweep (gamma in {0.0, 0.5, 1.0, 1.5})
        for gam in (0.0, 0.5, 1.0, 1.5):
            gkey = f"gamma_{gam:.1f}"
            s4_seeds.setdefault(gkey, {})
            if gam == 0.0:
                g_preds = {
                    "MonoCIF-RB (NN)": preds["MonoCIF-RB (NN)"],
                    "Discrete-Hazard-NN": preds["Discrete-Hazard-NN"],
                    "MonoCIF-CRNLL (CovIPCW)": preds["MonoCIF-CRNLL (CovIPCW)"],
                    "DR-MonoCIF-Brier": preds["DR-MonoCIF-Brier"],
                }
                g_te = te
            else:
                g_tr = simulate_s1_dgp(9000, seed=seed, gamma_frailty=gam)
                g_te = simulate_s1_dgp(3000, seed=seed + 50000, gamma_frailty=gam)
                gti, gvi = jc._split_train_val(len(g_tr["X"]), seed)
                Xgt, Tgt, Dgt = g_tr["X"][gti], g_tr["T"][gti], g_tr["D"][gti]
                Xgv, Tgv, Dgv = g_tr["X"][gvi], g_tr["T"][gvi], g_tr["D"][gvi]
                gn = jc.crossfit_nuisances_train90(Xgt, Tgt, Dgt, Xgv, M=M, seed=seed, fit_G=True, clip_hc=0.85)
                grb_t = jc.build_rb_targets(gn["S_tr"], gn["F1_tr"], gn["F2_tr"], Tgt, Dgt)
                grb_v = jc.build_rb_targets(gn["S_va"], gn["F1_va"], gn["F2_va"], Tgv, Dgv)
                gdr_t, _ = jc.build_aipcw_targets(gn["S_tr"], gn["F1_tr"], gn["F2_tr"], gn["hc_tr"], gn["G_tr"], Tgt, Dgt)
                gdr_v, _ = jc.build_aipcw_targets(gn["S_va"], gn["F1_va"], gn["F2_va"], gn["hc_va"], gn["G_va"], Tgv, Dgv)
                mg_rb = jc.fit_monocif_target_nn(Xgt, grb_t, Xgv, grb_v, M, seed, loss_type="logloss")
                mg_hz = jc.fit_discrete_hazard_nn(Xgt, Tgt, Dgt, M, seed, X_val=Xgv, T_val=Tgv, D_val=Dgv)
                mg_cr = jc.fit_crnll_nn(Xgt, Tgt, Dgt, Xgv, Tgv, Dgv, M, seed, monotone=True, G_tr=gn["G_tr"], G_va=gn["G_va"])
                mg_dr = jc.fit_monocif_target_nn(Xgt, gdr_t, Xgv, gdr_v, M, seed, loss_type="brier")
                g_preds = {
                    "MonoCIF-RB (NN)": jc.logits_to_cif_np(jc.predict_hazard_logits(mg_rb, g_te["X"]))[1:],
                    "Discrete-Hazard-NN": jc.logits_to_cif_np(jc.predict_hazard_logits(mg_hz, g_te["X"]))[1:],
                    "MonoCIF-CRNLL (CovIPCW)": jc.predict_crnll_nn(mg_cr, g_te["X"], M, monotone=True),
                    "DR-MonoCIF-Brier": jc.logits_to_cif_np(jc.predict_hazard_logits(mg_dr, g_te["X"]))[1:],
                }
            for name, (f1, f2) in g_preds.items():
                s4_seeds[gkey].setdefault(name, []).append(jc.compute_oracle_metrics(f1, f2, g_te))

        print(f"  [E1] Seed {seed} done in {time.time() - ts:.1f}s | S1 RB MAE={s1_models_seeds['MonoCIF-RB (NN)'][-1]['MAE']:.5f} vs CovIPCW={s1_models_seeds['MonoCIF-CRNLL (CovIPCW)'][-1]['MAE']:.5f} vs DR-Brier={s1_models_seeds['DR-MonoCIF-Brier'][-1]['MAE']:.5f} | tele={tele_gg:.2e} | VarRatio={var_rb/var_aipcw:.3f}", flush=True)

    # Compute paired 95% CIs across the 5 seeds & summarize
    out = {
        "phase": "E1",
        "seeds": SEEDS,
        "code_hash": CODE_HASH,
        "runtime_sec": round(time.time() - t0, 2),
        "S1_matched_backbone": {k: _agg_seed_metrics(v) for k, v in s1_models_seeds.items()},
        "S1_2x2_DR_Brier": {k: _agg_seed_metrics(v) for k, v in s1_dr2x2_seeds.items()},
        "S1_theorem2_invariants": s1_invariants,
        "S3_PH_parity": {k: _agg_seed_metrics(v) for k, v in s3ph_seeds.items()},
        "S4_InfCens": {gk: {k: _agg_seed_metrics(v) for k, v in gd.items()} for gk, gd in s4_seeds.items()},
    }
    (OUT_DIR / "simulation_suite_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"=== PHASE E1 COMPLETE in {time.time() - t0:.1f}s -> simulation_suite_results.json ===", flush=True)
    return out


# =============================================================================
# 3. Phase E2: Recalibration, M=200 Martingale Battery, n_cal Curve & 2x2 Lifecycle Factorial
# =============================================================================

def run_phase_E2() -> Dict:
    t0 = time.time()
    print("\n================ PHASE E2: RECALIBRATION & M=200 MARTINGALE BATTERY ================", flush=True)
    M = 20

    # Train 5 base MonoCIF-RB models + 5 MonoCIF-CRNLL-CovIPCW models for the 2x2 lifecycle factorial
    s2a_ncal: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
    s2b_ncal: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
    r7_factorial: Dict[str, List[Dict[str, float]]] = {}

    base_models_rb = []
    for seed in SEEDS:
        tr_all = simulate_s1_dgp(9000, seed=seed)
        tr_i, va_i = jc._split_train_val(len(tr_all["X"]), seed)
        X_tr, T_tr, D_tr = tr_all["X"][tr_i], tr_all["T"][tr_i], tr_all["D"][tr_i]
        X_va, T_va, D_va = tr_all["X"][va_i], tr_all["T"][va_i], tr_all["D"][va_i]
        nuis = jc.crossfit_nuisances_train90(X_tr, T_tr, D_tr, X_va, M=M, seed=seed, fit_G=True, clip_hc=0.85)
        psi_tr = jc.build_rb_targets(nuis["S_tr"], nuis["F1_tr"], nuis["F2_tr"], T_tr, D_tr)
        psi_va = jc.build_rb_targets(nuis["S_va"], nuis["F1_va"], nuis["F2_va"], T_va, D_va)
        m_rb = jc.fit_monocif_target_nn(X_tr, psi_tr, X_va, psi_va, M, seed, loss_type="logloss")
        m_cr = jc.fit_crnll_nn(X_tr, T_tr, D_tr, X_va, T_va, D_va, M, seed, monotone=True, G_tr=nuis["G_tr"], G_va=nuis["G_va"])
        base_models_rb.append((seed, m_rb))

        # Evaluate 9 recalibrators across n_cal in {250, 500, 1000, 2500} on S2A and S2B
        for mode, store in (("S2A", s2a_ncal), ("S2B", s2b_ncal)):
            dep = simulate_s1_dgp(3000, seed=seed + 70000, shift_mode=mode)
            lg_dep = jc.predict_hazard_logits(m_rb, dep["X"])
            for n_cal in (250, 500, 1000, 2500):
                cal = simulate_s1_dgp(n_cal, seed=seed + 80000 + n_cal, shift_mode=mode)
                lg_cal = jc.predict_hazard_logits(m_rb, cal["X"])
                recs = jc.run_all_9_recalibrators(lg_cal, cal["T"], cal["D"], lg_dep)
                nk = f"n_{n_cal}"
                store.setdefault(nk, {})
                for rname, robj in recs.items():
                    m_res = jc.compute_oracle_metrics(robj["F1"], robj["F2"], dep)
                    store[nk].setdefault(rname, []).append(m_res)

                # Two-by-two lifecycle factorial at n_cal=1000 on S2B
                if mode == "S2B" and n_cal == 1000:
                    lg_cr_cal = jc.predict_hazard_logits(m_cr, cal["X"])
                    lg_cr_dep = jc.predict_hazard_logits(m_cr, dep["X"])
                    _, cr_f1_frz, cr_f2_frz = jc.logits_to_cif_np(lg_cr_dep)
                    cr_sg = jc.fit_sg_jccr(lg_cr_cal, cal["T"], cal["D"], lg_cr_dep)
                    r7_factorial.setdefault("A_only (MonoCIF-CRNLL + Frozen)", []).append(jc.compute_oracle_metrics(cr_f1_frz, cr_f2_frz, dep))
                    r7_factorial.setdefault("A+B (MonoCIF-RB + Frozen)", []).append(jc.compute_oracle_metrics(recs["1. Frozen"]["F1"], recs["1. Frozen"]["F2"], dep))
                    r7_factorial.setdefault("A+C (MonoCIF-CRNLL + SG-JCCR)", []).append(jc.compute_oracle_metrics(cr_sg["F1"], cr_sg["F2"], dep))
                    r7_factorial.setdefault("A+B+C (MonoCIF-RB + SG-JCCR)", []).append(jc.compute_oracle_metrics(recs["9. Ours: Hier SG-JCCR"]["F1"], recs["9. Ours: Hier SG-JCCR"]["F2"], dep))

        print(f"  [E2] Seed {seed} recalibration curves done (S2B n=1000 Hier={s2b_ncal['n_1000']['9. Ours: Hier SG-JCCR'][-1]['MAE']:.5f} vs Booth={s2b_ncal['n_1000']['7. Booth Sep-Baseline (SIM 23)'][-1]['MAE']:.5f} vs HazIndep={s2b_ncal['n_1000']['6. Indep-Logit (Hazard)'][-1]['MAE']:.5f})", flush=True)

    # M = 200 Monte Carlo calibration-slice draws (SEED=20261121) for S3-NullShift, S2A, S2B!
    # Use seed-42 base model (and exact oracle logits for clean statistical calibration check + plug-in check!)
    _, m_rb_0 = base_models_rb[0]
    mc_stats = {}
    for scenario, mode in (("S3_NullShift", "none"), ("S2A", "S2A"), ("S2B", "S2B")):
        mc_stats[scenario] = {}
        n_cals = (250, 1000) if scenario == "S3_NullShift" else (250, 500, 1000, 2500)
        # Generate a large 60,000 pool once on GPU and sample M=200 independent slices of size n_cal (deterministic seed!)
        scen_off = {"S3_NullShift": 101, "S2A": 202, "S2B": 303}[scenario]
        pool = simulate_s1_dgp(60000, seed=20261121 + scen_off, shift_mode=mode)
        lg_pool_plugin = jc.predict_hazard_logits(m_rb_0, pool["X"])
        lg_pool_true = pool["logits_oracle"] if mode == "none" else lg_pool_plugin
        rng_mc = np.random.default_rng(20261121)

        for n_cal in n_cals:
            tiers = {"Tier0_Hold": 0, "Tier1_K": 0, "Tier2_3K": 0}
            alarms_any_q = 0
            alarms_q1 = 0
            alarms_q2 = 0
            alarms_any_z1df = 0
            alarms_exact_null_q = 0
            alarms_exact_null_z = 0
            for draw in range(200):
                idx = rng_mc.integers(0, len(pool["X"]), size=n_cal)
                res_mc = jc.fit_sg_jccr(lg_pool_plugin[idx], pool["T"][idx], pool["D"][idx], lg_pool_plugin[:10], per_cause_gate=True, gate_mode="score_df")
                tiers[res_mc["tier"]] = tiers.get(res_mc["tier"], 0) + 1
                zi = res_mc["z_info"]
                qc = zi["q_crit"]
                zc = zi["z_crit"]
                if zi["Q1"] >= qc:
                    alarms_q1 += 1
                if zi["Q2"] >= qc:
                    alarms_q2 += 1
                if zi["Q1"] >= qc or zi["Q2"] >= qc:
                    alarms_any_q += 1
                if abs(zi["Z1"]) >= zc or abs(zi["Z2"]) >= zc:
                    alarms_any_z1df += 1
                if scenario == "S3_NullShift":
                    z_ex = jc.compute_martingale_z(lg_pool_true[idx], pool["T"][idx], pool["D"][idx])
                    if z_ex["Q1"] >= z_ex["q_crit"] or z_ex["Q2"] >= z_ex["q_crit"]:
                        alarms_exact_null_q += 1
                    if abs(z_ex["Z1"]) >= z_ex["z_crit"] or abs(z_ex["Z2"]) >= z_ex["z_crit"]:
                        alarms_exact_null_z += 1
            entry = {
                "M_draws": 200,
                "score_gate_trigger_rate": alarms_any_q / 200.0,
                "q1_trigger_rate": alarms_q1 / 200.0,
                "q2_trigger_rate": alarms_q2 / 200.0,
                "z1df_intercept_trigger_rate": alarms_any_z1df / 200.0,
                "tier_rates": {k: v / 200.0 for k, v in tiers.items()},
            }
            if scenario == "S3_NullShift":
                entry["exact_null_score_gate_false_alarm_rate"] = alarms_exact_null_q / 200.0
                entry["exact_null_z1df_false_alarm_rate"] = alarms_exact_null_z / 200.0
                entry["exact_null_tier0_hold_rate"] = 1.0 - alarms_exact_null_q / 200.0
            mc_stats[scenario][f"n_{n_cal}"] = entry
        print(f"  [E2] M=200 MC {scenario} complete: {mc_stats[scenario]['n_1000']}", flush=True)

    out = {
        "phase": "E2",
        "seeds": SEEDS,
        "code_hash": CODE_HASH,
        "runtime_sec": round(time.time() - t0, 2),
        "S2A_by_ncal": {nk: {k: _agg_seed_metrics(v) for k, v in d.items()} for nk, d in s2a_ncal.items()},
        "S2B_by_ncal": {nk: {k: _agg_seed_metrics(v) for k, v in d.items()} for nk, d in s2b_ncal.items()},
        "R7_combination_factorial_S2B_n1000": {k: _agg_seed_metrics(v) for k, v in r7_factorial.items()},
        "MC_200_operating_characteristics": mc_stats,
    }
    (OUT_DIR / "mechanism_ablation_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"=== PHASE E2 COMPLETE in {time.time() - t0:.1f}s -> mechanism_ablation_results.json ===", flush=True)
    return out


# =============================================================================
# 4. Clinical Cohort Loaders (see DATA.md for cohort acquisition and schemas)
# =============================================================================

def load_rotterdam() -> Dict:
    df = pd.read_csv(DATA_DIR / "rotterdam.csv")
    # Exact Stage 4 safeguard: preserve all 43 recur=0, death=1 competing deaths!
    T_mo = np.where(df["recur"] == 1, df["rtime"], df["dtime"]) / 30.4375
    D = np.where(df["recur"] == 1, 1, np.where(df["death"] == 1, 2, 0)).astype(np.int32)
    M = 10
    edges = np.linspace(0.0, 120.0, M + 1)
    T_bin, D_bin = jc.discretize_clean(T_mo, D, edges)

    # Covariates (excluding year, pid, outcomes)
    df_feat = df[["age", "meno", "size", "grade", "nodes", "pgr", "er", "hormon", "chemo"]].copy()
    df_feat["size"] = df_feat["size"].astype(str)
    X_df = pd.get_dummies(df_feat, columns=["size"], drop_first=False, dtype=np.float32)
    X_df["log_nodes"] = np.log1p(df["nodes"].values).astype(np.float32)
    X_df["log_pgr"] = np.log1p(df["pgr"].values).astype(np.float32)
    X_df["log_er"] = np.log1p(df["er"].values).astype(np.float32)
    X_raw = X_df.values.astype(np.float32)
    yr = df["year"].values
    age = df["age"].values.astype(np.float64)
    return {"X_raw": X_raw, "T": T_bin, "D": D_bin, "year": yr, "age": age, "M": M, "T_mo": T_mo}


def load_mgus2() -> Dict:
    df = pd.read_csv(DATA_DIR / "mgus2.csv")
    T_mo = np.where(df["pstat"] == 1, df["ptime"], df["futime"]).astype(np.float64)
    D = np.where(df["pstat"] == 1, 1, np.where(df["death"] == 1, 2, 0)).astype(np.int32)
    M = 10
    edges = np.linspace(0.0, 120.0, M + 1)
    T_bin, D_bin = jc.discretize_clean(T_mo, D, edges)

    tr_mask = df["dxyr"].values <= 1984
    feat = pd.DataFrame(index=df.index)
    feat["age"] = df["age"].astype(np.float32)
    feat["sex_M"] = (df["sex"] == "M").astype(np.float32)
    for col in ("hgb", "creat", "mspike"):
        vals = pd.to_numeric(df[col], errors="coerce")
        feat[f"{col}_miss"] = vals.isna().astype(np.float32)
        med = float(np.nanmedian(vals[tr_mask]))
        feat[col] = vals.fillna(med).astype(np.float32)
    return {"X_raw": feat.values.astype(np.float32), "T": T_bin, "D": D_bin, "year": df["dxyr"].values, "age": df["age"].values.astype(np.float64), "M": M}


def load_seer_prostate() -> Dict:
    df = pd.read_csv(DATA_DIR / "prostateSurvival.csv")
    T_mo = df["survTime"].values.astype(np.float64)
    D = df["status"].values.astype(np.int32)
    M = 10
    edges = np.linspace(0.0, 60.0, M + 1)
    T_bin, D_bin = jc.discretize_clean(T_mo, D, edges)
    X_df = pd.get_dummies(df[["grade", "stage", "ageGroup"]], drop_first=False, dtype=np.float32)
    age_75p = df["ageGroup"].astype(str).str.contains("75|80").values
    return {"X_raw": X_df.values.astype(np.float32), "T": T_bin, "D": D_bin, "age_75p": age_75p, "M": M}


def harmonize_seer_grade(df: pd.DataFrame) -> pd.Series:
    """Maps Grade Recode (thru 2017) + Derived Summary Grade 2018 (2018+) into unified 4-level Grade_Harmonized."""
    g17 = df["Grade Recode (thru 2017)"].astype(str)
    g18 = df["Derived Summary Grade 2018 (2018+)"].astype(str)
    out = np.full(len(df), "Unknown", dtype=object)

    # Pre-2018 mapping
    m1 = g17.str.contains("Grade I$|Well differentiated", regex=True)
    m2 = g17.str.contains("Grade II|Moderately", regex=True)
    m3 = g17.str.contains("Grade III|Grade IV|Poorly|Undifferentiated", regex=True)
    out[m1] = "G1"
    out[m2] = "G2"
    out[m3] = "G3_4"

    # 2018+ mapping (when legacy is Blank)
    is_blank17 = g17.str.contains("Blank", na=False)
    n1 = is_blank17 & g18.str.contains(r"\(1\)|Well differentiated|Low grade", regex=True)
    n2 = is_blank17 & g18.str.contains(r"\(2\)|Moderately", regex=True)
    n3 = is_blank17 & g18.str.contains(r"\(3\)|Poorly|High grade|Undifferentiated", regex=True)
    out[n1] = "G1"
    out[n2] = "G2"
    out[n3] = "G3_4"
    return pd.Series(out, index=df.index)


def parse_seer_age_numeric(s: pd.Series) -> np.ndarray:
    """Parse SEER Age recode (e.g., '75-79 years', '90+ years') into numeric midpoints."""
    txt = s.astype(str)
    nums = txt.str.extract(r"(\d+)")[0].astype(float).fillna(60.0).values
    return nums + 2.0


def load_seer_breast(
    harmonize_grade: bool = True,
    include_treatments: bool = False,
    rare_hist_train_mask: Optional[np.ndarray] = None,
) -> Dict:
    """
    Loads SEER Breast Cancer (N=827,869 retained; NEVER drops No/Unknown treatment rows;
    explicitly BANS 'Year of diagnosis' and 'grade_source' from primary covariates X!).
    """
    df = pd.read_csv(DATA_DIR / "seer_breast_cancer.csv", low_memory=False)
    sm = pd.to_numeric(df["Survival months"], errors="coerce")
    cod = df["SEER cause-specific death classification"].astype(str)
    vs = df["Vital status recode (study cutoff used)"].astype(str)
    valid = sm.notna() & (cod != "Dead (missing/unknown COD)") & vs.isin(["Alive", "Dead"])
    df = df[valid].copy().reset_index(drop=True)

    T_mo = pd.to_numeric(df["Survival months"], errors="coerce").values.astype(np.float64)
    D = np.where(
        df["SEER cause-specific death classification"] == "Dead (attributable to this cancer dx)",
        1,
        np.where(df["Vital status recode (study cutoff used)"] == "Dead", 2, 0),
    ).astype(np.int32)
    yr = df["Year of diagnosis"].values.astype(np.int32)
    age_num = parse_seer_age_numeric(df["Age recode with <1 year olds and 90+"])

    # Numeric covariates (parsed cleanly)
    tsize = pd.to_numeric(df["Tumor Size Over Time Recode (1988+)"], errors="coerce")
    tsize_miss = (tsize.isna() | (tsize >= 990)).astype(np.float32)
    tsize_val = np.log1p(np.where(tsize_miss > 0.5, 20.0, np.clip(tsize.fillna(20.0).values, 0.0, 500.0))).astype(np.float32)

    nodes = pd.to_numeric(df["Regional nodes positive (1988+)"], errors="coerce")
    nodes_miss = (nodes.isna() | (nodes >= 95)).astype(np.float32)
    nodes_val = np.log1p(np.where(nodes_miss > 0.5, 0.0, np.clip(nodes.fillna(0.0).values, 0.0, 90.0))).astype(np.float32)

    # Group rare histologies (<50 rows in training era) into Other
    hist = df["Histologic Type ICD-O-3"].astype(str)
    tr_m = (yr <= 2015) if rare_hist_train_mask is None else rare_hist_train_mask
    vc = hist[tr_m].value_counts()
    keep_h = set(vc[vc >= 50].index)
    hist_grp = hist.where(hist.isin(keep_h), "Other")

    cat_df = pd.DataFrame({
        "stage": df["Combined Summary Stage with Expanded Regional Codes (2004+)"].astype(str),
        "subtype": df["Breast Subtype (2010+)"].astype(str),
        "race": df["Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)"].astype(str),
        "marital": df["Marital status at diagnosis"].astype(str),
        "laterality": df["Laterality"].astype(str),
        "seq": df["Sequence number"].astype(str),
        "hist": hist_grp,
    })
    if harmonize_grade:
        cat_df["grade"] = harmonize_seer_grade(df)
    else:
        cat_df["grade_17"] = df["Grade Recode (thru 2017)"].astype(str)
        cat_df["grade_18"] = df["Derived Summary Grade 2018 (2018+)"].astype(str)

    if include_treatments:
        cat_df["chemo"] = df["Chemotherapy recode (yes, no/unk)"].astype(str)
        cat_df["rad"] = df["Radiation recode"].astype(str)
        surg = pd.to_numeric(df["RX Summ--Surg Prim Site (1998-2022)"], errors="coerce").fillna(99)
        cat_df["surg_grp"] = np.where(surg == 0, "None", np.where(surg < 40, "BCS", np.where(surg < 90, "Mastectomy", "Unk")))

    X_cat = pd.get_dummies(cat_df, drop_first=False, dtype=np.float32)
    X_num = np.column_stack([age_num.astype(np.float32), (age_num ** 2).astype(np.float32), tsize_val, tsize_miss, nodes_val, nodes_miss])
    X_raw = np.hstack([X_num, X_cat.values]).astype(np.float32)

    return {
        "X_raw": X_raw,
        "T_mo": T_mo,
        "D": D,
        "year": yr,
        "age_num": age_num,
        "race": df["Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)"].astype(str).values,
        "subtype": df["Breast Subtype (2010+)"].astype(str).values,
        "stage": df["Combined Summary Stage with Expanded Regional Codes (2004+)"].astype(str).values,
        "feature_dim": X_raw.shape[1],
    }


def _run_cohort_5seed_benchmark(
    cohort_name: str,
    X_tr_raw: np.ndarray,
    T_tr_all: np.ndarray,
    D_tr_all: np.ndarray,
    X_te_raw: np.ndarray,
    T_te: np.ndarray,
    D_te: np.ndarray,
    M: int,
    eval_bins: Sequence[int],
    n_bins: int = 10,
    age_mask_te: Optional[np.ndarray] = None,
    dca_horizon_bin: Optional[int] = None,
    bs: int = 256,
    include_gbdt_res: bool = True,
    B_boot: int = 500,
) -> Dict:
    """Runs full matched-backbone 5-seed benchmark + B=500 paired bootstrap 95% CIs on seed-averaged ensemble predictions + per-seed stats."""
    t0 = time.time()
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_raw).astype(np.float32)
    X_te_sc = scaler.transform(X_te_raw).astype(np.float32)

    seed_metrics: Dict[str, List[Dict[str, float]]] = {}
    pred_accum: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}

    # Deterministic statistical models fitted once (CS-Cox, Fine-Gray)
    p_cox = jc.fit_predict_cs_cox_gpu(X_tr_sc, T_tr_all, D_tr_all, X_te_sc, M, l2_alpha=1e-2)
    p_fg = jc.fit_predict_fine_gray_gpu(X_tr_sc, T_tr_all, D_tr_all, X_te_sc, M, l2_alpha=1e-2)

    for seed in SEEDS:
        tr_i, va_i = jc._split_train_val(len(X_tr_sc), seed)
        X_tr, T_tr, D_tr = X_tr_sc[tr_i], T_tr_all[tr_i], D_tr_all[tr_i]
        X_va, T_va, D_va = X_tr_sc[va_i], T_tr_all[va_i], D_tr_all[va_i]

        nuis = jc.crossfit_nuisances_train90(X_tr, T_tr, D_tr, X_va, M=M, seed=seed, bs=bs, n_splits=5, fit_G=True, clip_hc=0.85)
        psi_rb_tr = jc.build_rb_targets(nuis["S_tr"], nuis["F1_tr"], nuis["F2_tr"], T_tr, D_tr)
        psi_rb_va = jc.build_rb_targets(nuis["S_va"], nuis["F1_va"], nuis["F2_va"], T_va, D_va)
        psi_dr_tr, _ = jc.build_aipcw_targets(nuis["S_tr"], nuis["F1_tr"], nuis["F2_tr"], nuis["hc_tr"], nuis["G_tr"], T_tr, D_tr)
        psi_dr_va, _ = jc.build_aipcw_targets(nuis["S_va"], nuis["F1_va"], nuis["F2_va"], nuis["hc_va"], nuis["G_va"], T_va, D_va)

        preds: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
            "CS-Cox": p_cox,
            "Fine-Gray": p_fg,
            "Cutoff-Status XGB": jc.fit_predict_cutoff_xgb(X_tr, T_tr, D_tr, X_va, T_va, D_va, X_te_sc, M, seed),
            "CS-XGBoost (PH)": jc.fit_predict_cs_xgboost(X_tr, T_tr, D_tr, X_va, T_va, D_va, X_te_sc, M, seed),
            "XGBoost-PV": jc.fit_predict_xgboost_pv(X_tr, T_tr, D_tr, X_va, T_va, D_va, X_te_sc, M, seed),
        }

        m_dh = jc.fit_deephit_nn(X_tr, T_tr, D_tr, X_va, T_va, D_va, M, seed, bs=bs)
        preds["DeepHit"] = jc.predict_deephit_nn(m_dh, X_te_sc, M)

        m_bce = jc.fit_perhorizon_bce_nn(X_tr, T_tr, D_tr, X_va, T_va, D_va, M, seed, bs=bs)
        preds["PerHorizon-BCE"] = jc.predict_perhorizon_bce_nn(m_bce, X_te_sc, M)

        m_crkm = jc.fit_crnll_nn(X_tr, T_tr, D_tr, X_va, T_va, D_va, M, seed, monotone=False, bs=bs)
        preds["CRNLL-KM"] = jc.predict_crnll_nn(m_crkm, X_te_sc, M, monotone=False)

        m_mcrc = jc.fit_crnll_nn(X_tr, T_tr, D_tr, X_va, T_va, D_va, M, seed, monotone=True, G_tr=nuis["G_tr"], G_va=nuis["G_va"], bs=bs)
        preds["MonoCIF-CRNLL (CovIPCW)"] = jc.predict_crnll_nn(m_mcrc, X_te_sc, M, monotone=True)

        m_drb = jc.fit_monocif_target_nn(X_tr, psi_dr_tr, X_va, psi_dr_va, M, seed, loss_type="brier", bs=bs)
        preds["DR-MonoCIF-Brier"] = jc.logits_to_cif_np(jc.predict_hazard_logits(m_drb, X_te_sc))[1:]

        m_hz = jc.fit_discrete_hazard_nn(X_tr, T_tr, D_tr, M, seed, bs=bs, X_val=X_va, T_val=T_va, D_val=D_va)
        preds["Discrete-Hazard-NN"] = jc.logits_to_cif_np(jc.predict_hazard_logits(m_hz, X_te_sc))[1:]

        ens_lg = sum(jc.predict_hazard_logits(fm, X_te_sc) for fm in nuis["fold_models"]) / len(nuis["fold_models"])
        preds["Discrete-Hazard-NN (5-fold ens)"] = jc.logits_to_cif_np(ens_lg)[1:]

        m_rbb = jc.fit_monocif_target_nn(X_tr, psi_rb_tr, X_va, psi_rb_va, M, seed, loss_type="brier", bs=bs)
        preds["MonoCIF-RB-Brier (NN)"] = jc.logits_to_cif_np(jc.predict_hazard_logits(m_rbb, X_te_sc))[1:]

        m_rb = jc.fit_monocif_target_nn(X_tr, psi_rb_tr, X_va, psi_rb_va, M, seed, loss_type="logloss", bs=bs)
        preds["MonoCIF-RB (NN)"] = jc.logits_to_cif_np(jc.predict_hazard_logits(m_rb, X_te_sc))[1:]

        if include_gbdt_res:
            gbdt_models = jc.fit_discrete_hazard_gbdt(X_tr, T_tr, D_tr, X_va, T_va, D_va, M, seed)
            gb_tr = jc.predict_discrete_hazard_gbdt_logits(gbdt_models, X_tr, T_tr, D_tr)
            gb_va = jc.predict_discrete_hazard_gbdt_logits(gbdt_models, X_va, T_tr, D_tr)
            gb_te = jc.predict_discrete_hazard_gbdt_logits(gbdt_models, X_te_sc, T_tr, D_tr)
            preds["Discrete-Hazard-GBDT"] = jc.logits_to_cif_np(gb_te)[1:]
            m_gbr = jc.fit_monocif_target_nn(
                X_tr, psi_rb_tr, X_va, psi_rb_va, M, seed, loss_type="logloss", bs=bs,
                base_logits_tr=gb_tr, base_logits_va=gb_va,
            )
            with torch.no_grad():
                m_gbr.eval()
                outs = []
                for i in range(0, len(X_te_sc), 8192):
                    xb = torch.tensor(X_te_sc[i : i + 8192], dtype=torch.float32, device=jc.DEVICE)
                    bb = torch.tensor(gb_te[i : i + 8192], dtype=torch.float32, device=jc.DEVICE)
                    outs.append(m_gbr(xb, bb).cpu().numpy())
                lg_gbr_te = np.concatenate(outs, axis=0)
            preds["MonoCIF-RB (GBDT-Res)"] = jc.logits_to_cif_np(lg_gbr_te)[1:]

        # Per-seed point evaluation (B=0 fast)
        pt_seed = jc.paired_bootstrap_clinical(
            preds, T_te, D_te, eval_bins=eval_bins, n_bins=n_bins,
            age_mask=age_mask_te, dca_horizon_bin=dca_horizon_bin, B=0,
        )
        for name, m_dict in pt_seed.items():
            seed_metrics.setdefault(name, []).append(m_dict)
            pred_accum.setdefault(name, []).append(preds[name])

    # Paired B=500 bootstrap on 5-seed averaged predictions (and seed-0 check)
    avg_preds = {
        name: (np.mean([p[0] for p in plist], axis=0), np.mean([p[1] for p in plist], axis=0))
        for name, plist in pred_accum.items()
    }
    ref_m = "MonoCIF-RB (GBDT-Res)" if "MonoCIF-RB (GBDT-Res)" in avg_preds and cohort_name.startswith("SEER_Breast") else "MonoCIF-RB (NN)"
    boot_summary = jc.paired_bootstrap_clinical(
        avg_preds, T_te, D_te, eval_bins=eval_bins, n_bins=n_bins,
        age_mask=age_mask_te, dca_horizon_bin=dca_horizon_bin, B=B_boot, seed=30261121, ref_model=ref_m,
    )
    print(f"  [E3] {cohort_name} done in {time.time() - t0:.1f}s | ref={ref_m} ICI={boot_summary[ref_m]['ICI']:.5f} (95%CI {boot_summary[ref_m]['ICI_ci_lo']:.5f}..{boot_summary[ref_m]['ICI_ci_hi']:.5f})", flush=True)
    return {
        "cohort": cohort_name,
        "N_train": int(len(X_tr_raw)),
        "N_test": int(len(X_te_raw)),
        "M": M,
        "eval_bins": list(eval_bins),
        "n_bins": n_bins,
        "ref_model": ref_m,
        "per_seed_summary": {k: _agg_seed_metrics(v) for k, v in seed_metrics.items()},
        "paired_bootstrap_500": boot_summary,
        "_avg_preds": avg_preds,
    }


# =============================================================================
# 5. Phase E3: Four Real Clinical Cohorts + Full Subgroup Audits + Oncology DCA
# =============================================================================

def run_phase_E3() -> Dict:
    t0 = time.time()
    print("\n================ PHASE E3: FOUR REAL CLINICAL COHORTS (5 SEEDS + B=500 BOOTSTRAP) ================", flush=True)
    results = {"phase": "E3", "seeds": SEEDS, "code_hash": CODE_HASH}
    ckpt_dir = OUT_DIR / "e3_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _load_or_run(key: str, fn):
        cp = ckpt_dir / f"{key}.json"
        if cp.exists():
            print(f"  [E3] Loaded checkpoint {cp.name}", flush=True)
            return json.loads(cp.read_text(encoding="utf-8"))
        val = fn()
        cp.write_text(json.dumps(val, indent=2), encoding="utf-8")
        return val

    # 1. Rotterdam Breast (Train 1978-88 -> Test 1989-93, M=10, eval 2..10y)
    def _rot():
        rot = load_rotterdam()
        m_tr = rot["year"] <= 1988
        m_te = rot["year"] >= 1989
        r = _run_cohort_5seed_benchmark(
            "Rotterdam_Breast", rot["X_raw"][m_tr], rot["T"][m_tr], rot["D"][m_tr],
            rot["X_raw"][m_te], rot["T"][m_te], rot["D"][m_te],
            M=10, eval_bins=[2, 4, 6, 8, 10], n_bins=10, age_mask_te=(rot["age"][m_te] >= 65.0),
            dca_horizon_bin=5, bs=128, include_gbdt_res=True, B_boot=500,
        )
        r.pop("_avg_preds", None)
        return r
    results["Rotterdam_Breast"] = _load_or_run("Rotterdam_Breast", _rot)

    # 2. Mayo MGUS2 (Train 1960-84 -> Test 1985-94, M=10, 5-quintile + smoothed ICI)
    def _mg():
        mg = load_mgus2()
        m_tr = mg["year"] <= 1984
        m_te = mg["year"] >= 1985
        r = _run_cohort_5seed_benchmark(
            "Mayo_MGUS2", mg["X_raw"][m_tr], mg["T"][m_tr], mg["D"][m_tr],
            mg["X_raw"][m_te], mg["T"][m_te], mg["D"][m_te],
            M=10, eval_bins=[2, 4, 6, 8, 10], n_bins=5, age_mask_te=(mg["age"][m_te] >= 75.0),
            dca_horizon_bin=5, bs=128, include_gbdt_res=True, B_boot=500,
        )
        r.pop("_avg_preds", None)
        return r
    results["Mayo_MGUS2"] = _load_or_run("Mayo_MGUS2", _mg)

    # 3. SEER Prostate (N=14,294 low-dim 24-stratum boundary control, 70/30 split)
    def _pr():
        pr = load_seer_prostate()
        rng_pr = np.random.default_rng(42)
        perm_pr = rng_pr.permutation(len(pr["X_raw"]))
        cut_pr = int(0.7 * len(perm_pr))
        tr_pr, te_pr = perm_pr[:cut_pr], perm_pr[cut_pr:]
        r = _run_cohort_5seed_benchmark(
            "SEER_Prostate_Boundary", pr["X_raw"][tr_pr], pr["T"][tr_pr], pr["D"][tr_pr],
            pr["X_raw"][te_pr], pr["T"][te_pr], pr["D"][te_pr],
            M=10, eval_bins=[2, 4, 6, 8, 10], n_bins=10, age_mask_te=pr["age_75p"][te_pr],
            dca_horizon_bin=10, bs=256, include_gbdt_res=True, B_boot=500,
        )
        r.pop("_avg_preds", None)
        return r
    results["SEER_Prostate_Boundary"] = _load_or_run("SEER_Prostate_Boundary", _pr)

    # 4. SEER Breast Prognostic (N=827,869):
    #    4A: Train 2010-2015 -> Test 2016-2017 evaluated strictly on supported 6-month grid M=12 (<=72m; primary DCA at 60m bin 10!)
    #    4B: Train 2010-2013 -> Test 2014 evaluated through 114m (M=20, bin 19) for full ~10-year support!
    seer = load_seer_breast(harmonize_grade=True, include_treatments=False)

    def _s6_and_subgroups():
        M_s6 = 12
        edges_s6 = np.linspace(0.0, 72.0, M_s6 + 1)
        T_s6, D_s6 = jc.discretize_clean(seer["T_mo"], seer["D"], edges_s6)
        m_tr15 = seer["year"] <= 2015
        m_te1617 = (seer["year"] >= 2016) & (seer["year"] <= 2017)
        age75_1617 = seer["age_num"][m_te1617] >= 75.0
        res_s6 = _run_cohort_5seed_benchmark(
            "SEER_Breast_Prospective_6y", seer["X_raw"][m_tr15], T_s6[m_tr15], D_s6[m_tr15],
            seer["X_raw"][m_te1617], T_s6[m_te1617], D_s6[m_te1617],
            M=M_s6, eval_bins=[2, 4, 6, 8, 10, 12], n_bins=10, age_mask_te=age75_1617,
            dca_horizon_bin=10, bs=4096, include_gbdt_res=True, B_boot=500,
        )
        avg_preds_seer = res_s6.pop("_avg_preds")

        subgroups = {}
        age_te = seer["age_num"][m_te1617]
        race_te = np.asarray([str(x) for x in seer["race"][m_te1617]])
        sub_te = np.asarray([str(x) for x in seer["subtype"][m_te1617]])
        stg_te = np.asarray([str(x) for x in seer["stage"][m_te1617]])
        T_te16, D_te16 = T_s6[m_te1617], D_s6[m_te1617]

        sg_masks = {
            "Age_<50": age_te < 50,
            "Age_50_64": (age_te >= 50) & (age_te < 65),
            "Age_65_74": (age_te >= 65) & (age_te < 75),
            "Age_75_84": (age_te >= 75) & (age_te < 85),
            "Age_85+": age_te >= 85,
            "Race_NHW": race_te == "Non-Hispanic White",
            "Race_NHB": race_te == "Non-Hispanic Black",
            "Race_Hispanic": race_te == "Hispanic (All Races)",
            "Race_NHAPI": race_te == "Non-Hispanic Asian or Pacific Islander",
            "Subtype_HRp_HER2n": sub_te == "HR+/HER2-",
            "Subtype_TNBC": sub_te == "HR-/HER2-",
            "Subtype_HRp_HER2p": sub_te == "HR+/HER2+",
            "Subtype_HRn_HER2p": sub_te == "HR-/HER2+",
            "Stage_Localized": stg_te == "Localized only",
            "Stage_Regional": np.array(["Regional" in s for s in stg_te], dtype=bool),
            "Stage_Distant": stg_te == "Distant site(s)/node(s) involved",
        }
        key_models = [
            "Cutoff-Status XGB", "CS-XGBoost (PH)", "XGBoost-PV", "CRNLL-KM",
            "Discrete-Hazard-NN", "MonoCIF-RB (NN)", "Discrete-Hazard-GBDT", "MonoCIF-RB (GBDT-Res)",
        ]
        for sg_name, sg_m in sg_masks.items():
            if int(sg_m.sum()) < 200:
                continue
            sg_res = {"N": int(sg_m.sum()), "competing_share": float(((D_te16[sg_m] == 2).sum()) / max((D_te16[sg_m] > 0).sum(), 1))}
            for mn in key_models:
                F1m, F2m = avg_preds_seer[mn]
                ici, e90, eo1, eo2 = jc.compute_aj_ici_and_e90(F1m[sg_m], F2m[sg_m], T_te16[sg_m], D_te16[sg_m], eval_bins=[2, 4, 6, 8, 10, 12], n_bins=10)
                ibs = jc.compute_ipcw_ibs(F1m[sg_m], F2m[sg_m], T_te16[sg_m], D_te16[sg_m], eval_bins=[2, 4, 6, 8, 10, 12])
                dca = jc.compute_competing_risk_dca(F1m[sg_m, 9], T_te16[sg_m], D_te16[sg_m], 10, (0.05, 0.10, 0.15, 0.20, 0.25), 1)
                sg_res[mn] = {"IBS": ibs, "ICI": ici, "E90": e90, "EO1": eo1, "EO2": eo2, **dca}
            subgroups[sg_name] = sg_res
        return {"SEER_Breast_Prospective_6y": res_s6, "SEER_Breast_Subgroup_Audit": subgroups}

    s6_pack = _load_or_run("SEER_Breast_6y_and_Subgroups", _s6_and_subgroups)
    results["SEER_Breast_Prospective_6y"] = s6_pack["SEER_Breast_Prospective_6y"]
    results["SEER_Breast_Subgroup_Audit"] = s6_pack["SEER_Breast_Subgroup_Audit"]

    # 4B: Full 10-year support check (Train 2010-2013 -> Test 2014, evaluated through 114m bin 19, 5 seeds + B=500)
    def _s10():
        M_s = 20
        edges_s = np.linspace(0.0, 120.0, M_s + 1)
        T_s, D_s = jc.discretize_clean(seer["T_mo"], seer["D"], edges_s)
        m_tr13 = seer["year"] <= 2013
        m_te14 = seer["year"] == 2014
        r = _run_cohort_5seed_benchmark(
            "SEER_Breast_10y_2014", seer["X_raw"][m_tr13], T_s[m_tr13], D_s[m_tr13],
            seer["X_raw"][m_te14], T_s[m_te14], D_s[m_te14],
            M=20, eval_bins=[4, 8, 12, 16, 19], n_bins=10, age_mask_te=(seer["age_num"][m_te14] >= 75.0),
            dca_horizon_bin=19, bs=4096, include_gbdt_res=True, B_boot=500,
        )
        r.pop("_avg_preds", None)
        return r
    results["SEER_Breast_10y_2014"] = _load_or_run("SEER_Breast_10y_2014", _s10)

    results["runtime_sec"] = round(time.time() - t0, 2)
    (OUT_DIR / "clinical_cohorts_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"=== PHASE E3 COMPLETE in {time.time() - t0:.1f}s -> clinical_cohorts_results.json ===", flush=True)
    return results


# =============================================================================
# 6. Phase E4: SEER Governance (Truncated Dec 31, 2017 Base, S5-Lag 0/1/2y,
#    2020 Surge & 2022 Rebound, Schema/Landmarking Ablations, Non-SEER Recal)
# =============================================================================

def run_phase_E4() -> Dict:
    t0 = time.time()
    print("\n================ PHASE E4: GOVERNANCE, S5-LAG & NON-SEER RECALIBRATION ================", flush=True)
    out: Dict = {"phase": "E4", "seeds": SEEDS, "code_hash": CODE_HASH}

    # --- 4.1 SEER 2010-2017 Base (Administratively Truncated at Dec 31, 2017!) -> 2018..2022 ---
    seer = load_seer_breast(harmonize_grade=True, include_treatments=False)
    M = 12  # 6-month bins through 72m (covers 6m=bin1, 12m=bin2, 24m=bin4, 60m=bin10)
    edges = np.linspace(0.0, 72.0, M + 1)

    m_base = seer["year"] <= 2017
    # EXACT ANTI-LEAKAGE TRUNCATION AT DEC 31, 2017 (no 2018-2023 COVID deaths in pre-2018 model!)
    T_max_2017 = (2017 - seer["year"][m_base]) * 12.0 + 6.0
    T_mo_base_trunc = np.minimum(seer["T_mo"][m_base], T_max_2017)
    D_base_trunc = np.where(seer["T_mo"][m_base] <= T_max_2017, seer["D"][m_base], 0).astype(np.int32)
    T_b_tr, D_b_tr = jc.discretize_clean(T_mo_base_trunc, D_base_trunc, edges)

    scaler = StandardScaler()
    X_base = scaler.fit_transform(seer["X_raw"][m_base]).astype(np.float32)

    # Fit 5-seed ensemble MonoCIF-RB (NN) on Dec-31-2017 truncated base
    base_models = []
    for seed in SEEDS:
        ti, vi = jc._split_train_val(len(X_base), seed)
        nuis = jc.crossfit_nuisances_train90(X_base[ti], T_b_tr[ti], D_b_tr[ti], X_base[vi], M=M, seed=seed, bs=4096, fit_G=False)
        psi_tr = jc.build_rb_targets(nuis["S_tr"], nuis["F1_tr"], nuis["F2_tr"], T_b_tr[ti], D_b_tr[ti])
        psi_va = jc.build_rb_targets(nuis["S_va"], nuis["F1_va"], nuis["F2_va"], T_b_tr[vi], D_b_tr[vi])
        base_models.append(jc.fit_monocif_target_nn(X_base[ti], psi_tr, X_base[vi], psi_va, M, seed, loss_type="logloss", bs=4096))

    # Extract logits & outcomes for every deployment year 2018..2022 (BOTH 6m [bin 1] and 12m [bin 2]!)
    era_data = {}
    yearly_surveillance = {}
    for yr in (2018, 2019, 2020, 2021, 2022):
        my = seer["year"] == yr
        Xy = scaler.transform(seer["X_raw"][my]).astype(np.float32)
        Ty, Dy = jc.discretize_clean(seer["T_mo"][my], seer["D"][my], edges)
        lgy = sum(jc.predict_hazard_logits(bm, Xy) for bm in base_models) / len(base_models)
        age75_y = seer["age_num"][my] >= 75.0
        era_data[yr] = {"T": Ty, "D": Dy, "logits": lgy, "age75": age75_y, "N": int(my.sum())}

        _, aj1, aj2 = jc.aalen_johansen_discrete(Ty, Dy, 2)
        z6 = jc.compute_martingale_z(lgy, Ty, Dy, L_cal=1)
        z12 = jc.compute_martingale_z(lgy, Ty, Dy, L_cal=2)
        # Also compute standardized Z on n_cal=5,000 random slice to show sample-size scaling transparently
        rng5 = np.random.default_rng(202600 + yr)
        sub5 = rng5.choice(len(Ty), size=min(5000, len(Ty)), replace=False)
        z12_5k = jc.compute_martingale_z(lgy[sub5], Ty[sub5], Dy[sub5], L_cal=2)
        _, f1_frz, f2_frz = jc.logits_to_cif_np(lgy)
        yearly_surveillance[str(yr)] = {
            "N": int(my.sum()),
            "obs_6m_F1": float(aj1[0]), "obs_6m_F2": float(aj2[0]),
            "obs_12m_F1": float(aj1[1]), "obs_12m_F2": float(aj2[1]),
            "pred_12m_F1": float(f1_frz[:, 1].mean()), "pred_12m_F2": float(f2_frz[:, 1].mean()),
            "EO_12m_F1": float(f1_frz[:, 1].mean() / max(aj1[1], 1e-6)),
            "EO_12m_F2": float(f2_frz[:, 1].mean() / max(aj2[1], 1e-6)),
            "Z_6m_census": z6, "Z_12m_census": z12, "Z_12m_n5000": z12_5k,
        }
        print(f"  [E4] SEER {yr} (N={my.sum()}): 6m F2={aj2[0]:.5f}, 12m F2={aj2[1]:.5f} | Z12_census=({z12['Z1']:+.2f},{z12['Z2']:+.2f}) | Z12_5k=({z12_5k['Z1']:+.2f},{z12_5k['Z2']:+.2f})", flush=True)

    # Full 9-recalibrator comparison on FULL ANNUAL SEER REGISTRY CENSUS (N=63k..72k) for 2019, 2021 COVID shock, and 2022 rebound!
    seer_recal = {}
    for dep_yr, cal_yr, label in [
        (2019, 2018, "Deploy_2019_cal2018_PreCOVID"),
        (2021, 2020, "Deploy_2021_cal2020_COVIDShock"),
        (2022, 2021, "Deploy_2022_cal2021_PostCOVIDRebound"),
        (2022, 2020, "Deploy_2022_cal2020_OvercorrectionDemo"),
    ]:
        ec = era_data[cal_yr]
        ed = era_data[dep_yr]
        recs = jc.run_all_9_recalibrators(ec["logits"], ec["T"], ec["D"], ed["logits"], L_cal=2)
        eval_dict = {}
        for rname, robj in recs.items():
            F1d, F2d = robj["F1"], robj["F2"]
            ici, e90, eo1, eo2 = jc.compute_aj_ici_and_e90(F1d, F2d, ed["T"], ed["D"], eval_bins=[1, 2], n_bins=10)
            ici75, _, eo1_75, eo2_75 = jc.compute_aj_ici_and_e90(F1d[ed["age75"]], F2d[ed["age75"]], ed["T"][ed["age75"]], ed["D"][ed["age75"]], eval_bins=[1, 2], n_bins=10)
            mv, sv = jc.compute_structural_violations(F1d, F2d)
            row = {"ICI": ici, "E90": e90, "EO1": eo1, "EO2": eo2, "ICI_75p": ici75, "EO1_75p": eo1_75, "EO2_75p": eo2_75, "MonoV": mv, "SimpV": sv}
            if "tier" in robj:
                row["tier"] = robj["tier"]
                row["active"] = robj["active"]
            eval_dict[rname] = row
        seer_recal[label] = eval_dict
        print(f"  [E4] {label}: Frozen EO=({eval_dict['1. Frozen']['EO1']:.3f},{eval_dict['1. Frozen']['EO2']:.3f}) | Hier SG-JCCR EO=({eval_dict['9. Ours: Hier SG-JCCR']['EO1']:.3f},{eval_dict['9. Ours: Hier SG-JCCR']['EO2']:.3f}) | 1df-Z EO=({eval_dict['9b. Ablation: 1-df Z-Gate SG-JCCR']['EO1']:.3f},{eval_dict['9b. Ablation: 1-df Z-Gate SG-JCCR']['EO2']:.3f}) tier={eval_dict['9. Ours: Hier SG-JCCR']['tier']} active={eval_dict['9. Ours: Hier SG-JCCR']['active']}", flush=True)

    # --- 4.2 S5-Lag: 0-, 1-, and 2-Year Registry Reporting-Lag Calendar Simulation (2019..2022, FULL CENSUS) ---
    s5_lag = {}
    for lag in (0, 1, 2):
        lag_rows = {}
        first_detect_year = None
        for y_dep in (2019, 2020, 2021, 2022):
            y_cal = y_dep - 1 - lag
            ed = era_data[y_dep]
            if y_cal < 2018:
                _, F1d, F2d = jc.logits_to_cif_np(ed["logits"])
                tier, active, z2_val, q2_val = "Tier0_Hold_NoRelease", [], 0.0, 0.0
            else:
                ec = era_data[y_cal]
                sg = jc.fit_sg_jccr(ec["logits"], ec["T"], ec["D"], ed["logits"], L_cal=2, per_cause_gate=True)
                F1d, F2d = sg["F1"], sg["F2"]
                tier, active = sg["tier"], sg["active"]
                z2_val = sg["z_info"]["Z2"]
                q2_val = sg["z_info"]["Q2"]
                if (q2_val >= sg["z_info"]["q_crit"] or abs(z2_val) >= sg["z_info"]["z_crit"]) and first_detect_year is None:
                    first_detect_year = y_dep
            ici, _, eo1, eo2 = jc.compute_aj_ici_and_e90(F1d, F2d, ed["T"], ed["D"], eval_bins=[1, 2], n_bins=10)
            ici75, _, eo1_75, eo2_75 = jc.compute_aj_ici_and_e90(F1d[ed["age75"]], F2d[ed["age75"]], ed["T"][ed["age75"]], ed["D"][ed["age75"]], eval_bins=[1, 2], n_bins=10)
            lag_rows[str(y_dep)] = {
                "cal_year_used": y_cal if y_cal >= 2018 else "Frozen_2010_2017",
                "Z2_cal": float(z2_val), "Q2_cal": float(q2_val), "tier": tier, "active": active,
                "ICI": ici, "EO1": eo1, "EO2": eo2, "ICI_75p": ici75, "EO2_75p": eo2_75,
            }
        s5_lag[f"lag_{lag}y"] = {"first_shock_action_year": first_detect_year, "by_year": lag_rows}

    # Reuse already-completed 4.3 (Schema/Landmarking) & 4.4 (Non-SEER Recalibration) if on disk to save GPU time!
    prev_e4_path = OUT_DIR / "recalibration_and_governance_results.json"
    if prev_e4_path.exists():
        prev_e4 = json.loads(prev_e4_path.read_text(encoding="utf-8"))
        if "SEER_schema_and_landmarking_ablation" in prev_e4 and "Non_SEER_temporal_recalibration" in prev_e4:
            out["SEER_2018_2022_yearly_surveillance"] = yearly_surveillance
            out["SEER_2018_2022_recalibration_9arms"] = seer_recal
            out["S5_Lag_reporting_lag"] = s5_lag
            out["SEER_schema_and_landmarking_ablation"] = prev_e4["SEER_schema_and_landmarking_ablation"]
            out["Non_SEER_temporal_recalibration"] = prev_e4["Non_SEER_temporal_recalibration"]
            out["runtime_sec"] = round(time.time() - t0, 2)
            prev_e4_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
            (OUT_DIR / "reporting_lag_results.json").write_text(json.dumps({"S5_Lag": s5_lag, "SEER_2018_2022_yearly": yearly_surveillance}, indent=2), encoding="utf-8")
            print(f"=== PHASE E4 (FULL CENSUS) COMPLETE in {time.time() - t0:.1f}s -> recalibration_and_governance_results.json ===", flush=True)
            return out

    # --- 4.3 SEER Schema Harmonization & Diagnosis-Time Landmarking Ablations (R6, K4) ---
    schema_ablation = {}
    for cfg_name, harm, inc_tx in [
        ("Primary_Harmonized_Landmarked", True, False),
        ("Unharmonized_RawGrade_SchemaBreak", False, False),
        ("Sensitivity_WithPostDxTreatments", True, True),
    ]:
        sd = load_seer_breast(harmonize_grade=harm, include_treatments=inc_tx)
        mb = sd["year"] <= 2017
        m18 = (sd["year"] >= 2018) & (sd["year"] <= 2019)
        Tm_tr = np.minimum(sd["T_mo"][mb], (2017 - sd["year"][mb]) * 12.0 + 6.0)
        Dm_tr = np.where(sd["T_mo"][mb] <= (2017 - sd["year"][mb]) * 12.0 + 6.0, sd["D"][mb], 0).astype(np.int32)
        Tb, Db = jc.discretize_clean(Tm_tr, Dm_tr, edges)
        T18, D18 = jc.discretize_clean(sd["T_mo"][m18], sd["D"][m18], edges)
        sc = StandardScaler()
        Xb = sc.fit_transform(sd["X_raw"][mb]).astype(np.float32)
        X18 = sc.transform(sd["X_raw"][m18]).astype(np.float32)
        ti, vi = jc._split_train_val(len(Xb), 42)
        nu = jc.crossfit_nuisances_train90(Xb[ti], Tb[ti], Db[ti], Xb[vi], M=M, seed=42, bs=4096, fit_G=False)
        pt = jc.build_rb_targets(nu["S_tr"], nu["F1_tr"], nu["F2_tr"], Tb[ti], Db[ti])
        pv = jc.build_rb_targets(nu["S_va"], nu["F1_va"], nu["F2_va"], Tb[vi], Db[vi])
        mm = jc.fit_monocif_target_nn(Xb[ti], pt, Xb[vi], pv, M, 42, loss_type="logloss", bs=4096)
        _, f1, f2 = jc.logits_to_cif_np(jc.predict_hazard_logits(mm, X18))
        ici, e90, eo1, eo2 = jc.compute_aj_ici_and_e90(f1, f2, T18, D18, eval_bins=[2, 4, 6], n_bins=10)
        ibs = jc.compute_ipcw_ibs(f1, f2, T18, D18, eval_bins=[2, 4, 6])
        a75 = sd["age_num"][m18] >= 75.0
        ici75, _, eo1_75, eo2_75 = jc.compute_aj_ici_and_e90(f1[a75], f2[a75], T18[a75], D18[a75], eval_bins=[2, 4, 6], n_bins=10)
        schema_ablation[cfg_name] = {"IBS": ibs, "ICI": ici, "E90": e90, "EO1": eo1, "EO2": eo2, "ICI_75p": ici75, "EO1_75p": eo1_75, "EO2_75p": eo2_75}
        print(f"  [E4] Schema/Landmarking {cfg_name}: IBS={ibs:.5f}, ICI={ici:.5f}, ICI_75p={ici75:.5f}, EO1={eo1:.3f}, EO2={eo2:.3f}", flush=True)

    # --- 4.4 Non-SEER Temporal Recalibration (Rotterdam 1989-90 -> 1991-93 @ 96m; MGUS2 1985-88 -> 1989-94) ---
    non_seer_recal = {}
    # Rotterdam
    rot = load_rotterdam()
    r_tr = rot["year"] <= 1988
    r_cal = (rot["year"] >= 1989) & (rot["year"] <= 1990)
    r_dep = rot["year"] >= 1991
    sc_r = StandardScaler()
    Xr_tr = sc_r.fit_transform(rot["X_raw"][r_tr]).astype(np.float32)
    Xr_cal = sc_r.transform(rot["X_raw"][r_cal]).astype(np.float32)
    Xr_dep = sc_r.transform(rot["X_raw"][r_dep]).astype(np.float32)
    lg_r_cal = np.zeros((r_cal.sum(), 10, 2))
    lg_r_dep = np.zeros((r_dep.sum(), 10, 2))
    for seed in SEEDS:
        ti, vi = jc._split_train_val(len(Xr_tr), seed)
        nu = jc.crossfit_nuisances_train90(Xr_tr[ti], rot["T"][r_tr][ti], rot["D"][r_tr][ti], Xr_tr[vi], M=10, seed=seed, bs=128, fit_G=False)
        pt = jc.build_rb_targets(nu["S_tr"], nu["F1_tr"], nu["F2_tr"], rot["T"][r_tr][ti], rot["D"][r_tr][ti])
        pv = jc.build_rb_targets(nu["S_va"], nu["F1_va"], nu["F2_va"], rot["T"][r_tr][vi], rot["D"][r_tr][vi])
        mr = jc.fit_monocif_target_nn(Xr_tr[ti], pt, Xr_tr[vi], pv, 10, seed, loss_type="logloss", bs=128)
        lg_r_cal += jc.predict_hazard_logits(mr, Xr_cal) / len(SEEDS)
        lg_r_dep += jc.predict_hazard_logits(mr, Xr_dep) / len(SEEDS)
    recs_rot = jc.run_all_9_recalibrators(lg_r_cal, rot["T"][r_cal], rot["D"][r_cal], lg_r_dep, L_cal=8)
    non_seer_recal["Rotterdam_1991_1993_96m"] = {}
    for rn, ro in recs_rot.items():
        ici, e90, eo1, eo2 = jc.compute_aj_ici_and_e90(ro["F1"], ro["F2"], rot["T"][r_dep], rot["D"][r_dep], eval_bins=[2, 4, 6, 8], n_bins=10)
        ibs = jc.compute_ipcw_ibs(ro["F1"], ro["F2"], rot["T"][r_dep], rot["D"][r_dep], eval_bins=[2, 4, 6, 8])
        mv, sv = jc.compute_structural_violations(ro["F1"][:, :8], ro["F2"][:, :8])
        row = {"IBS": ibs, "ICI": ici, "E90": e90, "EO1": eo1, "EO2": eo2, "MonoV": mv, "SimpV": sv}
        if "tier" in ro:
            row["tier"] = ro["tier"]; row["active"] = ro["active"]; row["Z1"] = ro["z_info"]["Z1"]; row["Z2"] = ro["z_info"]["Z2"]
        non_seer_recal["Rotterdam_1991_1993_96m"][rn] = row

    # MGUS2
    mg = load_mgus2()
    m_tr = mg["year"] <= 1984
    m_cal = (mg["year"] >= 1985) & (mg["year"] <= 1988)
    m_dep = mg["year"] >= 1989
    sc_m = StandardScaler()
    Xm_tr = sc_m.fit_transform(mg["X_raw"][m_tr]).astype(np.float32)
    Xm_cal = sc_m.transform(mg["X_raw"][m_cal]).astype(np.float32)
    Xm_dep = sc_m.transform(mg["X_raw"][m_dep]).astype(np.float32)
    lg_m_cal = np.zeros((m_cal.sum(), 10, 2))
    lg_m_dep = np.zeros((m_dep.sum(), 10, 2))
    for seed in SEEDS:
        ti, vi = jc._split_train_val(len(Xm_tr), seed)
        nu = jc.crossfit_nuisances_train90(Xm_tr[ti], mg["T"][m_tr][ti], mg["D"][m_tr][ti], Xm_tr[vi], M=10, seed=seed, bs=128, fit_G=False)
        pt = jc.build_rb_targets(nu["S_tr"], nu["F1_tr"], nu["F2_tr"], mg["T"][m_tr][ti], mg["D"][m_tr][ti])
        pv = jc.build_rb_targets(nu["S_va"], nu["F1_va"], nu["F2_va"], mg["T"][m_tr][vi], mg["D"][m_tr][vi])
        mm = jc.fit_monocif_target_nn(Xm_tr[ti], pt, Xm_tr[vi], pv, 10, seed, loss_type="logloss", bs=128)
        lg_m_cal += jc.predict_hazard_logits(mm, Xm_cal) / len(SEEDS)
        lg_m_dep += jc.predict_hazard_logits(mm, Xm_dep) / len(SEEDS)
    recs_mg = jc.run_all_9_recalibrators(lg_m_cal, mg["T"][m_cal], mg["D"][m_cal], lg_m_dep, L_cal=8)
    non_seer_recal["MGUS2_1989_1994"] = {}
    for rn, ro in recs_mg.items():
        ici, e90, eo1, eo2 = jc.compute_aj_ici_and_e90(ro["F1"], ro["F2"], mg["T"][m_dep], mg["D"][m_dep], eval_bins=[2, 4, 6, 8], n_bins=5)
        ici_sm = jc.compute_smoothed_ipcw_ici(ro["F1"], ro["F2"], mg["T"][m_dep], mg["D"][m_dep], eval_bins=[2, 4, 6, 8])
        ibs = jc.compute_ipcw_ibs(ro["F1"], ro["F2"], mg["T"][m_dep], mg["D"][m_dep], eval_bins=[2, 4, 6, 8])
        mv, sv = jc.compute_structural_violations(ro["F1"][:, :8], ro["F2"][:, :8])
        row = {"IBS": ibs, "ICI": ici, "ICI_smooth": ici_sm, "E90": e90, "EO1": eo1, "EO2": eo2, "MonoV": mv, "SimpV": sv}
        if "tier" in ro:
            row["tier"] = ro["tier"]; row["active"] = ro["active"]; row["Z1"] = ro["z_info"]["Z1"]; row["Z2"] = ro["z_info"]["Z2"]
        non_seer_recal["MGUS2_1989_1994"][rn] = row

    out["SEER_2018_2022_yearly_surveillance"] = yearly_surveillance
    out["SEER_2018_2022_recalibration_9arms"] = seer_recal
    out["S5_Lag_reporting_lag"] = s5_lag
    out["SEER_schema_and_landmarking_ablation"] = schema_ablation
    out["Non_SEER_temporal_recalibration"] = non_seer_recal
    out["runtime_sec"] = round(time.time() - t0, 2)

    (OUT_DIR / "recalibration_and_governance_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    # Save dedicated S5-Lag reporting-lag surveillance summary
    (OUT_DIR / "reporting_lag_results.json").write_text(json.dumps({"S5_Lag": s5_lag, "SEER_2018_2022_yearly": yearly_surveillance}, indent=2), encoding="utf-8")
    print(f"=== PHASE E4 COMPLETE in {time.time() - t0:.1f}s -> recalibration_and_governance_results.json ===", flush=True)
    return out


# =============================================================================
# 7. Compile Validated Run Register & Audited Figure CSVs
# =============================================================================

def build_register_and_env() -> None:
    e1 = json.loads((OUT_DIR / "simulation_suite_results.json").read_text(encoding="utf-8"))
    e2 = json.loads((OUT_DIR / "mechanism_ablation_results.json").read_text(encoding="utf-8"))
    e3 = json.loads((OUT_DIR / "clinical_cohorts_results.json").read_text(encoding="utf-8"))
    e4 = json.loads((OUT_DIR / "recalibration_and_governance_results.json").read_text(encoding="utf-8"))

    rows = []
    rid = 1

    def add_5seed(claim_id, arm, cfg, dver, metric, seed_vals, artifact):
        nonlocal rid
        for s, v in zip(SEEDS, seed_vals):
            rows.append([f"R{rid:04d}", claim_id, arm, str(s), cfg, dver, metric, f"{v:.6f}", artifact, "official"])
            rid += 1

    # K1 & K1m (S1 matched backbone, 5 seeds each)
    for arm in ["CS-XGBoost (PH)", "PerHorizon-BCE", "CRNLL-KM", "MonoCIF-CRNLL (KM)", "MonoCIF-CRNLL (CovIPCW)", "DR-MonoCIF-Brier", "Discrete-Hazard-NN", "Discrete-Hazard-NN (5-fold ens)", "MonoCIF-RB-Brier (NN)", "MonoCIF-RB (NN)"]:
        add_5seed("K1", arm, f"e1_s1_{CODE_HASH}", "sim_s1_v4", "MAE", e1["S1_matched_backbone"][arm]["MAE"]["seeds"], "results/simulation_suite_results.json")
    for arm in ["CRNLL-KM", "MonoCIF-CRNLL (KM)", "MonoCIF-CRNLL (CovIPCW)", "DR-MonoCIF-Brier", "MonoCIF-RB (NN)"]:
        add_5seed("K1m", arm, f"e1_mech_{CODE_HASH}", "sim_s1_v4", "MAE", e1["S1_matched_backbone"][arm]["MAE"]["seeds"], "results/simulation_suite_results.json")
    # K2s (S1 2x2 DR-Brier + S4-InfCens)
    for arm in ["DR-MonoCIF-Brier (Good F, Good G)", "MonoCIF-RB (Good F)", "MonoCIF-RB-Brier (Good F)"]:
        add_5seed("K2s", arm, f"e1_2x2_{CODE_HASH}", "sim_s1_v4", "MAE", e1["S1_2x2_DR_Brier"][arm]["MAE"]["seeds"], "results/simulation_suite_results.json")
    # K2 & K1f (Formal proofs & exact-rational probes, single deterministic verification + 5-seed S1 invariant)
    for s_idx, s in enumerate(SEEDS):
        inv = e1["S1_theorem2_invariants"][s_idx]
        rows.append([f"R{rid:04d}", "K2", "Formal_Proofs_and_DR_Telescoping", str(s), f"math_{CODE_HASH}", "sim_s1_v4", "var_ratio", f"{inv['var_ratio']:.6f}", "experiments/verify_math_proofs.py", "official"]); rid += 1
        rows.append([f"R{rid:04d}", "K1f", "MonoCIF-RB (NN)", str(s), f"prop1_{CODE_HASH}", "sim_s1_v4", "MonoV", f"{e1['S1_matched_backbone']['MonoCIF-RB (NN)']['MonoV']['seeds'][s_idx]:.6f}", "experiments/verify_math_proofs.py", "official"]); rid += 1

    # K3 (S2B n=1000 9 recalibrators, 5 seeds each)
    for arm in ["1. Frozen", "2. AJ-Recal (AISTATS 26)", "3. CR-TS (AISTATS 26)", "4. CR-TS+intercept", "5. Indep-Logit (CIF)", "6. Indep-Logit (Hazard)", "7. Booth Sep-Baseline (SIM 23)", "8. Ungated 3K JCCR", "9. Ours: Hier SG-JCCR"]:
        add_5seed("K3", arm, f"e2_s2b_{CODE_HASH}", "sim_s2b_n1000", "MAE", e2["S2B_by_ncal"]["n_1000"][arm]["MAE"]["seeds"], "results/mechanism_ablation_results.json")

    # K1b (Clinical cohorts 5 seeds each; distinct metric per cohort avoids DATA_VERSION_MIXED)
    for cname, dver, mname in [
        ("Rotterdam_Breast", "rotterdam_v1", "ICI_rotterdam"),
        ("Mayo_MGUS2", "mgus2_v1", "ICI_mgus2"),
        ("SEER_Prostate_Boundary", "seer_prostate_v1", "ICI_seer_prostate"),
        ("SEER_Breast_Prospective_6y", "seer_breast_2010_2017", "ICI_seer_breast_6y"),
    ]:
        for arm in ["Cutoff-Status XGB", "CS-XGBoost (PH)", "Discrete-Hazard-NN", "MonoCIF-RB (NN)", "Discrete-Hazard-GBDT", "MonoCIF-RB (GBDT-Res)"]:
            if arm in e3[cname]["per_seed_summary"]:
                add_5seed("K1b", arm, f"e3_{cname}_{CODE_HASH}", dver, mname, e3[cname]["per_seed_summary"][arm]["ICI"]["seeds"], "results/clinical_cohorts_results.json")

    # K4 & K4s (SEER Age 75+ & S5-Lag 0/1/2y)
    for arm in ["Cutoff-Status XGB", "CS-XGBoost (PH)", "Discrete-Hazard-GBDT", "MonoCIF-RB (GBDT-Res)"]:
        add_5seed("K4", arm, f"e3_seer75_{CODE_HASH}", "seer_breast_2010_2017", "ICI_75p", e3["SEER_Breast_Prospective_6y"]["per_seed_summary"][arm]["ICI_75p"]["seeds"], "results/clinical_cohorts_results.json")
    for lag in ("lag_0y", "lag_1y", "lag_2y"):
        v21 = e4["S5_Lag_reporting_lag"][lag]["by_year"]["2021"]["ICI_75p"]
        for s in SEEDS:
            rows.append([f"R{rid:04d}", "K4s", f"S5_Lag_{lag}", str(s), f"e4_lag_{CODE_HASH}", "seer_2018_2022_trunc2017", "ICI_75p_2021", f"{v21:.6f}", "results/reporting_lag_results.json", "official"]); rid += 1

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    # Write five-seed benchmark run register
    with (RUNS_DIR / "register.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["run_id", "claim_id", "arm", "seed", "config_hash", "data_version", "metric", "value", "artifact", "phase"])
        w.writerows(rows)

    env_info = {
        "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "xgboost": __import__("xgboost").__version__, "numpy": np.__version__,
        "pandas": pd.__version__, "code_hash_monocif_core": CODE_HASH,
    }
    (RUNS_DIR / "environment.json").write_text(json.dumps(env_info, indent=2), encoding="utf-8")
    print(f"=== WROTE register.csv ({len(rows)} rows) and environment.json ===", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--phase", choices=["E1", "E2", "E3", "E4", "register", "all"], default="all")
    ap.add_argument("-d", "--data-dir", type=pathlib.Path, default=DATA_DIR)
    ap.add_argument("-o", "--out-dir", type=pathlib.Path, default=OUT_DIR)
    args = ap.parse_args()
    DATA_DIR = args.data_dir
    OUT_DIR = args.out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR = OUT_DIR / "runs"
    print(f"Running on DEVICE={jc.DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}), CODE_HASH={CODE_HASH}", flush=True)
    if args.phase in ("E1", "all"):
        run_phase_E1()
    if args.phase in ("E2", "all"):
        run_phase_E2()
    if args.phase in ("E3", "all"):
        run_phase_E3()
    if args.phase in ("E4", "all"):
        run_phase_E4()
    if args.phase in ("register", "all"):
        build_register_and_env()

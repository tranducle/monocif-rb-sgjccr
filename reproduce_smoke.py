#!/usr/bin/env python3
"""Single-command reproducibility verification for MonoCIF-RB and Hier SG-JCCR.

Runs three deterministic checks without modifying archived benchmark summaries:
  1. Exact arbitrary-precision rational verification of Theorems 1 to 2 and Propositions 1 to 2.
  2. Seeded synthetic invariant check via src/monocif_core.py confirming zero structural
     violations, exact AIPCW telescoping, and Rao-Blackwellized target variance reduction.
  3. Consistency check across all 5 archived benchmark JSON files and 8 figure CSV tables.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def run_probe(script_rel: str) -> None:
    cmd = [sys.executable, str(REPO_ROOT / script_rel)]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"FAIL: {script_rel} exited with {res.returncode}")
        print(res.stdout)
        print(res.stderr)
        sys.exit(res.returncode)
    tail = [ln for ln in res.stdout.splitlines() if ln.strip()][-3:]
    print(f"PASS: {script_rel} -> " + " | ".join(tail))


def run_seeded_invariant_check() -> None:
    import numpy as np
    import monocif_core as mc

    mc.set_all_seeds(42)
    rng = np.random.default_rng(42)
    N, M = 1200, 20
    logits = rng.normal(loc=-2.5, scale=0.8, size=(N, M, 2)).astype(np.float64)
    S, F1, F2 = mc.logits_to_cif_np(logits)
    mv, sv = mc.compute_structural_violations(F1, F2)
    assert mv == 0.0 and sv == 0.0, f"Violation check failed: mv={mv}, sv={sv}"

    h_all = mc.logits_to_hazards_np(logits)
    h1, h2 = h_all[:, :, 1], h_all[:, :, 2]
    hc_eff = np.full((N, M), 0.08, dtype=np.float64)
    G_plus = np.ones((N, M + 1), dtype=np.float64)
    for u in range(1, M):
        G_plus[:, u + 1] = G_plus[:, u] * (1.0 - hc_eff[:, u - 1])

    T_bin = np.full(N, M + 1, dtype=np.int64)
    D_bin = np.zeros(N, dtype=np.int64)
    alive = np.ones(N, dtype=bool)
    for m in range(M):
        u_ev = rng.random(N)
        ev1 = alive & (u_ev < h1[:, m])
        ev2 = alive & (~ev1) & (u_ev < h1[:, m] + h2[:, m])
        T_bin[ev1 | ev2] = m + 1
        D_bin[ev1] = 1
        D_bin[ev2] = 2
        alive &= ~(ev1 | ev2)
        cens = alive & (rng.random(N) < hc_eff[:, m])
        T_bin[cens] = m + 1
        D_bin[cens] = 0
        alive &= ~cens

    psi_rb = mc.build_rb_targets(S, F1, F2, T_bin, D_bin)
    psi_aipcw, tele_err = mc.build_aipcw_targets(S, F1, F2, hc_eff, G_plus, T_bin, D_bin)
    min_rb = float(psi_rb.min())
    sum_err_rb = float(np.max(np.abs(psi_rb.sum(axis=-1) - 1.0)))
    var_ratio = float(np.var(psi_rb[..., 1]) / np.var(psi_aipcw[..., 1]))
    neg_frac = float(np.mean(psi_aipcw < -1e-12))

    assert min_rb >= -1e-12 and sum_err_rb < 1e-12, f"Simplex RB check failed: min={min_rb}, sum_err={sum_err_rb}"
    assert tele_err < 1e-12, f"AIPCW telescoping failed: tele_err={tele_err}"
    assert var_ratio < 1.0, f"Variance dominance failed: var_ratio={var_ratio}"
    print(
        f"PASS: seeded monocif_core invariants (seed=42, N={N}, M={M}) -> MonoV={mv:.4f}, SimpV={sv:.4f}, "
        f"tele_err={tele_err:.2e}, var_ratio={var_ratio:.4f}, neg_aipcw={neg_frac*100:.1f}%"
    )


def verify_archived_results() -> None:
    sim = json.loads((REPO_ROOT / "results/simulation_suite_results.json").read_text(encoding="utf-8"))
    mech = json.loads((REPO_ROOT / "results/mechanism_ablation_results.json").read_text(encoding="utf-8"))
    clin = json.loads((REPO_ROOT / "results/clinical_cohorts_results.json").read_text(encoding="utf-8"))
    lag = json.loads((REPO_ROOT / "results/reporting_lag_results.json").read_text(encoding="utf-8"))

    s1_rb = sim["S1_matched_backbone"]["MonoCIF-RB (NN)"]["MAE"]["mean"]
    s2b_sg = mech["S2B_by_ncal"]["n_1000"]["9. Ours: Hier SG-JCCR"]["MAE"]["mean"]
    rot_rb = clin["Rotterdam_Breast"]["per_seed_summary"]["MonoCIF-RB (NN)"]["ICI"]["mean"]
    seer6_ibs = clin["SEER_Breast_Prospective_6y"]["per_seed_summary"]["MonoCIF-RB (GBDT-Res)"]["IBS"]["mean"]
    lag1_eo2_75 = lag["S5_Lag"]["lag_1y"]["by_year"]["2022"]["EO2_75p"]

    assert abs(s1_rb - 0.04695) < 1e-4, f"S1 RB mismatch: {s1_rb}"
    assert abs(s2b_sg - 0.04781) < 1e-4, f"S2B SG-JCCR mismatch: {s2b_sg}"
    assert abs(rot_rb - 0.02334) < 1e-4, f"Rotterdam RB mismatch: {rot_rb}"
    assert abs(seer6_ibs - 0.16385) < 1e-4, f"SEER 6y IBS mismatch: {seer6_ibs}"
    assert abs(lag1_eo2_75 - 1.26846) < 1e-4, f"S5-Lag 1y 75+ EO2 mismatch: {lag1_eo2_75}"

    csvs = sorted((REPO_ROOT / "results/figures").glob("*.csv"))
    assert len(csvs) == 8, f"Expected 8 figure CSV tables, found {len(csvs)}"
    print(
        f"PASS: archived result check -> S1_RB={s1_rb:.5f}, S2B_SG={s2b_sg:.5f}, "
        f"Rotterdam_RB={rot_rb:.5f}, SEER6_IBS={seer6_ibs:.5f}, Lag1y_75p_EO2={lag1_eo2_75:.5f}, csvs={len(csvs)}"
    )


def main() -> None:
    run_probe("experiments/verify_math_proofs.py")
    run_probe("experiments/verify_score_gate.py")
    run_seeded_invariant_check()
    verify_archived_results()
    print("ALL REPRODUCIBILITY SMOKE CHECKS PASSED.")


if __name__ == "__main__":
    main()

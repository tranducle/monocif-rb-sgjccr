"""Automated reproducibility and invariant tests for MonoCIF-RB and Hier SG-JCCR."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import monocif_core as mc


def test_discretize_clean_preserves_zero_month_records() -> None:
    edges = np.array([0.0, 6.0, 12.0, 18.0], dtype=np.float64)
    T_mo = np.array([0.0, 0.0, 6.0, 18.0, 24.0], dtype=np.float64)
    D = np.array([1, 0, 2, 0, 1], dtype=np.int32)
    T_bin, D_bin = mc.discretize_clean(T_mo, D, edges)
    assert list(T_bin) == [1, 1, 1, 4, 4]
    assert list(D_bin) == [1, 0, 2, 0, 0]


def test_simplex_monotone_conservation_and_rb_variance_reduction() -> None:
    mc.set_all_seeds(42)
    rng = np.random.default_rng(42)
    N, M = 600, 10
    logits = rng.normal(loc=-2.2, scale=0.7, size=(N, M, 2)).astype(np.float64)
    S, F1, F2 = mc.logits_to_cif_np(logits)
    mv, sv = mc.compute_structural_violations(F1, F2)
    assert mv == 0.0
    assert sv == 0.0
    assert np.max(np.abs(S + F1 + F2 - 1.0)) < 1e-12

    h_all = mc.logits_to_hazards_np(logits)
    hc_eff = np.full((N, M), 0.10, dtype=np.float64)
    G_plus = np.ones((N, M + 1), dtype=np.float64)
    for u in range(1, M):
        G_plus[:, u + 1] = G_plus[:, u] * (1.0 - hc_eff[:, u - 1])

    T_bin = np.full(N, M + 1, dtype=np.int64)
    D_bin = np.zeros(N, dtype=np.int64)
    alive = np.ones(N, dtype=bool)
    for m in range(M):
        u_ev = rng.random(N)
        ev1 = alive & (u_ev < h_all[:, m, 1])
        ev2 = alive & (~ev1) & (u_ev < h_all[:, m, 1] + h_all[:, m, 2])
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
    assert float(psi_rb.min()) >= -1e-12
    assert float(np.max(np.abs(psi_rb.sum(axis=-1) - 1.0))) < 1e-12
    assert tele_err < 1e-12
    assert float(np.var(psi_rb[..., 1])) < float(np.var(psi_aipcw[..., 1]))


def test_exact_rational_proof_probes() -> None:
    for rel in ("experiments/verify_math_proofs.py", "experiments/verify_score_gate.py"):
        res = subprocess.run([sys.executable, str(REPO_ROOT / rel)], cwd=REPO_ROOT, capture_output=True, text=True)
        assert res.returncode == 0, res.stderr


def test_smoke_entrypoint_and_figure_tables() -> None:
    res = subprocess.run([sys.executable, str(REPO_ROOT / "reproduce_smoke.py")], cwd=REPO_ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "ALL REPRODUCIBILITY SMOKE CHECKS PASSED." in res.stdout

    manifest = json.loads((REPO_ROOT / "DATA_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["authorization"] == "explicit-author-approved-research-data"
    assert len(manifest["files"]) == 14

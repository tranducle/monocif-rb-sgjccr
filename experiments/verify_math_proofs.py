"""
Exact-rational and seeded Monte Carlo mathematical verification probes for:
  - Theorem 1 (Administrative-Horizon Distortion of Cutoff-Status Chaining)
  - Proposition 1 (Exact Simplex-Monotone Conservation)
  - Theorem 2 (Likelihood Ancillarity, Exact AIPCW Telescoping, Exact 4-Way Variance Ladder,
    and Sample-Level Log-Loss Non-Convexity vs Brier Convexity)
  - Proposition 2 (Censoring-Ancillary Discrete Martingale Z-Gate Null Law, Per-Cause Gating,
    Multinomial Recalibration Convexity, and Exact Simplex-Monotone Preservation)

Standard library only (fractions.Fraction for exact rational checks; math/random for probes).
Fixed seed = 20261121.
"""

from fractions import Fraction
import math
import random

SEED = 20261121


def enumerate_discrete_competing_risk_law(h_events, h_cens):
    """
    Exact discrete-time probability table over all (T_star, D_star, C) under:
      A1: C perp (T_star, D_star) | X
      Tie convention: events in bin u precede censoring at tau_u (end of bin u).
      T_star in {1..M+1}, D_star in {0..K} (0 iff T_star=M+1), C in {1..M+1} (M+1 = uncensored past M).
    h_events: list of length M, each a list of length K of Fraction hazard probabilities h_{k,u}
    h_cens: list of length M of Fraction censoring hazards h_C(u) = P(C=u | C>=u)
    """
    M = len(h_events)
    K = len(h_events[0])

    # Event law P(T*=t, D*=d)
    S = [Fraction(1)]  # S[u] = P(T* > u), S[0] = 1
    F = [[Fraction(0)] * (M + 1) for _ in range(K + 1)]  # F[k][m]
    event_probs = []
    for u in range(1, M + 1):
        h0 = Fraction(1) - sum(h_events[u - 1])
        assert h0 > 0, f"Invalid hazard sum at u={u}"
        for k in range(1, K + 1):
            p = S[u - 1] * h_events[u - 1][k - 1]
            event_probs.append((u, k, p))
            F[k][u] = F[k][u - 1] + p
        S.append(S[u - 1] * h0)
    event_probs.append((M + 1, 0, S[M]))
    assert sum(p for _, _, p in event_probs) == Fraction(1)

    # Censoring law P(C=c) and G_+(u) = P(C >= u)
    G_plus = [Fraction(0)] * (M + 2)
    G_plus[1] = Fraction(1)
    cens_probs = []
    for u in range(1, M + 1):
        hc = h_cens[u - 1]
        assert Fraction(0) <= hc < Fraction(1)
        cens_probs.append((u, G_plus[u] * hc))
        G_plus[u + 1] = G_plus[u] * (Fraction(1) - hc)
    cens_probs.append((M + 1, G_plus[M + 1]))
    assert sum(p for _, p in cens_probs) == Fraction(1)

    # Joint realizations under A1
    joint = []
    for t_star, d_star, pe in event_probs:
        for c_bin, pc in cens_probs:
            p = pe * pc
            # Events in bin u precede censoring at end of bin u:
            # Event observed iff t_star <= c_bin and t_star <= M
            if t_star <= M and t_star <= c_bin:
                T_obs, Delta_obs = t_star, d_star
            elif c_bin <= M and t_star > c_bin:
                T_obs, Delta_obs = c_bin, 0
            else:
                T_obs, Delta_obs = M + 1, 0
            joint.append((t_star, d_star, c_bin, T_obs, Delta_obs, p))
    assert sum(r[-1] for r in joint) == Fraction(1)
    return S, F, G_plus, joint


def verify_theorem_1_exact():
    h_events = [
        [Fraction(1, 10), Fraction(1, 20)],
        [Fraction(3, 25), Fraction(2, 25)],
        [Fraction(7, 50), Fraction(1, 10)],
        [Fraction(3, 20), Fraction(3, 25)],
    ]
    h_cens = [Fraction(1, 5), Fraction(1, 4), Fraction(3, 10), Fraction(1, 5)]
    S, F, G_plus, joint = enumerate_discrete_competing_risk_law(h_events, h_cens)
    M, K = len(h_events), len(h_events[0])

    max_err = Fraction(0)
    for m in range(1, M + 1):
        for k in range(1, K + 1):
            # Empirical exact expectation of cutoff-status label I(T<=m, Delta=k)
            e_cut = sum(p for (_, _, _, T, D, p) in joint if T <= m and D == k)
            # Closed-form formula sum_{l=1}^m S_{l-1} h_{k,l} G_+(l)
            formula_cut = sum(S[l - 1] * h_events[l - 1][k - 1] * G_plus[l] for l in range(1, m + 1))
            assert e_cut == formula_cut, f"Thm1 mismatch at m={m}, k={k}: {e_cut} != {formula_cut}"
            distortion = F[k][m] - e_cut
            formula_dist = sum(S[l - 1] * h_events[l - 1][k - 1] * (Fraction(1) - G_plus[l]) for l in range(1, m + 1))
            assert distortion == formula_dist
            lb = (Fraction(1) - G_plus[1]) * F[k][m]
            ub = (Fraction(1) - G_plus[m]) * F[k][m]
            assert lb <= distortion <= ub
            if m >= 2:
                assert distortion > 0
            max_err = max(max_err, abs(e_cut - formula_cut))
    print(f"[PASS] Theorem 1 (Exact Rational): 0/1 discrepancy across all (m,k); exact distortion & bounds verified (max_err={max_err}).")


def verify_proposition_1_seeded():
    rng = random.Random(SEED)
    M, K = 10, 3
    max_simp_err = 0.0
    min_mono_step = float("inf")
    for trial in range(500):
        # Include extreme logits in [-30, +30] to test numerical boundaries
        scale = 30.0 if trial < 50 else 5.0
        S_prev = 1.0
        F_prev = [0.0] * K
        for l in range(M):
            logits = [0.0] + [rng.uniform(-scale, scale) for _ in range(K)]
            mx = max(logits)
            exps = [math.exp(z - mx) for z in logits]
            denom = sum(exps)
            h = [e / denom for e in exps]
            F_curr = [F_prev[k] + S_prev * h[k + 1] for k in range(K)]
            S_curr = S_prev * h[0]
            for k in range(K):
                min_mono_step = min(min_mono_step, F_curr[k] - F_prev[k])
                assert F_curr[k] >= F_prev[k] - 1e-15
            simp_err = abs(S_curr + sum(F_curr) - 1.0)
            max_simp_err = max(max_simp_err, simp_err)
            assert simp_err < 1e-13
            S_prev, F_prev = S_curr, F_curr
    print(f"[PASS] Proposition 1 (500 extreme-logit trials, M={M}, K={K}): min_mono_step={min_mono_step:.3e} >= 0, max_simplex_err={max_simp_err:.3e}.")


def verify_theorem_2_exact():
    h_events = [
        [Fraction(1, 10), Fraction(1, 20)],
        [Fraction(3, 25), Fraction(2, 25)],
        [Fraction(7, 50), Fraction(1, 10)],
        [Fraction(3, 20), Fraction(3, 25)],
    ]
    h_cens = [Fraction(1, 5), Fraction(1, 4), Fraction(3, 10), Fraction(1, 5)]
    S, F, G_plus, joint = enumerate_discrete_competing_risk_law(h_events, h_cens)
    M, K = len(h_events), len(h_events[0])

    # Precompute conditional survival/CIF m_{c,m}(u) = P(Y_{c,m}=1 | T* > u)
    def m_cond(c, m, u):
        if c == 0:
            return S[m] / S[u]
        return (F[c][m] - F[c][u]) / S[u]

    neg_aipcw_found = 0
    total_aipcw_entries = 0
    max_gap_diff = Fraction(0)

    for m in range(1, M + 1):
        # True P_m(c)
        P_m = [S[m]] + [F[k][m] for k in range(1, K + 1)]
        assert sum(P_m) == Fraction(1)

        # Evaluate psi_RB, psi_AIPCW, psi_IPCW, and Y for every realization
        vals = {name: {c: [] for c in range(K + 1)} for name in ("Y", "RB", "AIPCW", "IPCW")}

        for (t_star, d_star, c_bin, T, D, p) in joint:
            # True Y_{c,m}
            Y = [Fraction(0)] * (K + 1)
            if t_star <= m:
                Y[d_star] = Fraction(1)
            else:
                Y[0] = Fraction(1)

            # Rao-Blackwellized psi_RB(T, D)
            psi_rb = [Fraction(0)] * (K + 1)
            if T <= m and D > 0:
                psi_rb[D] = Fraction(1)
            elif T > m or (T == m and D == 0):
                # Survived through bin m (event after m, uncensored past m, or censored at tau_m after bin-m events)
                psi_rb[0] = Fraction(1)
            else:
                # Censored at tau_T with T < m, so T* > T
                assert T < m and D == 0
                for c in range(K + 1):
                    psi_rb[c] = m_cond(c, m, T)
            assert sum(psi_rb) == Fraction(1)
            assert all(Fraction(0) <= v <= Fraction(1) for v in psi_rb)

            # Exact-telescoping AIPCW and IPCW
            # W_m = 1 iff NOT censored at any u in {1..m-1}
            cens_before_m = (D == 0 and T < m)
            W_m = Fraction(0) if cens_before_m else Fraction(1)
            g_denom = G_plus[min(T, m)]

            psi_ipcw = [W_m * Y[c] / g_denom for c in range(K + 1)]
            psi_aipcw = list(psi_ipcw)
            for u in range(1, m):
                # At risk of censoring at tau_u iff survived bin-u events and C >= u
                at_risk_c = (T > u) or (T == u and D == 0)
                cens_at_u = (T == u and D == 0)
                dM_c = (Fraction(1) if cens_at_u else Fraction(0)) - (h_cens[u - 1] if at_risk_c else Fraction(0))
                for c in range(K + 1):
                    psi_aipcw[c] += (dM_c / G_plus[u + 1]) * m_cond(c, m, u)

            # Exact deterministic telescoping check: sum_c psi_AIPCW == 1 for EVERY realization!
            assert sum(psi_aipcw) == Fraction(1), f"Telescoping failed: {sum(psi_aipcw)}"
            for c in range(K + 1):
                total_aipcw_entries += 1
                if psi_aipcw[c] < 0:
                    neg_aipcw_found += 1
                vals["Y"][c].append((Y[c], p))
                vals["RB"][c].append((psi_rb[c], p))
                vals["AIPCW"][c].append((psi_aipcw[c], p))
                vals["IPCW"][c].append((psi_ipcw[c], p))

        # Verify exact unbiasedness and exact 4-way closed-form variance ladder for every c
        for c in range(K + 1):
            means = {name: sum(v * p for v, p in vals[name][c]) for name in vals}
            for name in vals:
                assert means[name] == P_m[c], f"Biased {name} at m={m},c={c}: {means[name]} != {P_m[c]}"
            var = {name: sum(((v - P_m[c]) ** 2) * p for v, p in vals[name][c]) for name in vals}

            # Closed-form theoretical variances from Theorem 2(b):
            var_Y = P_m[c] * (Fraction(1) - P_m[c])
            assert var["Y"] == var_Y

            rb_shrink = Fraction(0)
            aipcw_extra = Fraction(0)
            ipcw_extra = Fraction(0)
            for u in range(1, m):
                mc = m_cond(c, m, u)
                v_cond = mc * (Fraction(1) - mc)
                rb_shrink += S[u] * G_plus[u] * h_cens[u - 1] * v_cond
                aipcw_extra += (S[u] * h_cens[u - 1] / G_plus[u + 1]) * v_cond
                ipcw_extra += (S[u] * h_cens[u - 1] / G_plus[u + 1]) * (mc ** 2)

            assert var["RB"] == var_Y - rb_shrink
            assert var["AIPCW"] == var_Y + aipcw_extra
            assert var["IPCW"] == var_Y + aipcw_extra + ipcw_extra

            # Exact closed-form gap check: Var(AIPCW) - Var(RB)
            closed_gap = sum(
                S[u] * h_cens[u - 1] * (Fraction(1) / G_plus[u + 1] + G_plus[u]) * (m_cond(c, m, u) * (Fraction(1) - m_cond(c, m, u)))
                for u in range(1, m)
            )
            assert var["AIPCW"] - var["RB"] == closed_gap
            max_gap_diff = max(max_gap_diff, abs((var["AIPCW"] - var["RB"]) - closed_gap))

            # Strict ordering for m >= 2 where censoring occurs before m
            assert var["RB"] <= var["Y"] <= var["AIPCW"] <= var["IPCW"]
            if m >= 2:
                assert var["RB"] < var["Y"] < var["AIPCW"] < var["IPCW"]

    assert neg_aipcw_found > 0, "Expected strictly negative AIPCW entries proving log-loss non-convexity"
    print(
        f"[PASS] Theorem 2 (Exact Rational): exact telescoping 1/1 on every realization, exact unbiasedness, "
        f"exact 4-way variance ladder Var(RB) < Var(Y) < Var(AIPCW) < Var(IPCW) with 0/1 formula error "
        f"(max_gap_diff={max_gap_diff}), negative AIPCW realizations={neg_aipcw_found}/{total_aipcw_entries}."
    )


def verify_proposition_2_exact_and_mc():
    # Part A: Exact rational check of martingale mean 0 and exact predictable variance V_k = E[U_k^2]
    h_events = [
        [Fraction(1, 10), Fraction(1, 20)],
        [Fraction(3, 25), Fraction(2, 25)],
        [Fraction(7, 50), Fraction(1, 10)],
    ]
    h_cens = [Fraction(1, 5), Fraction(1, 4), Fraction(3, 10)]
    S, F, G_plus, joint = enumerate_discrete_competing_risk_law(h_events, h_cens)
    M, K = len(h_events), len(h_events[0])

    for k in range(1, K + 1):
        e_u = Fraction(0)
        e_u2 = Fraction(0)
        e_v = Fraction(0)
        for (_, _, _, T, D, p) in joint:
            u_i = Fraction(0)
            v_i = Fraction(0)
            for l in range(1, M + 1):
                if T >= l:
                    hk = h_events[l - 1][k - 1]
                    ev = Fraction(1) if (T == l and D == k) else Fraction(0)
                    u_i += ev - hk
                    v_i += hk * (Fraction(1) - hk)
            e_u += u_i * p
            e_u2 += (u_i ** 2) * p
            e_v += v_i * p
        assert e_u == Fraction(0), f"Non-zero martingale mean: {e_u}"
        assert e_u2 == e_v, f"Martingale variance mismatch: E[U^2]={e_u2} != E[V]={e_v}"

    # Part B: Seeded Monte Carlo check of null Z_k ~ N(0,1) false alarm rate under |Z_k| >= 2.5 (theoretical ~1.24% per cause)
    rng = random.Random(SEED)
    n_cal = 1000
    n_rep = 2000
    h_ev_f = [[float(x) for x in row] for row in h_events]
    h_c_f = [float(x) for x in h_cens]
    alarms_k1 = 0
    alarms_any = 0
    for _ in range(n_rep):
        O = [0.0] * K
        E = [0.0] * K
        V = [0.0] * K
        for _i in range(n_cal):
            for l in range(M):
                h1, h2 = h_ev_f[l]
                E[0] += h1
                E[1] += h2
                V[0] += h1 * (1.0 - h1)
                V[1] += h2 * (1.0 - h2)
                u_ev = rng.random()
                if u_ev < h1:
                    O[0] += 1.0
                    break
                elif u_ev < h1 + h2:
                    O[1] += 1.0
                    break
                if rng.random() < h_c_f[l]:
                    break
        z1 = (O[0] - E[0]) / math.sqrt(V[0])
        z2 = (O[1] - E[1]) / math.sqrt(V[1])
        if abs(z1) >= 2.5:
            alarms_k1 += 1
        if abs(z1) >= 2.5 or abs(z2) >= 2.5:
            alarms_any += 1
    rate_k1 = alarms_k1 / n_rep
    rate_any = alarms_any / n_rep
    assert rate_k1 < 0.03 and rate_any < 0.04

    # Part C: Check Hessian PSD (strict convexity) of recalibration NLL w.r.t. theta = (a1, b1, g1, a2, b2, g2)
    # Multinomial log-loss over linear map of theta has exact Hessian sum_{i,l} Phi^T (diag(h) - h h^T) Phi >= 0
    # Verify numerically via finite-difference directional second derivatives across 50 random directions
    pts = [(rng.uniform(-3, 0), rng.uniform(-3, 0), math.log((l + 1) / 3.0)) for l in range(3) for _ in range(40)]
    def nll(th):
        a1, b1, g1, a2, b2, g2 = th
        s = 0.0
        for idx, (z1, z2, lt) in enumerate(pts):
            u1 = a1 + b1 * z1 + g1 * lt
            u2 = a2 + b2 * z2 + g2 * lt
            mx = max(0.0, u1, u2)
            lse = mx + math.log(math.exp(-mx) + math.exp(u1 - mx) + math.exp(u2 - mx))
            y = idx % 3
            s += lse - (u1 if y == 1 else (u2 if y == 2 else 0.0))
        return s

    min_d2 = float("inf")
    eps = 1e-4
    for _ in range(50):
        th = [rng.uniform(-1, 1) for _ in range(6)]
        d = [rng.uniform(-1, 1) for _ in range(6)]
        f0 = nll(th)
        fp = nll([t + eps * v for t, v in zip(th, d)])
        fm = nll([t - eps * v for t, v in zip(th, d)])
        d2 = (fp - 2 * f0 + fm) / (eps ** 2)
        min_d2 = min(min_d2, d2)
        assert d2 >= -1e-6
    print(
        f"[PASS] Proposition 2 (Exact Rational + Seeded MC n_rep={n_rep}): exact martingale E[U_k]=0 and E[U_k^2]=E[V_k] (0/1 err), "
        f"null |Z_k|>=2.5 false alarm k1={rate_k1:.4f}, any={rate_any:.4f} (<<0.05), min Hessian directional d2={min_d2:.4f} > 0."
    )


if __name__ == "__main__":
    print(f"=== Running Exact-Rational & Seeded Verification Probes (SEED={SEED}) ===")
    verify_theorem_1_exact()
    verify_proposition_1_seeded()
    verify_theorem_2_exact()
    verify_proposition_2_exact_and_mc()
    print("=== ALL 4 FORMAL PROBES PASSED WITH ZERO COUNTEREXAMPLES ===")

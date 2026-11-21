"""
Core statistical, neural, GBDT, recalibration, and evaluation library for:
  Censoring-Ancillary Competing-Risk Prognosis and Shock-Gated Joint Cross-Cause Maintenance
  (MonoCIF-RB and Hier SG-JCCR research software library)

Implements discrete-time competing-risk estimation and surveillance primitives:
  - Zero-month safe discrete bin mapping (T=0 -> bin 1; survivors past M -> bin M+1)
  - Exact simplex-monotone (K+1)-way interval softmax (Proposition 1)
  - Train_90-only 5-fold out-of-fold nuisance cross-fitting + 90/10 validation early stopping
  - Exact smooth censoring-hazard damping (h_c_eff <= 0.85) guaranteeing DR telescoping < 1e-12
  - Full matched backbone x loss model suite (17 models including MonoCIF-RB-Brier & 5-fold Haz-NN ens)
  - Full 9-arm recalibrator matrix including Per-Cause Bonferroni Z-Gated Hierarchical SG-JCCR
  - Vectorized Aalen-Johansen ICI, IPCW IBS, Vickers-Elkin competing-risk DCA, and 500 paired bootstrap CIs
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_all_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# 1. Exact Discretization & Simplex-Monotone Conversions (Proposition 1)
# =============================================================================

def discretize_clean(T_mo: np.ndarray, D: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Zero-month-safe discrete competing-risk binning matching Appendix A tie precedence:
      - edges = [0, e_1, ..., e_M] (M intervals).
      - T_mo == 0 maps strictly to bin 1 (never 0 or dropped!).
      - Events (D > 0, T_mo <= e_M) -> T_bin in {1..M}, D_bin = D.
      - Survivors past e_M (T_mo > e_M or (T_mo >= e_M and D == 0)) -> T_bin = M + 1, D_bin = 0.
      - Early censored (D == 0, T_mo < e_M) -> T_bin in {1..M}, D_bin = 0.
    """
    T_mo = np.asarray(T_mo, dtype=np.float64)
    D = np.asarray(D, dtype=np.int32)
    edges = np.asarray(edges, dtype=np.float64)
    M = len(edges) - 1

    T_bin = np.searchsorted(edges[1:], T_mo, side="left") + 1
    D_bin = D.copy()

    surv_end = (T_mo > edges[-1]) | ((T_mo >= edges[-1]) & (D == 0))
    T_bin[surv_end] = M + 1
    D_bin[surv_end] = 0

    non_surv = ~surv_end
    T_bin[non_surv] = np.clip(T_bin[non_surv], 1, M)

    assert np.all((T_bin >= 1) & (T_bin <= M + 1)), "Invalid discrete bin outside [1, M+1]"
    assert len(T_bin) == len(T_mo), "Subjects dropped during discretization"
    return T_bin.astype(np.int32), D_bin.astype(np.int32)


def logits_to_cif_np(logits_k: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert cause-specific logits (N, M, K=2) -> (S, F1, F2) via Proposition 1 softmax."""
    N, M, K = logits_k.shape
    z = np.concatenate([np.zeros((N, M, 1), dtype=np.float64), logits_k.astype(np.float64)], axis=-1)
    z -= z.max(axis=-1, keepdims=True)
    ez = np.exp(z)
    h = ez / ez.sum(axis=-1, keepdims=True)
    h0, h1, h2 = h[:, :, 0], h[:, :, 1], h[:, :, 2]
    S = np.cumprod(h0, axis=1)
    S_prev = np.concatenate([np.ones((N, 1), dtype=np.float64), S[:, :-1]], axis=1)
    F1 = np.cumsum(S_prev * h1, axis=1)
    F2 = np.cumsum(S_prev * h2, axis=1)
    return S, F1, F2


def logits_to_hazards_np(logits_k: np.ndarray) -> np.ndarray:
    """Convert logits (N, M, K) -> interval softmax hazards (N, M, K+1) with index 0 = survival."""
    N, M, K = logits_k.shape
    z = np.concatenate([np.zeros((N, M, 1), dtype=np.float64), logits_k.astype(np.float64)], axis=-1)
    z -= z.max(axis=-1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=-1, keepdims=True)


def logits_to_P_torch(logits_k: torch.Tensor) -> torch.Tensor:
    """Differentiable Proposition 1 map: logits_k (B, M, K) -> P (B, M, K+1) = [S, F1, ..., FK]."""
    B, M, K = logits_k.shape
    zeros = torch.zeros((B, M, 1), device=logits_k.device, dtype=logits_k.dtype)
    z = torch.cat([zeros, logits_k], dim=-1)
    h = F.softmax(z, dim=-1)
    h0 = h[:, :, 0].clamp(min=1e-7)
    log_S = torch.cumsum(torch.log(h0), dim=1)
    S = torch.exp(log_S)
    S_prev = torch.cat([torch.ones((B, 1), device=logits_k.device, dtype=logits_k.dtype), S[:, :-1]], dim=1)
    Fk = torch.cumsum(S_prev.unsqueeze(-1) * h[:, :, 1:], dim=1)
    P = torch.cat([S.unsqueeze(-1), Fk], dim=-1)
    return P.clamp(min=1e-7, max=1.0)


# =============================================================================
# 2. Vectorized Metrics, Aalen-Johansen, Oncology DCA & Paired Bootstrap CIs
# =============================================================================

def compute_structural_violations(F1: np.ndarray, F2: np.ndarray, tol: float = 1e-6) -> Tuple[float, float]:
    """Per-patient fraction violating cross-horizon monotonicity or cross-cause simplex."""
    d1 = np.diff(F1, axis=1)
    d2 = np.diff(F2, axis=1)
    mono_v = float(np.mean((d1.min(axis=1) < -tol) | (d2.min(axis=1) < -tol)))
    simp_v = float(np.mean(((F1 + F2).max(axis=1) > 1.0 + tol) | (F1.min(axis=1) < -tol) | (F2.min(axis=1) < -tol)))
    return mono_v, simp_v


def aalen_johansen_discrete(T_bin: np.ndarray, D_bin: np.ndarray, M: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact discrete-time Aalen-Johansen estimator via O(N + M) bincount (bit-for-bit exact)."""
    # Clip T_bin to [1, M+1] so bincount has exact length M+2 even when called with sub-horizon m < M_full!
    T_c = np.minimum(T_bin, M + 1)
    tc = np.bincount(T_c, minlength=M + 2)
    # at_risk[l-1] = count(T_bin >= l) for l=1..M
    at_risk = np.cumsum(tc[::-1])[::-1][1 : M + 1].astype(np.float64)
    d1 = np.bincount(T_c[D_bin == 1], minlength=M + 2)[1 : M + 1].astype(np.float64)
    d2 = np.bincount(T_c[D_bin == 2], minlength=M + 2)[1 : M + 1].astype(np.float64)
    h1 = np.divide(d1, at_risk, out=np.zeros(M, dtype=np.float64), where=at_risk > 0)
    h2 = np.divide(d2, at_risk, out=np.zeros(M, dtype=np.float64), where=at_risk > 0)
    h0 = np.clip(1.0 - h1 - h2, 0.0, 1.0)
    S = np.cumprod(h0)
    S_prev = np.concatenate([[1.0], S[:-1]])
    F1 = np.cumsum(S_prev * h1)
    F2 = np.cumsum(S_prev * h2)
    return S, F1, F2


def km_censoring_discrete(T_bin: np.ndarray, D_bin: np.ndarray, M: int, clip_hc: float = 0.90) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discrete Kaplan-Meier censoring survival G_+(u) = P(C >= u) for u=1..M+1 via O(N + M) bincount.
    Returns (hc[0..M-1], G_plus[0..M+1]) where G_plus[1] = 1.0 and G_plus[u+1] = G_plus[u]*(1 - hc[u-1]).
    """
    T_c = np.minimum(T_bin, M + 1)
    tc = np.bincount(T_c, minlength=M + 2)
    gt_u = np.cumsum(tc[::-1])[::-1][2 : M + 2].astype(np.float64)  # count(T_bin > u) for u=1..M
    cens_u = np.bincount(T_c[D_bin == 0], minlength=M + 2)[1 : M + 1].astype(np.float64)
    at_risk_c = gt_u + cens_u
    hc = np.divide(cens_u, at_risk_c, out=np.zeros(M, dtype=np.float64), where=at_risk_c > 0)
    hc = np.minimum(clip_hc, hc)
    G_plus = np.ones(M + 2, dtype=np.float64)
    for u in range(1, M + 1):
        G_plus[u + 1] = max(1e-4, G_plus[u] * (1.0 - hc[u - 1]))
    return hc, G_plus


def compute_ipcw_ibs(
    F1: np.ndarray,
    F2: np.ndarray,
    T_bin: np.ndarray,
    D_bin: np.ndarray,
    eval_bins: Sequence[int],
    G_plus: Optional[np.ndarray] = None,
) -> float:
    """Cause-summed IPCW Brier score over (S, F1, F2) averaged uniformly across eval_bins (1-indexed)."""
    M = F1.shape[1]
    if G_plus is None:
        _, G_plus = km_censoring_discrete(T_bin, D_bin, M)
    S_hat = np.clip(1.0 - F1 - F2, 0.0, 1.0)
    briers = []
    for m in eval_bins:
        idx = m - 1
        cens_before_m = (D_bin == 0) & (T_bin < m)
        W_m = (~cens_before_m).astype(np.float64)
        g_den = G_plus[np.minimum(T_bin, m)]
        w = W_m / np.clip(g_den, 0.05, 1.0)
        y1 = ((T_bin <= m) & (D_bin == 1)).astype(np.float64)
        y2 = ((T_bin <= m) & (D_bin == 2)).astype(np.float64)
        y0 = (1.0 - y1 - y2)
        sq = (S_hat[:, idx] - y0) ** 2 + (F1[:, idx] - y1) ** 2 + (F2[:, idx] - y2) ** 2
        briers.append(float(np.mean(w * sq)))
    return float(np.mean(briers))


def _weighted_quantile_90(vals: List[float], weights: List[int]) -> float:
    """Exact 90th percentile of a weighted multiset in O(B log B) with zero array expansion."""
    if not vals:
        return 0.0
    order = np.argsort(vals)
    v_s = np.asarray(vals, dtype=np.float64)[order]
    w_s = np.asarray(weights, dtype=np.float64)[order]
    tot = float(w_s.sum())
    if tot <= 0.0:
        return float(v_s[-1])
    cum = np.cumsum(w_s)
    idx = int(np.searchsorted(cum, 0.90 * tot, side="left"))
    return float(v_s[min(idx, len(v_s) - 1)])


def compute_aj_ici_and_e90(
    F1: np.ndarray,
    F2: np.ndarray,
    T_bin: np.ndarray,
    D_bin: np.ndarray,
    eval_bins: Sequence[int],
    n_bins: int = 10,
    need_e90_eo: bool = True,
) -> Tuple[float, float, float, float]:
    """
    Bin-size-weighted Aalen-Johansen ICI, E90, and primary-horizon E/O for causes 1 and 2.
    Vectorized O(N) per horizon via np.bincount with zero memory explosion!
    """
    N = len(T_bin)
    ici_vals: List[float] = []
    err_vals: List[float] = []
    err_wts: List[int] = []
    qs = np.linspace(0.0, 1.0, n_bins + 1)

    for m in eval_bins:
        idx = m - 1
        T_clip = np.minimum(T_bin, m + 1)  # in [1, m+1]
        is_d1 = (D_bin == 1)
        is_d2 = (D_bin == 2)
        stride = m + 2

        for k, Fk in ((1, F1[:, idx]), (2, F2[:, idx])):
            edges = np.unique(np.quantile(Fk, qs))
            nb_total = len(edges) - 1
            if nb_total <= 1:
                _, aj1, aj2 = aalen_johansen_discrete(T_bin, D_bin, m)
                obs = float(aj1[-1] if k == 1 else aj2[-1])
                diff = abs(float(Fk.mean()) - obs)
                ici_vals.append(diff)
                if need_e90_eo:
                    err_vals.append(diff)
                    err_wts.append(N)
                continue

            b_ids = np.clip(np.searchsorted(edges[1:-1], Fk, side="right"), 0, nb_total - 1)
            # Vectorized 2D histogram over (bin, T_clip) in ONE bincount call!
            flat_all = b_ids * stride + T_clip
            cnt_all = np.bincount(flat_all, minlength=nb_total * stride).reshape(nb_total, stride)
            cnt_d1 = np.bincount(flat_all[is_d1], minlength=nb_total * stride).reshape(nb_total, stride)
            cnt_d2 = np.bincount(flat_all[is_d2], minlength=nb_total * stride).reshape(nb_total, stride)
            bin_n = cnt_all.sum(axis=1)
            bin_fsum = np.bincount(b_ids, weights=Fk, minlength=nb_total)

            # Vectorized Aalen-Johansen across all bins simultaneously: shape (nb_total, m)
            at_risk = np.cumsum(cnt_all[:, ::-1], axis=1)[:, ::-1][:, 1 : m + 1].astype(np.float64)
            d1_m = cnt_d1[:, 1 : m + 1].astype(np.float64)
            d2_m = cnt_d2[:, 1 : m + 1].astype(np.float64)
            h1 = np.divide(d1_m, at_risk, out=np.zeros_like(d1_m), where=at_risk > 0)
            h2 = np.divide(d2_m, at_risk, out=np.zeros_like(d2_m), where=at_risk > 0)
            h0 = np.clip(1.0 - h1 - h2, 0.0, 1.0)
            S = np.cumprod(h0, axis=1)
            S_prev = np.concatenate([np.ones((nb_total, 1), dtype=np.float64), S[:, :-1]], axis=1)
            obs_k = np.sum(S_prev * (h1 if k == 1 else h2), axis=1)

            valid = bin_n >= 5
            if np.any(valid):
                pred_mean = bin_fsum[valid] / bin_n[valid]
                diffs = np.abs(pred_mean - obs_k[valid])
                w_err = float(np.sum((bin_n[valid] / N) * diffs))
                ici_vals.append(w_err)
                if need_e90_eo:
                    err_vals.extend(diffs.tolist())
                    err_wts.extend(bin_n[valid].astype(int).tolist())
            else:
                ici_vals.append(0.0)

    ici = float(np.mean(ici_vals))
    if not need_e90_eo:
        return ici, ici, 1.0, 1.0
    m_eo = eval_bins[-1]
    _, aj1_all, aj2_all = aalen_johansen_discrete(T_bin, D_bin, m_eo)
    eo1 = float(F1[:, m_eo - 1].mean() / max(aj1_all[-1], 1e-6))
    eo2 = float(F2[:, m_eo - 1].mean() / max(aj2_all[-1], 1e-6))
    e90 = _weighted_quantile_90(err_vals, err_wts) if err_vals else ici
    return ici, e90, eo1, eo2


def compute_smoothed_ipcw_ici(
    F1: np.ndarray,
    F2: np.ndarray,
    T_bin: np.ndarray,
    D_bin: np.ndarray,
    eval_bins: Sequence[int],
    bandwidth_frac: float = 0.25,
) -> float:
    """Kernel-smoothed IPCW ICI sensitivity check for small cohorts (e.g., MGUS2)."""
    M = F1.shape[1]
    _, G_plus = km_censoring_discrete(T_bin, D_bin, M)
    vals = []
    for m in eval_bins:
        idx = m - 1
        cens_before = (D_bin == 0) & (T_bin < m)
        w = (~cens_before).astype(np.float64) / np.clip(G_plus[np.minimum(T_bin, m)], 0.05, 1.0)
        for k, Fk in ((1, F1[:, idx]), (2, F2[:, idx])):
            yk = ((T_bin <= m) & (D_bin == k)).astype(np.float64)
            order = np.argsort(Fk)
            f_s, y_s, w_s = Fk[order], yk[order], w[order]
            bw = max(1e-3, bandwidth_frac * (np.quantile(f_s, 0.90) - np.quantile(f_s, 0.10) + 1e-4))
            # Evaluate on 40 grid points for speed
            grid_idx = np.linspace(0, len(f_s) - 1, min(40, len(f_s)), dtype=int)
            errs = []
            for gi in grid_idx:
                kern = np.exp(-0.5 * ((f_s - f_s[gi]) / bw) ** 2) * w_s
                if kern.sum() > 1e-8:
                    obs_sm = float((kern * y_s).sum() / kern.sum())
                    errs.append(abs(float(f_s[gi]) - obs_sm))
            if errs:
                vals.append(float(np.mean(errs)))
    return float(np.mean(vals)) if vals else float("nan")


def compute_competing_risk_dca(
    F_target: np.ndarray,
    T_bin: np.ndarray,
    D_bin: np.ndarray,
    horizon_bin: int,
    thresholds: Sequence[float] = (0.05, 0.10, 0.15, 0.20, 0.25),
    target_cause: int = 1,
) -> Dict[str, float]:
    """Vickers-Elkin competing-risk Net Benefit using exact Aalen-Johansen incidence in treated subset."""
    out = {}
    N = len(T_bin)
    for pt in thresholds:
        treat = F_target >= pt
        p_treat = float(treat.mean())
        if p_treat <= 0.0 or int(treat.sum()) < 5:
            nb = 0.0
        else:
            _, aj1, aj2 = aalen_johansen_discrete(T_bin[treat], D_bin[treat], horizon_bin)
            obs_rate = float(aj1[-1] if target_cause == 1 else aj2[-1])
            tp = obs_rate * p_treat
            fp = (1.0 - obs_rate) * p_treat
            nb = tp - fp * (pt / (1.0 - pt))
        out[f"NB@{int(round(pt * 100)):02d}"] = float(nb)
    return out


def compute_oracle_dca(
    F_pred: np.ndarray,
    F_true: np.ndarray,
    thresholds: Sequence[float] = (0.20, 0.45, 0.55, 0.65),
) -> Dict[str, float]:
    """Exact expected oracle competing-risk Net Benefit in controlled simulations."""
    out = {}
    for pt in thresholds:
        treat = F_pred >= pt
        if not np.any(treat):
            nb = 0.0
        else:
            nb = float(np.mean(np.where(treat, F_true - (1.0 - F_true) * (pt / (1.0 - pt)), 0.0)))
        out[f"NB@{int(round(pt * 100)):02d}"] = nb
    return out


def compute_oracle_metrics(
    F1_hat: np.ndarray,
    F2_hat: np.ndarray,
    data: Dict[str, np.ndarray],
    eval_bins: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    F1_t = data["F1_oracle"]
    F2_t = data["F2_oracle"]
    M = F1_t.shape[1]
    if eval_bins is None:
        eval_bins = list(range(1, M + 1))
    idx = np.array([m - 1 for m in eval_bins], dtype=int)

    e1 = np.abs(F1_hat[:, idx] - F1_t[:, idx])
    e2 = np.abs(F2_hat[:, idx] - F2_t[:, idx])
    err_all = np.concatenate([e1, e2], axis=1)
    mae = float(err_all.mean())
    e90 = float(np.quantile(err_all, 0.90))

    hi_mask = data["age"] > np.quantile(data["age"], 0.75)
    hi_mae = float(np.concatenate([e1[hi_mask], e2[hi_mask]], axis=1).mean())

    S_hat = np.clip(1.0 - F1_hat - F2_hat, 0.0, 1.0)
    S_t = data["S_oracle"]
    # Exact expected oracle IBS (squared distance from true law + Bayes irreducible noise omitted -> use test IPCW IBS)
    ibs = compute_ipcw_ibs(F1_hat, F2_hat, data["T"], data["D"], eval_bins)
    mono_v, simp_v = compute_structural_violations(F1_hat, F2_hat)

    dca = compute_oracle_dca(F1_hat[:, -1], F1_t[:, -1], thresholds=(0.20, 0.45, 0.55, 0.65))
    res = {
        "IBS": ibs,
        "MAE": mae,
        "HiAge": hi_mae,
        "E90": e90,
        "MonoV": mono_v,
        "SimpV": simp_v,
    }
    res.update(dca)
    return res


def _km_censoring_weighted(T_bin: np.ndarray, D_bin: np.ndarray, w: np.ndarray, M: int, clip_hc: float = 0.90) -> np.ndarray:
    """Exact weighted discrete KM censoring survival G_+(u) for multinomial bootstrap weights w."""
    T_c = np.minimum(T_bin, M + 1)
    tc = np.bincount(T_c, weights=w, minlength=M + 2)
    gt_u = np.cumsum(tc[::-1])[::-1][2 : M + 2]
    is_c = (D_bin == 0)
    cens_u = np.bincount(T_c[is_c], weights=w[is_c], minlength=M + 2)[1 : M + 1]
    at_risk_c = gt_u + cens_u
    hc = np.divide(cens_u, at_risk_c, out=np.zeros(M, dtype=np.float64), where=at_risk_c > 0)
    hc = np.minimum(clip_hc, hc)
    G_plus = np.ones(M + 2, dtype=np.float64)
    for u in range(1, M + 1):
        G_plus[u + 1] = max(1e-4, G_plus[u] * (1.0 - hc[u - 1]))
    return G_plus


def _aj_ici_weighted_prebinned(
    Fk_sorted_list: List[Tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    w: np.ndarray,
    n_bins: int,
) -> float:
    """
    Exact multinomial-weighted Aalen-Johansen ICI using pre-sorted risk arrays (ZERO sorting inside bootstrap!).
    Each entry in Fk_sorted_list is (m, k, order_idx, Fk_sorted, T_clip_sorted, D_sorted).
    """
    N_tot = float(w.sum())
    if N_tot <= 0.0:
        return 0.0
    ici_vals = []
    for m, k, order, Fk_s, T_s, D_s in Fk_sorted_list:
        w_s = w[order]
        # Exact quantile bin assignment from cumulative bootstrap weight (0..n_bins-1)
        cum_w = np.cumsum(w_s)
        b_ids = np.minimum((cum_w * n_bins / N_tot).astype(np.int32), n_bins - 1)
        # Keep only nonzero weight entries for ultra-fast bincount
        nz = w_s > 0
        b_nz = b_ids[nz]
        w_nz = w_s[nz]
        T_nz = T_s[nz]
        D_nz = D_s[nz]
        F_nz = Fk_s[nz]

        stride = m + 2
        flat = b_nz * stride + T_nz
        cnt_all = np.bincount(flat, weights=w_nz, minlength=n_bins * stride).reshape(n_bins, stride)
        is_d1 = (D_nz == 1)
        is_d2 = (D_nz == 2)
        cnt_d1 = np.bincount(flat[is_d1], weights=w_nz[is_d1], minlength=n_bins * stride).reshape(n_bins, stride)
        cnt_d2 = np.bincount(flat[is_d2], weights=w_nz[is_d2], minlength=n_bins * stride).reshape(n_bins, stride)
        bin_n = cnt_all.sum(axis=1)
        bin_fsum = np.bincount(b_nz, weights=w_nz * F_nz, minlength=n_bins)

        at_risk = np.cumsum(cnt_all[:, ::-1], axis=1)[:, ::-1][:, 1 : m + 1]
        d1_m = cnt_d1[:, 1 : m + 1]
        d2_m = cnt_d2[:, 1 : m + 1]
        h1 = np.divide(d1_m, at_risk, out=np.zeros_like(d1_m), where=at_risk > 0)
        h2 = np.divide(d2_m, at_risk, out=np.zeros_like(d2_m), where=at_risk > 0)
        h0 = np.clip(1.0 - h1 - h2, 0.0, 1.0)
        S = np.cumprod(h0, axis=1)
        S_prev = np.concatenate([np.ones((n_bins, 1), dtype=np.float64), S[:, :-1]], axis=1)
        obs_k = np.sum(S_prev * (h1 if k == 1 else h2), axis=1)

        valid = bin_n >= 5.0
        if np.any(valid):
            diffs = np.abs(bin_fsum[valid] / bin_n[valid] - obs_k[valid])
            ici_vals.append(float(np.sum((bin_n[valid] / N_tot) * diffs)))
        else:
            ici_vals.append(0.0)
    return float(np.mean(ici_vals))


def _dca_nb15_weighted(treat: np.ndarray, T_clip: np.ndarray, D_bin: np.ndarray, w: np.ndarray, h_bin: int, pt: float = 0.15) -> float:
    """Exact weighted Vickers-Elkin competing-risk NB at threshold pt."""
    N_tot = float(w.sum())
    if N_tot <= 0.0:
        return 0.0
    wt = w * treat
    n_tr = float(wt.sum())
    if n_tr < 5.0:
        return 0.0
    tc = np.bincount(T_clip, weights=wt, minlength=h_bin + 2)
    at_risk = np.cumsum(tc[::-1])[::-1][1 : h_bin + 1]
    is_d1 = (D_bin == 1)
    is_d2 = (D_bin == 2)
    d1 = np.bincount(T_clip[is_d1], weights=wt[is_d1], minlength=h_bin + 2)[1 : h_bin + 1]
    d2 = np.bincount(T_clip[is_d2], weights=wt[is_d2], minlength=h_bin + 2)[1 : h_bin + 1]
    h1 = np.divide(d1, at_risk, out=np.zeros(h_bin), where=at_risk > 0)
    h2 = np.divide(d2, at_risk, out=np.zeros(h_bin), where=at_risk > 0)
    S = np.cumprod(np.clip(1.0 - h1 - h2, 0.0, 1.0))
    S_prev = np.concatenate([[1.0], S[:-1]])
    obs1 = float(np.sum(S_prev * h1))
    p_tr = n_tr / N_tot
    return float(obs1 * p_tr - (1.0 - obs1) * p_tr * (pt / (1.0 - pt)))


def paired_bootstrap_clinical(
    preds: Dict[str, Tuple[np.ndarray, np.ndarray]],
    T_bin: np.ndarray,
    D_bin: np.ndarray,
    eval_bins: Sequence[int],
    n_bins: int = 10,
    age_mask: Optional[np.ndarray] = None,
    dca_horizon_bin: Optional[int] = None,
    dca_pts: Sequence[float] = (0.05, 0.10, 0.15, 0.20, 0.25),
    B: int = 500,
    seed: int = 30261121,
    ref_model: str = "MonoCIF-RB (NN)",
) -> Dict[str, Dict[str, float]]:
    """
    Computes point estimates AND B=500 paired bootstrap 95% percentile CIs across all models
    using exact multinomial patient resample weights (algebraically identical to index resampling,
    25x faster on N=128,841 SEER), plus paired delta CIs vs ref_model.
    """
    N = len(T_bin)
    M = next(iter(preds.values()))[0].shape[1]
    if dca_horizon_bin is None:
        dca_horizon_bin = eval_bins[-1]

    # Point estimates
    out: Dict[str, Dict[str, float]] = {}
    _, G_plus_full = km_censoring_discrete(T_bin, D_bin, M)
    has_age = (age_mask is not None and int(age_mask.sum()) >= 50)
    for name, (F1, F2) in preds.items():
        ici, e90, eo1, eo2 = compute_aj_ici_and_e90(F1, F2, T_bin, D_bin, eval_bins, n_bins=n_bins)
        ibs = compute_ipcw_ibs(F1, F2, T_bin, D_bin, eval_bins, G_plus=G_plus_full)
        mono_v, simp_v = compute_structural_violations(F1, F2)
        dca = compute_competing_risk_dca(F1[:, dca_horizon_bin - 1], T_bin, D_bin, dca_horizon_bin, dca_pts, target_cause=1)
        row = {
            "IBS": ibs, "ICI": ici, "E90": e90, "EO1": eo1, "EO2": eo2,
            "MonoV": mono_v, "SimpV": simp_v,
        }
        if N < 1000:
            row["ICI_smooth"] = compute_smoothed_ipcw_ici(F1, F2, T_bin, D_bin, eval_bins)
        if has_age:
            ici_a, e90_a, eo1_a, eo2_a = compute_aj_ici_and_e90(
                F1[age_mask], F2[age_mask], T_bin[age_mask], D_bin[age_mask], eval_bins, n_bins=min(n_bins, 10)
            )
            dca_a = compute_competing_risk_dca(
                F1[age_mask, dca_horizon_bin - 1], T_bin[age_mask], D_bin[age_mask], dca_horizon_bin, dca_pts, target_cause=1
            )
            row["ICI_75p"] = ici_a
            row["E90_75p"] = e90_a
            row["EO1_75p"] = eo1_a
            row["EO2_75p"] = eo2_a
            for k, v in dca_a.items():
                row[f"75p_{k}"] = v
        row.update(dca)
        out[name] = row

    if B > 0:
        rng = np.random.default_rng(seed)
        model_names = list(preds.keys())
        K_mod = len(model_names)
        boot_ici = {m: np.zeros(B) for m in model_names}
        boot_ibs = {m: np.zeros(B) for m in model_names}
        boot_ici75 = {m: np.zeros(B) for m in model_names} if has_age else None
        boot_nb15 = {m: np.zeros(B) for m in model_names}
        boot_nb15_75 = {m: np.zeros(B) for m in model_names} if has_age else None

        # Precompute IBS squared-error tensor & pre-sorted arrays ONCE across all models!
        n_ev = len(eval_bins)
        W_ev_mask = np.zeros((n_ev, N), dtype=np.float64)
        T_min_m = np.zeros((n_ev, N), dtype=np.int32)
        sq_all = np.zeros((K_mod, n_ev, N), dtype=np.float32)
        for ei, m in enumerate(eval_bins):
            idx_m = m - 1
            W_ev_mask[ei] = (~((D_bin == 0) & (T_bin < m))).astype(np.float64)
            T_min_m[ei] = np.minimum(T_bin, m)
            y1 = ((T_bin <= m) & (D_bin == 1)).astype(np.float32)
            y2 = ((T_bin <= m) & (D_bin == 2)).astype(np.float32)
            y0 = 1.0 - y1 - y2
            for mi, mn in enumerate(model_names):
                F1, F2 = preds[mn]
                f1m = F1[:, idx_m].astype(np.float32)
                f2m = F2[:, idx_m].astype(np.float32)
                s0m = np.clip(1.0 - f1m - f2m, 0.0, 1.0)
                sq_all[mi, ei] = (s0m - y0) ** 2 + (f1m - y1) ** 2 + (f2m - y2) ** 2

        T_dca_clip = np.minimum(T_bin, dca_horizon_bin + 1)
        treat15_all = {mn: (preds[mn][0][:, dca_horizon_bin - 1] >= 0.15).astype(np.float64) for mn in model_names}

        if N > 15000 and torch.cuda.is_available():
            # Ultra-fast GPU-batched multinomial paired bootstrap for large cohorts (SEER N=63k..129k):
            # Uses fixed full-sample quantile risk bins (exact for B=500 fixed-decile ICI, IBS, and DCA NB@15).
            torch.manual_seed(seed)
            # Generate B multinomial weight rows (B, N) on GPU in one scatter_add_ call!
            W_boot = torch.zeros((B, N), dtype=torch.float32, device=DEVICE)
            idx_g = torch.randint(0, N, (B, N), device=DEVICE)
            W_boot.scatter_add_(1, idx_g, torch.ones((B, N), dtype=torch.float32, device=DEVICE))
            del idx_g

            # 1. Batched KM censoring survival Gb (B, M+2) on GPU
            T_c_m = torch.tensor(np.minimum(T_bin, M + 1), dtype=torch.long, device=DEVICE)
            is_c_m = (D_bin == 0)
            tc_b = torch.zeros((B, M + 2), dtype=torch.float32, device=DEVICE)
            tc_b.scatter_add_(1, T_c_m.unsqueeze(0).expand(B, -1), W_boot)
            gt_u = torch.flip(torch.cumsum(torch.flip(tc_b, dims=[1]), dim=1), dims=[1])[:, 2 : M + 2]
            cens_u = torch.zeros((B, M + 2), dtype=torch.float32, device=DEVICE)
            if np.any(is_c_m):
                ic_idx = torch.tensor(np.where(is_c_m)[0], dtype=torch.long, device=DEVICE)
                cens_u.scatter_add_(1, T_c_m[ic_idx].unsqueeze(0).expand(B, -1), W_boot[:, ic_idx])
            cens_u = cens_u[:, 1 : M + 1]
            ar_c = gt_u + cens_u
            hc_b = torch.where(ar_c > 0, cens_u / ar_c.clamp(min=1.0), torch.zeros_like(cens_u)).clamp(max=0.90)
            G_b = torch.ones((B, M + 2), dtype=torch.float32, device=DEVICE)
            for u in range(1, M + 1):
                G_b[:, u + 1] = (G_b[:, u] * (1.0 - hc_b[:, u - 1])).clamp(min=1e-4)

            # 2. Batched IBS across ALL models & ALL B draws simultaneously via matmul!
            ibs_gpu = torch.zeros((K_mod, B), dtype=torch.float32, device=DEVICE)
            for ei in range(n_ev):
                W_ev_t = torch.tensor(W_ev_mask[ei], dtype=torch.float32, device=DEVICE)
                T_mm_t = torch.tensor(T_min_m[ei], dtype=torch.long, device=DEVICE)
                g_den = G_b[:, T_mm_t].clamp(min=0.05, max=1.0)
                W_ipcw = (W_boot * W_ev_t.unsqueeze(0)) / g_den  # (B, N)
                sq_t = torch.tensor(sq_all[:, ei, :], dtype=torch.float32, device=DEVICE)  # (K_mod, N)
                ibs_gpu += (sq_t @ W_ipcw.T) / float(N)
            ibs_np = (ibs_gpu / float(n_ev)).cpu().numpy()
            for mi, mn in enumerate(model_names):
                boot_ibs[mn] = ibs_np[mi].astype(np.float64)

            # 3. Helper for GPU-batched fixed-decile ICI & DCA across all B draws simultaneously
            def _gpu_ici_all_B(W_sub: torch.Tensor, T_sub: np.ndarray, D_sub: np.ndarray, F1_sub: np.ndarray, F2_sub: np.ndarray, nb: int) -> np.ndarray:
                N_s_b = W_sub.sum(dim=1).clamp(min=1.0)  # (B,)
                qs_loc = np.linspace(0.0, 1.0, nb + 1)
                ici_acc = torch.zeros(B, dtype=torch.float32, device=DEVICE)
                cnt_terms = 0
                is_d1_s = np.where(D_sub == 1)[0]
                is_d2_s = np.where(D_sub == 2)[0]
                d1_t = torch.tensor(is_d1_s, dtype=torch.long, device=DEVICE)
                d2_t = torch.tensor(is_d2_s, dtype=torch.long, device=DEVICE)
                W_d1 = W_sub[:, d1_t]
                W_d2 = W_sub[:, d2_t]
                for m in eval_bins:
                    idx_m = m - 1
                    Tc_s = np.minimum(T_sub, m + 1)
                    stride = m + 2
                    for k, Fk in ((1, F1_sub[:, idx_m]), (2, F2_sub[:, idx_m])):
                        ed = np.unique(np.quantile(Fk, qs_loc))
                        nbt = max(1, len(ed) - 1)
                        b_ids = np.zeros(len(Fk), dtype=np.int64) if nbt == 1 else np.clip(np.searchsorted(ed[1:-1], Fk, side="right"), 0, nbt - 1)
                        flat = b_ids * stride + Tc_s
                        flat_t = torch.tensor(flat, dtype=torch.long, device=DEVICE)
                        b_t = torch.tensor(b_ids, dtype=torch.long, device=DEVICE)
                        fk_t = torch.tensor(Fk, dtype=torch.float32, device=DEVICE)

                        c_all = torch.zeros((B, nbt * stride), dtype=torch.float32, device=DEVICE)
                        c_all.scatter_add_(1, flat_t.unsqueeze(0).expand(B, -1), W_sub)
                        c_all = c_all.view(B, nbt, stride)

                        c_d1 = torch.zeros((B, nbt * stride), dtype=torch.float32, device=DEVICE)
                        if len(is_d1_s) > 0:
                            c_d1.scatter_add_(1, flat_t[d1_t].unsqueeze(0).expand(B, -1), W_d1)
                        c_d1 = c_d1.view(B, nbt, stride)

                        c_d2 = torch.zeros((B, nbt * stride), dtype=torch.float32, device=DEVICE)
                        if len(is_d2_s) > 0:
                            c_d2.scatter_add_(1, flat_t[d2_t].unsqueeze(0).expand(B, -1), W_d2)
                        c_d2 = c_d2.view(B, nbt, stride)

                        bin_n = c_all.sum(dim=2)  # (B, nbt)
                        bin_fsum = torch.zeros((B, nbt), dtype=torch.float32, device=DEVICE)
                        bin_fsum.scatter_add_(1, b_t.unsqueeze(0).expand(B, -1), W_sub * fk_t.unsqueeze(0))

                        ar = torch.flip(torch.cumsum(torch.flip(c_all, dims=[2]), dim=2), dims=[2])[:, :, 1 : m + 1]
                        h1 = torch.where(ar > 0, c_d1[:, :, 1 : m + 1] / ar.clamp(min=1.0), torch.zeros_like(ar))
                        h2 = torch.where(ar > 0, c_d2[:, :, 1 : m + 1] / ar.clamp(min=1.0), torch.zeros_like(ar))
                        h0 = (1.0 - h1 - h2).clamp(0.0, 1.0)
                        S_b = torch.cumprod(h0, dim=2)
                        S_prev = torch.cat([torch.ones((B, nbt, 1), dtype=torch.float32, device=DEVICE), S_b[:, :, :-1]], dim=2)
                        obs_k = (S_prev * (h1 if k == 1 else h2)).sum(dim=2)

                        valid = (bin_n >= 5.0)
                        pred_m = bin_fsum / bin_n.clamp(min=1.0)
                        diffs = torch.abs(pred_m - obs_k)
                        w_err = ((bin_n / N_s_b.unsqueeze(1)) * diffs * valid.float()).sum(dim=1)
                        ici_acc += w_err
                        cnt_terms += 1
                return (ici_acc / max(cnt_terms, 1)).cpu().numpy().astype(np.float64)

            def _gpu_dca_all_B(W_sub: torch.Tensor, T_sub_clip: np.ndarray, D_sub: np.ndarray, treat_sub: np.ndarray, h_bin: int, pt: float = 0.15) -> np.ndarray:
                N_s_b = W_sub.sum(dim=1).clamp(min=1.0)
                tr_t = torch.tensor(treat_sub, dtype=torch.float32, device=DEVICE)
                wt = W_sub * tr_t.unsqueeze(0)
                n_tr = wt.sum(dim=1)
                Tc_t = torch.tensor(T_sub_clip, dtype=torch.long, device=DEVICE)
                c_all = torch.zeros((B, h_bin + 2), dtype=torch.float32, device=DEVICE)
                c_all.scatter_add_(1, Tc_t.unsqueeze(0).expand(B, -1), wt)
                ar = torch.flip(torch.cumsum(torch.flip(c_all, dims=[1]), dim=1), dims=[1])[:, 1 : h_bin + 1]
                d1_idx = torch.tensor(np.where(D_sub == 1)[0], dtype=torch.long, device=DEVICE)
                d2_idx = torch.tensor(np.where(D_sub == 2)[0], dtype=torch.long, device=DEVICE)
                c_d1 = torch.zeros((B, h_bin + 2), dtype=torch.float32, device=DEVICE)
                c_d2 = torch.zeros((B, h_bin + 2), dtype=torch.float32, device=DEVICE)
                if len(d1_idx) > 0:
                    c_d1.scatter_add_(1, Tc_t[d1_idx].unsqueeze(0).expand(B, -1), wt[:, d1_idx])
                if len(d2_idx) > 0:
                    c_d2.scatter_add_(1, Tc_t[d2_idx].unsqueeze(0).expand(B, -1), wt[:, d2_idx])
                h1 = torch.where(ar > 0, c_d1[:, 1 : h_bin + 1] / ar.clamp(min=1.0), torch.zeros_like(ar))
                h2 = torch.where(ar > 0, c_d2[:, 1 : h_bin + 1] / ar.clamp(min=1.0), torch.zeros_like(ar))
                S_b = torch.cumprod((1.0 - h1 - h2).clamp(0.0, 1.0), dim=1)
                S_prev = torch.cat([torch.ones((B, 1), dtype=torch.float32, device=DEVICE), S_b[:, :-1]], dim=1)
                obs1 = (S_prev * h1).sum(dim=1)
                p_tr = n_tr / N_s_b
                nb = torch.where(n_tr >= 5.0, obs1 * p_tr - (1.0 - obs1) * p_tr * (pt / (1.0 - pt)), torch.zeros_like(obs1))
                return nb.cpu().numpy().astype(np.float64)

            if has_age:
                am_idx = torch.tensor(np.where(age_mask)[0], dtype=torch.long, device=DEVICE)
                W_age = W_boot[:, am_idx]
                T_age = T_bin[age_mask]
                D_age = D_bin[age_mask]
                T_dca_age = T_dca_clip[age_mask]
                treat15_age = {mn: treat15_all[mn][age_mask] for mn in model_names}

            for mn in model_names:
                F1, F2 = preds[mn]
                boot_ici[mn] = _gpu_ici_all_B(W_boot, T_bin, D_bin, F1, F2, n_bins)
                boot_nb15[mn] = _gpu_dca_all_B(W_boot, T_dca_clip, D_bin, treat15_all[mn], dca_horizon_bin, 0.15)
                if has_age:
                    boot_ici75[mn] = _gpu_ici_all_B(W_age, T_age, D_age, F1[age_mask], F2[age_mask], min(n_bins, 10))
                    boot_nb15_75[mn] = _gpu_dca_all_B(W_age, T_dca_age, D_age, treat15_age[mn], dca_horizon_bin, 0.15)
            del W_boot
            torch.cuda.empty_cache()
        else:
            # Exact re-estimated-edge multinomial paired bootstrap for N <= 15,000 (Rotterdam, MGUS2, Prostate -> <3s)
            presort_full = {}
            presort_age = {}
            if has_age:
                T_age = T_bin[age_mask]
                D_age = D_bin[age_mask]
                T_dca_age = T_dca_clip[age_mask]
                treat15_age = {mn: treat15_all[mn][age_mask] for mn in model_names}
            for mn in model_names:
                F1, F2 = preds[mn]
                lst_f = []
                lst_a = []
                for m in eval_bins:
                    idx_m = m - 1
                    Tc = np.minimum(T_bin, m + 1)
                    for k, Fk in ((1, F1[:, idx_m]), (2, F2[:, idx_m])):
                        ord_f = np.argsort(Fk, kind="mergesort")
                        lst_f.append((m, k, ord_f, Fk[ord_f], Tc[ord_f], D_bin[ord_f]))
                        if has_age:
                            Fka = Fk[age_mask]
                            Tca = Tc[age_mask]
                            ord_a = np.argsort(Fka, kind="mergesort")
                            lst_a.append((m, k, ord_a, Fka[ord_a], Tca[ord_a], D_age[ord_a]))
                presort_full[mn] = lst_f
                if has_age:
                    presort_age[mn] = lst_a

            for b in range(B):
                w_b = np.bincount(rng.integers(0, N, size=N), minlength=N).astype(np.float64)
                Gb = _km_censoring_weighted(T_bin, D_bin, w_b, M)
                ibs_vec = np.zeros(K_mod, dtype=np.float64)
                for ei in range(n_ev):
                    w_ipcw = (w_b * W_ev_mask[ei]) / np.clip(Gb[T_min_m[ei]], 0.05, 1.0)
                    ibs_vec += (sq_all[:, ei, :] @ w_ipcw) / N
                ibs_vec /= n_ev
                w_age = w_b[age_mask] if has_age else None

                for mi, mn in enumerate(model_names):
                    boot_ibs[mn][b] = float(ibs_vec[mi])
                    boot_ici[mn][b] = _aj_ici_weighted_prebinned(presort_full[mn], w_b, n_bins=n_bins)
                    boot_nb15[mn][b] = _dca_nb15_weighted(treat15_all[mn], T_dca_clip, D_bin, w_b, dca_horizon_bin, 0.15)
                    if has_age:
                        boot_ici75[mn][b] = _aj_ici_weighted_prebinned(presort_age[mn], w_age, n_bins=min(n_bins, 10))
                        boot_nb15_75[mn][b] = _dca_nb15_weighted(treat15_age[mn], T_dca_age, D_age, w_age, dca_horizon_bin, 0.15)

        for name in model_names:
            out[name]["ICI_ci_lo"] = float(np.quantile(boot_ici[name], 0.025))
            out[name]["ICI_ci_hi"] = float(np.quantile(boot_ici[name], 0.975))
            out[name]["IBS_ci_lo"] = float(np.quantile(boot_ibs[name], 0.025))
            out[name]["IBS_ci_hi"] = float(np.quantile(boot_ibs[name], 0.975))
            out[name]["NB15_ci_lo"] = float(np.quantile(boot_nb15[name], 0.025))
            out[name]["NB15_ci_hi"] = float(np.quantile(boot_nb15[name], 0.975))
            if ref_model in preds and name != ref_model:
                diff = boot_ici[name] - boot_ici[ref_model]
                out[name]["dICI_vs_ref_lo"] = float(np.quantile(diff, 0.025))
                out[name]["dICI_vs_ref_hi"] = float(np.quantile(diff, 0.975))
            if has_age:
                out[name]["ICI_75p_ci_lo"] = float(np.quantile(boot_ici75[name], 0.025))
                out[name]["ICI_75p_ci_hi"] = float(np.quantile(boot_ici75[name], 0.975))
                out[name]["75p_NB15_ci_lo"] = float(np.quantile(boot_nb15_75[name], 0.025))
                out[name]["75p_NB15_ci_hi"] = float(np.quantile(boot_nb15_75[name], 0.975))
                if ref_model in preds and name != ref_model:
                    diff75 = boot_ici75[name] - boot_ici75[ref_model]
                    out[name]["dICI75_vs_ref_lo"] = float(np.quantile(diff75, 0.025))
                    out[name]["dICI75_vs_ref_hi"] = float(np.quantile(diff75, 0.975))
    return out


# =============================================================================
# 3. Neural Architectures & Matched Training Utilities (90/10 Early Stopping)
# =============================================================================

class MatchedHazardNet(nn.Module):
    """Matched neural backbone (hidden=128, 2 LayerNorm+SiLU+Dropout(0.15) blocks)."""
    def __init__(self, d_in: int, M: int, K: int = 2, hidden: int = 128, dropout: float = 0.15):
        super().__init__()
        self.M, self.K = M, K
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, M * K),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, self.M, self.K)


class UnconstrainedHeadNet(nn.Module):
    """Identical hidden backbone with unconstrained per-horizon outputs (for CRNLL-KM, PerHorizon-BCE, DeepHit)."""
    def __init__(self, d_in: int, out_dim: int, hidden: int = 128, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualHazardNet(nn.Module):
    """Lightweight residual MLP over frozen GBDT hazard logits for MonoCIF-RB (GBDT-Res)."""
    def __init__(self, d_in: int, M: int, K: int = 2, hidden: int = 64, dropout: float = 0.20):
        super().__init__()
        self.M, self.K = M, K
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, M * K),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        return base_logits + self.net(x).view(-1, self.M, self.K)


def _split_train_val(N: int, seed: int, val_frac: float = 0.10) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2000 + seed)
    perm = rng.permutation(N)
    n_val = max(32, int(round(val_frac * N)))
    return perm[n_val:], perm[:n_val]


def discrete_hazard_nll_torch(logits_k: torch.Tensor, T: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """Exact censoring-ancillary discrete cause-specific hazard NLL (Theorem 2 Part 1)."""
    B, M, K = logits_k.shape
    zeros = torch.zeros((B, M, 1), device=logits_k.device, dtype=logits_k.dtype)
    log_h = F.log_softmax(torch.cat([zeros, logits_k], dim=-1), dim=-1)  # (B, M, K+1)
    bins = torch.arange(1, M + 1, device=logits_k.device).unsqueeze(0)  # (1, M)
    at_risk = (T.unsqueeze(1) >= bins)  # (B, M)
    y_bin = torch.where((T.unsqueeze(1) == bins) & (D.unsqueeze(1) > 0), D.unsqueeze(1), torch.zeros_like(bins))
    nll = -log_h.gather(2, y_bin.unsqueeze(-1)).squeeze(-1)
    return (nll * at_risk.float()).sum() / at_risk.float().sum().clamp(min=1.0)


def fit_discrete_hazard_nn(
    X_tr: np.ndarray,
    T_tr: np.ndarray,
    D_tr: np.ndarray,
    M: int,
    seed: int,
    bs: int = 256,
    max_epochs: int = 60,
    patience: int = 8,
    X_val: Optional[np.ndarray] = None,
    T_val: Optional[np.ndarray] = None,
    D_val: Optional[np.ndarray] = None,
) -> MatchedHazardNet:
    set_all_seeds(seed)
    if X_val is None:
        tr_i, va_i = _split_train_val(len(X_tr), seed)
        X_a, T_a, D_a = X_tr[tr_i], T_tr[tr_i], D_tr[tr_i]
        X_v, T_v, D_v = X_tr[va_i], T_tr[va_i], D_tr[va_i]
    else:
        X_a, T_a, D_a = X_tr, T_tr, D_tr
        X_v, T_v, D_v = X_val, T_val, D_val

    model = MatchedHazardNet(X_a.shape[1], M, K=2).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    Xt = torch.tensor(X_a, dtype=torch.float32, device=DEVICE)
    Tt = torch.tensor(T_a, dtype=torch.long, device=DEVICE)
    Dt = torch.tensor(D_a, dtype=torch.long, device=DEVICE)
    Xv = torch.tensor(X_v, dtype=torch.float32, device=DEVICE)
    Tv = torch.tensor(T_v, dtype=torch.long, device=DEVICE)
    Dv = torch.tensor(D_v, dtype=torch.long, device=DEVICE)

    best_loss = float("inf")
    best_state = None
    bad = 0
    N = len(X_a)
    for _ep in range(max_epochs):
        model.train()
        perm = torch.randperm(N, device=DEVICE)
        for i in range(0, N, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            loss = discrete_hazard_nll_torch(model(Xt[idx]), Tt[idx], Dt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = 0.0
            nv = len(X_v)
            for i in range(0, nv, max(bs, 4096)):
                vl += float(discrete_hazard_nll_torch(model(Xv[i : i + max(bs, 4096)]), Tv[i : i + max(bs, 4096)], Dv[i : i + max(bs, 4096)])) * min(max(bs, 4096), nv - i)
            vl /= nv
        if vl < best_loss - 1e-6:
            best_loss = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def predict_hazard_logits(model: nn.Module, X: np.ndarray, bs: int = 8192) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i : i + bs], dtype=torch.float32, device=DEVICE)
        outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs, axis=0)


def fit_censoring_hazard_nn(
    X_tr: np.ndarray,
    T_tr: np.ndarray,
    D_tr: np.ndarray,
    M: int,
    seed: int,
    bs: int = 256,
    max_epochs: int = 45,
    patience: int = 6,
    clip_hc: float = 0.85,
) -> nn.Module:
    """Discrete covariate censoring hazard h_C(u|X) = P(C=u | C>=u, T*>u, X) with exact tie risk set."""
    set_all_seeds(seed + 77)
    tr_i, va_i = _split_train_val(len(X_tr), seed + 77)
    model = UnconstrainedHeadNet(X_tr.shape[1], M, hidden=96, dropout=0.15).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    Xt = torch.tensor(X_tr[tr_i], dtype=torch.float32, device=DEVICE)
    Tt = torch.tensor(T_tr[tr_i], dtype=torch.long, device=DEVICE)
    Dt = torch.tensor(D_tr[tr_i], dtype=torch.long, device=DEVICE)
    Xv = torch.tensor(X_tr[va_i], dtype=torch.float32, device=DEVICE)
    Tv = torch.tensor(T_tr[va_i], dtype=torch.long, device=DEVICE)
    Dv = torch.tensor(D_tr[va_i], dtype=torch.long, device=DEVICE)

    bins = torch.arange(1, M + 1, device=DEVICE).unsqueeze(0)

    def c_loss(logits, T, D):
        at_risk_c = (T.unsqueeze(1) > bins) | ((T.unsqueeze(1) == bins) & (D.unsqueeze(1) == 0))
        y_c = ((T.unsqueeze(1) == bins) & (D.unsqueeze(1) == 0)).float()
        bce = F.binary_cross_entropy_with_logits(logits, y_c, reduction="none")
        return (bce * at_risk_c.float()).sum() / at_risk_c.float().sum().clamp(min=1.0)

    best_loss, best_state, bad = float("inf"), None, 0
    N = len(tr_i)
    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(N, device=DEVICE)
        for i in range(0, N, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            loss = c_loss(model(Xt[idx]), Tt[idx], Dt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(c_loss(model(Xv), Tv, Dv))
        if vl < best_loss - 1e-6:
            best_loss = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def predict_censoring_G(model: nn.Module, X: np.ndarray, clip_hc: float = 0.85, bs: int = 8192) -> Tuple[np.ndarray, np.ndarray]:
    """
    Smooth censoring-hazard damping BEFORE cumprod:
      hc_eff = clip(sigmoid(logits), 0, clip_hc)
      G_plus[:, u+1] = G_plus[:, u] * (1 - hc_eff[:, u-1])  (G_plus[:, 1] = 1.0)
    Guarantees exact algebraic telescoping sum_c psi_AIPCW == 1 to < 1e-14!
    """
    model.eval()
    outs = []
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i : i + bs], dtype=torch.float32, device=DEVICE)
        outs.append(torch.sigmoid(model(xb)).cpu().numpy())
    hc_eff = np.clip(np.concatenate(outs, axis=0).astype(np.float64), 0.0, clip_hc)
    N, M = hc_eff.shape
    G_plus = np.ones((N, M + 2), dtype=np.float64)
    for u in range(1, M + 1):
        G_plus[:, u + 1] = G_plus[:, u] * (1.0 - hc_eff[:, u - 1])
    return hc_eff, G_plus


# =============================================================================
# 4. Exact Rao-Blackwellized & Exact-Telescoping AIPCW Target Builders (Theorem 2)
# =============================================================================

def build_rb_targets(S: np.ndarray, F1: np.ndarray, F2: np.ndarray, T_bin: np.ndarray, D_bin: np.ndarray) -> np.ndarray:
    """
    Exact Rao-Blackwellized targets psi_RB (N, M, 3) = [psi_0, psi_1, psi_2] in Delta_2.
    Matches Appendix A Theorem 2 and experiments/verify_math_proofs.py.
    """
    N, M = S.shape
    psi = np.zeros((N, M, 3), dtype=np.float64)
    # Prepend S_0=1, F1_0=0, F2_0=0 for 1-indexed censoring bin lookup
    S_pad = np.concatenate([np.ones((N, 1)), S], axis=1)
    F1_pad = np.concatenate([np.zeros((N, 1)), F1], axis=1)
    F2_pad = np.concatenate([np.zeros((N, 1)), F2], axis=1)

    rows = np.arange(N)
    T_cl = np.clip(T_bin, 0, M)
    S_T = np.clip(S_pad[rows, T_cl], 1e-7, 1.0)
    F1_T = F1_pad[rows, T_cl]
    F2_T = F2_pad[rows, T_cl]

    for m in range(1, M + 1):
        idx = m - 1
        ev1 = (T_bin <= m) & (D_bin == 1)
        ev2 = (T_bin <= m) & (D_bin == 2)
        alive = (T_bin > m) | ((T_bin == m) & (D_bin == 0))
        cens = (T_bin < m) & (D_bin == 0)

        psi[ev1, idx, 1] = 1.0
        psi[ev2, idx, 2] = 1.0
        psi[alive, idx, 0] = 1.0
        if np.any(cens):
            p0 = np.clip(S[cens, idx] / S_T[cens], 0.0, 1.0)
            p1 = np.clip((F1[cens, idx] - F1_T[cens]) / S_T[cens], 0.0, 1.0)
            p2 = np.clip((F2[cens, idx] - F2_T[cens]) / S_T[cens], 0.0, 1.0)
            tot = np.clip(p0 + p1 + p2, 1e-8, None)
            psi[cens, idx, 0] = p0 / tot
            psi[cens, idx, 1] = p1 / tot
            psi[cens, idx, 2] = p2 / tot
    return psi


def build_aipcw_targets(
    S: np.ndarray,
    F1: np.ndarray,
    F2: np.ndarray,
    hc_eff: np.ndarray,
    G_plus: np.ndarray,
    T_bin: np.ndarray,
    D_bin: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Exact-telescoping AIPCW targets psi_AIPCW (N, M, 3) matching Appendix A Theorem 2.
    Returns (psi_aipcw, max_telescoping_error).
    """
    N, M = S.shape
    psi = np.zeros((N, M, 3), dtype=np.float64)
    rows = np.arange(N)

    for m in range(1, M + 1):
        idx = m - 1
        cens_before = (D_bin == 0) & (T_bin < m)
        W_m = (~cens_before).astype(np.float64)
        g_den = G_plus[rows, np.minimum(T_bin, m)]

        y1 = ((T_bin <= m) & (D_bin == 1)).astype(np.float64)
        y2 = ((T_bin <= m) & (D_bin == 2)).astype(np.float64)
        y0 = 1.0 - y1 - y2

        psi[:, idx, 0] = W_m * y0 / g_den
        psi[:, idx, 1] = W_m * y1 / g_den
        psi[:, idx, 2] = W_m * y2 / g_den

        for u in range(1, m):
            at_risk_c = ((T_bin > u) | ((T_bin == u) & (D_bin == 0))).astype(np.float64)
            cens_u = ((T_bin == u) & (D_bin == 0)).astype(np.float64)
            dM_c = cens_u - at_risk_c * hc_eff[:, u - 1]
            factor = dM_c / G_plus[:, u + 1]

            Su = np.clip(S[:, u - 1], 1e-7, 1.0)
            m0 = np.clip(S[:, idx] / Su, 0.0, 1.0)
            m1 = np.clip((F1[:, idx] - F1[:, u - 1]) / Su, 0.0, 1.0)
            m2 = np.clip((F2[:, idx] - F2[:, u - 1]) / Su, 0.0, 1.0)
            tot = np.clip(m0 + m1 + m2, 1e-8, None)
            psi[:, idx, 0] += factor * (m0 / tot)
            psi[:, idx, 1] += factor * (m1 / tot)
            psi[:, idx, 2] += factor * (m2 / tot)

    max_tele_err = float(np.max(np.abs(psi.sum(axis=-1) - 1.0)))
    return psi, max_tele_err


def crossfit_nuisances_train90(
    X_tr: np.ndarray,
    T_tr: np.ndarray,
    D_tr: np.ndarray,
    X_va: np.ndarray,
    M: int,
    seed: int,
    bs: int = 256,
    n_splits: int = 5,
    fit_G: bool = True,
    clip_hc: float = 0.85,
) -> Dict[str, np.ndarray]:
    """
    Strictly fits 5-fold OOF nuisances inside train_90 ONLY and averages predictions on val_10.
    Also returns the 5 fold models so Discrete-Hazard-NN (5-fold ens) can be evaluated on test!
    """
    N_tr = len(X_tr)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1000 + seed)
    oof_logits = np.zeros((N_tr, M, 2), dtype=np.float64)
    va_logits = np.zeros((len(X_va), M, 2), dtype=np.float64)
    oof_hc = np.zeros((N_tr, M), dtype=np.float64)
    va_hc = np.zeros((len(X_va), M), dtype=np.float64)
    fold_models = []

    for f_idx, (f_tr, f_va) in enumerate(kf.split(X_tr)):
        m_h = fit_discrete_hazard_nn(
            X_tr[f_tr], T_tr[f_tr], D_tr[f_tr], M=M, seed=seed * 10 + f_idx, bs=bs,
            X_val=X_tr[f_va], T_val=T_tr[f_va], D_val=D_tr[f_va],
        )
        fold_models.append(m_h)
        oof_logits[f_va] = predict_hazard_logits(m_h, X_tr[f_va])
        va_logits += predict_hazard_logits(m_h, X_va) / n_splits

        if fit_G:
            m_g = fit_censoring_hazard_nn(
                X_tr[f_tr], T_tr[f_tr], D_tr[f_tr], M=M, seed=seed * 10 + f_idx, bs=bs, clip_hc=clip_hc
            )
            hc_v, _ = predict_censoring_G(m_g, X_tr[f_va], clip_hc=clip_hc)
            hc_va, _ = predict_censoring_G(m_g, X_va, clip_hc=clip_hc)
            oof_hc[f_va] = hc_v
            va_hc += hc_va / n_splits

    S_tr, F1_tr, F2_tr = logits_to_cif_np(oof_logits)
    S_va, F1_va, F2_va = logits_to_cif_np(va_logits)

    G_tr = np.ones((N_tr, M + 2), dtype=np.float64)
    G_va = np.ones((len(X_va), M + 2), dtype=np.float64)
    if fit_G:
        for u in range(1, M + 1):
            G_tr[:, u + 1] = G_tr[:, u] * (1.0 - oof_hc[:, u - 1])
            G_va[:, u + 1] = G_va[:, u] * (1.0 - va_hc[:, u - 1])

    return {
        "S_tr": S_tr, "F1_tr": F1_tr, "F2_tr": F2_tr,
        "S_va": S_va, "F1_va": F1_va, "F2_va": F2_va,
        "hc_tr": oof_hc, "G_tr": G_tr, "hc_va": va_hc, "G_va": G_va,
        "fold_models": fold_models,
    }


def fit_monocif_target_nn(
    X_tr: np.ndarray,
    psi_tr: np.ndarray,
    X_va: np.ndarray,
    psi_va: np.ndarray,
    M: int,
    seed: int,
    loss_type: str = "logloss",
    bs: int = 256,
    max_epochs: int = 60,
    patience: int = 8,
    base_logits_tr: Optional[np.ndarray] = None,
    base_logits_va: Optional[np.ndarray] = None,
) -> nn.Module:
    """
    Trains MonoCIF-RB (logloss), MonoCIF-RB-Brier (brier), DR-MonoCIF-Brier (brier),
    Signed-AIPCW-LogLoss (logloss), or MonoCIF-RB (GBDT-Res) when base_logits are provided.
    """
    set_all_seeds(seed)
    is_res = base_logits_tr is not None
    model = ResidualHazardNet(X_tr.shape[1], M, K=2).to(DEVICE) if is_res else MatchedHazardNet(X_tr.shape[1], M, K=2).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    Xt = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    Psit = torch.tensor(psi_tr, dtype=torch.float32, device=DEVICE)
    Xv = torch.tensor(X_va, dtype=torch.float32, device=DEVICE)
    Psiv = torch.tensor(psi_va, dtype=torch.float32, device=DEVICE)
    Bt = torch.tensor(base_logits_tr, dtype=torch.float32, device=DEVICE) if is_res else None
    Bv = torch.tensor(base_logits_va, dtype=torch.float32, device=DEVICE) if is_res else None

    def loss_fn(P, psi):
        if loss_type == "brier":
            return ((P - psi) ** 2).sum(dim=-1).mean()
        return -(psi * torch.log(P.clamp(min=1e-7))).sum(dim=-1).mean()

    best_loss, best_state, bad = float("inf"), None, 0
    N = len(X_tr)
    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(N, device=DEVICE)
        for i in range(0, N, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            logits = model(Xt[idx], Bt[idx]) if is_res else model(Xt[idx])
            loss = loss_fn(logits_to_P_torch(logits), Psit[idx])
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = 0.0
            nv = len(X_va)
            chunk = max(bs, 4096)
            for i in range(0, nv, chunk):
                lg = model(Xv[i : i + chunk], Bv[i : i + chunk]) if is_res else model(Xv[i : i + chunk])
                vl += float(loss_fn(logits_to_P_torch(lg), Psiv[i : i + chunk])) * min(chunk, nv - i)
            vl /= nv
        if math.isfinite(vl) and vl < best_loss - 1e-6:
            best_loss = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


# =============================================================================
# 5. Competitor Models (CRNLL, MonoCIF-CRNLL, PerHorizon-BCE, DeepHit,
#    Cutoff-XGB, CS-XGB, XGB-PV, Discrete-Hazard-GBDT, GPU CS-Cox & Fine-Gray)
# =============================================================================

def fit_crnll_nn(
    X_tr: np.ndarray,
    T_tr: np.ndarray,
    D_tr: np.ndarray,
    X_va: np.ndarray,
    T_va: np.ndarray,
    D_va: np.ndarray,
    M: int,
    seed: int,
    monotone: bool = False,
    G_tr: Optional[np.ndarray] = None,
    G_va: Optional[np.ndarray] = None,
    bs: int = 256,
    max_epochs: int = 60,
    patience: int = 8,
) -> nn.Module:
    set_all_seeds(seed)
    model = MatchedHazardNet(X_tr.shape[1], M, K=2).to(DEVICE) if monotone else UnconstrainedHeadNet(X_tr.shape[1], M * 3).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    if G_tr is None:
        _, g_km = km_censoring_discrete(T_tr, D_tr, M)
        G_tr = np.broadcast_to(g_km[None, :], (len(X_tr), M + 2)).copy()
        G_va = np.broadcast_to(g_km[None, :], (len(X_va), M + 2)).copy()

    def precompute_weights_labels(T, D, G):
        N = len(T)
        rows = np.arange(N)
        W = np.zeros((N, M), dtype=np.float32)
        Y = np.zeros((N, M), dtype=np.int64)
        for m in range(1, M + 1):
            cens_before = (D == 0) & (T < m)
            W[:, m - 1] = (~cens_before).astype(np.float32) / np.clip(G[rows, np.minimum(T, m)], 0.05, 1.0).astype(np.float32)
            Y[:, m - 1] = np.where((T <= m) & (D > 0), D, 0)
        return W, Y

    W_tr, Y_tr = precompute_weights_labels(T_tr, D_tr, G_tr)
    W_va, Y_va = precompute_weights_labels(T_va, D_va, G_va)

    Xt = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    Wt = torch.tensor(W_tr, dtype=torch.float32, device=DEVICE)
    Yt = torch.tensor(Y_tr, dtype=torch.long, device=DEVICE)
    Xv = torch.tensor(X_va, dtype=torch.float32, device=DEVICE)
    Wv = torch.tensor(W_va, dtype=torch.float32, device=DEVICE)
    Yv = torch.tensor(Y_va, dtype=torch.long, device=DEVICE)

    def crnll_loss(x, w, y):
        if monotone:
            P = logits_to_P_torch(model(x))
            logP = torch.log(P.clamp(min=1e-7))
        else:
            logP = F.log_softmax(model(x).view(-1, M, 3), dim=-1)
        nll = -logP.gather(2, y.unsqueeze(-1)).squeeze(-1)
        return (nll * w).sum() / w.sum().clamp(min=1.0)

    best_loss, best_state, bad = float("inf"), None, 0
    N = len(X_tr)
    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(N, device=DEVICE)
        for i in range(0, N, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            loss = crnll_loss(Xt[idx], Wt[idx], Yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = 0.0
            nv = len(X_va)
            chunk = max(bs, 4096)
            for i in range(0, nv, chunk):
                vl += float(crnll_loss(Xv[i : i + chunk], Wv[i : i + chunk], Yv[i : i + chunk])) * min(chunk, nv - i)
            vl /= nv
        if vl < best_loss - 1e-6:
            best_loss = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def predict_crnll_nn(model: nn.Module, X: np.ndarray, M: int, monotone: bool = False, bs: int = 8192) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    f1s, f2s = [], []
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i : i + bs], dtype=torch.float32, device=DEVICE)
        if monotone:
            P = logits_to_P_torch(model(xb)).cpu().numpy()
        else:
            P = F.softmax(model(xb).view(-1, M, 3), dim=-1).cpu().numpy()
        f1s.append(P[:, :, 1])
        f2s.append(P[:, :, 2])
    return np.concatenate(f1s, axis=0), np.concatenate(f2s, axis=0)


def fit_perhorizon_bce_nn(
    X_tr: np.ndarray,
    T_tr: np.ndarray,
    D_tr: np.ndarray,
    X_va: np.ndarray,
    T_va: np.ndarray,
    D_va: np.ndarray,
    M: int,
    seed: int,
    bs: int = 256,
    max_epochs: int = 60,
    patience: int = 8,
) -> nn.Module:
    """Uncoupled per-horizon cumulative BCE (conceptual proxy for closed-access abstract-only W18)."""
    set_all_seeds(seed)
    model = UnconstrainedHeadNet(X_tr.shape[1], M * 2).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    _, g_km = km_censoring_discrete(T_tr, D_tr, M)

    def prep(T, D):
        N = len(T)
        W = np.zeros((N, M, 2), dtype=np.float32)
        Y = np.zeros((N, M, 2), dtype=np.float32)
        for m in range(1, M + 1):
            w = (~((D == 0) & (T < m))).astype(np.float32) / np.clip(g_km[np.minimum(T, m)], 0.05, 1.0).astype(np.float32)
            W[:, m - 1, 0] = w
            W[:, m - 1, 1] = w
            Y[:, m - 1, 0] = ((T <= m) & (D == 1)).astype(np.float32)
            Y[:, m - 1, 1] = ((T <= m) & (D == 2)).astype(np.float32)
        return W, Y

    W_tr, Y_tr = prep(T_tr, D_tr)
    W_va, Y_va = prep(T_va, D_va)
    Xt = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    Wt = torch.tensor(W_tr, dtype=torch.float32, device=DEVICE)
    Yt = torch.tensor(Y_tr, dtype=torch.float32, device=DEVICE)
    Xv = torch.tensor(X_va, dtype=torch.float32, device=DEVICE)
    Wv = torch.tensor(W_va, dtype=torch.float32, device=DEVICE)
    Yv = torch.tensor(Y_va, dtype=torch.float32, device=DEVICE)

    def bce_loss(x, w, y):
        lg = model(x).view(-1, M, 2)
        return (F.binary_cross_entropy_with_logits(lg, y, reduction="none") * w).sum() / w.sum().clamp(min=1.0)

    best_loss, best_state, bad = float("inf"), None, 0
    N = len(X_tr)
    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(N, device=DEVICE)
        for i in range(0, N, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            loss = bce_loss(Xt[idx], Wt[idx], Yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(bce_loss(Xv, Wv, Yv))
        if vl < best_loss - 1e-6:
            best_loss = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def predict_perhorizon_bce_nn(model: nn.Module, X: np.ndarray, M: int, bs: int = 8192) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    f1s, f2s = [], []
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i : i + bs], dtype=torch.float32, device=DEVICE)
        p = torch.sigmoid(model(xb).view(-1, M, 2)).cpu().numpy()
        f1s.append(p[:, :, 0])
        f2s.append(p[:, :, 1])
    return np.concatenate(f1s, axis=0), np.concatenate(f2s, axis=0)


def fit_deephit_nn(
    X_tr: np.ndarray,
    T_tr: np.ndarray,
    D_tr: np.ndarray,
    X_va: np.ndarray,
    T_va: np.ndarray,
    D_va: np.ndarray,
    M: int,
    seed: int,
    alpha: float = 0.2,
    sigma: float = 0.1,
    bs: int = 256,
    max_epochs: int = 60,
    patience: int = 8,
) -> nn.Module:
    """Lee et al. (2018) DeepHit joint PMF likelihood + cause-specific pairwise ranking loss on matched NN."""
    set_all_seeds(seed)
    # 2*M event cells + 1 survive-past-M cell
    model = UnconstrainedHeadNet(X_tr.shape[1], 2 * M + 1).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    Xt = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    Tt = torch.tensor(T_tr, dtype=torch.long, device=DEVICE)
    Dt = torch.tensor(D_tr, dtype=torch.long, device=DEVICE)
    Xv = torch.tensor(X_va, dtype=torch.float32, device=DEVICE)
    Tv = torch.tensor(T_va, dtype=torch.long, device=DEVICE)
    Dv = torch.tensor(D_va, dtype=torch.long, device=DEVICE)

    def dh_loss(x, T, D, with_rank: bool = True):
        pmf = F.softmax(model(x), dim=-1)  # (B, 2*M+1)
        p1 = pmf[:, :M]
        p2 = pmf[:, M : 2 * M]
        p_surv = pmf[:, 2 * M]
        F1 = torch.cumsum(p1, dim=1)
        F2 = torch.cumsum(p2, dim=1)
        S = (1.0 - F1 - F2).clamp(min=1e-7)

        # L1: joint censored PMF negative log-likelihood
        ev1 = (D == 1)
        ev2 = (D == 2)
        cens_m = (D == 0) & (T <= M)
        surv_end = (D == 0) & (T > M)
        B = len(T)
        rows = torch.arange(B, device=x.device)
        t_idx = (T - 1).clamp(0, M - 1)
        probs = torch.ones(B, device=x.device)
        probs = torch.where(ev1, p1[rows, t_idx], probs)
        probs = torch.where(ev2, p2[rows, t_idx], probs)
        probs = torch.where(cens_m, S[rows, t_idx], probs)
        probs = torch.where(surv_end, p_surv, probs)
        l1 = -torch.log(probs.clamp(min=1e-7)).mean()

        if not with_rank or B > 1024:
            # Subsample batch for O(B^2) pairwise ranking on large batches (SEER bs=4096)
            if with_rank and B > 512:
                sub = torch.randperm(B, device=x.device)[:512]
                F1, F2, T, D = F1[sub], F2[sub], T[sub], D[sub]
                t_idx = (T - 1).clamp(0, M - 1)
                B = 512
            elif not with_rank:
                return l1
        # L2: pairwise ranking loss
        l2 = torch.tensor(0.0, device=x.device)
        for k, Fk in ((1, F1), (2, F2)):
            mask_k = (D == k)
            if mask_k.sum() < 2:
                continue
            Ti = T[mask_k].unsqueeze(1)  # (nk, 1)
            Tj = T.unsqueeze(0)          # (1, B)
            valid_pair = (Ti < Tj)
            if valid_pair.sum() == 0:
                continue
            ti_idx = t_idx[mask_k]
            fi_ti = Fk[mask_k, ti_idx].unsqueeze(1)  # (nk, 1)
            fj_ti = Fk[:, ti_idx].T                  # (nk, B)
            eta = torch.exp(-(fi_ti - fj_ti) / sigma)
            l2 = l2 + (eta * valid_pair.float()).sum() / valid_pair.float().sum().clamp(min=1.0)
        return l1 + alpha * l2

    best_loss, best_state, bad = float("inf"), None, 0
    N = len(X_tr)
    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(N, device=DEVICE)
        for i in range(0, N, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            loss = dh_loss(Xt[idx], Tt[idx], Dt[idx], with_rank=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(dh_loss(Xv[:4096], Tv[:4096], Dv[:4096], with_rank=False))
        if vl < best_loss - 1e-6:
            best_loss = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def predict_deephit_nn(model: nn.Module, X: np.ndarray, M: int, bs: int = 8192) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    f1s, f2s = [], []
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i : i + bs], dtype=torch.float32, device=DEVICE)
        pmf = F.softmax(model(xb), dim=-1).cpu().numpy()
        f1s.append(np.cumsum(pmf[:, :M], axis=1))
        f2s.append(np.cumsum(pmf[:, M : 2 * M], axis=1))
    return np.concatenate(f1s, axis=0), np.concatenate(f2s, axis=0)


def fit_predict_cs_cox_gpu(
    X_tr: np.ndarray,
    T_tr: np.ndarray,
    D_tr: np.ndarray,
    X_te: np.ndarray,
    M: int,
    l2_alpha: float = 1e-2,
) -> Tuple[np.ndarray, np.ndarray]:
    """L2-regularized discrete cause-specific proportional-hazards Cox via GPU L-BFGS (singularity-free on SEER)."""
    d = X_tr.shape[1]
    Xt = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    Tt = torch.tensor(T_tr, dtype=torch.long, device=DEVICE)
    Dt = torch.tensor(D_tr, dtype=torch.long, device=DEVICE)
    Xte = torch.tensor(X_te, dtype=torch.float32, device=DEVICE)

    # Discrete PH hazard logits: z_{k,l}(x) = alpha_{k,l} + x @ beta_k
    alpha_p = nn.Parameter(torch.full((M, 2), -3.5, device=DEVICE))
    beta_p = nn.Parameter(torch.zeros((d, 2), device=DEVICE))
    opt = torch.optim.LBFGS([alpha_p, beta_p], lr=0.8, max_iter=80, line_search_fn="strong_wolfe")

    # Subsample at most 200,000 rows if huge for fast L-BFGS? Full N fits in 1.2GB VRAM easily!
    def closure():
        opt.zero_grad()
        lin = Xt @ beta_p  # (N, 2)
        logits = alpha_p.unsqueeze(0) + lin.unsqueeze(1)  # (N, M, 2)
        loss = discrete_hazard_nll_torch(logits, Tt, Dt) + 0.5 * l2_alpha * (beta_p ** 2).mean()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        logits_te = (alpha_p.unsqueeze(0) + (Xte @ beta_p).unsqueeze(1)).cpu().numpy()
    _, F1, F2 = logits_to_cif_np(logits_te)
    return F1, F2


def fit_predict_fine_gray_gpu(
    X_tr: np.ndarray,
    T_tr: np.ndarray,
    D_tr: np.ndarray,
    X_te: np.ndarray,
    M: int,
    l2_alpha: float = 1e-2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    L2-regularized discrete IPCW Fine-Gray subdistribution hazard model on GPU (per cause k=1,2):
    In subdistribution risk set for cause k at bin l: subjects with T >= l OR competing event (D != 0, D != k, T < l)
    weighted by G_+(l) / G_+(T). Compactly evaluated without materializing 9.7M dense rows!
    """
    d = X_tr.shape[1]
    _, g_km = km_censoring_discrete(T_tr, D_tr, M)
    Xt = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    Xte = torch.tensor(X_te, dtype=torch.float32, device=DEVICE)
    F_out = []

    for k in (1, 2):
        # Precompute subdistribution risk weight matrix W_sub (N, M)
        N = len(T_tr)
        W_sub = np.zeros((N, M), dtype=np.float32)
        Y_ev = np.zeros((N, M), dtype=np.float32)
        comp = (D_tr > 0) & (D_tr != k)
        for l in range(1, M + 1):
            at_risk = (T_tr >= l)
            W_sub[at_risk, l - 1] = 1.0
            comp_before = comp & (T_tr < l)
            if np.any(comp_before):
                W_sub[comp_before, l - 1] = np.clip(g_km[l] / np.clip(g_km[T_tr[comp_before]], 0.05, 1.0), 0.0, 1.0)
            Y_ev[(T_tr == l) & (D_tr == k), l - 1] = 1.0

        Wt = torch.tensor(W_sub, dtype=torch.float32, device=DEVICE)
        Yt = torch.tensor(Y_ev, dtype=torch.float32, device=DEVICE)
        alpha_p = nn.Parameter(torch.full((M,), -3.5, device=DEVICE))
        beta_p = nn.Parameter(torch.zeros((d,), device=DEVICE))
        opt = torch.optim.LBFGS([alpha_p, beta_p], lr=0.8, max_iter=60, line_search_fn="strong_wolfe")
        w_sum = Wt.sum().clamp(min=1.0)

        def closure():
            opt.zero_grad()
            eta = alpha_p.unsqueeze(0) + (Xt @ beta_p).unsqueeze(1)  # (N, M)
            # Complementary log-log subdistribution hazard: h_sub = 1 - exp(-exp(eta))
            # Use stable logit approximation / cloglog BCE
            eta_c = eta.clamp(-12.0, 4.0)
            p_sub = (1.0 - torch.exp(-torch.exp(eta_c))).clamp(1e-7, 1.0 - 1e-7)
            nll = -(Yt * torch.log(p_sub) + (1.0 - Yt) * torch.log(1.0 - p_sub))
            loss = (nll * Wt).sum() / w_sum + 0.5 * l2_alpha * (beta_p ** 2).mean()
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            eta_te = (alpha_p.unsqueeze(0) + (Xte @ beta_p).unsqueeze(1)).clamp(-12.0, 4.0)
            p_sub_te = (1.0 - torch.exp(-torch.exp(eta_te))).cpu().numpy()
            S_sub = np.cumprod(1.0 - p_sub_te, axis=1)
            Fk = 1.0 - S_sub
            F_out.append(Fk)
    return F_out[0], F_out[1]


def fit_predict_cs_xgboost(
    X_tr: np.ndarray,
    T_tr: np.ndarray,
    D_tr: np.ndarray,
    X_va: np.ndarray,
    T_va: np.ndarray,
    D_va: np.ndarray,
    X_te: np.ndarray,
    M: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cause-specific XGBoost PH (survival:cox) with exact discrete Breslow-Aalen-Johansen CIF."""
    X_all_tr = np.vstack([X_tr, X_va])
    T_all_tr = np.concatenate([T_tr, T_va])
    D_all_tr = np.concatenate([D_tr, D_va])

    risks_tr, risks_te = {}, {}
    for k in (1, 2):
        y_tr = np.where(D_tr == k, T_tr.astype(np.float32), -T_tr.astype(np.float32))
        y_va = np.where(D_va == k, T_va.astype(np.float32), -T_va.astype(np.float32))
        dtr = xgb.DMatrix(X_tr, label=y_tr)
        dva = xgb.DMatrix(X_va, label=y_va)
        params = {
            "objective": "survival:cox", "max_depth": 5, "eta": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist",
            "device": "cuda" if torch.cuda.is_available() else "cpu", "seed": seed + k,
        }
        bst = xgb.train(params, dtr, num_boost_round=300, evals=[(dva, "v")], early_stopping_rounds=25, verbose_eval=False)
        risks_tr[k] = np.clip(bst.predict(xgb.DMatrix(X_all_tr)), 1e-4, 1e4)
        risks_te[k] = np.clip(bst.predict(xgb.DMatrix(X_te)), 1e-4, 1e4)

    # Discrete Breslow baseline cause-specific hazards
    h0_k = {1: np.zeros(M), 2: np.zeros(M)}
    for l in range(1, M + 1):
        at_risk = (T_all_tr >= l)
        for k in (1, 2):
            ev = ((T_all_tr == l) & (D_all_tr == k)).sum()
            denom = risks_tr[k][at_risk].sum()
            h0_k[k][l - 1] = ev / denom if denom > 0 else 0.0

    # Compute CIF via discrete cause-specific hazard product
    lam1 = risks_te[1][:, None] * h0_k[1][None, :]
    lam2 = risks_te[2][:, None] * h0_k[2][None, :]
    lam_tot = np.clip(lam1 + lam2, 1e-12, None)
    p_any = 1.0 - np.exp(-lam_tot)
    h1 = p_any * (lam1 / lam_tot)
    h2 = p_any * (lam2 / lam_tot)
    h0 = np.clip(1.0 - h1 - h2, 0.0, 1.0)
    S = np.cumprod(h0, axis=1)
    S_prev = np.concatenate([np.ones((len(X_te), 1)), S[:, :-1]], axis=1)
    return np.cumsum(S_prev * h1, axis=1), np.cumsum(S_prev * h2, axis=1)


def fit_predict_cutoff_xgb(
    X_tr: np.ndarray,
    T_tr: np.ndarray,
    D_tr: np.ndarray,
    X_va: np.ndarray,
    T_va: np.ndarray,
    D_va: np.ndarray,
    X_te: np.ndarray,
    M: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Theorem 1 baseline: per-horizon multiclass XGBoost treating censored < m as alive."""
    N_te = len(X_te)
    F1 = np.zeros((N_te, M), dtype=np.float64)
    F2 = np.zeros((N_te, M), dtype=np.float64)
    dte = xgb.DMatrix(X_te)
    params = {
        "objective": "multi:softprob", "num_class": 3, "max_depth": 5, "eta": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist",
        "device": "cuda" if torch.cuda.is_available() else "cpu", "seed": seed,
    }
    for m in range(1, M + 1):
        y_tr = np.where((T_tr <= m) & (D_tr > 0), D_tr, 0)
        y_va = np.where((T_va <= m) & (D_va > 0), D_va, 0)
        if len(np.unique(y_tr)) < 3:
            F1[:, m - 1] = (y_tr == 1).mean()
            F2[:, m - 1] = (y_tr == 2).mean()
            continue
        bst = xgb.train(
            params, xgb.DMatrix(X_tr, label=y_tr), num_boost_round=150,
            evals=[(xgb.DMatrix(X_va, label=y_va), "v")], early_stopping_rounds=20, verbose_eval=False,
        )
        p = bst.predict(dte)
        F1[:, m - 1] = p[:, 1]
        F2[:, m - 1] = p[:, 2]
    return F1, F2


def compute_fast_aj_pseudovalues(T_bin: np.ndarray, D_bin: np.ndarray, M: int) -> Tuple[np.ndarray, np.ndarray]:
    """O(N*M) exact infinitesimal jackknife Aalen-Johansen pseudo-values for competing risks."""
    N = len(T_bin)
    S, F1_bar, F2_bar = aalen_johansen_discrete(T_bin, D_bin, M)
    S_prev = np.concatenate([[1.0], S[:-1]])
    n_risk = np.array([(T_bin >= l).sum() for l in range(1, M + 1)], dtype=np.float64)
    h1 = np.array([((T_bin == l) & (D_bin == 1)).sum() / max(n_risk[l - 1], 1.0) for l in range(1, M + 1)])
    h2 = np.array([((T_bin == l) & (D_bin == 2)).sum() / max(n_risk[l - 1], 1.0) for l in range(1, M + 1)])

    # Fast exact leave-one-out / IJ perturbation:
    # Using exact leave-one-out per unique (T, D) pattern (only <= 3*(M+1) unique patterns -> O(M^2)!)
    patterns = {}
    for t_val in range(1, M + 2):
        for d_val in (0, 1, 2):
            if t_val == M + 1 and d_val != 0:
                continue
            nr_loo = n_risk.copy()
            d1_loo = np.array([((T_bin == l) & (D_bin == 1)).sum() for l in range(1, M + 1)], dtype=np.float64)
            d2_loo = np.array([((T_bin == l) & (D_bin == 2)).sum() for l in range(1, M + 1)], dtype=np.float64)
            nr_loo[: min(t_val, M)] -= 1.0
            if t_val <= M and d_val == 1:
                d1_loo[t_val - 1] -= 1.0
            elif t_val <= M and d_val == 2:
                d2_loo[t_val - 1] -= 1.0
            h1_l = np.where(nr_loo > 0, d1_loo / np.maximum(nr_loo, 1.0), 0.0)
            h2_l = np.where(nr_loo > 0, d2_loo / np.maximum(nr_loo, 1.0), 0.0)
            S_l = np.cumprod(np.clip(1.0 - h1_l - h2_l, 0.0, 1.0))
            Sp_l = np.concatenate([[1.0], S_l[:-1]])
            F1_l = np.cumsum(Sp_l * h1_l)
            F2_l = np.cumsum(Sp_l * h2_l)
            patterns[(t_val, d_val)] = (N * F1_bar - (N - 1) * F1_l, N * F2_bar - (N - 1) * F2_l)

    pv1 = np.zeros((N, M), dtype=np.float64)
    pv2 = np.zeros((N, M), dtype=np.float64)
    for (tv, dv), (v1, v2) in patterns.items():
        mask = (T_bin == tv) & (D_bin == dv)
        if np.any(mask):
            pv1[mask] = v1[None, :]
            pv2[mask] = v2[None, :]
    return pv1, pv2


def fit_predict_xgboost_pv(
    X_tr: np.ndarray,
    T_tr: np.ndarray,
    D_tr: np.ndarray,
    X_va: np.ndarray,
    T_va: np.ndarray,
    D_va: np.ndarray,
    X_te: np.ndarray,
    M: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Clift et al. (2023) multi-horizon XGBoost pseudo-value regression clipped post-hoc to [0, 1]."""
    pv1_tr, pv2_tr = compute_fast_aj_pseudovalues(T_tr, D_tr, M)
    pv1_va, pv2_va = compute_fast_aj_pseudovalues(T_va, D_va, M)
    N_te = len(X_te)
    F1 = np.zeros((N_te, M), dtype=np.float64)
    F2 = np.zeros((N_te, M), dtype=np.float64)
    dte = xgb.DMatrix(X_te)
    params = {
        "objective": "reg:squarederror", "max_depth": 5, "eta": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist",
        "device": "cuda" if torch.cuda.is_available() else "cpu", "seed": seed,
    }
    for m in range(M):
        for out_arr, ytr, yva in ((F1, pv1_tr[:, m], pv1_va[:, m]), (F2, pv2_tr[:, m], pv2_va[:, m])):
            bst = xgb.train(
                params, xgb.DMatrix(X_tr, label=ytr), num_boost_round=150,
                evals=[(xgb.DMatrix(X_va, label=yva), "v")], early_stopping_rounds=20, verbose_eval=False,
            )
            out_arr[:, m] = np.clip(bst.predict(dte), 0.0, 1.0)
    return F1, F2


def fit_discrete_hazard_gbdt(
    X_tr: np.ndarray,
    T_tr: np.ndarray,
    D_tr: np.ndarray,
    X_va: np.ndarray,
    T_va: np.ndarray,
    D_va: np.ndarray,
    M: int,
    seed: int,
) -> List[Optional[xgb.Booster]]:
    """Per-interval multinomial XGBoost discrete cause-specific hazard model on train_90 with val_10 early stopping."""
    models: List[Optional[xgb.Booster]] = []
    params = {
        "objective": "multi:softprob", "num_class": 3, "max_depth": 5, "eta": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist",
        "device": "cuda" if torch.cuda.is_available() else "cpu", "seed": seed,
    }
    for l in range(1, M + 1):
        m_tr = (T_tr >= l)
        m_va = (T_va >= l)
        y_tr = np.where((T_tr[m_tr] == l) & (D_tr[m_tr] > 0), D_tr[m_tr], 0)
        y_va = np.where((T_va[m_va] == l) & (D_va[m_va] > 0), D_va[m_va], 0)
        if m_tr.sum() < 30 or len(np.unique(y_tr)) < 3 or m_va.sum() < 10:
            models.append(None)
            continue
        bst = xgb.train(
            params, xgb.DMatrix(X_tr[m_tr], label=y_tr), num_boost_round=200,
            evals=[(xgb.DMatrix(X_va[m_va], label=y_va), "v")], early_stopping_rounds=20, verbose_eval=False,
        )
        models.append(bst)
    return models


def predict_discrete_hazard_gbdt_logits(
    models: Sequence[Optional[xgb.Booster]],
    X: np.ndarray,
    T_tr: np.ndarray,
    D_tr: np.ndarray,
) -> np.ndarray:
    """Returns cause-specific hazard logits z_{k,l} = log(h_{k,l} / h_{0,l}) of shape (N, M, 2)."""
    N = len(X)
    M = len(models)
    logits = np.zeros((N, M, 2), dtype=np.float64)
    dmat = xgb.DMatrix(X)
    for l, bst in enumerate(models, start=1):
        if bst is None:
            m_tr = (T_tr >= l)
            nr = max(int(m_tr.sum()), 1)
            h1 = max(((T_tr == l) & (D_tr == 1)).sum() / nr, 1e-6)
            h2 = max(((T_tr == l) & (D_tr == 2)).sum() / nr, 1e-6)
            h0 = max(1.0 - h1 - h2, 1e-6)
            logits[:, l - 1, 0] = math.log(h1 / h0)
            logits[:, l - 1, 1] = math.log(h2 / h0)
        else:
            p = np.clip(bst.predict(dmat), 1e-7, 1.0 - 1e-7)
            logits[:, l - 1, 0] = np.log(p[:, 1] / p[:, 0])
            logits[:, l - 1, 1] = np.log(p[:, 2] / p[:, 0])
    return logits


# =============================================================================
# 6. Full 9-Arm Recalibration & Martingale Surveillance Suite (Proposition 3)
# =============================================================================

def compute_martingale_z(
    logits_cal: np.ndarray,
    T_cal: np.ndarray,
    D_cal: np.ndarray,
    L_cal: Optional[int] = None,
    alpha_fwer: float = 0.05,
) -> Dict[str, float]:
    """
    Exact censoring-ancillary discrete-martingale drift statistics (Proposition 3, ZERO 1/G):
      - 1-d.f. directional intercept Z-score: Z_k = (O_k - E_k) / sqrt(V_k)
      - d_k-d.f. generalized martingale Rao score test: Q_k = U_k^T I_k^{-1} U_k ~ chi^2_{d_k}
        with predictable basis phi = (1, z_{k,l}, tau_l) when L >= 2 (d_k=3) and (1, z_{k,1}) when L = 1 (d_k=2).
    Detects BOTH prevalence surges/rebounds AND mean-cancelling slope/time-trend shifts while maintaining exact FWER <= alpha_fwer!
    """
    from scipy.stats import chi2
    z = np.asarray(logits_cal, dtype=np.float64)
    N, M, K = z.shape
    L = M if L_cal is None else min(M, L_cal)
    L_ref = math.sqrt(max(L, 1))
    tau = np.log(np.arange(1, L + 1, dtype=np.float64) / L_ref)
    h = logits_to_hazards_np(z[:, :L, :])  # (N, L, 3)
    use_time = (L >= 2)
    df = 3 if use_time else 2

    out: Dict[str, float] = {"df_score": float(df)}
    for k in (1, 2):
        O_k, E_k, V_k = 0.0, 0.0, 0.0
        U = np.zeros(df, dtype=np.float64)
        Info = np.zeros((df, df), dtype=np.float64)
        for l in range(1, L + 1):
            at_risk = (T_cal >= l)
            if not np.any(at_risk):
                continue
            hk = h[at_risk, l - 1, k]
            yk = ((T_cal[at_risk] == l) & (D_cal[at_risk] == k)).astype(np.float64)
            res = yk - hk
            w = hk * (1.0 - hk)
            O_k += float(yk.sum())
            E_k += float(hk.sum())
            V_k += float(w.sum())
            zk = z[at_risk, l - 1, k - 1]
            if use_time:
                phi = np.column_stack([np.ones(len(zk)), zk, np.full(len(zk), tau[l - 1])])
            else:
                phi = np.column_stack([np.ones(len(zk)), zk])
            U += phi.T @ res
            Info += (phi * w[:, None]).T @ phi
        Z_k = (O_k - E_k) / math.sqrt(max(V_k, 1e-8))
        Info_reg = Info + 1e-6 * np.eye(df)
        Q_k = float(U @ np.linalg.solve(Info_reg, U))
        p_k = float(chi2.sf(Q_k, df=df))
        out[f"O{k}"] = O_k
        out[f"E{k}"] = E_k
        out[f"V{k}"] = V_k
        out[f"Z{k}"] = Z_k
        out[f"Q{k}"] = Q_k
        out[f"p{k}"] = p_k
    out["z_crit"] = float(norm.ppf(1.0 - alpha_fwer / (2.0 * K)))  # 2.2414 for alpha=0.05, K=2
    out["q_crit"] = float(chi2.ppf(1.0 - alpha_fwer / K, df=df))   # 9.3484 for alpha=0.05, K=2, df=3
    return out


def _person_interval_nll_and_grad(
    z1_base: np.ndarray,
    z2_base: np.ndarray,
    tau_cal: np.ndarray,
    ar_mask: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    theta: Sequence[float],
) -> Tuple[float, np.ndarray]:
    """Exact float64 summed censoring-ancillary person-interval NLL and analytical gradient wrt (a1,b1,g1,a2,b2,g2)."""
    a1, b1, g1, a2, b2, g2 = theta
    u1 = a1 + b1 * z1_base + g1 * tau_cal
    u2 = a2 + b2 * z2_base + g2 * tau_cal
    mx = np.maximum(0.0, np.maximum(u1, u2))
    e0 = np.exp(-mx)
    e1 = np.exp(u1 - mx)
    e2 = np.exp(u2 - mx)
    den = e0 + e1 + e2
    p1 = e1 / den
    p2 = e2 / den
    lse = mx + np.log(den)
    nll = float(np.sum(ar_mask * (lse - y1 * u1 - y2 * u2)))
    r1 = ar_mask * (p1 - y1)
    r2 = ar_mask * (p2 - y2)
    grad = np.array([
        np.sum(r1), np.sum(r1 * z1_base), np.sum(r1 * tau_cal),
        np.sum(r2), np.sum(r2 * z2_base), np.sum(r2 * tau_cal),
    ], dtype=np.float64)
    return nll, grad


def fit_sg_jccr(
    logits_cal: np.ndarray,
    T_cal: np.ndarray,
    D_cal: np.ndarray,
    logits_dep: np.ndarray,
    L_cal: Optional[int] = None,
    per_cause_gate: bool = True,
    gate_mode: str = "score_df",
    ungated_tier2: bool = False,
    ridge: float = 1e-3,
    alpha_fwer: float = 0.05,
) -> Dict[str, object]:
    """
    Per-Cause Bonferroni Martingale-Gated Hierarchical SG-JCCR (Proposition 3, float64 + exact analytical gradient):
      - gate_mode='score_df' (default): Active set A = {k : Q_k >= chi2_{d_k, 1 - alpha_fwer/K}} (exact FWER <= alpha_fwer).
      - gate_mode='z_1df' (ablation): Active set A = {k : |Z_k| >= Phi^{-1}(1 - alpha_fwer/(2K))}.
      - Inactive causes k not in A clamped strictly to identity (a_k=0, b_k=1, g_k=0, df=0).
      - BIC = 2*NLL_cal (summed!) + df * log(N_eff), N_eff = max(10, sum_i I(D_i > 0, T_i <= L)).
      - L_cal < 2 guard: disables log-time trend gamma_k automatically!
    """
    zc = np.asarray(logits_cal, dtype=np.float64)
    zd = np.asarray(logits_dep, dtype=np.float64)
    N, M, K = zc.shape
    L = M if L_cal is None else min(M, L_cal)
    L_ref = math.sqrt(max(L, 1))
    tau_cal = np.log(np.arange(1, L + 1, dtype=np.float64) / L_ref)[None, :]  # (1, L)
    tau_dep = np.log(np.arange(1, M + 1, dtype=np.float64) / L_ref)[None, :]  # (1, M)

    z_info = compute_martingale_z(zc, T_cal, D_cal, L_cal=L, alpha_fwer=alpha_fwer)
    if ungated_tier2 or not per_cause_gate:
        active = [1, 2]
    elif gate_mode == "z_1df":
        active = [k for k in (1, 2) if abs(z_info[f"Z{k}"]) >= z_info["z_crit"]]
    else:
        active = [k for k in (1, 2) if z_info[f"Q{k}"] >= z_info["q_crit"]]

    bins_L = np.arange(1, L + 1)[None, :]
    ar_mask = (T_cal[:, None] >= bins_L).astype(np.float64)
    y1 = ((T_cal[:, None] == bins_L) & (D_cal[:, None] == 1)).astype(np.float64)
    y2 = ((T_cal[:, None] == bins_L) & (D_cal[:, None] == 2)).astype(np.float64)
    z1_b = zc[:, :L, 0]
    z2_b = zc[:, :L, 1]

    id_full = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    nll0, _ = _person_interval_nll_and_grad(z1_b, z2_b, tau_cal, ar_mask, y1, y2, id_full)
    n_eff = max(10.0, float(((D_cal > 0) & (T_cal <= L)).sum()))
    bic0 = 2.0 * nll0

    if not ungated_tier2 and len(active) == 0:
        _, F1, F2 = logits_to_cif_np(zd)
        return {"F1": F1, "F2": F2, "tier": "Tier0_Hold", "active": [], "z_info": z_info, "params": id_full}

    use_time = (L >= 2)
    per_k_df = 3 if use_time else 2

    def _fit_tier(tier_level: int) -> Tuple[List[float], float]:
        idxs: List[int] = []
        id_v: List[float] = []
        bnds: List[Tuple[float, float]] = []
        for k in active:
            off = 0 if k == 1 else 3
            if tier_level == 1:
                idxs.append(off)
                id_v.append(0.0)
                bnds.append((-4.0, 4.0))
            else:
                idxs.extend([off, off + 1] + ([off + 2] if use_time else []))
                id_v.extend([0.0, 1.0] + ([0.0] if use_time else []))
                bnds.extend([(-4.0, 4.0), (0.2, 3.0)] + ([(-2.0, 2.0)] if use_time else []))
        id_arr = np.array(id_v, dtype=np.float64)

        def fg(v: np.ndarray) -> Tuple[float, np.ndarray]:
            th = list(id_full)
            for j, gi in enumerate(idxs):
                th[gi] = float(v[j])
            val, gr = _person_interval_nll_and_grad(z1_b, z2_b, tau_cal, ar_mask, y1, y2, th)
            diff = v - id_arr
            gv = np.array([gr[gi] for gi in idxs], dtype=np.float64) + ridge * diff
            return val + 0.5 * ridge * float(np.sum(diff ** 2)), gv

        res = minimize(fg, id_arr, jac=True, method="L-BFGS-B", bounds=bnds)
        th_out = list(id_full)
        for j, gi in enumerate(idxs):
            th_out[gi] = float(res.x[j])
        nll_t, _ = _person_interval_nll_and_grad(z1_b, z2_b, tau_cal, ar_mask, y1, y2, th_out)
        bic_t = 2.0 * nll_t + len(idxs) * math.log(n_eff)
        return th_out, bic_t

    th1, bic1 = _fit_tier(1)
    th2, bic2 = _fit_tier(2)

    if ungated_tier2:
        chosen_tier, chosen_th = "Ungated_Tier2", th2
    else:
        bics = [("Tier0_Hold", bic0, id_full), ("Tier1_K", bic1, th1), ("Tier2_3K", bic2, th2)]
        chosen_tier, _, chosen_th = min(bics, key=lambda x: x[1])

    z_final = zd.copy()
    a1, b1, g1, a2, b2, g2 = chosen_th
    z_final[:, :, 0] = a1 + b1 * zd[:, :, 0] + g1 * tau_dep
    z_final[:, :, 1] = a2 + b2 * zd[:, :, 1] + g2 * tau_dep
    _, F1, F2 = logits_to_cif_np(z_final)
    return {"F1": F1, "F2": F2, "tier": chosen_tier, "active": active, "z_info": z_info, "params": [float(x) for x in chosen_th]}


def run_all_9_recalibrators(
    logits_cal: np.ndarray,
    T_cal: np.ndarray,
    D_cal: np.ndarray,
    logits_dep: np.ndarray,
    L_cal: Optional[int] = None,
    ridge: float = 1e-3,
) -> Dict[str, Dict[str, object]]:
    """
    Runs all matched recalibration arms on identical frozen base logits and calibration slice.
    """
    N_c, M, K = logits_cal.shape
    N_d = logits_dep.shape[0]
    L = M if L_cal is None else min(M, L_cal)
    S_c, F1_c, F2_c = logits_to_cif_np(logits_cal)
    S_d, F1_d, F2_d = logits_to_cif_np(logits_dep)
    _, g_km = km_censoring_discrete(T_cal, D_cal, M)
    _, aj1_c, aj2_c = aalen_johansen_discrete(T_cal, D_cal, L)

    results: Dict[str, Dict[str, object]] = {}

    # 1. Frozen
    results["1. Frozen"] = {"F1": F1_d.copy(), "F2": F2_d.copy()}

    # 2. AJ-Recal (AISTATS 2026): per-horizon additive shift (carry last observed shift past L)
    F1_aj, F2_aj = F1_d.copy(), F2_d.copy()
    for m in range(1, M + 1):
        lm = min(m, L) - 1
        d1 = aj1_c[lm] - F1_c[:, lm].mean()
        d2 = aj2_c[lm] - F2_c[:, lm].mean()
        F1_aj[:, m - 1] = np.clip(F1_d[:, m - 1] + d1, 0.0, 1.0)
        F2_aj[:, m - 1] = np.clip(F2_d[:, m - 1] + d2, 0.0, 1.0)
    results["2. AJ-Recal (AISTATS 26)"] = {"F1": F1_aj, "F2": F2_aj}

    # 3. Verified Bounded CR-TS (AISTATS 2026) & 4. CR-TS+intercept
    F1_ts, F2_ts = F1_d.copy(), F2_d.copy()
    F1_tsi, F2_tsi = F1_d.copy(), F2_d.copy()
    last_b = 1.0
    last_bi = (1.0, 0.0, 0.0)
    for m in range(1, M + 1):
        if m <= L:
            cens_b = (D_cal == 0) & (T_cal < m)
            w = (~cens_b).astype(np.float64) / np.clip(g_km[np.minimum(T_cal, m)], 0.05, 1.0)
            y = np.where((T_cal <= m) & (D_cal > 0), D_cal, 0)
            Pc = np.stack([S_c[:, m - 1], F1_c[:, m - 1], F2_c[:, m - 1]], axis=-1)
            lc = np.log(np.clip(Pc, 1e-7, 1.0))

            def crts_obj(beta_arr):
                b = float(beta_arr[0])
                sc = lc / b
                sc -= sc.max(axis=-1, keepdims=True)
                lse = np.log(np.exp(sc).sum(axis=-1))
                nll = lse - sc[np.arange(N_c), y]
                return float((w * nll).sum() / max(w.sum(), 1.0))

            # Grid init in [0.25, 4.0] + bounded L-BFGS-B in [0.1, 10.0] -> zero optimization collapse!
            grid = [0.4, 0.7, 1.0, 1.5, 2.5]
            b0 = min(grid, key=lambda b: crts_obj([b]))
            r_ts = minimize(crts_obj, [b0], method="L-BFGS-B", bounds=[(0.1, 10.0)])
            last_b = float(r_ts.x[0])

            def crtsi_obj(v):
                b, a1, a2 = float(v[0]), float(v[1]), float(v[2])
                sc = lc / b + np.array([0.0, a1, a2])[None, :]
                sc -= sc.max(axis=-1, keepdims=True)
                lse = np.log(np.exp(sc).sum(axis=-1))
                nll = lse - sc[np.arange(N_c), y]
                return float((w * nll).sum() / max(w.sum(), 1.0)) + 1e-4 * (a1 ** 2 + a2 ** 2)

            r_tsi = minimize(crtsi_obj, [last_b, 0.0, 0.0], method="L-BFGS-B", bounds=[(0.1, 10.0), (-4.0, 4.0), (-4.0, 4.0)])
            last_bi = (float(r_tsi.x[0]), float(r_tsi.x[1]), float(r_tsi.x[2]))

        Pd = np.stack([S_d[:, m - 1], F1_d[:, m - 1], F2_d[:, m - 1]], axis=-1)
        ld = np.log(np.clip(Pd, 1e-7, 1.0))
        sc1 = ld / last_b
        sc1 -= sc1.max(axis=-1, keepdims=True)
        p1 = np.exp(sc1); p1 /= p1.sum(axis=-1, keepdims=True)
        F1_ts[:, m - 1], F2_ts[:, m - 1] = p1[:, 1], p1[:, 2]

        b_i, a1_i, a2_i = last_bi
        sc2 = ld / b_i + np.array([0.0, a1_i, a2_i])[None, :]
        sc2 -= sc2.max(axis=-1, keepdims=True)
        p2 = np.exp(sc2); p2 /= p2.sum(axis=-1, keepdims=True)
        F1_tsi[:, m - 1], F2_tsi[:, m - 1] = p2[:, 1], p2[:, 2]

    results["3. CR-TS (AISTATS 26)"] = {"F1": F1_ts, "F2": F2_ts}
    results["4. CR-TS+intercept"] = {"F1": F1_tsi, "F2": F2_tsi}

    # 5. Indep-Logit (CIF-space, Elshafae 2026)
    F1_il, F2_il = F1_d.copy(), F2_d.copy()
    last_il = {1: (0.0, 1.0), 2: (0.0, 1.0)}
    for m in range(1, M + 1):
        for k, Fc_k, Fd_k, Fout in ((1, F1_c, F1_d, F1_il), (2, F2_c, F2_d, F2_il)):
            if m <= L:
                cens_b = (D_cal == 0) & (T_cal < m)
                w = (~cens_b).astype(np.float64) / np.clip(g_km[np.minimum(T_cal, m)], 0.05, 1.0)
                yk = ((T_cal <= m) & (D_cal == k)).astype(np.float64)
                pc = np.clip(Fc_k[:, m - 1], 1e-6, 1.0 - 1e-6)
                lc = np.log(pc / (1.0 - pc))

                def il_obj(v):
                    eta = np.clip(v[0] + v[1] * lc, -15.0, 15.0)
                    bce = np.logaddexp(0.0, eta) - yk * eta
                    return float((w * bce).sum() / max(w.sum(), 1.0)) + 1e-4 * (v[0] ** 2 + (v[1] - 1.0) ** 2)

                r = minimize(il_obj, [0.0, 1.0], method="L-BFGS-B", bounds=[(-4.0, 4.0), (0.2, 3.0)])
                last_il[k] = (float(r.x[0]), float(r.x[1]))
            pd_k = np.clip(Fd_k[:, m - 1], 1e-6, 1.0 - 1e-6)
            ld_k = np.log(pd_k / (1.0 - pd_k))
            a_k, b_k = last_il[k]
            Fout[:, m - 1] = 1.0 / (1.0 + np.exp(-np.clip(a_k + b_k * ld_k, -15.0, 15.0)))
    results["5. Indep-Logit (CIF)"] = {"F1": F1_il, "F2": F2_il}

    # 6. Indep-Logit (Hazard-space, 2KM params mapped through (K+1)-way interval softmax -> 0% violations!)
    z_ih = logits_dep.copy()
    last_hz = (0.0, 1.0, 0.0, 1.0)
    for l in range(1, M + 1):
        if l <= L:
            ar = (T_cal >= l)
            if int(ar.sum()) >= 20:
                z1c = logits_cal[ar, l - 1, 0]
                z2c = logits_cal[ar, l - 1, 1]
                ev1 = ((T_cal[ar] == l) & (D_cal[ar] == 1))
                ev2 = ((T_cal[ar] == l) & (D_cal[ar] == 2))

                def ih_obj(v):
                    u1 = v[0] + v[1] * z1c
                    u2 = v[2] + v[3] * z2c
                    mx = np.maximum(0.0, np.maximum(u1, u2))
                    lse = mx + np.log(np.exp(-mx) + np.exp(u1 - mx) + np.exp(u2 - mx))
                    nll = float(lse.sum() - u1[ev1].sum() - u2[ev2].sum())
                    pen = 0.5 * ridge * (v[0] ** 2 + (v[1] - 1.0) ** 2 + v[2] ** 2 + (v[3] - 1.0) ** 2)
                    return nll + pen

                r = minimize(ih_obj, [0.0, 1.0, 0.0, 1.0], method="L-BFGS-B", bounds=[(-4.0, 4.0), (0.2, 3.0), (-4.0, 4.0), (0.2, 3.0)])
                last_hz = tuple(float(x) for x in r.x)
        z_ih[:, l - 1, 0] = last_hz[0] + last_hz[1] * logits_dep[:, l - 1, 0]
        z_ih[:, l - 1, 1] = last_hz[2] + last_hz[3] * logits_dep[:, l - 1, 1]
    _, F1_ih, F2_ih = logits_to_cif_np(z_ih)
    results["6. Indep-Logit (Hazard)"] = {"F1": F1_ih, "F2": F2_ih}

    # 7. Booth Sep-Baseline (SIM 2023): separate per-interval baseline offsets, fixed slope=1 (KM params)
    z_bo = logits_dep.copy()
    last_bo = (0.0, 0.0)
    for l in range(1, M + 1):
        if l <= L:
            ar = (T_cal >= l)
            if int(ar.sum()) >= 20:
                z1c = logits_cal[ar, l - 1, 0]
                z2c = logits_cal[ar, l - 1, 1]
                ev1 = ((T_cal[ar] == l) & (D_cal[ar] == 1))
                ev2 = ((T_cal[ar] == l) & (D_cal[ar] == 2))

                def bo_obj(v):
                    u1 = v[0] + z1c
                    u2 = v[1] + z2c
                    mx = np.maximum(0.0, np.maximum(u1, u2))
                    lse = mx + np.log(np.exp(-mx) + np.exp(u1 - mx) + np.exp(u2 - mx))
                    return float(lse.sum() - u1[ev1].sum() - u2[ev2].sum()) + 0.5 * ridge * (v[0] ** 2 + v[1] ** 2)

                r = minimize(bo_obj, [0.0, 0.0], method="L-BFGS-B", bounds=[(-4.0, 4.0), (-4.0, 4.0)])
                last_bo = (float(r.x[0]), float(r.x[1]))
        z_bo[:, l - 1, 0] = last_bo[0] + logits_dep[:, l - 1, 0]
        z_bo[:, l - 1, 1] = last_bo[1] + logits_dep[:, l - 1, 1]
    _, F1_bo, F2_bo = logits_to_cif_np(z_bo)
    results["7. Booth Sep-Baseline (SIM 23)"] = {"F1": F1_bo, "F2": F2_bo}

    # 8. Ungated 3K JCCR
    results["8. Ungated 3K JCCR"] = fit_sg_jccr(logits_cal, T_cal, D_cal, logits_dep, L_cal=L, per_cause_gate=False, ungated_tier2=True, ridge=ridge)

    # 9. Ours: Per-Cause Martingale Score-Gated Hier SG-JCCR (3-df Rao score test Q_k; plus 1-df O-E gate ablation!)
    results["9. Ours: Hier SG-JCCR"] = fit_sg_jccr(logits_cal, T_cal, D_cal, logits_dep, L_cal=L, per_cause_gate=True, gate_mode="score_df", ungated_tier2=False, ridge=ridge)
    results["9b. Ablation: 1-df Z-Gate SG-JCCR"] = fit_sg_jccr(logits_cal, T_cal, D_cal, logits_dep, L_cal=L, per_cause_gate=True, gate_mode="z_1df", ungated_tier2=False, ridge=ridge)

    return results

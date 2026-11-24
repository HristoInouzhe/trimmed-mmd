"""Exploratory (AoS revision, 2026-11-21): cut-point rules and kernel choice for the sequential betting certificate.

Only :mod:`trimmedmmd.unequal` (unequal trimming fractions, whose keywords ``alpha_y`` and ``zeta`` default to the
symmetric case) imports this module, and no existing default changes. It evaluates, on ONE set of
witness fits per (split, level), three rules for the cut points of the ramp (results/aos/fallback/NOTES.md):

* ``"own"``      -- the paper: (1-alpha)-quantile of the fit values at the X's and alpha-quantile at the Y's, each
                    value computed without the point's own kernel term; skip the batch if the cuts cross;
* ``"fallback"`` -- own-term cuts; if they cross, the plain cuts of the same witness; if those cross too, skip;
* ``"plain"``    -- plain empirical quantiles of the in-sample fit values (own term kept), always.

All three are functions of the earlier batches only, so each is an admissible fitting rule and Theorem thm:betting
holds for it. The variance guess of the fractions uses the fit values the cuts were computed from.

The kernel is ``"gaussian"`` (exp(-|x-y|^2 / (2 bw^2)); with it :func:`fit_batch` reproduces
:func:`trimmedmmd.splitting._sequential_fit` exactly: the same Gram matrices, the same call of
:func:`trimmedmmd.solver.solve_gram` and the same random stream) or ``"laplace"`` (exp(-|x-y| / bw)). Both have
k(x, x) = 1, so the own term of a fit point is its weight over the witness norm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .kernels import gaussian_kernel, sqdist
from .solver import solve_gram
from .splitting import (SEQ_EDGES, pair_statistics, ramp, ramp_cut_points, ramp_prior_var,
                        sequential_horizon)

RULES = ("own", "fallback", "plain")
KERNELS = ("gaussian", "laplace")


def kernel_matrix(A, B, bw: float, kernel: str = "gaussian", same: bool = False) -> np.ndarray:
    """Gram matrix k(A_i, B_j). ``same=True`` (A is B) sets the diagonal distance to exactly 0, so the Laplace
    Gram has k(x, x) = 1 exactly (sqdist leaves rounding residue ~1e-13 on the diagonal, which the square root
    would amplify to ~1e-7); the Gaussian path is the package's gaussian_kernel, untouched."""
    if kernel == "gaussian":
        return gaussian_kernel(A, B, bw)
    if kernel == "laplace":
        d2 = sqdist(A, B)
        if same:
            np.fill_diagonal(d2, 0.0)
        return np.exp(-np.sqrt(d2) / bw)
    raise ValueError(f"unknown kernel {kernel!r}")


@dataclass
class BatchFit:
    """The witness fitted on the batches before batch b, its values on batch b, and both kinds of fit values."""
    norm: float
    fXb: np.ndarray           # witness on the X's / Y's of batch b
    fYb: np.ndarray
    u_own: np.ndarray         # fit values, own term removed (the paper)
    v_own: np.ndarray
    u_plain: np.ndarray       # fit values, own term kept
    v_plain: np.ndarray


def fit_batch(Xf, Yf, Xb, Yb, alpha, bw, rng, kernel="gaussian", alpha_y=None, **kw) -> BatchFit:
    """Fit the trimmed-MMD witness at ``alpha`` on (Xf, Yf) and evaluate it.

    Gaussian: witness_direction + evaluate_direction + fold1_witness_values of the package, operation for
    operation (Gram blocks gaussian_kernel(X1, X1), (Y1, Y1), (X1, Y1); evaluation gaussian_kernel(Z, X1) @ w -
    gaussian_kernel(Z, Y1) @ v over the norm), so the random stream and every value are the package's.
    ``alpha_y`` (default ``alpha``; unequal trimming fractions, :mod:`trimmedmmd.unequal`) trims the Y sample at its
    own level: the caps of the quadratic program are 1/(n (1 - alpha)) and 1/(m (1 - alpha_y)).
    """
    Kxx = kernel_matrix(Xf, Xf, bw, kernel, same=True)
    Kyy = kernel_matrix(Yf, Yf, bw, kernel, same=True)
    Kxy = kernel_matrix(Xf, Yf, bw, kernel)
    res = solve_gram(Kxx, Kyy, Kxy, alpha, rng=rng, alpha_y=alpha_y, **kw)
    w, v, norm = res.w, res.v, float(res.discrepancy)
    del Kxx, Kyy, Kxy

    def f(Z, same_as=None):
        if norm <= 0:
            return np.zeros(len(Z))
        kx = kernel_matrix(Z, Xf, bw, kernel, same=(same_as == "X"))
        ky = kernel_matrix(Z, Yf, bw, kernel, same=(same_as == "Y"))
        return (kx @ w - ky @ v) / norm

    fXb, fYb = f(Xb), f(Yb)
    if norm <= 0:
        zx, zy = np.zeros(len(Xf)), np.zeros(len(Yf))
        return BatchFit(norm, fXb, fYb, zx, zy, zx.copy(), zy.copy())
    u_pl, v_pl = f(Xf, "X"), f(Yf, "Y")
    # k(x, x) = 1 for both kernels: the own term of X_i is w_i / norm, of Y_j is -v_j / norm
    u_own, v_own = u_pl - np.asarray(w) / norm, v_pl + np.asarray(v) / norm
    return BatchFit(norm, fXb, fYb, u_own, v_own, u_pl, v_pl)


def choose_cuts(u_own, v_own, u_pl, v_pl, alpha, rule, cut_rule="linear", alpha_y=None):
    """Cut points of one batch under ``rule``: (s, sp, u, v, source) with source in {"own", "plain", "skip"}; u, v
    are the fit values the cuts came from (for the variance guess). Also returns both raw cut pairs.
    ``cut_rule`` (AoS main runs, default "linear" = every earlier run): the quantile rule of
    :func:`trimmedmmd.splitting.ramp_cut_points` ("order" = genuine empirical quantiles). ``alpha_y`` (default
    ``alpha``): the Y cut is the alpha_y-quantile of the Y fit values."""
    so, spo = ramp_cut_points(u_own, v_own, alpha, rule=cut_rule, alpha_y=alpha_y)
    sp_, spp = ramp_cut_points(u_pl, v_pl, alpha, rule=cut_rule, alpha_y=alpha_y)
    raw = {"own": (float(so), float(spo)), "plain": (float(sp_), float(spp))}
    own_ok, pl_ok = so > spo, sp_ > spp
    if rule == "own":
        pick = "own" if own_ok else "skip"
    elif rule == "plain":
        pick = "plain" if pl_ok else "skip"
    elif rule == "fallback":
        pick = "own" if own_ok else ("plain" if pl_ok else "skip")
    else:
        raise ValueError(rule)
    if pick == "own":
        return so, spo, u_own, v_own, pick, raw
    if pick == "plain":
        return sp_, spp, u_pl, v_pl, pick, raw
    return None, None, None, None, pick, raw


def bet(Zs, pri, alpha, N, delta=0.05, c=0.5, prior_w=20.0, zeta=1.0, return_path=False, v2_min=1e-8):
    """The wealth loop of splitting.sequential_betting_certificate (and scripts/aos_realdata_wealth.py), recording
    the path summary. Zs: list of arrays Z - alpha of the batches bet on; pri: their variance guesses.

    ``zeta`` (default 1, the paper; unequal trimming fractions, :mod:`trimmedmmd.unequal`): the weight of the Y side
    of the pairs, Z = G - zeta mean(H) - alpha >= -(zeta + alpha), so the fractions are capped at
    c / (zeta + alpha) and every factor 1 + lambda Z stays at least 1 - c > 0. ``return_path`` also returns the
    fractions ("lams") and the factors 1 + lambda Z ("factors").

    ``v2_min`` (default 1e-8, every run of the paper) is the floor v^2_min of the running variance of the fractions
    (eq:fractions of the supplement); scripts/aos_explore_*.py pass larger floors."""
    L = math.log(1.0 / delta)
    if not Zs or N == 0:
        out = {"certified": False, "log_final": 0.0, "log_max": 0.0, "log_stopped": 0.0, "stop_index": None,
               "pairs_bet": 0}
        if return_path:
            out.update(lams=[], factors=[])
        return out
    lam_max = c / (zeta + alpha)
    logK, best = 0.0, float("-inf")
    ssd, cnt, mu_sum = 0.0, 0, 0.0
    stopped, stop_index = None, None
    lams, factors = [], []
    for z, pv in zip(Zs, pri):
        for y in z:
            mu = mu_sum / (prior_w + cnt)
            sig2 = max((prior_w * pv + ssd) / (prior_w + cnt), v2_min)
            lam = min(lam_max, math.sqrt(2.0 * L / (N * sig2)))
            if return_path:
                lams.append(float(lam))
                factors.append(float(1.0 + lam * y))
            logK += math.log1p(lam * y)
            best = max(best, logK)
            if stopped is None and logK >= L:
                stopped, stop_index = logK, cnt + 1
            ssd += (y - mu) ** 2
            mu_sum += y
            cnt += 1
    out = {"certified": bool(best >= L), "log_final": float(logK), "log_max": float(best),
           "log_stopped": float(stopped if stopped is not None else logK), "stop_index": stop_index,
           "pairs_bet": int(sum(z.size for z in Zs))}
    if return_path:
        out.update(lams=lams, factors=factors)
    return out


def sequential_all_rules(order, alpha, kernel="gaussian", delta=0.05, edges=SEQ_EDGES, c=0.5, prior_w=20.0,
                         rules=RULES, cut_rule="linear", v2_min=1e-8, **kw):
    """The sequential certificate of Theorem thm:betting at one level under every rule in ``rules``, on one set of
    witness fits (``order`` from splitting.sequential_order; its rng is consumed by the fits exactly as by
    splitting._sequential_fit). Returns {"horizon", "sigma", "batches": [...], rule: {...}}."""
    X, Y = order.X, order.Y
    n, m = len(X), len(Y)
    bx = [int(round(e * n)) for e in edges]
    by = [int(round(e * m)) for e in edges]
    N = sequential_horizon(n, m, edges)
    fits, binfo = [], []
    for b in range(1, len(edges) - 1):
        Xf, Yf = X[:bx[b]], Y[:by[b]]
        Xb, Yb = X[bx[b]:bx[b + 1]], Y[by[b]:by[b + 1]]
        bf = fit_batch(Xf, Yf, Xb, Yb, alpha, order.sigma, order.rng, kernel=kernel, **kw)
        fits.append((bf, len(Xb), len(Yb)))
    out = {"horizon": int(N), "sigma": float(order.sigma), "kernel": kernel}
    if cut_rule != "linear":
        out["cut_rule"] = cut_rule
    for rule in rules:
        Zs, pri, src, cuts = [], [], [], []
        for bf, nb, mb in fits:
            s, sp, u, v, pick, raw = choose_cuts(bf.u_own, bf.v_own, bf.u_plain, bf.v_plain, alpha, rule,
                                                 cut_rule=cut_rule)
            if nb == 0 or mb == 0:
                src.append("empty")
                cuts.append(None)
                continue
            src.append(pick)
            if pick == "skip":
                cuts.append(None)
                continue
            cuts.append([float(s), float(sp)])
            G, H = ramp(bf.fXb, s, sp), ramp(bf.fYb, s, sp)
            Zs.append(pair_statistics(G, H) - alpha)
            pri.append(ramp_prior_var(u, v, s, sp, nb, mb))
        r = bet(Zs, pri, alpha, N, delta=delta, c=c, prior_w=prior_w, v2_min=v2_min)
        r.update(source=src, cuts=cuts)
        out[rule] = r
    for bf, nb, mb in fits:
        _, _, _, _, _, raw = choose_cuts(bf.u_own, bf.v_own, bf.u_plain, bf.v_plain, alpha, "own",
                                         cut_rule=cut_rule)
        binfo.append({"norm": bf.norm, "own_cut": list(raw["own"]), "plain_cut": list(raw["plain"]),
                      "own_crossed": bool(not raw["own"][0] > raw["own"][1]),
                      "plain_crossed": bool(not raw["plain"][0] > raw["plain"][1])})
    out["batches"] = binfo
    return out

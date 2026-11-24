"""Candidate thresholds for the analytic test of H_{0,alpha}: TV(P,Q) <= alpha.

All thresholds here are deterministic functions of (n, m, alpha, delta) and of
three kernel constants that are known before the data are seen:

  kappa  = sup_x sqrt(k(x,x))
  Dphi   = sup_{x,x'} ||phi(x) - phi(x')||_H      (<= 2 kappa always;
           = sqrt(2) for the Gaussian, Laplace, IMQ kernels, k>=0, k(x,x)=1)
  v2     = an upper bound on tr Cov_R(phi(X)) valid for every law R
           (<= min(kappa^2, Dphi^2/2); = 1 for the Gaussian kernel)
  rho    = an upper bound on sup_x ||phi(x) - mu_R|| valid for every R
           (<= Dphi, since mu_R lies in the closed convex hull of phi(X))

tau_old : the paper's r_n(alpha, delta/2) + r_m(alpha, delta/2)          (Thm 4.2)
tau_A   : one joint one-sided two-sample supremum                       (NOTE Thm A)
tau_P   : plug-in common trimming, no supremum over the unit ball       (NOTE Thm P)
tau_new : min(tau_A, tau_P), a deterministic choice for every alpha     (NOTE Cor C)

Every function is nondecreasing in alpha (checked numerically in
`check_monotone`), which is what makes one event serve every alpha.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq, minimize_scalar

GAUSS = dict(kappa=1.0, Dphi=math.sqrt(2.0), v2=1.0, rho=math.sqrt(2.0))


# --------------------------------------------------------------------------
# the paper's threshold
# --------------------------------------------------------------------------

def r_old(n, alpha, delta, kappa=1.0):
    return kappa / (1 - alpha) * (4 / math.sqrt(n) + math.sqrt(2 * math.log(2 / delta) / n))


def tau_old(n, m, alpha, delta, kappa=1.0, **_):
    return r_old(n, alpha, delta / 2, kappa) + r_old(m, alpha, delta / 2, kappa)


# --------------------------------------------------------------------------
# Theorem A: one joint one-sided supremum
# --------------------------------------------------------------------------

@lru_cache(maxsize=None)
def rademacher_abs_mean(n: int) -> float:
    """E|sigma_1 + ... + sigma_n| / n, exactly:
    E|S_n| = n * C(n-1, floor((n-1)/2)) / 2^(n-1)."""
    k = (n - 1) // 2
    logv = (math.log(n) + math.lgamma(n) - math.lgamma(k + 1)
            - math.lgamma(n - k) - (n - 1) * math.log(2.0))
    return math.exp(logv) / n


def tau_A(n, m, alpha, delta, kappa=1.0, Dphi=None, **_):
    Dphi = 2 * kappa if Dphi is None else Dphi
    beta = 1.0 / (1.0 - alpha)
    s = 1.0 / n + 1.0 / m
    mean = 2 * kappa * math.sqrt(s) + 2 * kappa * (rademacher_abs_mean(n)
                                                   + rademacher_abs_mean(m))
    dev = Dphi * math.sqrt(s * math.log(1.0 / delta) / 2.0)
    return beta * (mean + dev)


# --------------------------------------------------------------------------
# Theorem P: plug-in common trimming
# --------------------------------------------------------------------------

def _mcd_tail(t, mean, c2sum):
    """McDiarmid upper tail P(Z >= t) with E Z <= mean, sum of squared
    bounded differences c2sum."""
    if t <= mean:
        return 1.0
    return math.exp(-2.0 * (t - mean) ** 2 / c2sum)


def _pinelis_tail(t, a, b2):
    """Pinelis (1994, Thm 3.4, Hilbert space D=1):
    P(||sum|| >= t) <= 2 exp(t/a - (t/a + b2/a^2) log(1 + t a / b2)),
    for independent mean-zero summands with ||d_j|| <= a and
    sum E||d_j||^2 <= b2."""
    if t <= 0:
        return 1.0
    u = t / a
    val = u - (u + b2 / a ** 2) * math.log1p(t * a / b2)
    return min(1.0, 2.0 * math.exp(val))


def _hilbert_tail(t, beta, s, cmax, kappa, Dphi, v2, rho, which=("mcd", "pin")):
    """min of the valid tail bounds for ||sum of independent mean-zero
    plug-in vectors||, whose squared-norm sum is <= beta v2 s, bounded
    differences beta Dphi * (1/n_k), and norms <= beta rho * (1/n_k)."""
    out = 1.0
    if "mcd" in which:
        out = min(out, _mcd_tail(t, math.sqrt(beta * v2 * s), beta ** 2 * Dphi ** 2 * s))
    if "pin" in which:
        out = min(out, _pinelis_tail(t, beta * rho * cmax, beta * v2 * s))
    return out


def _two_point_log_mgf(lam, beta, k):
    """log of the two-point bound on E exp(lam * (1 - a(X))/k) for
    a in [0, beta] with E a = 1 (Hoeffding 1963, convexity of exp)."""
    s = lam / k
    p1 = 1.0 / beta
    # log( p1 e^{s(1-beta)} + (1-p1) e^{s} ) computed stably
    a1 = math.log(p1) + s * (1 - beta)
    if beta > 1:
        a2 = math.log(1 - p1) + s
        mx = max(a1, a2)
        return mx + math.log(math.exp(a1 - mx) + math.exp(a2 - mx))
    return a1


def _chernoff_tail(t, beta, sizes):
    """P( sum_{k in sizes} (1 - S_k) >= t ), S_k the mean of k i.i.d.
    a(X) in [0, beta] with mean one: inf_lam e^{-lam t} prod MGF."""
    if t <= 0:
        return 1.0
    if beta <= 1.0 + 1e-15:
        return 0.0

    def f(loglam):
        lam = math.exp(loglam)
        return -lam * t + sum(k * _two_point_log_mgf(lam, beta, k) for k in sizes)

    res = minimize_scalar(f, bounds=(-10, math.log(1e6 * max(sizes))), method="bounded",
                          options=dict(xatol=1e-6))
    return min(1.0, math.exp(min(res.fun, 0.0)))


def _root(fun, target, hi=10.0):
    """smallest t with fun(t) <= target, fun nonincreasing."""
    lo = 0.0
    if fun(hi) > target:
        return float("inf")
    # +1e-6: the bracketing root finder returns t within xtol of the root;
    # the margin makes the returned value sit on the valid side.
    return brentq(lambda t: fun(t) - target, lo, hi, xtol=1e-9) + 1e-6


def hilbert_quantile(n, m, beta, delta1, kappa, Dphi, v2, rho, three_events=True,
                     which=("mcd", "pin")):
    """t_1: P(M > t_1) <= delta1 with M = max(||A-B||, ||A||, ||B||)."""
    s = 1 / n + 1 / m
    cmax = max(1 / n, 1 / m)

    def tail(t):
        p = _hilbert_tail(t, beta, s, cmax, kappa, Dphi, v2, rho, which)
        if three_events:
            p += _hilbert_tail(t, beta, 1 / n, 1 / n, kappa, Dphi, v2, rho, which)
            p += _hilbert_tail(t, beta, 1 / m, 1 / m, kappa, Dphi, v2, rho, which)
        return p

    return _root(tail, delta1)


def repair_quantile(n, m, beta, delta2):
    """t_2: P((1-S_X)_+ + (1-S_Y)_+ > t_2) <= delta2, via
    {c_X + c_Y > t} subset {1-S_X > t} u {1-S_Y > t} u {2-S_X-S_Y > t}."""
    if beta <= 1.0 + 1e-15:
        return 0.0

    def tail(t):
        return (_chernoff_tail(t, beta, (n,)) + _chernoff_tail(t, beta, (m,))
                + _chernoff_tail(t, beta, (n, m)))

    return _root(tail, delta2, hi=1.0 + 1e-9)


SPLITS = (0.5, 0.6, 0.7, 0.8, 0.9)   # fixed in advance: delta_1 = f * delta


def tau_P(n, m, alpha, delta, kappa=1.0, Dphi=None, v2=None, rho=None,
          splits=SPLITS, which=("mcd", "pin"), detail=False):
    Dphi = 2 * kappa if Dphi is None else Dphi
    v2 = min(kappa ** 2, Dphi ** 2 / 2) if v2 is None else v2
    rho = Dphi if rho is None else rho
    beta = 1.0 / (1.0 - alpha)
    if alpha <= 0.0:
        t1 = hilbert_quantile(n, m, 1.0, delta, kappa, Dphi, v2, rho,
                              three_events=False, which=which)
        return (t1, dict(t1=t1, t2=0.0, split=1.0)) if detail else t1
    best = (float("inf"), None)
    for f in splits:
        t1 = hilbert_quantile(n, m, beta, f * delta, kappa, Dphi, v2, rho, which=which)
        t2 = repair_quantile(n, m, beta, (1 - f) * delta)
        val = t1 + rho * t2
        if val < best[0]:
            best = (val, dict(t1=t1, t2=t2, split=f))
    return best if detail else best[0]


def tau_P_coupled(n, m, alpha, delta, kappa=1.0, Dphi=None, grid=20, **kw):
    """Theorem P with the coupled constants of a kernel with k >= 0 and
    k(x,x) = kappa^2 (Gaussian, Laplace, IMQ): for the unknown common
    trimming R with s = ||mu_R||^2 in [0, kappa^2],
        tr Cov_R(phi) = kappa^2 - s,   sup_x ||phi(x) - mu_R||^2 <= kappa^2 + s.
    Each tail bound is nondecreasing in (v2, rho), so on a cell
    [s_j, s_{j+1}] tau(s) <= tau(v2 = kappa^2 - s_j, rho^2 = kappa^2 + s_{j+1});
    the max over cells is a rigorous bound on sup_s tau(s)."""
    Dphi = math.sqrt(2) * kappa if Dphi is None else Dphi
    if alpha <= 0:
        # rho does not enter the alpha = 0 McDiarmid bound; Pinelis uses rho
        pass
    ss = np.linspace(0, kappa ** 2, grid + 1)
    vals = []
    for j in range(grid):
        v2 = kappa ** 2 - ss[j]
        rho = min(math.sqrt(kappa ** 2 + ss[j + 1]), Dphi)
        vals.append(tau_P(n, m, alpha, delta, kappa=kappa, Dphi=Dphi, v2=v2,
                          rho=rho, **kw))
    return max(vals)


def tau_new(n, m, alpha, delta, coupled=False, **kw):
    cands = [tau_A(n, m, alpha, delta, **kw), tau_P(n, m, alpha, delta, **kw)]
    if coupled:
        cands.append(tau_P_coupled(n, m, alpha, delta, **kw))
    return min(cands)


# the closed form of Theorem P (Corollary P1 of the note)
def tau_P_closed(n, m, alpha, delta, kappa=1.0, Dphi=None, v2=None, rho=None, **_):
    """alpha = 0:  sqrt(v2 s) + Dphi sqrt(s log(1/delta)/2);
    alpha > 0:  T1 + rho * T2 with delta/2 for each part,
      T1 = sqrt(beta v2 s) + beta Dphi sqrt(s log(6/delta)/2)       (McDiarmid,
           three events max(||A-B||,||A||,||B||), delta/6 each),
      T2 = sqrt(2 (beta-1) s log(6/delta)) + cmax log(6/delta)/3   (Bernstein,
           three events, delta/6 each; BLM Thm 2.10 with c = b/3),
    s = 1/n + 1/m, cmax = max(1/n, 1/m)."""
    Dphi = 2 * kappa if Dphi is None else Dphi
    v2 = min(kappa ** 2, Dphi ** 2 / 2) if v2 is None else v2
    rho = Dphi if rho is None else rho
    beta = 1 / (1 - alpha)
    s = 1 / n + 1 / m
    if alpha <= 0:
        return math.sqrt(v2 * s) + Dphi * math.sqrt(s * math.log(1 / delta) / 2)
    L = math.log(6 / delta)
    T1 = math.sqrt(beta * v2 * s) + beta * Dphi * math.sqrt(s * L / 2)
    T2 = math.sqrt(2 * (beta - 1) * s * L) + max(1 / n, 1 / m) * L / 3
    return T1 + rho * T2


ALL = dict(old=tau_old, A=tau_A, P=tau_P, P_closed=tau_P_closed,
           P_coupled=tau_P_coupled, new=tau_new)


def check_monotone(n=300, m=300, delta=0.05, consts=GAUSS):
    grid = np.linspace(0, 0.9, 91)
    out = {}
    for name, fn in ALL.items():
        vals = np.array([fn(n, m, a, delta, **consts) for a in grid])
        out[name] = bool(np.all(np.diff(vals) >= -1e-12))
    return out


if __name__ == "__main__":
    for a in (0.0, 0.05, 0.1, 0.2, 0.5):
        row = {k: round(fn(300, 300, a, 0.05, **GAUSS), 4) for k, fn in ALL.items()}
        print(a, row, tau_P(300, 300, a, 0.05, detail=True, **GAUSS)[1])
    print(check_monotone())

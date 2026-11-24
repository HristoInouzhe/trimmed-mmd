"""The analytic threshold and the test of H_{0,alpha}: TV(P,Q) <= alpha."""

from __future__ import annotations

import numpy as np


def radius(n: int, alpha: float, delta: float, kappa: float = 1.0) -> float:
    """r_n(alpha, delta) of the paper (`thm:concentration`):

        kappa / (1 - alpha) * [ 4 / sqrt(n) + sqrt(2 log(2/delta) / n) ].
    """
    return (kappa / (1.0 - alpha)) * (
        4.0 / np.sqrt(n) + np.sqrt(2.0 * np.log(2.0 / delta) / n)
    )


def tau(n: int, m: int, alpha: float, delta: float, kappa: float = 1.0) -> float:
    """tau_{n,m}(alpha, delta) = r_n(alpha, delta/2) + r_m(alpha, delta/2)."""
    return radius(n, alpha, delta / 2.0, kappa) + radius(m, alpha, delta / 2.0, kappa)


def reject(d_hat: float, n: int, m: int, alpha: float, delta: float,
           kappa: float = 1.0) -> bool:
    """The test of `eq:test`: reject H_{0,alpha} if D-hat exceeds tau."""
    return bool(d_hat > tau(n, m, alpha, delta, kappa))


def leading_term_share(n: int, m: int, delta: float) -> float:
    """Fraction of tau contributed by the 4/sqrt(n) Rademacher term.

    Reported because it turns out to dominate: the threshold is large not
    because the confidence level is demanding but because the worst-case
    complexity term is.
    """
    a = 4.0 / np.sqrt(n) + 4.0 / np.sqrt(m)
    b = (np.sqrt(2.0 * np.log(4.0 / delta) / n)
         + np.sqrt(2.0 * np.log(4.0 / delta) / m))
    return float(a / (a + b))


def mmd_radius(n: int, m: int, delta: float, kappa: float = 1.0) -> float:
    """Deviation radius for the ordinary (untrimmed) MMD statistic.

    The classical uniform bound for the biased two-sample statistic, in the
    ONE-SIDED form (Gretton et al. 2012 state the two-sided version, whose
    confidence term carries log(2/delta)): with probability at least
    1 - delta,

        MMD_hat - MMD <= 2 kappa (1/sqrt(n) + 1/sqrt(m))
                         + kappa sqrt(2 log(1/delta) (n+m) / (nm)).

    One-sidedness is what the screen of eq. (17) needs -- it only ever asks
    whether the observed MMD is too large -- so the cheaper constant is the
    correct one here. Use log(2/delta) if a two-sided statement is wanted.

    Worth comparing term by term with :func:`tau`: for n = m the leading
    term here is 4 kappa / sqrt(n), against 8 kappa / sqrt(n) for the
    trimmed-region threshold at alpha = 0. The screen is a factor of about
    two cheaper in sample size, which is the price of tracking a whole
    region rather than a single mean embedding.
    """
    return (2.0 * kappa * (1.0 / np.sqrt(n) + 1.0 / np.sqrt(m))
            + kappa * np.sqrt(2.0 * np.log(1.0 / delta) * (n + m) / (n * m)))


def reject_mmd_screen(mmd_hat: float, n: int, m: int, alpha: float,
                      delta: float, kappa: float = 1.0) -> bool:
    """The necessary-condition screen of `eq:necessary`.

    Under the contamination model MMD_k(P,Q) <= 2 alpha kappa, so an
    observed MMD significantly above that forces TV(P,Q) > alpha. Cheap --
    no trimming optimization at all -- but it can only ever detect
    alternatives whose MMD exceeds 2 alpha kappa, which is a real
    restriction because MMD does not lower-bound total variation.
    """
    return bool(mmd_hat > 2.0 * alpha * kappa + mmd_radius(n, m, delta, kappa))


# ===========================================================================
# Revision v2: the plug-in threshold (Theorem P) and the two-sided Theorem A
# ===========================================================================
#
# Ported from explore/sharper_threshold/thresholds.py (research note NOTE.md,
# Sections 2-4). Everything above this line is unchanged; the functions below
# are new and are only used when a caller asks for them (``threshold="plugin"``
# in :mod:`trimmedmmd.estimate`, ``--threshold plugin`` in the scripts).
#
# All thresholds are deterministic functions of (n, m, alpha, delta) and of
# four kernel constants known before the data are seen (NOTE Section 1):
#
#   kappa = sup_x k(x,x)^{1/2}
#   Dphi  = sup_{x,x'} ||phi(x) - phi(x')||   (<= 2 kappa; = sqrt(2) kappa for
#           a kernel with k >= 0 and k(x,x) = kappa^2 on an unbounded domain)
#   v2    = a bound on tr Cov_R(phi) for EVERY law R  (min(kappa^2, Dphi^2/2))
#   rho   = a bound on sup_x ||phi(x) - mu_R|| for every R  (Dphi works)
#
# Gaussian (and Laplace, IMQ) kernel: kappa = 1, Dphi = sqrt 2, v2 = 1,
# rho = sqrt 2, and the coupled-constant refinement applies.
#
# The test of revision v2 rejects H_{0,alpha} iff D-hat_alpha > tau_new(alpha),
# tau_new = min of the valid thresholds (Corollary C). tau_A (Theorem A) is the
# two-sided tool for the estimator's lower side (consistency, rate); the test
# does not need it, but it is one of the candidates in the minimum.

import math as _math
from dataclasses import dataclass as _dataclass
from functools import lru_cache as _lru_cache


@_dataclass(frozen=True)
class KernelConstants:
    """Kernel constants entering the v2 thresholds (NOTE Section 1).

    ``coupled`` marks a kernel with ``k >= 0`` and constant diagonal
    ``k(x, x) = kappa^2`` (Gaussian, Laplace, inverse multiquadric), for which
    Lemma P1(iv) couples the variance and the repair range and the
    coupled-constant threshold is valid. It must be False for kernels that can
    take negative values, e.g. a random-feature kernel
    ``k_hat(x, y) = z(x)'z(y)``.
    """

    kappa: float = 1.0
    Dphi: float = _math.sqrt(2.0)
    v2: float = 1.0
    rho: float = _math.sqrt(2.0)
    coupled: bool = True

    @classmethod
    def gaussian(cls, kappa: float = 1.0) -> "KernelConstants":
        """k >= 0 with k(x, x) = kappa^2 (Gaussian, Laplace, IMQ; exact Gram)."""
        return cls(kappa=kappa, Dphi=_math.sqrt(2.0) * kappa, v2=kappa ** 2,
                   rho=_math.sqrt(2.0) * kappa, coupled=True)

    @classmethod
    def bounded(cls, kappa: float = 1.0) -> "KernelConstants":
        """Any kernel with sup_x k(x, x) <= kappa^2 (no sign condition).

        Dphi = 2 kappa, v2 = min(kappa^2, Dphi^2 / 2) = kappa^2, rho = Dphi,
        no coupling. This is the safe choice for random-feature kernels (the
        cos/sin map has k_hat(x, x) = 1 exactly but k_hat can be negative; the
        phased map needs ``kappa = kernels.realized_kappa(Z)``).
        """
        return cls(kappa=kappa, Dphi=2.0 * kappa, v2=kappa ** 2,
                   rho=2.0 * kappa, coupled=False)


GAUSSIAN = KernelConstants.gaussian(1.0)


def kernel_constants(kind: str = "gaussian", kappa: float = 1.0) -> KernelConstants:
    """``"gaussian"`` (also Laplace/IMQ: k >= 0, constant diagonal) or
    ``"bounded"`` (any kernel with k(x,x) <= kappa^2, e.g. random features)."""
    if kind in ("gaussian", "laplace", "imq", "nonneg"):
        return KernelConstants.gaussian(kappa)
    if kind in ("bounded", "general", "rff", "features"):
        return KernelConstants.bounded(kappa)
    raise ValueError(f"unknown kernel kind {kind!r}")


def _consts(kc) -> KernelConstants:
    if kc is None:
        return GAUSSIAN
    if isinstance(kc, KernelConstants):
        return kc
    if isinstance(kc, str):
        return kernel_constants(kc)
    if isinstance(kc, dict):
        return KernelConstants(**kc)
    raise TypeError("kernel_consts must be a KernelConstants, a dict or a kind string")


# --------------------------------------------------------------------------
# Theorem A: one joint one-sided two-sample supremum (NOTE Section 2)
# --------------------------------------------------------------------------

@_lru_cache(maxsize=None)
def rademacher_abs_mean(n: int) -> float:
    """e_n = E|sigma_1 + ... + sigma_n| / n, exactly:
    E|S_n| = n C(n-1, floor((n-1)/2)) / 2^(n-1)."""
    k = (n - 1) // 2
    logv = (_math.log(n) + _math.lgamma(n) - _math.lgamma(k + 1)
            - _math.lgamma(n - k) - (n - 1) * _math.log(2.0))
    return _math.exp(logv) / n


def tau_A(n: int, m: int, alpha: float, delta: float, kernel_consts=None) -> float:
    """Theorem A, ONE side at level delta:

        beta [ 2 kappa sqrt(s) + 2 kappa (e_n + e_m) + Dphi sqrt(s log(1/delta)/2) ],

    s = 1/n + 1/m, beta = 1/(1-alpha). On an event E+ of probability
    >= 1 - delta, D-hat_alpha <= D_alpha + tau_A(alpha, delta) for every alpha;
    on E-, D-hat_alpha >= D_alpha - tau_A(alpha, delta). Always <= :func:`tau`.
    """
    kc = _consts(kernel_consts)
    beta = 1.0 / (1.0 - alpha)
    s = 1.0 / n + 1.0 / m
    mean = 2 * kc.kappa * _math.sqrt(s) + 2 * kc.kappa * (rademacher_abs_mean(n)
                                                          + rademacher_abs_mean(m))
    dev = kc.Dphi * _math.sqrt(s * _math.log(1.0 / delta) / 2.0)
    return beta * (mean + dev)


def tau_twosided(n: int, m: int, alpha: float, delta: float, kernel_consts=None) -> float:
    """Two-sided radius of Theorem A: with probability >= 1 - delta,
    |D-hat_alpha - D_alpha| <= tau_A(alpha, delta/2) for EVERY alpha.

    This is the tool for the estimator's lower side (consistency, rate,
    confidence set, upper end); the test uses :func:`tau_plugin`.
    """
    return tau_A(n, m, alpha, delta / 2.0, kernel_consts)


# --------------------------------------------------------------------------
# Theorem P: the plug-in common trimming (NOTE Section 3)
# --------------------------------------------------------------------------

def tau_plugin_closed(n: int, m: int, alpha: float, delta: float,
                      kernel_consts=None) -> float:
    """Theorem P, closed form (NOTE Section 3.2).

    alpha = 0:  v sqrt(s) + Dphi sqrt(s log(1/delta) / 2);
    alpha > 0:  T1 + rho T2 with L = log(6/delta),
        T1 = sqrt(beta v2 s) + beta Dphi sqrt(s L / 2)        (McDiarmid, three
             events ||A-B||, ||A||, ||B||, delta/6 each),
        T2 = sqrt(2 (beta - 1) s L) + c L / 3                  (Bernstein, three
             events, delta/6 each), c = max(1/n, 1/m).
    """
    kc = _consts(kernel_consts)
    beta = 1.0 / (1.0 - alpha)
    s = 1.0 / n + 1.0 / m
    if alpha <= 0:
        return _math.sqrt(kc.v2 * s) + kc.Dphi * _math.sqrt(s * _math.log(1.0 / delta) / 2.0)
    L = _math.log(6.0 / delta)
    T1 = _math.sqrt(beta * kc.v2 * s) + beta * kc.Dphi * _math.sqrt(s * L / 2.0)
    T2 = _math.sqrt(2.0 * (beta - 1.0) * s * L) + max(1.0 / n, 1.0 / m) * L / 3.0
    return T1 + kc.rho * T2


def _mcd_tail(t, mean, c2sum):
    """McDiarmid upper tail P(Z >= t) with E Z <= mean and sum of squared
    bounded differences c2sum."""
    if t <= mean:
        return 1.0
    return _math.exp(-2.0 * (t - mean) ** 2 / c2sum)


def _pinelis_tail(t, a, b2):
    """Pinelis (1994, Thm 3.4; Hilbert space, D = 1):
    P(||sum|| >= t) <= 2 exp(t/a - (t/a + b2/a^2) log(1 + t a / b2)) for
    independent mean-zero summands with ||d_j|| <= a and sum E||d_j||^2 <= b2."""
    if t <= 0:
        return 1.0
    u = t / a
    val = u - (u + b2 / a ** 2) * _math.log1p(t * a / b2)
    return min(1.0, 2.0 * _math.exp(val))


def _hilbert_tail(t, beta, s, cmax, kc, which=("mcd", "pin")):
    """min of the valid tail bounds for the norm of a sum of independent
    mean-zero plug-in vectors: squared-norm sum <= beta v2 s, bounded
    differences beta Dphi / n_k, norms <= beta rho / n_k."""
    out = 1.0
    if "mcd" in which:
        out = min(out, _mcd_tail(t, _math.sqrt(beta * kc.v2 * s), beta ** 2 * kc.Dphi ** 2 * s))
    if "pin" in which:
        out = min(out, _pinelis_tail(t, beta * kc.rho * cmax, beta * kc.v2 * s))
    return out


def _two_point_log_mgf(lam, beta, k):
    """log of the two-point bound on E exp(lam (1 - a(X)) / k) for a in
    [0, beta] with E a = 1 (Hoeffding 1963, convexity of exp)."""
    s = lam / k
    p1 = 1.0 / beta
    a1 = _math.log(p1) + s * (1 - beta)
    if beta > 1:
        a2 = _math.log(1 - p1) + s
        mx = max(a1, a2)
        return mx + _math.log(_math.exp(a1 - mx) + _math.exp(a2 - mx))
    return a1


def _chernoff_tail(t, beta, sizes):
    """P( sum_{k in sizes} (1 - S_k) >= t ), S_k the mean of k i.i.d. a(X) in
    [0, beta] with mean one: inf_lam e^{-lam t} prod MGF. Any lam gives a valid
    bound, so an inexact numerical minimizer only makes the bound larger."""
    from scipy.optimize import minimize_scalar
    if t <= 0:
        return 1.0
    if beta <= 1.0 + 1e-15:
        return 0.0

    def f(loglam):
        lam = _math.exp(loglam)
        return -lam * t + sum(k * _two_point_log_mgf(lam, beta, k) for k in sizes)

    res = minimize_scalar(f, bounds=(-10, _math.log(1e6 * max(sizes))), method="bounded",
                          options=dict(xatol=1e-6))
    return min(1.0, _math.exp(min(res.fun, 0.0)))


def _root(fun, target, hi=10.0, hi_max=1e4):
    """Smallest t >= 0 with fun(t) <= target, fun nonincreasing (the text's
    "smallest such value in [0, infinity)"). The bracket starts at ``hi`` (the
    research code's value, so every finite research root is reproduced) and is
    doubled while fun(hi) > target, up to ``hi_max``; only then is the root
    reported as infinite (no rejection possible at that delta share). The
    bracketing root finder returns t within xtol of the root; the +1e-6 margin
    puts the returned value on the valid side, so the implemented threshold is
    never below the one defined in the text."""
    from scipy.optimize import brentq
    while fun(hi) > target:
        hi *= 2.0
        if hi > hi_max:
            return float("inf")
    return brentq(lambda t: fun(t) - target, 0.0, hi, xtol=1e-9) + 1e-6


def plugin_hilbert_quantile(n, m, beta, delta1, kernel_consts=None, three_events=True,
                            which=("mcd", "pin")) -> float:
    """t_1: P(M > t_1) <= delta1, M = max(||A-B||, ||A||, ||B||) (or ||A-B||
    alone when ``three_events`` is False, the alpha = 0 case)."""
    kc = _consts(kernel_consts)
    s = 1.0 / n + 1.0 / m
    cmax = max(1.0 / n, 1.0 / m)

    def tail(t):
        p = _hilbert_tail(t, beta, s, cmax, kc, which)
        if three_events:
            p += _hilbert_tail(t, beta, 1.0 / n, 1.0 / n, kc, which)
            p += _hilbert_tail(t, beta, 1.0 / m, 1.0 / m, kc, which)
        return p

    return _root(tail, delta1)


def plugin_repair_quantile(n, m, beta, delta2) -> float:
    """t_2: P((1-S_X)_+ + (1-S_Y)_+ > t_2) <= delta2, via
    {c_X + c_Y > t} in {1-S_X > t} u {1-S_Y > t} u {2-S_X-S_Y > t}.

    Depends on (n, m, beta, delta2) only, not on the kernel constants, so it
    is cached (the coupled threshold evaluates it once per cell otherwise)."""
    return _plugin_repair_quantile_cached(int(n), int(m), float(beta), float(delta2))


@_lru_cache(maxsize=65536)
def _plugin_repair_quantile_cached(n, m, beta, delta2) -> float:
    if beta <= 1.0 + 1e-15:
        return 0.0

    def tail(t):
        return (_chernoff_tail(t, beta, (n,)) + _chernoff_tail(t, beta, (m,))
                + _chernoff_tail(t, beta, (n, m)))

    return _root(tail, delta2, hi=1.0 + 1e-9)


PLUGIN_SPLITS = (0.5, 0.6, 0.7, 0.8, 0.9)   # delta_1 = f delta, fixed in advance


@_lru_cache(maxsize=65536)
def _tau_plugin_numeric_cached(n, m, alpha, delta, kc, splits, which):
    beta = 1.0 / (1.0 - alpha)
    if alpha <= 0.0:
        t1 = plugin_hilbert_quantile(n, m, 1.0, delta, kc, three_events=False, which=which)
        return t1, dict(t1=t1, t2=0.0, split=1.0)
    best = (float("inf"), dict(t1=float("inf"), t2=float("inf"), split=float("nan")))
    for f in splits:
        t1 = plugin_hilbert_quantile(n, m, beta, f * delta, kc, which=which)
        t2 = plugin_repair_quantile(n, m, beta, (1 - f) * delta)
        val = t1 + kc.rho * t2
        if val < best[0]:
            best = (val, dict(t1=t1, t2=t2, split=f))
    return best


def tau_plugin_numeric(n: int, m: int, alpha: float, delta: float, kernel_consts=None,
                       splits=PLUGIN_SPLITS, which=("mcd", "pin"), detail=False):
    """Theorem P with numerically inverted tails, tau_{P'} (NOTE Section 3.3).

    Hilbert part: min of McDiarmid and Pinelis (1994, Thm 3.4) for each of
    the three events; repair part: two-point Chernoff for each of the three
    events; delta_1 = f delta over the fixed grid ``splits``. At alpha = 0 the
    whole delta goes to ||A - B||. ``detail=True`` also returns the pieces.
    """
    kc = _consts(kernel_consts)
    val, det = _tau_plugin_numeric_cached(int(n), int(m), float(alpha), float(delta), kc,
                                          tuple(splits), tuple(which))
    return (val, dict(det)) if detail else val


@_lru_cache(maxsize=65536)
def _tau_plugin_coupled_cached(n, m, alpha, delta, kappa, Dphi, grid, splits, which):
    import numpy as _np
    us = _np.linspace(0, kappa ** 2, grid + 1)
    vals = []
    for j in range(grid):
        kc = KernelConstants(kappa=kappa, Dphi=Dphi, v2=kappa ** 2 - us[j],
                             rho=min(_math.sqrt(kappa ** 2 + us[j + 1]), Dphi), coupled=True)
        vals.append(_tau_plugin_numeric_cached(n, m, alpha, delta, kc, splits, which)[0])
    return max(vals)


def tau_plugin_coupled(n: int, m: int, alpha: float, delta: float, kernel_consts=None,
                       grid: int = 20, splits=PLUGIN_SPLITS, which=("mcd", "pin")) -> float:
    """tau_{P'} with the coupled constants of a kernel with k >= 0 and
    k(x, x) = kappa^2 (NOTE Section 3.3, "Coupled constants").

    With u = ||mu_R||^2 in [0, kappa^2] unknown: tr Cov_R(phi) = kappa^2 - u and
    sup_x ||phi(x) - mu_R||^2 <= kappa^2 + u. Each tail is nondecreasing in
    (v2, rho), so on the cell [u_j, u_{j+1}] the threshold is at most its value
    at (v2 = kappa^2 - u_j, rho = sqrt(kappa^2 + u_{j+1}) ^ Dphi); the max over
    ``grid`` cells bounds the supremum over u rigorously.
    """
    kc = _consts(kernel_consts)
    if not kc.coupled:
        raise ValueError("coupled constants need a kernel with k >= 0 and constant "
                         "diagonal (KernelConstants.coupled is False)")
    return _tau_plugin_coupled_cached(int(n), int(m), float(alpha), float(delta), kc.kappa,
                                      kc.Dphi, int(grid), tuple(splits), tuple(which))


PLUGIN_VARIANTS = ("closed", "numeric", "coupled", "sharp", "sharp_coupled", "auto", "closed_A")


def tau_plugin(n: int, m: int, alpha: float, delta: float, kernel_consts=None,
               variant: str = "auto") -> float:
    """The revision-v2 test threshold.

    ``variant``:
      * ``"closed"``  -- Theorem P, closed form (:func:`tau_plugin_closed`);
      * ``"numeric"`` -- tau_{P'}, numerically inverted tails;
      * ``"coupled"`` -- tau_{P'} with coupled constants (k >= 0, constant diagonal);
      * ``"sharp"``   -- Corollary C without coupling: min(tau_A, closed, numeric);
      * ``"sharp_coupled"`` -- Corollary C: min(tau_A, closed, numeric, coupled);
      * ``"auto"`` (default) -- ``"sharp_coupled"`` when ``kernel_consts.coupled``
        holds (the Gaussian default), else ``"sharp"``;
      * ``"closed_A"`` (AoS restructure, 2026-11-21) -- min(tau^P closed form, tau^A):
        the threshold of the restructured paper, without the sharper tau^{P'}
        (:func:`tau_aos`).

    tau_new = the ``"auto"`` value is valid in finite samples, uniformly over
    H_{0,alpha} = {TV(P,Q) <= alpha}, and nondecreasing in alpha, so one event
    of probability >= 1 - delta covers every alpha (NOTE Lemma 1). The minimum
    is a deterministic choice for each alpha, so it costs no extra delta.
    """
    kc = _consts(kernel_consts)
    if variant == "auto":
        variant = "sharp_coupled" if kc.coupled else "sharp"
    if variant == "closed":
        return tau_plugin_closed(n, m, alpha, delta, kc)
    if variant == "closed_A":
        return float(min(tau_plugin_closed(n, m, alpha, delta, kc), tau_A(n, m, alpha, delta, kc)))
    if variant == "numeric":
        return tau_plugin_numeric(n, m, alpha, delta, kc)
    if variant == "coupled":
        return tau_plugin_coupled(n, m, alpha, delta, kc)
    if variant in ("sharp", "sharp_coupled"):
        cands = [tau_A(n, m, alpha, delta, kc), tau_plugin_closed(n, m, alpha, delta, kc),
                 tau_plugin_numeric(n, m, alpha, delta, kc)]
        if variant == "sharp_coupled":
            cands.append(tau_plugin_coupled(n, m, alpha, delta, kc))
        return float(min(cands))
    raise ValueError(f"unknown variant {variant!r}; choose from {PLUGIN_VARIANTS}")


def tau_new(n: int, m: int, alpha: float, delta: float, kernel_consts=None) -> float:
    """Corollary C: ``tau_plugin(..., variant="auto")``, the v2 test threshold."""
    return tau_plugin(n, m, alpha, delta, kernel_consts, variant="auto")


def tau_aos(n: int, m: int, alpha: float, delta: float, kernel_consts=None) -> float:
    """AoS restructure (2026-11-21): tau(alpha) = min{tau^P(alpha), tau^A(alpha, delta)}, Theorem thm:plugin's closed
    form and the two-sided threshold eq:tauA; nondecreasing in alpha as both are. ``tau_plugin(..., "closed_A")``."""
    return tau_plugin(n, m, alpha, delta, kernel_consts, variant="closed_A")


def reject_plugin(d_hat: float, n: int, m: int, alpha: float, delta: float,
                  kernel_consts=None, variant: str = "auto") -> bool:
    """The v2 test: reject H_{0,alpha} iff D-hat_alpha > tau_plugin."""
    return bool(d_hat > tau_plugin(n, m, alpha, delta, kernel_consts, variant))


THRESHOLDS = ("paper", "plugin", "closed", "twosided", "aos")


def threshold_function(threshold: str, n: int, m: int, delta: float, kappa: float = 1.0,
                       kernel_consts=None, variant: str = "auto"):
    """``alpha -> threshold`` for the crossing search and the scripts.

    ``"paper"``: :func:`tau` (v1, Theorem `thm:concentration`, with ``kappa``);
    ``"plugin"``: :func:`tau_plugin` with ``variant`` (default tau_new);
    ``"closed"``: the closed form of Theorem P;
    ``"twosided"``: :func:`tau_twosided` (Theorem A at delta/2 per side);
    ``"aos"``: :func:`tau_aos`, min(closed form, tau_A) (AoS restructure).
    """
    if threshold == "paper":
        return lambda a: float(tau(n, m, a, delta, kappa))
    kc = _consts(kernel_consts) if kernel_consts is not None else (
        GAUSSIAN if kappa == 1.0 else KernelConstants.gaussian(kappa))
    if threshold == "plugin":
        return lambda a: float(tau_plugin(n, m, a, delta, kc, variant))
    if threshold == "closed":
        return lambda a: float(tau_plugin_closed(n, m, a, delta, kc))
    if threshold == "twosided":
        return lambda a: float(tau_twosided(n, m, a, delta, kc))
    if threshold == "aos":
        return lambda a: float(tau_aos(n, m, a, delta, kc))
    raise ValueError(f"unknown threshold {threshold!r}; choose from {THRESHOLDS}")


def sample_size_for(d_target: float, alpha: float, delta: float, threshold: str = "plugin",
                    m_over_n: float = 1.0, kappa: float = 1.0, kernel_consts=None,
                    n_max: int = 10 ** 9) -> int:
    """Smallest n with threshold(n, m, alpha, delta) < d_target, m = ceil(m_over_n * n).

    Replaces the closed-form "needs n" formulas of v1 (which hard-code the
    constant C = 4 + sqrt(2 log(4/delta)) of :func:`tau`, i.e. tau(0) = 2C/sqrt(n)
    at n = m): for the plug-in threshold there is no single constant once
    alpha > 0, so the size is found by bisection on n (every threshold here is
    nonincreasing in n). Returns ``n_max + 1`` if even ``n_max`` is not enough.
    ``threshold`` as in :func:`threshold_function`.
    """
    if d_target <= 0:
        return n_max + 1

    def thr(n):
        m = max(1, int(_math.ceil(m_over_n * n)))
        return threshold_function(threshold, n, m, delta, kappa=kappa,
                                  kernel_consts=kernel_consts)(alpha)

    lo, hi = 1, 2
    while thr(hi) >= d_target:
        lo, hi = hi, hi * 2
        if hi > n_max:
            return n_max + 1 if thr(n_max) >= d_target else _bisect_n(thr, lo, n_max, d_target)
    return _bisect_n(thr, lo, hi, d_target)


def _bisect_n(thr, lo, hi, d_target):
    # invariant: thr(hi) < d_target; find the smallest such n in (lo, hi]
    if thr(lo) < d_target:
        return lo
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if thr(mid) < d_target:
            hi = mid
        else:
            lo = mid
    return hi


def closed_form_constant(delta: float, threshold: str = "paper", kernel_consts=None) -> float:
    """The constant C with threshold(n, n, 0, delta) = 2 C / sqrt(n).

    ``"paper"``: C = kappa (4 + sqrt(2 log(4/delta))) (v1 formulas, e.g. 6.96 at
    delta = 0.05); ``"plugin"``/``"closed"``: Theorem P's closed form at alpha = 0,
    C = (v + Dphi sqrt(log(1/delta)/2)) / sqrt(2) (1.93 for the Gaussian kernel),
    which tau_new attains at alpha = 0 (McDiarmid is the binding tail there).
    Only valid at alpha = 0 for the plug-in threshold: tau_new(alpha) is NOT
    tau_new(0)/(1-alpha); use :func:`sample_size_for` for alpha > 0.
    """
    if threshold == "paper":
        kc = _consts(kernel_consts)
        return kc.kappa * (4.0 + _math.sqrt(2.0 * _math.log(4.0 / delta)))
    kc = _consts(kernel_consts)
    return (_math.sqrt(kc.v2) + kc.Dphi * _math.sqrt(_math.log(1.0 / delta) / 2.0)) / _math.sqrt(2.0)


def samplesize_constant(delta: float, kappa: float = 1.0) -> float:
    """C_0 of eq:samplesize (jmlr/sections/11_applications.tex):
    C_0 = kappa [1 + sqrt(log(1/delta))] (2.73 at delta = 0.05, kappa = 1).

    For the Gaussian kernel (v = kappa, Dphi = sqrt(2) kappa) Theorem P gives
    tau_{n,n}(0, delta) <= C_0 sqrt(2/n), with equality for the closed form,
    which is the member of tau_{n,m} that binds at alpha = 0. Equivalently
    C_0 = sqrt(2) * closed_form_constant(delta, "plugin") for kappa = 1.
    """
    return kappa * (1.0 + _math.sqrt(_math.log(1.0 / delta)))


def fire_sample_size(d0: float, delta: float, kappa: float = 1.0) -> float:
    """eq:samplesize: the test of H_{0,0} rejects (n = m) once
    n > 2 (C_0 / D_0)^2. Returns the real number 2 (C_0/D_0)^2 (inf if D_0 <= 0);
    callers round as they print (v1 printed ceil for the data contrasts)."""
    if not d0 > 0:
        return float("inf")
    return 2.0 * (samplesize_constant(delta, kappa) / d0) ** 2


def sharp_sample_size(d_a: float, a: float, delta: float, kernel_consts=None) -> int:
    """eq:sharpsize: n_a = min{n : tau_{n,n}(a, delta) < D_a}, solved numerically
    with the package's tau_{n,m} (:func:`tau_new`, Gaussian constants by
    default); D_a is read off the estimated path. -1 if D_a <= 0."""
    if not d_a > 0:
        return -1
    return int(sample_size_for(d_a, a, delta, threshold="plugin", kernel_consts=kernel_consts))

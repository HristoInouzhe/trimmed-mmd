"""Finite-sample, variance-adaptive split-sample certificates for TV(P,Q) > alpha.

Exploration code (not part of the package). Everything here works on the
SECOND fold only, given objects fixed by the first fold:

* a witness ``f`` (any bounded measurable map; here the unit-norm trimmed-MMD
  witness of eq:witness, so |f| <= kappa = 1 for the Gaussian kernel);
* two cut points ``s`` (for U = f(X)) and ``sp`` (for V = f(Y)), chosen on fold 1.

Two families of fold-2 statistics:

``psi``   A_i = psi_s(U_i) = s - beta (s - U_i)_+,  B_j = -sp - beta (V_j - sp)_+.
          E A + E B <= T(f) = inf_{C_a(P)} <f,a> - sup_{C_a(Q)} <f,b> <= 0 under H0.
          A priori ranges use |f| <= kappa.

``ramp``  g = clip((f - sp)/(s - sp), 0, 1) in [0,1]; Delta = E_P g - E_Q g <= TV.
          Reject H0: TV <= alpha iff a lower confidence bound on Delta exceeds alpha.
          Identity: beta (s - sp) (Delta - alpha) = E A' + E B' for the clipped witness
          f' = clip(f, sp, s), and E A' + E B' >= E A + E B (clipping only helps).

All "fs" tests below are valid in finite samples, for every pair (P,Q) in
H0 and every fold-1 outcome; "asy" tests are the asymptotic references.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm as _norm

# ---------------------------------------------------------------------------
# elementary statistics
# ---------------------------------------------------------------------------


def lower_trimmed_mean(x, alpha):
    """Mean of the smallest (1-alpha) fraction, boundary point fractional."""
    x = np.sort(np.asarray(x, float))
    n = x.size
    cap = 1.0 / (n * (1.0 - alpha))
    k = min(int(np.floor(1.0 / cap + 1e-12)), n)
    out = cap * x[:k].sum()
    if k < n:
        out += (1.0 - k * cap) * x[k]
    return float(out)


def trimmed_sep(u, v, alpha):
    """Paper's eq:splitstat: lower trimmed mean of u minus upper trimmed mean of v."""
    return lower_trimmed_mean(u, alpha) + lower_trimmed_mean(-np.asarray(v), alpha)


def psi_vars(u, v, s, sp, alpha):
    beta = 1.0 / (1.0 - alpha)
    A = s - beta * np.maximum(s - u, 0.0)
    B = -sp - beta * np.maximum(v - sp, 0.0)
    return A, B


def ramp(f, s, sp):
    return np.clip((f - sp) / (s - sp), 0.0, 1.0)


# ---------------------------------------------------------------------------
# one-sample bounds
# ---------------------------------------------------------------------------


def mp_lcb(z, b, delta):
    """Maurer & Pontil (2009) Thm 4, rescaled to an interval of length b.

    With prob >= 1-delta: E Z >= mean - sqrt(2 V ln(2/d)/n) - 7 b ln(2/d)/(3(n-1)),
    V the unbiased sample variance. i.i.d., n >= 2.
    """
    n = z.size
    L = math.log(2.0 / delta)
    V = float(np.var(z, ddof=1))
    return float(z.mean() - math.sqrt(2.0 * V * L / n) - 7.0 * b * L / (3.0 * (n - 1)))


def sd_ucb(z, b, delta):
    """Maurer & Pontil (2009) Thm 10 eq (3), rescaled: sigma <= sqrt(V) + b sqrt(2 ln(1/d)/(n-1)).

    Also sigma <= b/2 always (Popoviciu)."""
    n = z.size
    V = float(np.var(z, ddof=1))
    return min(b / 2.0, math.sqrt(V) + b * math.sqrt(2.0 * math.log(1.0 / delta) / (n - 1)))


# ---------------------------------------------------------------------------
# two-sample bounds on mu_A + mu_B (A_i iid, B_j iid, independent samples)
# ---------------------------------------------------------------------------


def lcb_mp_union(A, bA, B, bB, delta):
    """MP Thm 4 on each sample at delta/2 (union bound)."""
    return mp_lcb(A, bA, delta / 2) + mp_lcb(B, bB, delta / 2)


def lcb_bern2s(A, bA, B, bB, delta, share=0.8):
    """Two-sample Bernstein (BLM 2013 Thm 2.10) + MP Thm 10 variance UCBs.

    Summands (mu_A - A_i)/n <= bA/n and (mu_B - B_j)/m <= bB/m are independent;
    Bernstein: P(sum >= sqrt(2 v L) + c L/3) <= e^{-L}, v = sA^2/n + sB^2/m.
    delta = share*delta (Bernstein) + 2 * (1-share)/2 * delta (two sd UCBs).
    """
    n, m = A.size, B.size
    d1 = share * delta
    d2 = (1.0 - share) * delta / 2.0
    sA = sd_ucb(A, bA, d2)
    sB = sd_ucb(B, bB, d2)
    v = sA ** 2 / n + sB ** 2 / m
    c = max(bA / n, bB / m)
    L = math.log(1.0 / d1)
    return float(A.mean() + B.mean() - math.sqrt(2.0 * v * L) - c * L / 3.0)


def hoeffding2s_lcb(A, bA, B, bB, delta):
    """Hoeffding for the two-sample mean: ranges bA/n (n terms) and bB/m (m terms)."""
    n, m = A.size, B.size
    t = math.sqrt(math.log(1.0 / delta) * (bA ** 2 / n + bB ** 2 / m) / 2.0)
    return float(A.mean() + B.mean() - t)


def pair_up(A, B):
    """Independent pair statistics Z_i = A_i + mean(B over a block), or the mirror.

    The larger sample is cut into contiguous blocks, one per observation of the
    smaller sample, so Z_1..Z_N are independent with E Z_i = mu_A + mu_B and
    Var(mean Z) = Var A/n + Var B/m when the block sizes are equal.
    """
    n, m = A.size, B.size
    if n <= m:
        blocks = np.array_split(B, n)
        return A + np.array([blk.mean() for blk in blocks])
    blocks = np.array_split(A, m)
    return np.array([blk.mean() for blk in blocks]) + B


def equal_blocks(G, H):
    """i.i.d. pair statistics with EQUAL blocks: Z_k = G_k - mean(H over block k of size r).

    r = floor(m/n) (or the mirror); the remainder of the larger sample is dropped,
    which keeps the Z_k identically distributed (needed by Maurer-Pontil Thm 4).
    """
    n, m = G.size, H.size
    if n <= m:
        r = m // n
        return G - H[: n * r].reshape(n, r).mean(1)
    r = n // m
    return G[: m * r].reshape(m, r).mean(1) - H


def mp_pair_lcb(G, H, delta):
    """Maurer & Pontil Thm 4 on the i.i.d. equal-block pairs Z in [-1, 1] (range 2)."""
    return mp_lcb(equal_blocks(G, H), 2.0, delta)


def betting_test(Y, lo, delta, c=0.5, prior_var=None, prior_w=20.0, prior_mu=0.0):
    """Test H0: E[Y_i | past] <= 0 for Y_i >= lo (lo < 0) by betting.

    Wealth K_t = prod_{i<=t} (1 + lam_i Y_i), lam_i in [0, c/|lo|] predictable.
    Under H0, (K_t) is a nonnegative supermartingale with K_0 = 1, so by Ville's
    inequality P(max_t K_t >= 1/delta) <= delta, for every N.

    lam_i = min(c/|lo|, sqrt(2 ln(1/delta) / (N sig2_{i-1}))), the fixed-N plug-in
    of Waudby-Smith & Ramdas (2024, Remark 1 / eq. 27), one-sided, with sig2 a
    running variance estimate shrunk towards a fold-1 prior.

    Returns (reject, log max wealth, log final wealth).
    """
    Y = np.asarray(Y, float)
    N = Y.size
    lam_max = c / abs(lo)
    L = math.log(1.0 / delta)
    if prior_var is None:
        prior_var = (abs(lo) / 2.0) ** 2
    # predictable running mean / variance (Welford with prior pseudo-counts)
    cs = np.concatenate([[0.0], np.cumsum(Y)[:-1]])
    cnt = np.arange(N, dtype=float)
    mu_prev = (prior_w * prior_mu + cs) / (prior_w + cnt)
    # running sum of squared deviations from the predictable mean (WSR style)
    dev2 = (Y - mu_prev) ** 2
    cdev = np.concatenate([[0.0], np.cumsum(dev2)[:-1]])
    sig2 = (prior_w * prior_var + cdev) / (prior_w + cnt)
    sig2 = np.maximum(sig2, 1e-8)
    lam = np.minimum(lam_max, np.sqrt(2.0 * L / (N * sig2)))
    logK = np.cumsum(np.log1p(lam * Y))
    return bool(logK.max() >= L), float(logK.max()), float(logK[-1])


def betting_logwealth(Y, lo, delta, c=0.5, prior_var=None, prior_w=20.0):
    return betting_test(Y, lo, delta, c=c, prior_var=prior_var, prior_w=prior_w)[2]


# ---------------------------------------------------------------------------
# DKW-based bounds on the paper's estimand (no cut point)
# ---------------------------------------------------------------------------


def dkw_lower_trimmed_lcb(u, alpha, lo, delta):
    """Thomas & Learned-Miller (2019, Thm 4)-type lower bound on the lower trimmed mean.

    One-sided DKW (Massart): with prob >= 1-delta, F <= F_n + eps, eps =
    sqrt(ln(1/delta)/(2n)) (needs delta <= 1/2). The lower trimmed mean is
    monotone in the stochastic order, so it is >= that of the smallest law in
    the band: mass eps moved from the top of F_n to the known lower bound lo.
    """
    u = np.sort(np.asarray(u, float))
    n = u.size
    eps = math.sqrt(math.log(1.0 / delta) / (2.0 * n))
    if eps >= 1:
        return lo
    # discrete law: atom eps at lo, then 1/n at each u_(i), truncated at total mass 1
    pts = np.concatenate([[lo], u])
    wts = np.concatenate([[eps], np.full(n, 1.0 / n)])
    cum = np.cumsum(wts)
    over = cum - 1.0
    wts = np.where(over > 0, np.maximum(wts - over, 0.0), wts)
    # lower (1-alpha)-trimmed mean of the discrete law (pts sorted ascending)
    keep = 1.0 - alpha
    cw = np.cumsum(wts)
    take = np.minimum(wts, np.maximum(keep - (cw - wts), 0.0))
    return float((take * pts).sum() / keep)


# ---------------------------------------------------------------------------
# the certificate family on one fold-2 sample
# ---------------------------------------------------------------------------


def evaluate_all(u2, v2, s, sp, alpha, delta, kappa=1.0, n_boot=1000, rng=None,
                 prior=None):
    """All second-fold decisions for one witness and one pair of cut points.

    u2 = f(X^(2)), v2 = f(Y^(2)); s, sp fold-1 cut points; prior: dict of
    fold-1 variance guesses for the betting rule (keys 'ramp', 'psi').
    Returns a dict of booleans (rejections) and a few statistics.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    beta = 1.0 / (1.0 - alpha)
    n, m = u2.size, v2.size
    out = {}
    That = trimmed_sep(u2, v2, alpha)
    out["T_hat"] = That

    # --- paper: Theorem 16 percentile bootstrap (as in splitting.split_test)
    ia = rng.integers(0, n, size=(n_boot, n))
    ib = rng.integers(0, m, size=(n_boot, m))
    boots = np.array([trimmed_sep(u2[i], v2[j], alpha) for i, j in zip(ia, ib)])
    out["boot"] = bool(np.quantile(boots, delta) > 0)

    # --- paper: Proposition 17 (two-sided DKW, 2 kappa beta)
    e = lambda k: math.sqrt(math.log(4.0 / delta) / (2.0 * k))
    out["prop17"] = bool(That > 2 * kappa * beta * (e(n) + e(m)))

    # --- DKW one-sided band (Thomas & Learned-Miller type), a priori [-kappa, kappa]
    lcb = (dkw_lower_trimmed_lcb(u2, alpha, -kappa, delta / 2)
           + dkw_lower_trimmed_lcb(-v2, alpha, -kappa, delta / 2))
    out["tlm"] = bool(lcb > 0)

    valid_cut = s > sp
    # --- psi family (unclipped witness, a priori ranges)
    A, B = psi_vars(u2, v2, s, sp, alpha)
    bA = beta * max(s + kappa, 0.0)
    bB = beta * max(kappa - sp, 0.0)
    if valid_cut and bA > 0 and bB > 0:
        out["psi_mpu"] = bool(lcb_mp_union(A, bA, B, bB, delta) > 0)
        out["psi_b2s"] = bool(lcb_bern2s(A, bA, B, bB, delta) > 0)
        Z = pair_up(A, B)
        zlo = (s - beta * (s + kappa)) + (-sp - beta * (kappa - sp))
        pv = None if prior is None else prior.get("psi")
        if zlo < 0:
            out["psi_bet"] = betting_test(Z, zlo, delta, c=0.5, prior_var=pv)[0]
        else:
            out["psi_bet"] = True  # Z >= zlo >= 0 surely (cannot happen under H0 unless E Z = 0)
        out["psi_clt"] = bool(A.mean() + B.mean()
                              - _norm.ppf(1 - delta) * math.sqrt(A.var(ddof=1) / n + B.var(ddof=1) / m) > 0)
    else:
        for k in ("psi_mpu", "psi_b2s", "psi_bet", "psi_clt"):
            out[k] = False

    # --- ramp family (clipped witness == soft classifier g in [0,1])
    if valid_cut:
        G, H = ramp(u2, s, sp), ramp(v2, s, sp)
        out["Delta_hat"] = float(G.mean() - H.mean())
        out["ramp_hoef"] = bool(hoeffding2s_lcb(G, 1.0, -H, 1.0, delta) > alpha)
        out["ramp_mpu"] = bool(lcb_mp_union(G, 1.0, -H, 1.0, delta) > alpha)
        out["ramp_b2s"] = bool(lcb_bern2s(G, 1.0, -H, 1.0, delta) > alpha)
        out["ramp_mp_pair"] = bool(mp_pair_lcb(G, H, delta) > alpha)
        Z = pair_up(G, -H) - alpha
        pv = None if prior is None else prior.get("ramp")
        rej, lmax, lfin = betting_test(Z, -(1.0 + alpha), delta, c=0.5, prior_var=pv)
        out["ramp_bet"] = rej
        out["ramp_bet_logK"] = lfin
        out["ramp_bet75"] = betting_test(Z, -(1.0 + alpha), delta, c=0.75, prior_var=pv)[0]
        out["ramp_clt"] = bool(out["Delta_hat"] - _norm.ppf(1 - delta)
                               * math.sqrt(G.var(ddof=1) / n + H.var(ddof=1) / m) > alpha)
    else:
        out["Delta_hat"] = float("nan")
        out["ramp_bet_logK"] = float("-inf")
        for k in ("ramp_hoef", "ramp_mpu", "ramp_b2s", "ramp_mp_pair", "ramp_bet", "ramp_bet75", "ramp_clt"):
            out[k] = False
    return out


FS_METHODS = ["prop17", "tlm", "psi_mpu", "psi_b2s", "psi_bet",
              "ramp_hoef", "ramp_mpu", "ramp_b2s", "ramp_mp_pair", "ramp_bet", "ramp_bet75"]
ASY_METHODS = ["boot", "psi_clt", "ramp_clt"]

"""Sample splitting: certify bounds on TV without a uniform bound.

The analytic threshold pays for a supremum over the whole RKHS unit ball --
that is where the $4\\kappa/\\sqrt n$ Rademacher term comes from, and it is
what makes the test cost roughly twelve times the sample size a correct
calibration needs. Splitting removes the supremum: choose one direction on
the first fold, then on the second fold the statistic is a fixed scalar
functional of i.i.d. data, with an ordinary CLT and an estimable variance.

The construction rests on a separation identity. For closed convex
$A,B\\subset\\Hh$,

    dist(A, B) >= inf_{a in A} <f, a> - sup_{b in B} <f, b>

for any unit $f$, with equality at the optimal separating direction. Taking
$A=\\mathcal C_\\alpha(P)$ and $B=\\mathcal C_\\alpha(Q)$, both terms are support
functions of trimmed regions, i.e. tail means -- and the optimum over
trimmings is attained by putting the maximum allowed weight on the extreme
values. So the support function in a fixed direction is simply a **trimmed
mean**:

    sup over alpha-trimmings of E[g f] = mean of the largest (1-alpha)
                                          fraction of the values f(X_i),

and symmetrically for the infimum. The fold-two statistic is therefore a
difference of two trimmed means of bounded scalars, which is about as
classical an object as one could hope for.

Crucially the inequality runs the safe way. Any direction gives a *lower*
bound on $D_{\\alpha,k}$, so a significantly positive statistic certifies
$D_{\\alpha,k}>0$ and hence $\\TV(P,Q)>\\alpha$ -- with no margin condition, and
no appeal to the optimality of the direction. A poorly chosen direction
costs power, never validity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kernels import gaussian_kernel, make_rff, median_heuristic
from .simplex import project_capped_simplex
from .solver import solve_features, solve_gram


def trimmed_mean(values: np.ndarray, alpha: float, upper: bool) -> float:
    """Extreme trimmed mean: the support function of the trimmed region.

    Returns the largest (``upper=True``) or smallest (``upper=False``) value
    attainable by ``E_g[values]`` over weights ``0 <= g_i <= 1/(n(1-alpha))``
    summing to one. The optimum loads the cap onto the extreme order
    statistics, so this is a mean over the extreme ``1-alpha`` fraction with
    the boundary point weighted fractionally.
    """
    v = np.sort(np.asarray(values, dtype=float))
    if upper:
        v = v[::-1]
    n = v.size
    cap = 1.0 / (n * (1.0 - alpha))
    k = int(np.floor(1.0 / cap))          # number of fully-weighted points
    k = min(k, n)
    total = k * cap
    out = float(np.dot(v[:k], np.full(k, cap)))
    if k < n and total < 1.0:
        out += v[k] * (1.0 - total)
    return out


@dataclass
class SplitResult:
    separation: float      # point estimate of the fold-two statistic
    lower_ci: float        # one-sided (1-delta) lower confidence limit
    certified: bool        # lower_ci > 0, i.e. TV > alpha is certified
    alpha: float


def witness_direction(X1, Y1, alpha, sigma, rng=None, rff=None, **kw):
    """Optimal separating direction, fitted on fold one.

    Returns ``(coef_X, coef_Y, norm)`` describing
    ``u = sum_i w_i phi(X_i) - sum_j v_j phi(Y_j)``; the unit direction is
    ``u / ||u||``, and it is evaluated out of sample through the kernel.

    Passing a fitted :class:`~trimmedmmd.kernels.RFF` solves the direction in
    feature space instead, at O((n+m)r) rather than O((n+m)^2) in both time
    and memory. The returned coefficients mean the same thing; only the
    inner products are approximated, and ``evaluate_direction`` must then be
    given the same ``rff`` so the witness is evaluated where it was fitted.
    """
    if rff is not None:
        res = solve_features(rff.transform(X1), rff.transform(Y1), alpha,
                             rng=rng, **kw)
        return res.w, res.v, res.discrepancy
    Kxx = gaussian_kernel(X1, X1, sigma)
    Kyy = gaussian_kernel(Y1, Y1, sigma)
    Kxy = gaussian_kernel(X1, Y1, sigma)
    res = solve_gram(Kxx, Kyy, Kxy, alpha, rng=rng, **kw)
    norm = res.discrepancy
    return res.w, res.v, norm


def evaluate_direction(w, v, X1, Y1, Z, sigma, norm, rff=None):
    """Evaluate the unit witness at new points ``Z`` (the out-of-sample map)."""
    if norm <= 0:
        return np.zeros(len(Z))
    if rff is not None:
        d = rff.transform(X1).T @ w - rff.transform(Y1).T @ v
        return (rff.transform(Z) @ d) / norm
    return (gaussian_kernel(Z, X1, sigma) @ w
            - gaussian_kernel(Z, Y1, sigma) @ v) / norm


def split_test(X, Y, alpha, delta=0.05, sigma=None, rng=None,
               split_frac=0.5, n_boot=1000, use_rff=False, n_features=512,
               rff_map="phased", **kw) -> SplitResult:
    """One split-sample certificate that ``TV(P,Q) > alpha``.

    Fold one picks the direction; fold two computes the trimmed-mean
    difference and a bootstrap lower confidence limit for it. The bootstrap
    resamples scalars, not kernel matrices, so it costs nothing next to the
    quadratic program.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    X, Y = np.asarray(X), np.asarray(Y)
    n, m = len(X), len(Y)
    ix, iy = rng.permutation(n), rng.permutation(m)
    n1, m1 = int(n * split_frac), int(m * split_frac)
    X1, X2 = X[ix[:n1]], X[ix[n1:]]
    Y1, Y2 = Y[iy[:m1]], Y[iy[m1:]]
    sigma = median_heuristic(X1, Y1, rng=rng) if sigma is None else sigma

    rff = (make_rff(X.shape[1], n_features, sigma, rng=rng, rff_map=rff_map)
           if use_rff else None)
    w, v, norm = witness_direction(X1, Y1, alpha, sigma, rng=rng, rff=rff, **kw)
    a = evaluate_direction(w, v, X1, Y1, X2, sigma, norm, rff=rff)
    b = evaluate_direction(w, v, X1, Y1, Y2, sigma, norm, rff=rff)

    def stat(av, bv):
        # P-region's minimum along the witness, minus Q-region's maximum.
        return trimmed_mean(av, alpha, upper=False) - trimmed_mean(bv, alpha, upper=True)

    sep = stat(a, b)
    idx_a = rng.integers(0, a.size, size=(n_boot, a.size))
    idx_b = rng.integers(0, b.size, size=(n_boot, b.size))
    boots = np.array([stat(a[ia], b[ib]) for ia, ib in zip(idx_a, idx_b)])
    lower = float(np.quantile(boots, delta))
    return SplitResult(separation=float(sep), lower_ci=lower,
                       certified=bool(lower > 0.0), alpha=float(alpha))


def lower_bound_tv(X, Y, alphas, delta=0.05, sigma=None, rng=None, **kw):
    """Largest grid level whose rejection is certified: a lower bound on TV.

    The split is drawn once and reused across the grid. Drawing a fresh split
    per level would give each level its own witness and its own fold two, so
    the certified set need not be downward-closed and the no-multiplicity
    argument would not apply; we measured roughly one non-nested realization
    in twelve when the split was redrawn.

    Even with one split we enforce downward-closedness explicitly rather than
    assume it, taking the largest level such that every smaller grid level is
    also certified. The guarantee is then that of the single smallest
    uncertified level, as in the paper's Corollary on lower confidence
    bounds; otherwise a union bound over the grid applies.
    """
    alphas = np.sort(np.asarray(alphas, dtype=float))
    rng = np.random.default_rng(0) if rng is None else rng
    split_seed = int(rng.integers(0, 2**31 - 1))
    out = []
    for a in alphas:
        out.append(split_test(X, Y, a, delta=delta, sigma=sigma,
                              rng=np.random.default_rng(split_seed), **kw))
    best = 0.0
    for a, r in zip(alphas, out):
        if r.certified:
            best = float(a)
        else:
            break            # downward-closed: stop at the first failure
    return best, out


def upper_bound_tv(d_hat_alpha: float, alpha: float, deviation: float,
                   c0: float, gamma: float) -> float:
    """Upper bound on TV from a margin condition.

    Under ass:margin, ``D_alpha >= c0 (TV - alpha)^gamma`` for
    ``alpha < TV``, so an upper bound on ``D_alpha`` inverts to one on TV:

        TV <= alpha + ((D_alpha_upper) / c0) ** (1/gamma).

    ``deviation`` is whatever valid upper bound on ``D_alpha - d_hat_alpha``
    is available. This direction genuinely needs the margin function --
    without it, a small discrepancy is compatible with any TV above alpha,
    which is why upper bounds are the harder side.
    """
    d_up = max(d_hat_alpha + deviation, 0.0)
    return float(alpha + (d_up / c0) ** (1.0 / gamma))


def trimming_weights(values: np.ndarray, alpha: float, upper: bool) -> np.ndarray:
    """The optimal trimming weights behind :func:`trimmed_mean`.

    Applied out of sample these define the map ``x -> g_alpha(x)`` that plain
    weights on training atoms cannot provide: threshold the witness, load the
    cap onto the extreme side, and renormalize.
    """
    v = np.asarray(values, dtype=float)
    n = v.size
    order = np.argsort(-v if upper else v)
    cap = 1.0 / (n * (1.0 - alpha))
    w = np.zeros(n)
    remaining = 1.0
    for idx in order:
        take = min(cap, remaining)
        w[idx] = take
        remaining -= take
        if remaining <= 1e-15:
            break
    return w


def split_upper_bound(X, Y, alpha, delta=0.05, sigma=None, rng=None,
                      split_frac=0.5, n_boot=400, **kw):
    """Upper-bound ``D_{alpha,k}`` by exhibiting one feasible trimming pair.

    Any feasible pair gives ``D_alpha <= || mu_{g_P . P} - mu_{g_Q . Q} ||``,
    so this direction needs no optimality either -- a mediocre pair costs
    tightness, not validity. Both the direction *and the cut points* are
    chosen on fold one, and the resulting fixed threshold rule is applied to
    fold two.

    Choosing the cut points on fold two instead would be a subtle trap: the
    weights would then be selected to minimise the very discrepancy being
    estimated, biasing the point estimate downward and making an *upper*
    bound anti-conservative -- while looking tighter and more stable, i.e.\
    like success.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    X, Y = np.asarray(X), np.asarray(Y)
    n, m = len(X), len(Y)
    ix, iy = rng.permutation(n), rng.permutation(m)
    n1, m1 = int(n * split_frac), int(m * split_frac)
    X1, X2 = X[ix[:n1]], X[ix[n1:]]
    Y1, Y2 = Y[iy[:m1]], Y[iy[m1:]]
    sigma = median_heuristic(X1, Y1, rng=rng) if sigma is None else sigma

    w1, v1, norm = witness_direction(X1, Y1, alpha, sigma, rng=rng, **kw)
    if norm <= 0:
        return 0.0, 0.0

    # Cut points from fold one only.
    fX1 = evaluate_direction(w1, v1, X1, Y1, X1, sigma, norm)
    fY1 = evaluate_direction(w1, v1, X1, Y1, Y1, sigma, norm)
    tX = float(np.quantile(fX1, 1.0 - alpha)) if alpha > 0 else np.inf
    tY = float(np.quantile(fY1, alpha)) if alpha > 0 else -np.inf

    fX2 = evaluate_direction(w1, v1, X1, Y1, X2, sigma, norm)
    fY2 = evaluate_direction(w1, v1, X1, Y1, Y2, sigma, norm)

    def rule_weights(vals, thresh, keep_below):
        """Fixed threshold rule from fold one, projected onto the cap.

        The rule keeps the fold-two points on one side of a threshold fixed by
        fold one and weights them uniformly. That vector is feasible only when
        the kept fraction is at least ``1 - alpha``; the threshold is a fold-one
        quantile, so on fold two the realized fraction fluctuates around it and
        falls below it about half the time.

        Capping and renormalizing does not fix this -- on a vector that is
        uniform on the kept set, ``minimum(w, cap)`` followed by a division by
        the new sum returns the original vector exactly, so the guard is a
        no-op and the pair silently leaves the feasible set. We therefore take
        the Euclidean projection onto ``{0 <= w <= cap, sum w = 1}``.

        When the kept set is too small the projection puts the cap on it and
        spreads the residual mass uniformly over the discarded points. That is
        a worse trimming than the rule intended, so the resulting pair gives a
        *looser* upper bound -- the safe direction, since any feasible pair
        bounds ``D_alpha`` from above and only optimality, not validity, is at
        stake. Crucially the cut point still comes from fold one alone:
        choosing it on fold two would select the weights against the very
        discrepancy being bounded and bias the bound downward.
        """
        keep = (vals <= thresh) if keep_below else (vals >= thresh)
        w = keep.astype(float)
        if w.sum() <= 0:
            w = np.ones_like(vals, dtype=float)
        w = w / w.sum()
        cap = 1.0 / (len(vals) * (1.0 - alpha))
        if w.max() > cap + 1e-12:
            w = project_capped_simplex(w, cap)
        assert w.max() <= cap + 1e-9 and abs(w.sum() - 1.0) < 1e-9, (
            f"infeasible trimming: max {w.max():.6g} > cap {cap:.6g} "
            f"or sum {w.sum():.6g} != 1"
        )
        return w

    Kxx = gaussian_kernel(X2, X2, sigma)
    Kyy = gaussian_kernel(Y2, Y2, sigma)
    Kxy = gaussian_kernel(X2, Y2, sigma)

    def wmmd(a, b, Kaa, Kbb, Kab):
        val = a @ Kaa @ a + b @ Kbb @ b - 2.0 * a @ Kab @ b
        return float(np.sqrt(max(val, 0.0)))

    gX = rule_weights(fX2, tX, keep_below=True)
    gY = rule_weights(fY2, tY, keep_below=False)
    point = wmmd(gX, gY, Kxx, Kyy, Kxy)

    boots = np.empty(n_boot)
    for t in range(n_boot):
        ia = rng.integers(0, len(fX2), len(fX2))
        ib = rng.integers(0, len(fY2), len(fY2))
        # Re-derive the weights on each resample; resampling and
        # renormalizing the original weights would let a repeatedly drawn
        # point carry several times the cap, i.e. bootstrap a different and
        # spikier functional than the one being estimated.
        a = rule_weights(fX2[ia], tX, keep_below=True)
        b = rule_weights(fY2[ib], tY, keep_below=False)
        boots[t] = wmmd(a, b, Kxx[np.ix_(ia, ia)], Kyy[np.ix_(ib, ib)],
                        Kxy[np.ix_(ia, ib)])
    return point, float(np.quantile(boots, 1.0 - delta))


def interval_tv(X, Y, alphas, c0, gamma, delta=0.05, sigma=None, rng=None,
                alpha_for_upper=None, **kw):
    """A two-sided interval for ``TV(P,Q)``.

    The lower end is certified directly and needs no margin condition. The
    upper end inverts the margin condition, so it is only as trustworthy as
    the constants supplied -- which is the honest asymmetry between the two
    sides of this problem, not an artifact of the construction.
    """
    lo, details = lower_bound_tv(X, Y, alphas, delta=delta / 2, sigma=sigma,
                                 rng=rng, **kw)
    a_up = lo if alpha_for_upper is None else alpha_for_upper
    _, d_up = split_upper_bound(X, Y, a_up, delta=delta / 2, sigma=sigma,
                                rng=rng, **kw)
    hi = upper_bound_tv(d_up, a_up, 0.0, c0, gamma)
    return lo, float(min(hi, 1.0)), details


# ===========================================================================
# Revision v2: the finite-sample betting certificate on the witness ramp
# ===========================================================================
#
# Ported from explore/split_adaptive/{sacert,fold1,seqcert}.py (research note
# NOTE.md, Sections 2.3-2.4 and 4.6). Everything above this line (the
# bootstrap certificate of Theorem 16 and its helpers) is unchanged.
#
# Fold one fixes a witness f, cut points s (the (1-alpha)-quantile of f on the
# fold-one X's) and s' (the alpha-quantile on the fold-one Y's), and a variance
# guess. The ramp g = clip((f - s')/(s - s'), 0, 1) in [0, 1] satisfies
# E_P g - E_Q g <= TV(P, Q) for any g, so a test of "E_P g - E_Q g <= alpha"
# on fold two is a test of H_{0,alpha} with no kernel condition, no kappa and
# no assumption on the laws of the witness values (Lemmas 1-2). Pairs
# Z_k = g(X_k) - mean g(Y over block k) are independent given fold one with
# mean Delta = E_P g - E_Q g, and the wealth
#
#     K_t = prod_{k <= t} (1 + lam_k (Z_k - alpha)),  lam_k in [0, c/(1+alpha)]
#
# predictable, is a nonnegative supermartingale under the null; "reject iff
# max_t K_t >= 1/delta" has level delta for every sample size (Ville's
# inequality on a finite horizon; Theorem 1).

import math as _math
from typing import Callable as _Callable


def ramp(f, s: float, sp: float) -> np.ndarray:
    """The [0, 1]-valued ramp of ``f`` between the cut points sp < s."""
    return np.clip((np.asarray(f, dtype=float) - sp) / (s - sp), 0.0, 1.0)


def pair_statistics(G, H, zeta: float = 1.0) -> np.ndarray:
    """Independent pair statistics with mean E_P g - zeta E_Q g.

    If n <= m, cut ``H`` into n consecutive blocks of sizes floor(m/n) or
    ceil(m/n) (``numpy.array_split``) and return Z_k = G_k - zeta mean(H over
    block k); otherwise the mirror, Z_k = mean(G over block k) - zeta H_k.
    Given fold one the Z_k are independent, lie in [-zeta, 1] for [0, 1]-valued
    g, and have mean E_P g - zeta E_Q g; with equal blocks and zeta = 1,
    Var(mean Z) = Var G / n + Var H / m.

    ``zeta`` (default 1, the paper's statistic; unequal trimming fractions,
    :mod:`trimmedmmd.unequal`) weights the Y side; at 1.0 every value is the
    unweighted one bit for bit, since multiplying by 1.0 is exact.
    """
    G = np.asarray(G, dtype=float)
    H = np.asarray(H, dtype=float)
    n, m = G.size, H.size
    if n <= m:
        return G + np.array([blk.mean() for blk in np.array_split(-zeta * H, n)])
    return np.array([blk.mean() for blk in np.array_split(G, m)]) - zeta * H


def pair_variance(var_g: float, var_h: float, n2: int, m2: int, zeta: float = 1.0) -> float:
    """Variance of one pair statistic G_k - zeta mean(H over a block) (or mirror)."""
    if n2 <= m2:
        return var_g + zeta * zeta * var_h * n2 / m2
    return var_g * m2 / n2 + zeta * zeta * var_h


@dataclass
class BettingResult:
    certified: bool           # max_t K_t >= 1/delta
    log_max_wealth: float
    log_final_wealth: float   # log K_N: an e-value (E K_N <= 1 under H0)
    n_pairs: int


def betting_test(Y, lo: float, delta: float, c: float = 0.5, prior_var=None,
                 prior_w: float = 20.0, prior_mu: float = 0.0, v2_min: float = 1e-8) -> BettingResult:
    """Test H0: E[Y_k | past] <= 0 for Y_k >= lo (lo < 0) by betting.

    Wealth K_t = prod_{k <= t} (1 + lam_k Y_k), lam_k in [0, c/|lo|]
    predictable. Under H0 (K_t) is a nonnegative supermartingale with K_0 = 1,
    so P(max_{t <= N} K_t >= 1/delta) <= delta for every N (Ville).

    lam_k = min(c/|lo|, sqrt(2 log(1/delta) / (N sig2_{k-1}))), the fixed-N
    plug-in of Waudby-Smith and Ramdas (2024, Remark 1, eq. 16; one-sided,
    so log(1/delta)), with
      mu_{k-1}   = (prior_w prior_mu + sum_{i<k} Y_i) / (prior_w + k - 1),
      sig2_{k-1} = (prior_w prior_var + sum_{i<k} (Y_i - mu_{i-1})^2) / (prior_w + k - 1),
    a running variance shrunk towards the fold-one guess ``prior_var``
    (default (|lo|/2)^2) with ``prior_w`` pseudo-observations, floored at
    ``v2_min`` (default 1e-8, every run of the paper).
    """
    Y = np.asarray(Y, dtype=float)
    N = Y.size
    if N == 0:
        return BettingResult(False, float("-inf"), 0.0, 0)
    lam_max = c / abs(lo)
    L = _math.log(1.0 / delta)
    if prior_var is None:
        prior_var = (abs(lo) / 2.0) ** 2
    cs = np.concatenate([[0.0], np.cumsum(Y)[:-1]])
    cnt = np.arange(N, dtype=float)
    mu_prev = (prior_w * prior_mu + cs) / (prior_w + cnt)
    dev2 = (Y - mu_prev) ** 2
    cdev = np.concatenate([[0.0], np.cumsum(dev2)[:-1]])
    sig2 = (prior_w * prior_var + cdev) / (prior_w + cnt)
    sig2 = np.maximum(sig2, v2_min)
    lam = np.minimum(lam_max, np.sqrt(2.0 * L / (N * sig2)))
    logK = np.cumsum(np.log1p(lam * Y))
    return BettingResult(bool(logK.max() >= L), float(logK.max()), float(logK[-1]), int(N))


def sd_ucb(z, b: float, delta: float) -> float:
    """Maurer and Pontil (2009, Thm 10, eq. 3), rescaled to an interval of
    length b: with probability >= 1 - delta, sigma <= sqrt(V) + b sqrt(2 log(1/delta)/(n-1)),
    V the unbiased sample variance (i.i.d., n >= 2). Also sigma <= b/2 (Popoviciu)."""
    z = np.asarray(z, dtype=float)
    n = z.size
    V = float(np.var(z, ddof=1))
    return min(b / 2.0, _math.sqrt(V) + b * _math.sqrt(2.0 * _math.log(1.0 / delta) / (n - 1)))


def lcb_bernstein_2s(A, bA: float, B, bB: float, delta: float, share: float = 0.8) -> float:
    """Two-sample empirical-Bernstein lower confidence bound for mu_A + mu_B
    (Corollary 1 of the betting note; closed form).

    A_1..A_n i.i.d. with range of length bA, B_1..B_m i.i.d. with range bB,
    independent samples. The n + m summands (A_i - mu_A)/n <= bA/n and
    (B_j - mu_B)/m <= bB/m are independent, centred and bounded ABOVE, with
    total variance v = sigma_A^2/n + sigma_B^2/m. Bernstein's inequality
    (Boucheron, Lugosi and Massart 2013, Thm 2.10 with c = b/3) bounds the
    upper tail of their sum:

        P( (mean A + mean B) - (mu_A + mu_B) >= sqrt(2 v L1) + L1 b / 3 ) <= delta_1,

    L1 = log(1/delta_1), b = max(bA/n, bB/m). The variances are replaced by
    Maurer-Pontil upper confidence bounds (:func:`sd_ucb`) at delta_2 each, so
    with delta = delta_1 + 2 delta_2 (delta_1 = share * delta),

        P( mean A + mean B - sqrt(2 v_bar L1) - L1 b / 3 > mu_A + mu_B ) <= delta.

    (The research code's docstring had this tail with the opposite sign; the
    formula was right.) For the ramp, call with (G, 1, -H, 1): the bound is a
    (1 - delta) lower confidence bound for Delta = E_P g - E_Q g <= TV(P, Q).
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    n, m = A.size, B.size
    d1 = share * delta
    d2 = (1.0 - share) * delta / 2.0
    sA = sd_ucb(A, bA, d2)
    sB = sd_ucb(B, bB, d2)
    v = sA ** 2 / n + sB ** 2 / m
    c = max(bA / n, bB / m)
    L = _math.log(1.0 / d1)
    return float(A.mean() + B.mean() - _math.sqrt(2.0 * v * L) - c * L / 3.0)


CUT_RULES = ("linear", "order")
_CUT_EPS = 1e-9    # n * p within 1e-9 of an integer counts as that integer (floating-point levels such as j a*/8)


def cut_ranks(n: int, m: int, alpha: float, alpha_y: float | None = None):
    """Ranks (1-based) of the order statistics used by ``ramp_cut_points(..., rule="order")``:
    k_s = ceil(n (1 - alpha)) (the smallest (1-alpha)-quantile of n values) and
    k_u = floor(m alpha) + 1 (the largest alpha-quantile of m values), each clipped to [1, n] resp. [1, m].
    ``alpha_y`` (default ``alpha``) is the trimming fraction of the Y side, k_u = floor(m alpha_y) + 1."""
    alpha_y = alpha if alpha_y is None else alpha_y
    k_s = int(np.ceil(n * (1.0 - alpha) - _CUT_EPS))
    k_u = int(np.floor(m * alpha_y + _CUT_EPS)) + 1
    return min(max(k_s, 1), n), min(max(k_u, 1), m)


def ramp_cut_points(u1, v1, alpha: float, rule: str = "linear", alpha_y: float | None = None):
    """Fold-one cut points: s = (1-alpha)-quantile of the fold-one witness values
    on X, s' = alpha-quantile on Y (the alpha_y-quantile when ``alpha_y`` is
    given, for unequal trimming fractions; default ``alpha``).

    ``rule="linear"`` (default, every published run): ``numpy.quantile`` with
    linear interpolation. Any fold-one rule is valid; this is the pre-specified
    one (rule A).

    ``rule="order"`` (AoS restructure, 2026-11-21): genuine empirical quantiles,
    i.e. order statistics, s = the SMALLEST (1-alpha)-quantile of the X values,
    u_(k_s) with k_s = ceil(n (1 - alpha)), and s' = the LARGEST alpha-quantile of
    the Y values, v_(k_u) with k_u = floor(m alpha) + 1 (:func:`cut_ranks`).
    At alpha = 0 both rules give (max u, min v)."""
    alpha_y = alpha if alpha_y is None else alpha_y
    if rule == "linear":
        return float(np.quantile(u1, 1.0 - alpha)), float(np.quantile(v1, alpha_y))
    if rule == "order":
        u1, v1 = np.asarray(u1, dtype=float), np.asarray(v1, dtype=float)
        k_s, k_u = cut_ranks(u1.size, v1.size, alpha, alpha_y)
        return (float(np.partition(u1, k_s - 1)[k_s - 1]),
                float(np.partition(v1, k_u - 1)[k_u - 1]))
    raise ValueError(f"unknown cut rule {rule!r}; choose from {CUT_RULES}")


def ramp_prior_var(u1, v1, s: float, sp: float, n2: int, m2: int,
                   floor: float = 1e-4, zeta: float = 1.0) -> float:
    """Fold-one guess of the variance of one pair statistic, for the bets
    (``zeta``: the weight of the Y side, as in :func:`pair_statistics`)."""
    Gf, Hf = ramp(u1, s, sp), ramp(v1, s, sp)
    return max(pair_variance(float(Gf.var()), float(Hf.var()), n2, m2, zeta=zeta), floor)


def fold1_witness_values(w, v, X1, Y1, sigma, norm, rff=None, self_exclude=True):
    """Witness values on fold one, each point's own kernel term removed.

    f(X1_i) contains w_i k(X1_i, X1_i)/||u||; removing it is a cheap
    approximate leave-one-out that reduces the in-sample bias of the fold-one
    quantiles. Uses fold one only, so validity is unaffected. With ``rff``
    the self term is w_i ||z(X1_i)||^2 / ||u||.
    """
    if norm <= 0:
        return np.zeros(len(X1)), np.zeros(len(Y1))
    u1 = evaluate_direction(w, v, X1, Y1, X1, sigma, norm, rff=rff)
    v1 = evaluate_direction(w, v, X1, Y1, Y1, sigma, norm, rff=rff)
    if not self_exclude:
        return u1, v1
    if rff is None:
        # Gaussian kernel: k(x, x) = 1
        return u1 - np.asarray(w) / norm, v1 + np.asarray(v) / norm
    zx, zy = rff.transform(X1), rff.transform(Y1)
    return (u1 - np.asarray(w) * (zx ** 2).sum(1) / norm,
            v1 + np.asarray(v) * (zy ** 2).sum(1) / norm)


CERTIFICATES = ("betting", "bernstein", "bootstrap", "sequential")


@dataclass
class RampCertificate:
    alpha: float
    certified: bool           # TV(P, Q) > alpha certified at level delta
    method: str               # "betting" or "bernstein"
    s: float                  # fold-one cut points
    sp: float
    delta_hat: float          # mean G - mean H on fold two (nan if s <= sp)
    log_max_wealth: float     # betting only
    log_final_wealth: float   # betting only
    lcb: float                # bernstein only: (1-delta) LCB for E_P g - E_Q g
    prior_var: float


def betting_certificate(u2, v2, alpha: float, s: float, sp: float, delta: float = 0.05,
                        prior_var=None, c: float = 0.5, prior_w: float = 20.0,
                        method: str = "betting") -> RampCertificate:
    """Fold-two decision of the ramp certificate for one level.

    ``u2 = f(X^(2))``, ``v2 = f(Y^(2))``; ``s``, ``sp`` and ``prior_var`` come
    from fold one (:func:`ramp_cut_points`, :func:`ramp_prior_var`). If
    ``s <= sp`` the level is not tested (Lemma 2(4): no power). ``method``:
    ``"betting"`` (Theorem 1, default c = 1/2, w0 = 20) or ``"bernstein"``
    (Corollary 1, the closed-form empirical-Bernstein bound, delta_1 = 0.8 delta).
    """
    nan = float("nan")
    if not s > sp:
        return RampCertificate(float(alpha), False, method, float(s), float(sp), nan,
                               float("-inf"), float("-inf"), nan,
                               nan if prior_var is None else float(prior_var))
    G, H = ramp(u2, s, sp), ramp(v2, s, sp)
    dh = float(G.mean() - H.mean())
    if method == "betting":
        Z = pair_statistics(G, H) - alpha
        r = betting_test(Z, -(1.0 + alpha), delta, c=c, prior_var=prior_var, prior_w=prior_w)
        return RampCertificate(float(alpha), r.certified, method, float(s), float(sp), dh,
                               r.log_max_wealth, r.log_final_wealth, nan,
                               nan if prior_var is None else float(prior_var))
    if method == "bernstein":
        lcb = lcb_bernstein_2s(G, 1.0, -H, 1.0, delta)
        return RampCertificate(float(alpha), bool(lcb > alpha), method, float(s), float(sp), dh,
                               nan, nan, lcb, nan if prior_var is None else float(prior_var))
    raise ValueError(f"unknown method {method!r}")


@dataclass
class SplitFold:
    """Everything fold one fixes, plus the fold-two witness values."""
    alpha: float
    sigma: float
    norm: float
    T_hat: float              # fold-two trimmed-mean difference (Theorem 16 statistic)
    u1: np.ndarray            # fold-one witness values (self-excluded)
    v1: np.ndarray
    u2: np.ndarray            # fold-two witness values
    v2: np.ndarray
    boot_certified: bool | None   # Theorem 16 bootstrap decision (None if not run)
    boot_lower: float | None


def split_fold(X, Y, alpha, delta=0.05, sigma=None, rng=None, split_frac=0.5,
               n_boot=1000, use_rff=False, n_features=512, self_exclude=True,
               rff_map="phased", **kw) -> SplitFold:
    """Fit the fold-one witness at ``alpha`` with EXACTLY the random stream of
    :func:`split_test` (permutation, fold-one median bandwidth, random
    features, QP), so that the betting certificate and the bootstrap of
    Theorem 16 are computed on the same split and the same witness, and with
    ``n_boot > 0`` the bootstrap decision equals ``split_test``'s bit for bit.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    X, Y = np.asarray(X), np.asarray(Y)
    n, m = len(X), len(Y)
    ix, iy = rng.permutation(n), rng.permutation(m)
    n1, m1 = int(n * split_frac), int(m * split_frac)
    X1, X2 = X[ix[:n1]], X[ix[n1:]]
    Y1, Y2 = Y[iy[:m1]], Y[iy[m1:]]
    sigma = median_heuristic(X1, Y1, rng=rng) if sigma is None else sigma
    rff = (make_rff(X.shape[1], n_features, sigma, rng=rng, rff_map=rff_map)
           if use_rff else None)
    w, v, norm = witness_direction(X1, Y1, alpha, sigma, rng=rng, rff=rff, **kw)
    a = evaluate_direction(w, v, X1, Y1, X2, sigma, norm, rff=rff)
    b = evaluate_direction(w, v, X1, Y1, Y2, sigma, norm, rff=rff)

    def stat(av, bv):
        return trimmed_mean(av, alpha, upper=False) - trimmed_mean(bv, alpha, upper=True)

    sep = stat(a, b)
    boot_cert = boot_lower = None
    if n_boot and n_boot > 0:
        idx_a = rng.integers(0, a.size, size=(n_boot, a.size))
        idx_b = rng.integers(0, b.size, size=(n_boot, b.size))
        boots = np.array([stat(a[ia], b[ib]) for ia, ib in zip(idx_a, idx_b)])
        boot_lower = float(np.quantile(boots, delta))
        boot_cert = bool(boot_lower > 0.0)
    u1, v1 = fold1_witness_values(w, v, X1, Y1, sigma, norm, rff=rff,
                                  self_exclude=self_exclude)
    return SplitFold(float(alpha), float(sigma), float(norm), float(sep), u1, v1, a, b,
                     boot_cert, boot_lower)


def ramp_certificate_on_fold(fold: SplitFold, alpha: float, delta: float = 0.05,
                             method: str = "betting", c: float = 0.5,
                             prior_w: float = 20.0, cut_rule: str = "linear") -> RampCertificate:
    """The ramp certificate at level ``alpha`` on a fitted fold (the fold's
    witness may have been fitted at another level, e.g. alpha = 0).
    ``cut_rule``: see :func:`ramp_cut_points`."""
    if fold.norm <= 0:
        nan = float("nan")
        return RampCertificate(float(alpha), False, method, nan, nan, nan,
                               float("-inf"), float("-inf"), nan, nan)
    s, sp = ramp_cut_points(fold.u1, fold.v1, alpha, rule=cut_rule)
    pv = (ramp_prior_var(fold.u1, fold.v1, s, sp, fold.u2.size, fold.v2.size)
          if s > sp else None)
    return betting_certificate(fold.u2, fold.v2, alpha, s, sp, delta, prior_var=pv,
                               c=c, prior_w=prior_w, method=method)


@dataclass
class SplitCertificate:
    alpha: float
    certified: bool           # the requested finite-sample certificate
    ramp: RampCertificate
    separation: float         # Theorem 16 statistic on the same split
    boot_certified: bool | None
    boot_lower: float | None


def split_certificate(X, Y, alpha, delta=0.05, sigma=None, rng=None, split_frac=0.5,
                      n_boot=0, use_rff=False, n_features=512, method="betting",
                      c=0.5, prior_w=20.0, rff_map="phased", **kw) -> SplitCertificate:
    """One finite-sample split certificate that ``TV(P, Q) > alpha``.

    Same split, bandwidth and witness as :func:`split_test` with the same
    ``rng``; fold one also fixes the cut points (self-excluded quantiles) and
    the variance guess; fold two bets (Theorem 1). ``n_boot > 0`` also
    returns the Theorem 16 bootstrap decision on the same witness.
    """
    fold = split_fold(X, Y, alpha, delta=delta, sigma=sigma, rng=rng,
                      split_frac=split_frac, n_boot=n_boot, use_rff=use_rff,
                      n_features=n_features, rff_map=rff_map, **kw)
    rc = ramp_certificate_on_fold(fold, alpha, delta, method=method, c=c, prior_w=prior_w)
    return SplitCertificate(float(alpha), rc.certified, rc, fold.T_hat,
                            fold.boot_certified, fold.boot_lower)


def _down_closed(alphas, decisions, none_value):
    best = none_value
    for a, ok in zip(alphas, decisions):
        if ok:
            best = float(a)
        else:
            break
    return best


def lower_bound_betting(X, Y, alphas, delta=0.05, sigma=None, rng=None, split_seed=None,
                        witness="per_level", methods=("betting",), n_boot=0,
                        none_value=0.0, split_frac=0.5, use_rff=False, n_features=512,
                        c=0.5, prior_w=20.0, edges=None, standardize_batch0=False,
                        rff_map="phased", cut_rule="linear", **kw):
    """Finite-sample lower confidence bound for TV by the down-closed rule
    (Corollary 2 of the betting note).

    Split and seeds as :func:`lower_bound_tv`: ``split_seed`` (default: one
    draw ``rng.integers(0, 2**31 - 1)`` from ``rng = default_rng(0)``) and, for
    every level, a fresh ``default_rng(split_seed)``, so the folds are the same
    at every level and, per level, the witness is the one ``split_test``
    fits. ``witness``:

    * ``"per_level"`` -- refit the witness at each level (as published);
    * ``"alpha0"``    -- ONE witness fitted at alpha = 0 (the constraint set
      is a point, so the fit is immediate) with level-specific cut points.
      Valid for the same reason (any fold-one g), and one QP per split.

    ``methods`` is any subset of ``("sequential", "betting", "bernstein",
    "bootstrap")``. ``"sequential"`` is the paper's main certificate (batches
    ``edges``, default :data:`SEQ_EDGES` = (0, 1/4, 1/2, 1); random stream of
    :func:`sequential_order` from the same ``default_rng(split_seed)``, so its
    order is the single split's permutation); the others use the single split.
    ``"bootstrap"`` (Theorem 16, asymptotic) needs ``n_boot > 0`` and is only
    computed with per-level witnesses unless ``witness="alpha0"`` (then it is
    the bootstrap of the alpha = 0 witness, a matched comparison). Levels are
    evaluated in increasing order until every method has failed; each method
    keeps its own down-closed bound. ``none_value`` is returned when not even
    the smallest level is certified (0.0 as in ``lower_bound_tv``; the
    fairness scripts use -1).

    ``standardize_batch0`` is passed to :func:`sequential_order` (sequential method only).
    ``rff_map`` (``"phased"`` default, or ``"cossin"``) is the random-feature map of every
    method when ``use_rff`` (:class:`~trimmedmmd.kernels.RFF`).

    Returns ``(bounds, levels)``: ``bounds[method]`` is the certified level and
    ``levels`` a list of per-level dicts (decisions and statistics).
    """
    alphas = np.sort(np.asarray(alphas, dtype=float))
    methods = tuple(methods)
    for mth in methods:
        if mth not in CERTIFICATES:
            raise ValueError(f"unknown method {mth!r}")
    if "bootstrap" in methods and not n_boot:
        raise ValueError("methods include 'bootstrap' but n_boot = 0")
    if split_seed is None:
        rng = np.random.default_rng(0) if rng is None else rng
        split_seed = int(rng.integers(0, 2**31 - 1))
    common = dict(delta=delta, sigma=sigma, split_frac=split_frac, use_rff=use_rff,
                  n_features=n_features, rff_map=rff_map)
    fold0 = None
    if witness == "alpha0":
        fold0 = split_fold(X, Y, 0.0, rng=np.random.default_rng(split_seed),
                           n_boot=0, **common, **kw)
    elif witness != "per_level":
        raise ValueError(f"unknown witness {witness!r}")
    alive = set(methods)
    rows = []
    edges = SEQ_EDGES if edges is None else tuple(edges)
    single = [mth for mth in methods if mth != "sequential"]
    seq_cache = {} if witness == "alpha0" else None
    seq_order0 = None
    for a in alphas:
        seq_row = {}
        if "sequential" in methods and "sequential" in alive:
            if witness == "alpha0" and seq_order0 is not None:
                order = seq_order0
            else:
                order = sequential_order(X, Y, np.random.default_rng(split_seed), edges=edges,
                                         sigma=sigma, use_rff=use_rff, n_features=n_features,
                                         standardize_batch0=standardize_batch0, rff_map=rff_map)
                seq_order0 = order
            rs = sequential_certificate(X, Y, a, delta=delta, edges=edges, c=c, prior_w=prior_w,
                                        order=order, fit_alpha=0.0 if witness == "alpha0" else None,
                                        cache=seq_cache, cut_rule=cut_rule, **kw)
            seq_row = {"sequential": rs.certified, "seq_log_final_wealth": rs.log_final_wealth,
                       "seq_horizon": rs.horizon, "seq_pairs_bet": rs.pairs_bet,
                       "seq_skipped": list(rs.skipped_batches)}
        elif "sequential" in methods:
            seq_row = {"sequential": False}
        if not single:
            row = {"alpha": float(a)}
            row.update(seq_row)
            rows.append(row)
            alive = {mth for mth in alive if row.get(mth, False)}
            if not alive:
                break
            continue
        if witness == "per_level":
            fold = split_fold(X, Y, a, rng=np.random.default_rng(split_seed),
                              n_boot=n_boot if "bootstrap" in methods else 0, **common, **kw)
            boot = fold.boot_certified
        else:
            fold = fold0
            boot = None
            if "bootstrap" in methods:
                # bootstrap of the alpha = 0 witness at level a (matched comparison)
                rb = np.random.default_rng([split_seed, int(round(a * 1e6))])
                def stat(av, bv):
                    return (trimmed_mean(av, a, upper=False)
                            - trimmed_mean(bv, a, upper=True))
                ia = rb.integers(0, fold.u2.size, size=(n_boot, fold.u2.size))
                ib = rb.integers(0, fold.v2.size, size=(n_boot, fold.v2.size))
                boots = np.array([stat(fold.u2[i], fold.v2[j]) for i, j in zip(ia, ib)])
                boot = bool(np.quantile(boots, delta) > 0.0)
        row = {"alpha": float(a), "norm": fold.norm, "sigma": fold.sigma}
        row.update(seq_row)
        if witness == "per_level":
            row["T_hat"] = fold.T_hat
        for mth in single:
            if mth == "bootstrap":
                row["bootstrap"] = bool(boot)
                continue
            rc = ramp_certificate_on_fold(fold, a, delta, method=mth, c=c, prior_w=prior_w,
                                          cut_rule=cut_rule)
            row[mth] = bool(rc.certified)
            row["s"], row["sp"], row["delta_hat"] = rc.s, rc.sp, rc.delta_hat
            if mth == "betting":
                row["log_max_wealth"] = rc.log_max_wealth
            else:
                row["lcb"] = rc.lcb
        rows.append(row)
        alive = {mth for mth in alive if row.get(mth, False)}
        if not alive:
            break
    bounds = {mth: _down_closed([r["alpha"] for r in rows], [r.get(mth, False) for r in rows],
                                none_value) for mth in methods}
    return bounds, rows


def sequential_horizon(n: int, m: int, edges) -> int:
    """Number of betting pairs of the sequential certificate, fixed in advance.

    Batch b >= 1 holds rows [round(e_b n), round(e_{b+1} n)) of X and likewise
    of Y and contributes min(n_b, m_b) pairs (:func:`pair_statistics`). The
    horizon depends on the sample sizes and the batch fractions only, never on
    the data, which is what makes the WSR fractions predictable.
    """
    bx = [int(round(e * n)) for e in edges]
    by = [int(round(e * m)) for e in edges]
    return int(sum(min(bx[b + 1] - bx[b], by[b + 1] - by[b]) for b in range(1, len(edges) - 1)))


def sequential_betting_certificate(X, Y, alpha: float, edges, sigma: float, rng=None,
                                   delta: float = 0.05, c: float = 0.5, prior_w: float = 20.0,
                                   fit: _Callable | None = None, return_detail: bool = False,
                                   cut_rule: str = "linear", v2_min: float = 1e-8, **kw):
    """Sequential-batch certificate (NOTE Section 2.4): refit on everything
    seen so far, bet on the next batch, keep multiplying the same wealth.

    ``X``, ``Y`` must already be in a random order fixed independently of the
    data; they are cut at the fractions ``edges = [0, e_1, ..., 1]`` (e.g.
    ``(0, 1/4, 1/2, 1)``: fit on the first quarter, bet on the second, refit on
    the first half, bet on the second half). Batch b >= 1 is scored with a
    witness, cut points and variance guess computed from batches 0..b-1 only;
    ``sigma`` must be a function of batch 0 only. Valid in finite samples,
    uniformly over H_{0,alpha} (the supermartingale argument of Theorem 1 with
    the filtration of all earlier batches).

    The horizon N of the fraction schedule lam_k = min(c/(1+alpha),
    sqrt(2 log(1/delta)/(N sig2_{k-1}))) is fixed in advance as the total
    number of pairs over ALL betting batches (:func:`sequential_horizon`). A
    batch whose fold-before cut points cross (s <= s') is not skipped: its
    pairs get lam = 0 (the wealth and the running variance are unchanged), so
    the bets on every batch stay predictable. (The research code
    explore/split_adaptive/seqcert.py set N to the pairs of the batches with a
    valid cut, which depends on earlier data; the two agree whenever no batch
    is skipped. explore/harder_sims/methods.py::bet_stream is the corrected
    reference.)

    ``fit(Xf, Yf, alpha) -> (witness callable, u_fit, v_fit)`` with
    self-excluded fit values; default: the exact-Gram trimmed-MMD witness.
    ``cut_rule`` (default ``"linear"``, every published run; ``"order"`` for genuine
    empirical quantiles) is passed to :func:`ramp_cut_points`. ``v2_min`` (default
    1e-8, every run of the paper) floors the running variance of the fractions.
    Returns ``(certified, log_final_wealth)``, or with ``return_detail`` also a
    dict with the horizon, the pairs bet on and the skipped batches.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    X, Y = np.asarray(X), np.asarray(Y)
    n, m = len(X), len(Y)
    bx = [int(round(e * n)) for e in edges]
    by = [int(round(e * m)) for e in edges]
    N = sequential_horizon(n, m, edges)
    if fit is None:
        def fit(Xf, Yf, a):
            w, v, norm = witness_direction(Xf, Yf, a, sigma, rng=rng, **kw)

            def f(Z, w=w, v=v, norm=norm, Xf=Xf, Yf=Yf):
                if norm <= 0:
                    return np.zeros(len(Z))
                return evaluate_direction(w, v, Xf, Yf, Z, sigma, norm)
            u1, v1 = fold1_witness_values(w, v, Xf, Yf, sigma, norm)
            return f, u1, v1
    Zs, pri, skipped = [], [], []
    for b in range(1, len(edges) - 1):
        Xf, Yf = X[:bx[b]], Y[:by[b]]
        Xb, Yb = X[bx[b]:bx[b + 1]], Y[by[b]:by[b + 1]]
        wit, u1, v1 = fit(Xf, Yf, alpha)   # fitted first, as in the research code (same RNG use)
        s, sp = ramp_cut_points(u1, v1, alpha, rule=cut_rule)
        if len(Xb) == 0 or len(Yb) == 0:
            continue                       # no pairs in this batch (it counts 0 in N)
        if not s > sp:
            skipped.append(b)              # lam = 0 on its min(n_b, m_b) pairs
            continue
        G, H = ramp(wit(Xb), s, sp), ramp(wit(Yb), s, sp)
        Zs.append(pair_statistics(G, H) - alpha)
        pri.append(ramp_prior_var(u1, v1, s, sp, len(Xb), len(Yb)))
    detail = {"horizon": N, "pairs_bet": int(sum(z.size for z in Zs)), "skipped_batches": skipped}
    if not Zs or N == 0:
        out = (False, float("-inf") if not Zs else 0.0)
        return (*out, detail) if return_detail else out
    L = _math.log(1.0 / delta)
    lam_max = c / (1.0 + alpha)
    logK, best = 0.0, float("-inf")
    ssd, cnt, mu_sum = 0.0, 0, 0.0
    for z, pv in zip(Zs, pri):
        for y in z:
            mu = mu_sum / (prior_w + cnt)
            sig2 = max((prior_w * pv + ssd) / (prior_w + cnt), v2_min)
            lam = min(lam_max, _math.sqrt(2.0 * L / (N * sig2)))
            logK += _math.log1p(lam * y)
            best = max(best, logK)
            ssd += (y - mu) ** 2
            mu_sum += y
            cnt += 1
    out = (bool(best >= L), float(logK))
    return (*out, detail) if return_detail else out


# ---------------------------------------------------------------------------
# The sequential certificate as the paper's main split result (revision v2,
# author decision Q7; jmlr/sections/06_splitting.tex "The certificate.",
# Theorem thm:betting, appendix B): batches e = (0, 1/4, 1/2, 1).
# ---------------------------------------------------------------------------

SEQ_EDGES = (0.0, 0.25, 0.5, 1.0)


@dataclass
class SequentialOrder:
    """The data-independent row order, batch-0 bandwidth and features of one
    sequential certificate (shared by every level of a lower bound)."""
    X: np.ndarray             # X in the fixed random order
    Y: np.ndarray
    sigma: float              # median heuristic on batch 0 only (or given)
    rff: object               # RFF fitted with that sigma, or None
    rng: object               # the stream after the order, bandwidth and features


def sequential_order(X, Y, rng, edges=SEQ_EDGES, sigma=None, use_rff=False,
                     n_features=512, standardize_batch0=False,
                     rff_map="phased") -> SequentialOrder:
    """Random stream of the sequential certificate, fixed in advance.

    ``rng`` (e.g. ``default_rng(split_seed)``) draws the row permutations of X
    and Y first -- the same two draws as :func:`split_test`, so batches 0-1
    together are the single split's fold one and the last batch (with
    e = (0, 1/4, 1/2, 1)) its fold two -- then the bandwidth is the median
    heuristic on batch 0 only (it must not see the batches bet on), then the
    random features if ``use_rff``. The witness fits draw from the same rng.

    ``standardize_batch0`` (JMLR v3; default off, so every earlier result is unchanged): standardize every coordinate
    by the mean and standard deviation of batch 0 of both samples pooled, after the permutation and before the
    bandwidth. The map then depends on batch 0 only, so the betting guarantee covers it, whereas standardizing by the
    moments of all the data (as ``load_rhc`` does) makes the ramps depend on the batches bet on. No rng is consumed,
    so the order, the bandwidth stream and the witness fits are those of the default run.
    """
    X, Y = np.asarray(X), np.asarray(Y)
    n, m = len(X), len(Y)
    ix, iy = rng.permutation(n), rng.permutation(m)
    Xo, Yo = X[ix], Y[iy]
    b1x, b1y = int(round(edges[1] * n)), int(round(edges[1] * m))
    if standardize_batch0:
        B0 = np.vstack([Xo[:b1x], Yo[:b1y]]).astype(float)
        mu, sd = B0.mean(axis=0), B0.std(axis=0)
        sd[sd == 0] = 1.0
        Xo, Yo = (Xo - mu) / sd, (Yo - mu) / sd
    if sigma is None:
        sigma = median_heuristic(Xo[:b1x], Yo[:b1y], rng=rng)
    rff = (make_rff(X.shape[1], n_features, sigma, rng=rng, rff_map=rff_map)
           if use_rff else None)
    return SequentialOrder(Xo, Yo, float(sigma), rff, rng)


def _sequential_fit(order: SequentialOrder, fit_alpha=None, cache=None, **kw):
    """``fit(Xf, Yf, a)`` for :func:`sequential_betting_certificate`: the
    trimmed-MMD witness on the earlier batches (exact Gram, or the order's
    random features), with self-excluded fit values. ``fit_alpha`` fixes the
    level of the QP (0.0 for the one-witness variant; None = the tested level);
    ``cache`` reuses fits across levels when ``fit_alpha`` is fixed."""
    sigma, rff, rng = order.sigma, order.rff, order.rng

    def fit(Xf, Yf, a):
        a_fit = a if fit_alpha is None else fit_alpha
        key = (len(Xf), len(Yf), float(a_fit))
        if cache is not None and key in cache:
            return cache[key]
        w, v, norm = witness_direction(Xf, Yf, a_fit, sigma, rng=rng, rff=rff, **kw)

        def f(Z, w=w, v=v, norm=norm, Xf=Xf, Yf=Yf):
            return evaluate_direction(w, v, Xf, Yf, Z, sigma, norm, rff=rff)
        u1, v1 = fold1_witness_values(w, v, Xf, Yf, sigma, norm, rff=rff)
        out = (f, u1, v1)
        if cache is not None:
            cache[key] = out
        return out
    return fit


@dataclass
class SequentialCertificate:
    alpha: float
    certified: bool           # max_t K_t >= 1/delta (Theorem thm:betting)
    log_final_wealth: float
    horizon: int              # N, fixed in advance
    pairs_bet: int
    skipped_batches: list     # batches without a ramp (lambda = 0)
    sigma: float


def sequential_certificate(X, Y, alpha, delta=0.05, rng=None, edges=SEQ_EDGES, sigma=None,
                           use_rff=False, n_features=512, c=0.5, prior_w=20.0,
                           order: SequentialOrder | None = None, fit_alpha=None, cache=None,
                           rff_map="phased", cut_rule: str = "linear", **kw) -> SequentialCertificate:
    """One sequential betting certificate that ``TV(P, Q) > alpha`` (the
    paper's main split result): order fixed by ``rng`` (:func:`sequential_order`),
    refit on all earlier batches, one wealth, horizon fixed in advance,
    lambda = 0 on a batch without a valid cut. ``order`` reuses a prepared
    order (e.g. across the levels of a lower bound, each level then drawing its
    witness fits from a fresh copy of the stream is the caller's choice)."""
    if order is None:
        rng = np.random.default_rng(0) if rng is None else rng
        order = sequential_order(X, Y, rng, edges=edges, sigma=sigma, use_rff=use_rff,
                                 n_features=n_features, rff_map=rff_map)
    fit = _sequential_fit(order, fit_alpha=fit_alpha, cache=cache, **kw)
    rej, logk, det = sequential_betting_certificate(order.X, order.Y, alpha, edges, order.sigma,
                                                    rng=order.rng, delta=delta, c=c,
                                                    prior_w=prior_w, fit=fit, return_detail=True,
                                                    cut_rule=cut_rule)
    return SequentialCertificate(float(alpha), bool(rej), float(logk), det["horizon"],
                                 det["pairs_bet"], det["skipped_batches"], order.sigma)


def lower_bound_sequential(X, Y, alphas, delta=0.05, sigma=None, rng=None, split_seed=None,
                           edges=SEQ_EDGES, witness="per_level", none_value=0.0,
                           use_rff=False, n_features=512, c=0.5, prior_w=20.0,
                           rff_map="phased", cut_rule="linear", **kw):
    """Corollary cor:lower with the sequential certificate: test each grid level
    on the SAME batches (same order and batch-0 bandwidth at every level) and
    read the grid upward, stopping at the first failure.

    Seeds as :func:`lower_bound_tv`: ``split_seed`` (default one draw
    ``rng.integers(0, 2**31 - 1)`` from ``rng = default_rng(0)``) and a fresh
    ``default_rng(split_seed)`` per level, so the order, the bandwidth and the
    features are identical at every level. ``witness="alpha0"`` fits each
    batch's witness once at alpha = 0 and only moves the cut points with the
    level. Returns ``(bound, rows)``.
    """
    alphas = np.sort(np.asarray(alphas, dtype=float))
    if split_seed is None:
        rng = np.random.default_rng(0) if rng is None else rng
        split_seed = int(rng.integers(0, 2**31 - 1))
    if witness not in ("per_level", "alpha0"):
        raise ValueError(f"unknown witness {witness!r}")
    cache = {} if witness == "alpha0" else None
    order0 = None
    rows = []
    for a in alphas:
        if witness == "alpha0" and order0 is not None:
            order = order0
        else:
            order = sequential_order(X, Y, np.random.default_rng(split_seed), edges=edges,
                                     sigma=sigma, use_rff=use_rff, n_features=n_features,
                                     rff_map=rff_map)
            order0 = order
        r = sequential_certificate(X, Y, a, delta=delta, edges=edges, c=c, prior_w=prior_w,
                                   order=order, fit_alpha=0.0 if witness == "alpha0" else None,
                                   cache=cache, cut_rule=cut_rule, **kw)
        rows.append({"alpha": float(a), "sequential": r.certified,
                     "seq_log_final_wealth": r.log_final_wealth, "seq_horizon": r.horizon,
                     "seq_pairs_bet": r.pairs_bet, "seq_skipped": list(r.skipped_batches),
                     "seq_sigma": r.sigma})
        if not r.certified:
            break
    return _down_closed([r["alpha"] for r in rows], [r["sequential"] for r in rows],
                        none_value), rows

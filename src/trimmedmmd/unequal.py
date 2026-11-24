"""Unequal trimming fractions: how much of one sample must go when the other loses a given fraction.

Let P be the law of X and Q the law of Y. Trimming P by alpha_x and Q by alpha_y can leave a common remainder R, with
(1 - alpha_x) R <= P and (1 - alpha_y) R <= Q, exactly when the integral of min{p / (1 - alpha_x), q / (1 - alpha_y)}
is at least one. With zeta = (1 - alpha_x) / (1 - alpha_y) this reads

    E_zeta(P || Q) := sup_A {P(A) - zeta Q(A)} = integral of (p - zeta q)_+  <=  alpha_x,

the hockey-stick divergence. Equal fractions give zeta = 1 and E_1 = TV(P, Q), the level alpha* of the paper.

The test of the null H0(alpha_x, alpha_y), "the pair can leave a common remainder", is the paper's sequential betting
certificate with three changes. The witness solves the quadratic program (2.10) with caps 1/(n (1 - alpha_x)) on the
X weights and 1/(m (1 - alpha_y)) on the Y weights. The cut points are t, the (1 - alpha_x)-quantile of the witness
values on X, and u, the alpha_y-quantile on Y; for t > u the ramp psi of the witness clipped to [u, t] satisfies

    inf over alpha_x-trimmings P' of E_P' f  -  sup over alpha_y-trimmings Q' of E_Q' f
        = (t - u) / (1 - alpha_x) * (E_P psi - zeta E_Q psi - alpha_x)

at the clipped witness f, and under H0 every [0, 1]-valued psi has E_P psi - zeta E_Q psi <= alpha_x (the best psi
is 1{p > zeta q}). The bets are on Z_k = G_k - zeta mean(H over block k) - alpha_x, which lies in
[-zeta - alpha_x, 1 - alpha_x] and has mean at most 0 under H0, so the fractions are capped at c / (zeta + alpha_x)
and the wealth is a nonnegative supermartingale (Ville's inequality, as in the paper's Theorem 4.2).

Everything else is the paper's AoS method, as in scripts/aos_main_realdata.py::level_record: the random order and
batch-0 median bandwidth of :func:`trimmedmmd.splitting.sequential_order` (optionally with batch-0 standardization),
batches :data:`~trimmedmmd.splitting.SEQ_EDGES`, the Laplace kernel by default, witnesses fitted on the earlier
batches (:func:`trimmedmmd.aos_fallback.fit_batch`), cut points from the own-term-removed fit values with genuine
empirical quantiles, a batch whose cuts cross is not bet on, c = 1/2 and prior_w = 20. With alpha_y = alpha_x the
certificate is that of level_record, number for number.

For fixed alpha_x the null is monotone in alpha_y (a common remainder at alpha_y is one at every larger alpha_y), so
reading an increasing alpha_y grid upward and stopping at the first failure gives a lower confidence bound on the
frontier alpha_y*(alpha_x) = min{alpha_y : H0(alpha_x, alpha_y) holds} (:func:`frontier_lower_bound`), valid at level
delta by the argument of the paper's Corollary 4.3. Along a ray of fixed zeta the null is monotone in alpha_x, which
gives a lower confidence bound on E_zeta(P || Q) (:func:`hockey_stick_lower_bound`). The paper's rates are proved for
equal fractions only; validity holds for every pair.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .aos_fallback import KERNELS, bet, choose_cuts, fit_batch
from .splitting import (CUT_RULES, SEQ_EDGES, pair_statistics, ramp, ramp_prior_var,
                        sequential_horizon, sequential_order)

_RAY_EPS = 1e-12    # alpha_y = 1 - (1 - alpha_x) / zeta within this of 0 counts as 0


@dataclass
class PairCertificate:
    """The sequential certificate of H0(alpha_x, alpha_y) on one random order."""
    alpha_x: float
    alpha_y: float
    zeta: float               # (1 - alpha_x) / (1 - alpha_y), the weight of the Y side
    certified: bool           # H0 rejected at level delta: the pair cannot leave a common remainder
    log_final_wealth: float   # log K_N, an e-value under H0
    log_max_wealth: float     # max_t log K_t; certified iff >= log(1/delta)
    horizon: int              # N, the number of pairs over all betting batches, fixed in advance
    pairs_bet: int
    cuts: list                # per betting batch [t, u], or None if the batch was not bet on
    batches: list             # per betting batch: fit sizes, witness norm, raw cuts, source, mean Z, variance guess
    sigma: float              # batch-0 median bandwidth (or the one given)
    split_seed: int
    kernel: str
    delta: float
    path: dict | None = None  # with return_path=True: the fractions and the factors 1 + lambda Z of the wealth


def _check_level(name, a):
    a = float(a)
    if not 0.0 <= a < 1.0:
        raise ValueError(f"{name} must lie in [0, 1), got {a!r}")
    return a


def _check_options(kernel, cut_rule, edges):
    if kernel not in KERNELS:
        raise ValueError(f"unknown kernel {kernel!r}; choose from {KERNELS}")
    if cut_rule not in CUT_RULES:
        raise ValueError(f"unknown cut rule {cut_rule!r}; choose from {CUT_RULES}")
    edges = tuple(float(e) for e in edges)
    if len(edges) < 3 or edges[0] != 0.0 or edges[-1] != 1.0 or any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError(f"edges must increase from 0 to 1 with at least one betting batch, got {edges!r}")
    return edges


def _check_betting(delta, c):
    """Level and betting constant: 0 < delta < 1, and 0 < c < 1 so that every wealth factor stays positive."""
    delta, c = float(delta), float(c)
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must lie in (0, 1), got {delta!r}")
    if not 0.0 < c < 1.0:
        raise ValueError(f"c must lie in (0, 1) (c >= 1 can make the wealth negative), got {c!r}")
    return delta, c


def _check_sizes(X, Y, edges):
    """Every batch of the order needs at least two observations of each sample."""
    for name, k in (("X", len(X)), ("Y", len(Y))):
        cuts = [int(round(e * k)) for e in edges]
        if min(b - a for a, b in zip(cuts, cuts[1:])) < 2:
            raise ValueError(f"{name} has {k} observations, too few for the batches {edges}: each batch needs at "
                             "least two")


def _as_2d(Z):
    Z = np.asarray(Z, dtype=float)
    return Z[:, None] if Z.ndim == 1 else Z


def _resolve_split_seed(split_seed, rng):
    """As :func:`trimmedmmd.splitting.lower_bound_sequential`: one draw from ``rng`` (default ``default_rng(0)``)."""
    if split_seed is None:
        rng = np.random.default_rng(0) if rng is None else rng
        split_seed = int(rng.integers(0, 2**31 - 1))
    return int(split_seed)


def _order(X, Y, split_seed, edges, sigma, standardize_batch0):
    """The order, bandwidth and stream of one certificate; a fresh ``default_rng(split_seed)`` every call, so every
    level of a grid sees the same batches and the same bandwidth, and its witness fits draw from the same stream."""
    return sequential_order(X, Y, np.random.default_rng(split_seed), edges=edges, sigma=sigma,
                            standardize_batch0=standardize_batch0)


def _certify(order, alpha_x, alpha_y, zeta, delta, kernel, edges, cut_rule, c, prior_w, split_seed,
             return_path=False, **kw) -> PairCertificate:
    """The certificate of H0(alpha_x, alpha_y) on a prepared order. ``zeta`` is passed separately so that a ray of
    fixed zeta tests exactly E_zeta <= alpha_x; the fit (caps, cuts) uses alpha_y, which any rule of the earlier
    batches may do without touching validity."""
    X, Y = order.X, order.Y
    n, m = len(X), len(Y)
    bx = [int(round(e * n)) for e in edges]
    by = [int(round(e * m)) for e in edges]
    N = sequential_horizon(n, m, edges)
    fits = []
    for b in range(1, len(edges) - 1):
        Xf, Yf = X[:bx[b]], Y[:by[b]]
        Xb, Yb = X[bx[b]:bx[b + 1]], Y[by[b]:by[b + 1]]
        bf = fit_batch(Xf, Yf, Xb, Yb, alpha_x, order.sigma, order.rng, kernel=kernel, alpha_y=alpha_y, **kw)
        fits.append((b, bf, len(Xb), len(Yb), bx[b], by[b]))
    Zs, pri, cuts, binfo = [], [], [], []
    for b, bf, nb, mb, nf, mf in fits:
        s, sp, u, v, pick, raw = choose_cuts(bf.u_own, bf.v_own, bf.u_plain, bf.v_plain, alpha_x, "own",
                                             cut_rule=cut_rule, alpha_y=alpha_y)
        info = {"batch": b, "n_fit": [nf, mf], "n_bet": [nb, mb], "norm": float(bf.norm),
                "cut": list(raw["own"]), "source": pick}
        if nb == 0 or mb == 0:
            info["source"] = "empty"
            pick = "skip"
        if pick == "skip":
            cuts.append(None)
            binfo.append(info)
            continue
        cuts.append([float(s), float(sp)])
        G, H = ramp(bf.fXb, s, sp), ramp(bf.fYb, s, sp)
        Z = pair_statistics(G, H, zeta=zeta) - alpha_x
        pv = ramp_prior_var(u, v, s, sp, nb, mb, zeta=zeta)
        Zs.append(Z)
        pri.append(pv)
        info.update(n_pairs=int(Z.size), mean_Z=float(Z.mean()), min_Z=float(Z.min()), prior_var=float(pv))
        binfo.append(info)
    r = bet(Zs, pri, alpha_x, N, delta=delta, c=c, prior_w=prior_w, zeta=zeta, return_path=return_path)
    path = {"lams": r["lams"], "factors": r["factors"]} if return_path else None
    return PairCertificate(alpha_x=float(alpha_x), alpha_y=float(alpha_y), zeta=float(zeta),
                           certified=bool(r["certified"]), log_final_wealth=float(r["log_final"]),
                           log_max_wealth=float(r["log_max"]), horizon=int(N), pairs_bet=int(r["pairs_bet"]),
                           cuts=cuts, batches=binfo, sigma=float(order.sigma), split_seed=int(split_seed),
                           kernel=kernel, delta=float(delta), path=path)


def pair_certificate(X, Y, alpha_x, alpha_y, delta=0.05, kernel="laplace", sigma=None, split_seed=None, rng=None,
                     edges=SEQ_EDGES, cut_rule="order", standardize_batch0=False, c=0.5, prior_w=20.0,
                     return_path=False, **kw) -> PairCertificate:
    """Test H0(alpha_x, alpha_y): trimming the law of X by alpha_x and that of Y by alpha_y can leave a common
    remainder, i.e. E_zeta(P || Q) <= alpha_x with zeta = (1 - alpha_x) / (1 - alpha_y).

    ``certified`` is True when H0 is rejected. If H0 holds, this happens with probability at most ``delta``, at every
    sample size and for every pair of laws; so a certified pair is, with confidence 1 - delta, one at which no
    alpha_x-trimming of the X law and alpha_y-trimming of the Y law have the same remainder.

    Validity needs the bandwidth and any standardization to depend on the fitting data only. The defaults (the median
    distance on batch 0) and ``standardize_batch0=True`` do. A ``sigma`` passed by the user must be fixed without
    looking at the data (in advance, or on separate data), and X and Y must not be standardized beforehand with the
    moments of the full samples.

    ``split_seed`` fixes the random order (default: one draw from ``rng``, itself ``default_rng(0)`` by default, as
    :func:`trimmedmmd.splitting.lower_bound_sequential`). ``kernel`` is ``"laplace"`` (exp(-|x - y| / h), default)
    or ``"gaussian"``; ``sigma`` is the bandwidth h (default: the median distance on batch 0). ``cut_rule="order"``
    takes genuine empirical quantiles of the own-term-removed fit values, ``"linear"`` numpy's interpolated ones.
    ``standardize_batch0`` standardizes every coordinate by the moments of batch 0 (both samples pooled). ``c`` and
    ``prior_w`` are the betting constants of the paper; ``return_path`` keeps the fractions and the wealth factors.
    Further keywords go to :func:`trimmedmmd.solver.solve_gram`. With ``alpha_y == alpha_x`` (zeta = 1) this is the
    paper's certificate of TV(P, Q) > alpha_x.
    """
    alpha_x, alpha_y = _check_level("alpha_x", alpha_x), _check_level("alpha_y", alpha_y)
    edges = _check_options(kernel, cut_rule, edges)
    delta, c = _check_betting(delta, c)
    X, Y = _as_2d(X), _as_2d(Y)
    _check_sizes(X, Y, edges)
    split_seed = _resolve_split_seed(split_seed, rng)
    order = _order(X, Y, split_seed, edges, sigma, standardize_batch0)
    zeta = (1.0 - alpha_x) / (1.0 - alpha_y)
    return _certify(order, alpha_x, alpha_y, zeta, delta, kernel, edges, cut_rule, c, prior_w, split_seed,
                    return_path=return_path, **kw)


def frontier_lower_bound(X, Y, alpha_x, alphas_y, delta=0.05, kernel="laplace", sigma=None, split_seed=None,
                         rng=None, edges=SEQ_EDGES, cut_rule="order", standardize_batch0=False, c=0.5,
                         prior_w=20.0, none_value=None, **kw):
    """Lower confidence bound on the frontier alpha_y*(alpha_x): how much of the Y law must go if the X law loses
    ``alpha_x``, before the two can coincide.

    The grid ``alphas_y`` is sorted and read upward with :func:`pair_certificate` on the same order, bandwidth and
    witness stream at every level, stopping at the first level not certified. The bound is the largest level
    certified, or ``none_value`` if not even the smallest is. With probability at least 1 - delta,
    alpha_y*(alpha_x) > bound: any alpha_x-trimming of the X law and any trimming of the Y law that leave the same
    remainder remove more than the bound from the Y law. Validity needs no correction for the grid, because the null
    is monotone in alpha_y (the argument of the paper's Corollary 4.3).

    Returns ``(bound, rows)``, rows the :class:`PairCertificate` of every level read. Options as in
    :func:`pair_certificate`.
    """
    alpha_x = _check_level("alpha_x", alpha_x)
    grid = np.sort(np.asarray(alphas_y, dtype=float).ravel())
    for a in grid:
        _check_level("alphas_y", a)
    edges = _check_options(kernel, cut_rule, edges)
    delta, c = _check_betting(delta, c)
    X, Y = _as_2d(X), _as_2d(Y)
    _check_sizes(X, Y, edges)
    split_seed = _resolve_split_seed(split_seed, rng)
    rows, bound = [], none_value
    for ay in grid:
        order = _order(X, Y, split_seed, edges, sigma, standardize_batch0)
        zeta = (1.0 - alpha_x) / (1.0 - float(ay))
        r = _certify(order, alpha_x, float(ay), zeta, delta, kernel, edges, cut_rule, c, prior_w, split_seed, **kw)
        rows.append(r)
        if not r.certified:
            break
        bound = float(ay)
    return bound, rows


def hockey_stick_lower_bound(X, Y, zeta, alphas_x, delta=0.05, kernel="laplace", sigma=None, split_seed=None,
                             rng=None, edges=SEQ_EDGES, cut_rule="order", standardize_batch0=False, c=0.5,
                             prior_w=20.0, none_value=None, **kw):
    """Lower confidence bound on the hockey-stick divergence E_zeta(P || Q) = sup_A {P(A) - zeta Q(A)}.

    Along the ray of fixed ``zeta`` the pair (alpha_x, alpha_y) with alpha_y = 1 - (1 - alpha_x) / zeta has the null
    E_zeta <= alpha_x, monotone in alpha_x. Grid levels with alpha_y outside [0, 1), i.e. alpha_x < 1 - zeta, are
    dropped (there E_zeta >= 1 - zeta > alpha_x holds for every pair of laws). The rest are sorted and read upward on
    one order, stopping at the first failure; with probability at least 1 - delta, E_zeta(P || Q) > bound, the
    largest level certified (``none_value`` if none). ``zeta = 1`` gives the paper's lower bound on TV(P, Q).

    Returns ``(bound, rows)``, rows the :class:`PairCertificate` of every level read. Options as in
    :func:`pair_certificate`.
    """
    zeta = float(zeta)
    if not (zeta > 0.0 and np.isfinite(zeta)):
        raise ValueError(f"zeta must be positive and finite, got {zeta!r}")
    grid = np.sort(np.asarray(alphas_x, dtype=float).ravel())
    for a in grid:
        _check_level("alphas_x", a)
    edges = _check_options(kernel, cut_rule, edges)
    delta, c = _check_betting(delta, c)
    X, Y = _as_2d(X), _as_2d(Y)
    _check_sizes(X, Y, edges)
    split_seed = _resolve_split_seed(split_seed, rng)
    rows, bound = [], none_value
    for ax in grid:
        ay = ((zeta - 1.0) + float(ax)) / zeta      # = 1 - (1 - alpha_x) / zeta, and exactly alpha_x at zeta = 1
        if -_RAY_EPS < ay < 0.0:
            ay = 0.0
        if not 0.0 <= ay < 1.0:
            continue
        order = _order(X, Y, split_seed, edges, sigma, standardize_batch0)
        r = _certify(order, float(ax), ay, zeta, delta, kernel, edges, cut_rule, c, prior_w, split_seed, **kw)
        rows.append(r)
        if not r.certified:
            break
        bound = float(ax)
    return bound, rows

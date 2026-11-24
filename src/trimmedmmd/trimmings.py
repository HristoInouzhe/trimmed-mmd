"""Recovering the trimming itself, not just its cost.

Everything else in this package answers *how much* contamination there is:
the path ``alpha -> D-hat(alpha)``, the test, the estimator ``alpha-hat``.
This module answers the next question --- *which* observations the estimated
contamination sits on --- by returning the minimizing pair in

    D_alpha(P_n, Q_m) = min { dist(P', Q') : P' in R_alpha(P_n),
                                             Q' in R_alpha(Q_m) },

for two geometries:

``geometry="mmd"``
    ``dist`` is the RKHS distance between kernel mean embeddings. The
    problem is the convex QP of :mod:`trimmedmmd.solver`, whose minimizer
    was already computed and then thrown away; here it is kept.

``geometry="wasserstein"``
    ``dist`` is ``W_p``. The problem is then the *partial* (or incomplete)
    transportation problem of Agullo-Antolin's thesis, Chapter 3: minimize
    ``<Pi, C>`` over ``Pi >= 0`` with ``Pi 1 <= a``, ``Pi' 1 <= b`` and
    total mass ``1``, where ``a_i = 1/(n(1-alpha_x))`` and
    ``b_j = 1/(m(1-alpha_y))``. Her Proposition 3.3 reduces it to a
    balanced transportation problem by adding one dummy source and one
    dummy sink; POT's ``ot.partial.partial_wasserstein`` performs exactly
    that reduction, so it is used when available and a HiGHS LP on the same
    dummy-augmented program is the fallback.

In both cases an ``alpha``-trimming of the empirical measure is a weight
vector on the *observed points*,

    w in Delta_alpha(n) := { w : 0 <= w_i <= 1/(n(1-alpha)), sum_i w_i = 1 },

so the two geometries optimize different objectives over the *same*
polytope. That is what makes them comparable, and what makes
``Trimming`` a single return type for both.

Reading the output
------------------
``Trimming.w`` is a probability vector on ``X``: the trimmed sample is the
weighted sample ``(X_i, w_i)``. Three derived objects are usually what is
actually wanted, and all three are provided because they are not
interchangeable:

* ``kept_x`` --- the retained observations, ``w_i > 0``. At the MMD optimum
  with a nonzero witness this is a genuine *subset* of size about
  ``n(1-alpha)``, with at most one observation carrying a fractional
  weight (:func:`fractional_count`, and Proposition "bang-bang" in the
  paper), so ``X[t.kept_x]`` really is a trimmed sample.
* ``t.reweighted("x")`` --- the pair ``(X[kept], w[kept]/sum)``, which is
  the exact minimizer and should be used whenever downstream code accepts
  weights.
* ``t.resample("x", size, rng)`` --- an i.i.d. draw from the trimmed
  empirical measure, for downstream code that does not accept weights.

Caveat, stated here because it is easy to forget
------------------------------------------------
Above the contact level the minimizer is *not unique*: once
``alpha >= alpha^star`` the two regions intersect and every trimming pair
landing on the intersection attains the minimum, so "the" optimal trimming
is a whole face of the polytope and any particular one is an arbitrary
element of it. ``select="least"`` picks the canonical element --- the
feasible minimizer closest to the untrimmed weights, i.e. the one that
trims as little as possible --- and is the default for that reason.
``Trimming.unique`` reports whether the minimizer was identified.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .kernels import gaussian_kernel, make_rff, median_heuristic
from .solver import solve_features, solve_gram

try:  # POT is the preferred backend for the transport geometry.
    import ot as _pot
    import ot.partial as _pot_partial
    HAVE_POT = True
except Exception:  # pragma: no cover - exercised only where POT is absent
    _pot = _pot_partial = None
    HAVE_POT = False


# --------------------------------------------------------------------------
# result type
# --------------------------------------------------------------------------

@dataclass
class Trimming:
    """An optimal trimming pair and everything read off it.

    ``w`` and ``v`` are probability vectors on ``X`` and ``Y``, feasible for
    the capped simplices at levels ``alpha_x`` and ``alpha_y``.
    """

    alpha_x: float
    alpha_y: float
    w: np.ndarray
    v: np.ndarray
    value: float                      # the discrepancy at the optimum
    geometry: str                     # "mmd" or "wasserstein"
    plan: np.ndarray | None = None    # transport plan, geometry="wasserstein"
    unique: bool | None = None        # was the minimizer identified?
    meta: dict = field(default_factory=dict)

    # -- the feasible set -------------------------------------------------
    @property
    def n(self) -> int:
        return self.w.size

    @property
    def m(self) -> int:
        return self.v.size

    @property
    def cap_x(self) -> float:
        return 1.0 / (self.n * (1.0 - self.alpha_x))

    @property
    def cap_y(self) -> float:
        return 1.0 / (self.m * (1.0 - self.alpha_y))

    # -- what was kept ----------------------------------------------------
    def _side(self, side):
        side = side.lower()
        if side in ("x", "0", "p"):
            return self.w, self.cap_x, self.alpha_x
        if side in ("y", "1", "q"):
            return self.v, self.cap_y, self.alpha_y
        raise ValueError(f"side must be 'x' or 'y', got {side!r}")

    def kept(self, side: str = "x", tol: float = 1e-10) -> np.ndarray:
        """Indices of observations the trimming retains (weight > tol*cap)."""
        g, cap, _ = self._side(side)
        return np.nonzero(g > tol * cap)[0]

    @property
    def kept_x(self) -> np.ndarray:
        return self.kept("x")

    @property
    def kept_y(self) -> np.ndarray:
        return self.kept("y")

    def dropped(self, side: str = "x", tol: float = 1e-10) -> np.ndarray:
        g, cap, _ = self._side(side)
        return np.nonzero(g <= tol * cap)[0]

    def reweighted(self, side: str = "x", tol: float = 1e-10):
        """``(indices, weights)`` of the retained part, weights summing to 1."""
        idx = self.kept(side, tol)
        g, _, _ = self._side(side)
        gk = g[idx]
        return idx, gk / gk.sum()

    def apply(self, Z: np.ndarray, side: str = "x", tol: float = 1e-10):
        """``(Z_trimmed, weights)`` --- the trimmed sample, as a weighted set."""
        idx, gk = self.reweighted(side, tol)
        return np.asarray(Z)[idx], gk

    # -- turning the weighted measure into an actual sample ---------------
    def hard(self, side: str = "x") -> np.ndarray:
        """The ``ceil(n(1-alpha))`` highest-weight indices: a hard subsample.

        Use when downstream code cannot take weights *and* must not
        resample. It is a rounding of the optimum, not the optimum: when
        :meth:`fractional_count` is 0 or 1 the two coincide up to the single
        boundary observation, and when it is large they need not.
        """
        g, _, alpha = self._side(side)
        k = int(np.ceil(g.size * (1.0 - alpha)))
        k = max(1, min(k, g.size))
        return np.sort(np.argsort(-g, kind="stable")[:k])

    def resample(self, side: str = "x", size: int | None = None, rng=None,
                 tol: float = 1e-10) -> np.ndarray:
        """Indices of an i.i.d. draw from the trimmed empirical measure.

        This is the honest way to hand a trimmed *sample* to a procedure
        that assumes i.i.d. uniform weights: it draws with replacement from
        ``(X_i, w_i)``, so the draw has the right law by construction but a
        smaller effective sample size than ``size`` suggests --- see
        :meth:`ess`.
        """
        rng = np.random.default_rng() if rng is None else rng
        idx, gk = self.reweighted(side, tol)
        g, _, alpha = self._side(side)
        if size is None:
            size = int(np.floor(g.size * (1.0 - alpha)))
        return rng.choice(idx, size=size, replace=True, p=gk)

    # -- diagnostics ------------------------------------------------------
    def ess(self, side: str = "x") -> float:
        """Kish effective sample size ``(sum w)^2 / sum w^2`` of the trimming.

        For a trimming that is exactly bang-bang at level ``alpha`` this is
        ``n(1-alpha)``; it falls below that when weights spread out.
        """
        g, _, _ = self._side(side)
        return float(1.0 / np.sum(g ** 2))

    def budget_used(self, side: str = "x") -> float:
        """Fraction of the trimming budget actually spent.

        ``1 - (ESS/size)`` measured against ``alpha``: the *realized*
        trimming level. Well below ``alpha`` means the cap was slack, which
        above the contact level is the normal state of affairs.
        """
        g, _, alpha = self._side(side)
        realized = 1.0 - self.ess(side) / g.size
        return float(realized / alpha) if alpha > 0 else 0.0

    def fractional_count(self, side: str = "x", tol: float = 1e-6) -> int:
        """How many weights are strictly between 0 and the cap.

        The bang-bang proposition predicts at most one whenever the witness
        is nonzero and its values at the sample are distinct, so a large
        count is a signal that the solver is at a nonunique optimum (or has
        not converged).
        """
        g, cap, _ = self._side(side)
        return int(np.sum((g > tol * cap) & (g < (1.0 - tol) * cap)))

    def summary(self) -> dict:
        return {
            "geometry": self.geometry,
            "alpha_x": self.alpha_x, "alpha_y": self.alpha_y,
            "value": self.value,
            "n": self.n, "m": self.m,
            "kept_x": int(self.kept_x.size), "kept_y": int(self.kept_y.size),
            "ess_x": self.ess("x"), "ess_y": self.ess("y"),
            "budget_used_x": self.budget_used("x"),
            "budget_used_y": self.budget_used("y"),
            "fractional_x": self.fractional_count("x"),
            "fractional_y": self.fractional_count("y"),
            "unique": self.unique,
            **self.meta,
        }

    def __repr__(self) -> str:
        return (f"Trimming({self.geometry}, alpha=({self.alpha_x:.4g},"
                f"{self.alpha_y:.4g}), value={self.value:.6g}, "
                f"kept={self.kept_x.size}/{self.n} and "
                f"{self.kept_y.size}/{self.m}, unique={self.unique})")


# --------------------------------------------------------------------------
# kernel geometry
# --------------------------------------------------------------------------

def witness(Kxx, Kyy, Kxy, w, v):
    """Witness values ``d(X_i)`` and ``d(Y_j)`` for ``d = mu_{P'} - mu_{Q'}``.

    These are the gradients that drive the first-order conditions: ``w``
    loads its cap on the *smallest* ``d(X_i)`` and ``v`` on the *largest*
    ``d(Y_j)``. Returned unnormalized (the paper's ``f^star`` is this
    divided by its RKHS norm, which does not change the ordering).
    """
    dx = Kxx @ w - Kxy @ v
    dy = Kxy.T @ w - Kyy @ v
    return dx, dy


def witness_at(trim, X, Y, Z, sigma=None, kernel=None):
    """Evaluate the fitted witness at new points.

    The witness ``d = mu_{P'} - mu_{Q'}`` is an element of the RKHS, so it is
    a function on the whole sample space, not only on the training points:

        d(z) = sum_i w_i k(X_i, z) - sum_j v_j k(Y_j, z).

    Two uses. It is the out-of-sample separating direction the split-sample
    certificate needs (Section 6 of the paper), and it is what makes the
    trimming rule *transferable*: a trimming fitted on one sample induces a
    keep/drop rule on any other, which is how the stability of the discarded
    set can be measured across disjoint subsamples.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Z = np.asarray(Z, dtype=float)
    X = X[:, None] if X.ndim == 1 else X
    Y = Y[:, None] if Y.ndim == 1 else Y
    Z = Z[:, None] if Z.ndim == 1 else Z
    rff = trim.meta.get("rff")
    if kernel is None and rff is not None:
        # Same feature map the trimming was fitted with, so the witness is
        # evaluated in the space it was optimized in.
        Zx, Zy, Zz = rff.transform(X), rff.transform(Y), rff.transform(Z)
        return Zz @ (Zx.T @ trim.w - Zy.T @ trim.v)
    if kernel is None:
        sigma = trim.meta.get("sigma") if sigma is None else sigma
        if sigma is None:
            raise ValueError("need sigma or kernel")

        def kernel(A, B):
            return gaussian_kernel(A, B, sigma)

    return kernel(Z, X) @ trim.w - kernel(Z, Y) @ trim.v


def optimal_trimming_mmd(X, Y, alpha, alpha_y=None, sigma=None, rng=None,
                         kernel=None, select="least", ridge0=None,
                         n_ridge=4, tol=1e-12, max_iter=8000,
                         zero_tol=1e-3, tau=None,
                         use_rff=False, n_features=512, rff_map="phased") -> Trimming:
    """The minimizing trimming pair for the empirical trimmed MMD.

    Parameters
    ----------
    alpha, alpha_y
        Trimming levels for ``X`` and ``Y``; ``alpha_y`` defaults to ``alpha``.
        Distinct levels are the asymmetric problem ``W(R_a(P), R_b(Q))``.
    tau : float, optional
        The analytic threshold at this level,
        ``trimmedmmd.threshold.tau(n, m, alpha, delta)``. When given it
        replaces ``zero_tol`` in the uniqueness diagnostic: the minimizer
        counts as identified exactly when ``D-hat > tau``. This is the
        right boundary to use, because ``tau`` is already the level below
        which this package declines to distinguish the discrepancy from
        zero, and because at finite ``n`` with distinct observations and a
        characteristic kernel ``D-hat`` is never exactly zero --- so a test
        against zero can only ever be a test against some tolerance.
    zero_tol : float
        Fallback relative tolerance, used when ``tau`` is not given, below
        which the optimal witness counts as zero
        and the minimizer as unidentified: the test is
        ``D-hat <= zero_tol * MMD(P_n, Q_m)``, measuring the residual
        discrepancy against the untrimmed one rather than against an
        absolute scale the kernel does not fix. ``fractional_count`` is an
        independent check on the same question and does not use it.
    select : {"least", "none"}
        How to break ties when the minimizer is not unique --- which is the
        generic situation once ``alpha >= alpha^star``, where the regions
        already intersect. ``"least"`` runs a short decreasing-``ridge``
        continuation on

            min || mu(w) - mu(v) ||^2 + ridge * (||w - 1/n||^2 + ||v - 1/m||^2),

        strictly convex for every ``ridge > 0``, whose limit as
        ``ridge -> 0`` is the minimizer closest to the untrimmed weights:
        among all trimmings that achieve the minimum, the one that trims
        least. ``"none"`` returns whatever the plain QP lands on.
    rff_map : {"phased", "cossin"}
        Random-feature map when ``use_rff`` (default ``"phased"``); see
        :class:`~trimmedmmd.kernels.RFF`.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    X = X[:, None] if X.ndim == 1 else X
    Y = Y[:, None] if Y.ndim == 1 else Y
    alpha_y = alpha if alpha_y is None else alpha_y
    rng = np.random.default_rng(0) if rng is None else rng

    if kernel is None:
        sigma = median_heuristic(X, Y, rng=rng) if sigma is None else sigma

        def kernel(A, B):
            return gaussian_kernel(A, B, sigma)

    # Two backends, one interface. The exact kernel costs O((n+m)^2) in
    # time and memory; the random-feature one costs O((n+m)r) in both, which
    # is what makes the trimming computable at the sizes Section 7 is about.
    # The weights, the witness and every diagnostic below are the same
    # objects either way -- only the inner products are approximated.
    rff = None
    if use_rff:
        rff = make_rff(X.shape[1], n_features, sigma, rng=rng, rff_map=rff_map)
        Zx, Zy = rff.transform(X), rff.transform(Y)

        def _solve(**kw):
            return solve_features(Zx, Zy, alpha, alpha_y=alpha_y, rng=rng,
                                  tol=tol, max_iter=max_iter, **kw)

        def _witness(w, v):
            d = Zx.T @ w - Zy.T @ v
            return Zx @ d, Zy @ d

        def _value(w, v):
            d = Zx.T @ w - Zy.T @ v
            return float(d @ d)

        scale = float(np.linalg.norm(Zx.mean(0) - Zy.mean(0)))
    else:
        Kxx, Kyy, Kxy = kernel(X, X), kernel(Y, Y), kernel(X, Y)

        def _solve(**kw):
            return solve_gram(Kxx, Kyy, Kxy, alpha, alpha_y=alpha_y, rng=rng,
                              tol=tol, max_iter=max_iter, **kw)

        def _witness(w, v):
            return witness(Kxx, Kyy, Kxy, w, v)

        def _value(w, v):
            return float(w @ (Kxx @ w) + v @ (Kyy @ v) - 2.0 * w @ (Kxy @ v))

        scale = float(np.sqrt(max(
            Kxx.mean() + Kyy.mean() - 2 * Kxy.mean(), 0.0)))

    base = _solve()
    w, v, value = base.w, base.v, base.value
    converged = base.converged

    # Is the minimizer identified? It is exactly when the optimal witness is
    # nonzero -- then the first-order condition pins w down (up to ties in
    # the witness values). A zero witness means the regions already touch
    # and the whole contact face is optimal.
    dx, dy = _witness(w, v)
    scale = scale if scale > 0 else 1.0
    d_hat = float(np.sqrt(max(value, 0.0)))
    cutoff = float(tau) if tau is not None else zero_tol * scale
    unique = bool(d_hat > cutoff)

    if select == "least" and not unique:
        r = float(max(value, 1e-12)) if ridge0 is None else float(ridge0)
        r = max(r, 1e-8)
        ww, vv = w.copy(), v.copy()   # warm start from the plain minimizer
        for _ in range(n_ridge):
            res = _solve(ridge=r, w0=ww, v0=vv)
            ww, vv = res.w, res.v
            r /= 10.0
        # Keep the continuation limit only if it is still a minimizer.
        cand = _value(ww, vv)
        if cand <= value + (zero_tol * scale) ** 2:
            w, v, value = ww, vv, max(cand, 0.0)
            converged = converged and res.converged
        dx, dy = _witness(w, v)
    elif select not in ("least", "none"):
        raise ValueError("select must be 'least' or 'none'")

    return Trimming(
        alpha_x=float(alpha), alpha_y=float(alpha_y), w=w, v=v,
        value=float(np.sqrt(max(value, 0.0))), geometry="mmd",
        unique=unique,
        meta={"sigma": sigma, "converged": bool(converged),
              "select": select, "n_iter": base.n_iter,
              "unique_cutoff": cutoff, "use_rff": bool(use_rff),
              "n_features": int(n_features) if use_rff else None,
              "rff": rff if use_rff else None,
              "witness_x": dx, "witness_y": dy},
    )


# --------------------------------------------------------------------------
# transport geometry
# --------------------------------------------------------------------------

def _cost_matrix(X, Y, p, cost):
    if cost is not None:
        C = np.asarray(cost, dtype=float)
        if C.shape != (X.shape[0], Y.shape[0]):
            raise ValueError("cost must have shape (n, m)")
        return C
    d2 = np.maximum(
        np.einsum("ij,ij->i", X, X)[:, None]
        + np.einsum("ij,ij->i", Y, Y)[None, :]
        - 2.0 * (X @ Y.T), 0.0)
    return d2 if p == 2 else np.sqrt(d2) ** p


def _partial_ot_highs(a, b, C, tol=1e-12):
    """Fallback for :func:`optimal_trimming_wasserstein` without POT.

    Solves the partial program directly (mass-1 constraint, two inequality
    marginals) with HiGHS, so it needs no dummy nodes at all -- the dummy
    construction exists to feed network-simplex codes that require a
    balanced problem, which this is not.
    """
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix, vstack

    n, m = C.shape
    N = n * m
    # row sums <= a
    rows = np.repeat(np.arange(n), m)
    Arow = coo_matrix((np.ones(N), (rows, np.arange(N))), shape=(n, N))
    # column sums <= b
    cols = np.tile(np.arange(m), n)
    Acol = coo_matrix((np.ones(N), (cols, np.arange(N))), shape=(m, N))
    A_ub = vstack([Arow, Acol]).tocsr()
    b_ub = np.concatenate([a, b])
    A_eq = coo_matrix((np.ones(N), (np.zeros(N, int), np.arange(N))),
                      shape=(1, N)).tocsr()
    res = linprog(C.ravel(), A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                  bounds=(0, None), method="highs")
    if not res.success:
        raise RuntimeError(f"partial transport LP failed: {res.message}")
    G = res.x.reshape(n, m)
    G[G < tol] = 0.0
    return G


def optimal_trimming_wasserstein(X, Y, alpha, alpha_y=None, p=2, cost=None,
                                 nb_dummies=1, backend="auto") -> Trimming:
    """The minimizing trimming pair for the empirical trimmed ``W_p``.

    This is the discrete partial transportation problem of Agullo-Antolin
    (2018), Chapter 3: the marginals are capped rather than fixed, and the
    total transported mass is pinned to 1. ``w = Pi 1`` and ``v = Pi' 1``
    are then the optimal trimming weights, feasible for the same capped
    simplices the kernel problem uses.

    ``backend`` is ``"pot"``, ``"highs"`` or ``"auto"`` (POT if importable).
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    X = X[:, None] if X.ndim == 1 else X
    Y = Y[:, None] if Y.ndim == 1 else Y
    alpha_y = alpha if alpha_y is None else alpha_y
    n, m = X.shape[0], Y.shape[0]

    C = _cost_matrix(X, Y, p, cost)
    a = np.full(n, 1.0 / (n * (1.0 - alpha)))       # total 1/(1-alpha)
    b = np.full(m, 1.0 / (m * (1.0 - alpha_y)))     # total 1/(1-alpha_y)

    if backend == "auto":
        backend = "pot" if HAVE_POT else "highs"
    if backend == "pot":
        if not HAVE_POT:
            raise ImportError("POT is not installed; use backend='highs'")
        G = _pot_partial.partial_wasserstein(a, b, C, m=1.0,
                                             nb_dummies=nb_dummies)
    elif backend == "highs":
        G = _partial_ot_highs(a, b, C)
    else:
        raise ValueError("backend must be 'pot', 'highs' or 'auto'")

    G = np.asarray(G, dtype=float)
    mass = float(G.sum())
    if abs(mass - 1.0) > 1e-6:
        warnings.warn(f"transported mass {mass:.6g} differs from 1; "
                      "increase nb_dummies or use backend='highs'")
    w, v = G.sum(axis=1), G.sum(axis=0)
    w = np.clip(w, 0.0, None)
    v = np.clip(v, 0.0, None)
    w, v = w / w.sum(), v / v.sum()
    raw = float((G * C).sum() / max(mass, 1e-300))

    return Trimming(
        alpha_x=float(alpha), alpha_y=float(alpha_y), w=w, v=v,
        value=float(raw ** (1.0 / p)), geometry="wasserstein", plan=G,
        unique=None,
        meta={"p": p, "backend": backend, "cost_raw": raw, "mass": mass},
    )


# --------------------------------------------------------------------------
# front doors
# --------------------------------------------------------------------------

def optimal_trimming(X, Y, alpha, geometry="mmd", **kwargs) -> Trimming:
    """Dispatch to the kernel or the transport trimming at level ``alpha``."""
    if geometry == "mmd":
        return optimal_trimming_mmd(X, Y, alpha, **kwargs)
    if geometry in ("wasserstein", "w2", "ot"):
        return optimal_trimming_wasserstein(X, Y, alpha, **kwargs)
    raise ValueError("geometry must be 'mmd' or 'wasserstein'")


def trimming_path(X, Y, alphas, geometry="mmd", **kwargs):
    """The trimming at each level of a grid, as a list of :class:`Trimming`.

    The feasible sets are nested in ``alpha``, so the retained sets are
    *not* guaranteed nested but usually nearly are;
    :func:`removal_order` turns the path into a ranking.
    """
    return [optimal_trimming(X, Y, float(a), geometry=geometry, **kwargs)
            for a in np.asarray(alphas, dtype=float)]


def removal_order(path, side: str = "x", tol: float = 1e-10):
    """Order in which observations leave along an increasing ``alpha`` path.

    Returns ``rank``, an integer array with ``rank[i]`` the index of the
    first level at which observation ``i`` is dropped, and ``len(path)`` for
    observations never dropped. A contamination diagnostic: the points that
    go first are the ones the discrepancy blames.
    """
    n = path[0].w.size if side.lower() in ("x", "0", "p") else path[0].v.size
    rank = np.full(n, len(path), dtype=int)
    for t, trim in enumerate(path):
        gone = trim.dropped(side, tol)
        fresh = gone[rank[gone] == len(path)]
        rank[fresh] = t
    return rank


def trim_at(X, Y, alpha_hat, geometry="mmd", **kwargs) -> Trimming:
    """Produce the trimmed samples at an estimated contamination level.

    The intended use: run :func:`trimmedmmd.estimate.alpha_hat_from_path`
    (or ``bisect_alpha_hat``) to get evidence on ``alpha``, then call this to
    turn that number into data. Because ``alpha-hat`` is by construction at
    or just past the crossing, the minimizer there is typically *not*
    unique, which is why the default ``select="least"`` matters; check
    ``Trimming.unique`` on the result.
    """
    return optimal_trimming(X, Y, alpha_hat, geometry=geometry, **kwargs)

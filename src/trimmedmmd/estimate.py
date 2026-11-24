"""The threshold-crossing estimator of the contamination level."""

from __future__ import annotations

import numpy as np

from .threshold import tau, threshold_function


def _resolve(threshold_fn, threshold, n, m, delta, kappa, kernel_consts):
    """The threshold the crossing search uses.

    ``threshold_fn`` (a callable) wins; otherwise ``threshold`` names one of
    :data:`trimmedmmd.threshold.THRESHOLDS`: ``"paper"`` (default, v1's
    tau_{n,m}) or ``"plugin"`` (revision v2's tau_new, Theorem P /
    Corollary C, with ``kernel_consts``; default Gaussian constants).
    """
    if threshold_fn is not None:
        return threshold_fn
    if threshold == "paper":
        return lambda a: tau(n, m, a, delta, kappa)
    return threshold_function(threshold, n, m, delta, kappa=kappa,
                              kernel_consts=kernel_consts)


def alpha_hat_from_path(alphas, d_hats, n, m, delta, kappa=1.0,
                        threshold_fn=None, threshold="paper",
                        kernel_consts=None) -> float:
    """First alpha on the grid at which the discrepancy drops to the threshold.

    This is `eq:alphaest` evaluated on a grid: the infimum of the crossing
    set. ``threshold_fn(alpha) -> float`` overrides the analytic threshold,
    which is how the oracle-calibrated comparison in the experiments is run.
    ``threshold="plugin"`` uses the revision-v2 threshold tau_new
    (:func:`trimmedmmd.threshold.tau_plugin`); the default ``"paper"`` is
    unchanged.
    """
    alphas = np.asarray(alphas, dtype=float)
    d_hats = np.asarray(d_hats, dtype=float)
    fn = _resolve(threshold_fn, threshold, n, m, delta, kappa, kernel_consts)
    thr = np.array([fn(a) for a in alphas])
    crossed = np.nonzero(d_hats <= thr)[0]
    if crossed.size == 0:
        return float(alphas[-1])  # no crossing on the grid
    return float(alphas[crossed[0]])


def bisect_alpha_hat(d_hat_at, lo=0.0, hi=0.95, n=None, m=None, delta=0.05,
                     kappa=1.0, tol=1e-3, max_iter=40, threshold_fn=None,
                     threshold="paper", kernel_consts=None, start=None):
    """Bracketed bisection for the crossing level (milestone M4).

    Maintains the largest rejected and smallest non-rejected level
    explicitly, which is the bookkeeping the earlier implementation got
    wrong (work programme, WP1.5). ``threshold`` as in
    :func:`alpha_hat_from_path` (default ``"paper"``, unchanged). Bisection
    is valid for any threshold that is nondecreasing in alpha, which both
    the paper's tau and tau_new are.

    ``start`` (JMLR v3, Algorithm alg:bisect): the first level tested after
    ``lo``, e.g. a guess from earlier data; ``None`` (default) keeps the
    midpoint, so existing results are unchanged. Returns the smallest
    accepted level found; :func:`bisect_alpha_hat_report` also returns the
    largest rejected level and the number of tests.
    """
    return bisect_alpha_hat_report(d_hat_at, lo=lo, hi=hi, n=n, m=m, delta=delta, kappa=kappa,
                                   tol=tol, max_iter=max_iter, threshold_fn=threshold_fn,
                                   threshold=threshold, kernel_consts=kernel_consts,
                                   start=start)["accepted"]


def bisect_alpha_hat_report(d_hat_at, lo=0.0, hi=0.95, n=None, m=None, delta=0.05,
                            kappa=1.0, tol=1e-3, max_iter=40, threshold_fn=None,
                            threshold="paper", kernel_consts=None, start=None):
    """Algorithm alg:bisect of the JMLR v3 paper, with its bookkeeping exposed.

    ``hi`` must be a level that is accepted (``d_hat_at(hi) <= tau(hi)``),
    e.g. the smallest level with ``tau >= Delta_phi``; it is not tested.
    The acceptance set ``{alpha : d_hat_at(alpha) <= tau(alpha)}`` is an
    interval ``[alpha_hat, 1)`` when the path is nonincreasing and the
    threshold nondecreasing (Lemma lem:crossing), so at every step
    ``alpha_hat`` lies in ``(last_rejected, accepted]``.

    Returns a dict with ``accepted`` (an accepted level within ``tol`` of
    alpha-hat, or ``lo`` if ``lo`` is accepted), ``last_rejected`` (the
    largest rejected level visited, ``None`` if ``lo`` is accepted; on the
    event of Lemma lem:oneevent it is below alpha-star), ``tests`` (number of
    levels tested, i.e. convex solves) and ``visited`` (list of
    ``(alpha, rejected)`` in order).
    """
    fn = _resolve(threshold_fn, threshold, n, m, delta, kappa, kernel_consts)
    visited = []

    def rejected(a):
        r = bool(d_hat_at(a) > fn(a))
        visited.append((float(a), r))
        return r

    if not rejected(lo):
        return {"accepted": float(lo), "last_rejected": None, "tests": 1, "visited": visited}
    largest_rejected = lo
    smallest_accepted = hi
    a = start if (start is not None and lo < start < hi) else 0.5 * (lo + hi)
    while smallest_accepted - largest_rejected >= tol and len(visited) < max_iter + 1:
        if rejected(a):
            largest_rejected = a
        else:
            smallest_accepted = a
        a = 0.5 * (largest_rejected + smallest_accepted)
    return {"accepted": float(smallest_accepted), "last_rejected": float(largest_rejected),
            "tests": len(visited), "visited": visited}

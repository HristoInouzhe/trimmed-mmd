"""Euclidean projection onto the capped simplex."""

from __future__ import annotations

import numpy as np


def project_capped_simplex(y: np.ndarray, cap: float, total: float = 1.0,
                           tol: float = 1e-12, max_iter: int = 200) -> np.ndarray:
    """Project ``y`` onto ``{w : 0 <= w_i <= cap, sum_i w_i = total}``.

    The solution has the form ``w(theta) = clip(y - theta, 0, cap)`` for a
    scalar ``theta`` fixed by the sum constraint; since ``sum w(theta)`` is
    continuous and nonincreasing in ``theta``, a bisection is exact up to
    floating point.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if cap * n < total - 1e-12:
        raise ValueError(
            f"infeasible: cap*n = {cap * n:.6g} < total = {total:.6g}"
        )

    def s(theta: float) -> float:
        return float(np.clip(y - theta, 0.0, cap).sum())

    # sum is nonincreasing in theta; bracket the root of s(theta) = total.
    lo, hi = y.min() - cap - 1.0, y.max() + 1.0
    if s(lo) < total:  # degenerate, cap binding everywhere
        return np.clip(y - lo, 0.0, cap)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if s(mid) > total:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    w = np.clip(y - 0.5 * (lo + hi), 0.0, cap)
    # Guard against tiny sum drift from the bisection tolerance.
    ssum = w.sum()
    if ssum > 0:
        w *= total / ssum
        np.clip(w, 0.0, cap, out=w)
    return w

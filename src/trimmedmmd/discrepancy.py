"""Trimmed MMD from samples, exact-kernel and random-feature versions."""

from __future__ import annotations

import numpy as np

from .kernels import gaussian_kernel, make_rff, median_heuristic
from .solver import solve_features, solve_gram, solve_path


def trimmed_mmd_gram(X, Y, alpha, sigma=None, rng=None, **kw):
    """Unsquared trimmed MMD at a single alpha, via the exact kernel."""
    sigma = median_heuristic(X, Y, rng=rng) if sigma is None else sigma
    Kxx = gaussian_kernel(X, X, sigma)
    Kyy = gaussian_kernel(Y, Y, sigma)
    Kxy = gaussian_kernel(X, Y, sigma)
    return solve_gram(Kxx, Kyy, Kxy, alpha, rng=rng, **kw)


def trimmed_mmd_path(X, Y, alphas, sigma=None, rng=None, use_rff=False,
                     n_features=512, rff_map="phased", **kw):
    """Trimmed MMD along a monotone alpha-path, with warm starts.

    Set ``use_rff`` for large samples: the feature backend never forms a
    dense Gram matrix, which is what keeps memory linear in the sample size.
    ``rff_map``: ``"phased"`` (default) or ``"cossin"`` (diagonal exactly 1),
    see :class:`~trimmedmmd.kernels.RFF`.
    """
    sigma = median_heuristic(X, Y, rng=rng) if sigma is None else sigma
    alphas = np.asarray(sorted(alphas), dtype=float)

    if use_rff:
        rff = make_rff(X.shape[1], n_features, sigma, rng=rng, rff_map=rff_map)
        Zx, Zy = rff.transform(X), rff.transform(Y)

        def solve_one(a, w0, v0):
            return solve_features(Zx, Zy, a, w0=w0, v0=v0, rng=rng, **kw)
    else:
        Kxx = gaussian_kernel(X, X, sigma)
        Kyy = gaussian_kernel(Y, Y, sigma)
        Kxy = gaussian_kernel(X, Y, sigma)

        def solve_one(a, w0, v0):
            return solve_gram(Kxx, Kyy, Kxy, a, w0=w0, v0=v0, rng=rng, **kw)

    results = solve_path(solve_one, alphas, **{})
    return alphas, np.array([r.discrepancy for r in results]), results


def plain_mmd(X, Y, sigma=None, rng=None) -> float:
    """Ordinary (untrimmed) MMD, i.e. the alpha = 0 case in closed form.

    Used for the cheap necessary-condition screen of `eq:necessary`.
    """
    sigma = median_heuristic(X, Y, rng=rng) if sigma is None else sigma
    n, m = len(X), len(Y)
    Kxx = gaussian_kernel(X, X, sigma)
    Kyy = gaussian_kernel(Y, Y, sigma)
    Kxy = gaussian_kernel(X, Y, sigma)
    val = Kxx.sum() / (n * n) + Kyy.sum() / (m * m) - 2.0 * Kxy.sum() / (n * m)
    return float(np.sqrt(max(val, 0.0)))

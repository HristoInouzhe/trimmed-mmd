"""Kernels, bandwidth selection, and random Fourier features."""

from __future__ import annotations

import numpy as np


def sqdist(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    xx = np.einsum("ij,ij->i", X, X)[:, None]
    yy = np.einsum("ij,ij->i", Y, Y)[None, :]
    d = xx + yy - 2.0 * (X @ Y.T)
    np.maximum(d, 0.0, out=d)
    return d


def gaussian_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian kernel with k(x,x) = 1, so the paper's bound kappa equals 1."""
    return np.exp(-sqdist(X, Y) / (2.0 * sigma * sigma))


def median_heuristic(X: np.ndarray, Y: np.ndarray | None = None,
                     max_points: int = 2000, rng=None) -> float:
    """Median pairwise distance, excluding the zero diagonal.

    Including the diagonal biases the median downward (and, for small
    samples with duplicates, can return zero); both are guarded here. This
    is defect WP1.8 of the work programme.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    Z = X if Y is None else np.vstack([X, Y])
    if Z.shape[0] > max_points:
        idx = rng.choice(Z.shape[0], size=max_points, replace=False)
        Z = Z[idx]
    d2 = sqdist(Z, Z)
    iu = np.triu_indices_from(d2, k=1)  # strictly upper: excludes diagonal
    d = np.sqrt(d2[iu])
    d = d[d > 0]
    if d.size == 0:
        return 1.0  # degenerate sample; any positive bandwidth will do
    med = float(np.median(d))
    return med if med > 0 else 1.0


def linear_kernel(X: np.ndarray, Y: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Normalized linear kernel, used deliberately as a NON-characteristic
    kernel to test what happens when ass:regularity(A3) fails."""
    return (X @ Y.T) / (scale * scale)


def matern_kernel(X: np.ndarray, Y: np.ndarray, rho: float, nu: float = 1.5) -> np.ndarray:
    """Matern kernel, k(x,x)=1, half-integer nu (closed form, no Bessel
    functions needed -- avoids a scipy.special.kv dependency and the extra
    numerical-verification burden that would bring for what is otherwise
    just a spectral-decay-rate testbed).

    Used only for CLAIMS.md n14 (cor:marginderived): unlike the Gaussian
    kernel (whose spectral density decays faster than any polynomial), the
    Matern spectral density is comparable to (1+||xi||^2)^{-s} with
    s = nu + d/2 -- exactly the form Assumption "Sobolev smoothness" and
    Corollary "the margin condition, derived" require, which the Gaussian
    kernel structurally cannot supply.
    """
    d = np.sqrt(sqdist(X, Y))
    if nu == 0.5:
        return np.exp(-d / rho)
    if nu == 1.5:
        z = np.sqrt(3.0) * d / rho
        return (1.0 + z) * np.exp(-z)
    if nu == 2.5:
        z = np.sqrt(5.0) * d / rho
        return (1.0 + z + z * z / 3.0) * np.exp(-z)
    raise ValueError("matern_kernel: only nu in {0.5, 1.5, 2.5} implemented "
                     "(half-integer closed forms)")


class RFF:
    """Random Fourier features for the Gaussian kernel, normalized so that
    E[z(x)'z(y)] = k(x,y) with k(x,x) = 1.

    ``paired=False`` (default): the phased map ``sqrt(2/r) cos(W'x + b)``
    with r = ``n_features`` frequencies; its diagonal ``z(x)'z(x)`` is 1 only
    on average. ``paired=True``: the cos/sin map
    ``sqrt(1/r') [cos(W'x), sin(W'x)]`` with r' = ``n_features // 2``
    frequencies (2r' features in all, so the dimension is ``n_features``
    when it is even); its diagonal is exactly 1, so the feature kernel is
    bounded by kappa = 1 exactly. It draws only W (no phases), so its random
    stream differs from the phased map's.
    """

    def __init__(self, dim: int, n_features: int, sigma: float, rng=None,
                 paired: bool = False):
        rng = np.random.default_rng(0) if rng is None else rng
        self.n_features = n_features
        self.sigma = sigma
        self.paired = bool(paired)
        # Gaussian kernel exp(-||x-y||^2/(2 sigma^2)) has spectral measure
        # N(0, sigma^{-2} I).
        if self.paired:
            self.n_freq = n_features // 2
            if self.n_freq < 1:
                raise ValueError("paired RFF needs n_features >= 2")
            self.W = rng.standard_normal((dim, self.n_freq)) / sigma
            self.b = None
        else:
            self.W = rng.standard_normal((dim, n_features)) / sigma
            self.b = rng.uniform(0.0, 2.0 * np.pi, size=n_features)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.paired:
            P = X @ self.W
            return np.sqrt(1.0 / self.n_freq) * np.hstack([np.cos(P), np.sin(P)])
        return np.sqrt(2.0 / self.n_features) * np.cos(X @ self.W + self.b)


RFF_MAPS = ("phased", "cossin")


def make_rff(dim: int, n_features: int, sigma: float, rng=None,
             rff_map: str = "phased") -> RFF:
    """:class:`RFF` by map name: ``"phased"`` (default) or ``"cossin"`` (paired)."""
    if rff_map not in RFF_MAPS:
        raise ValueError(f"rff_map must be one of {RFF_MAPS}, not {rff_map!r}")
    return RFF(dim=dim, n_features=n_features, sigma=sigma, rng=rng,
               paired=(rff_map == "cossin"))


def realized_kappa(Z: np.ndarray) -> float:
    """``sup_x k_hat(x, x)^{1/2}`` for an explicit feature matrix ``Z``.

    The phased random-feature map ``sqrt(2/r) cos(W'x + b)`` is unbiased for a
    kernel with ``k(x, x) = kappa^2``, but its realized diagonal is not exactly
    ``kappa^2``: it fluctuates by ``O(r^{-1/2})`` around it. The analytic
    threshold of the paper's `thm:concentration` is stated in terms of a bound on
    ``k(x, x)``, so a test run on the feature kernel must use the realized
    bound rather than the nominal one, or its level is not the stated level.

    Pass the value returned here as ``kappa`` to
    :func:`trimmedmmd.threshold.tau`. For the cos/sin map, whose diagonal is
    exactly ``kappa^2``, it returns ``kappa`` up to floating point.
    """
    Z = np.asarray(Z, dtype=float)
    return float(np.sqrt(np.max(np.einsum("ij,ij->i", Z, Z))))

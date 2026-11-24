"""Data-generating processes with exactly known total-variation distance.

Everything here rests on one identity. Under the contamination model

    P = (1 - a) R + a P0,     Q = (1 - a) R + a Q0,

the shared component cancels exactly in the density difference, so

    TV(P, Q) = a * TV(P0, Q0),

whatever R is. Two choices of (P0, Q0) then give exact ground truth:

* ``disjoint``  -- P0 and Q0 are uniform on disjoint boxes, so
  TV(P0, Q0) = 1 and TV(P, Q) = a exactly;
* ``gaussian``  -- P0 = N(mu, I), Q0 = N(-mu, I), for which
  TV(P0, Q0) = 2 Phi(||mu||) - 1 in closed form.

Having two families matters: the first pins alpha* to the contamination
fraction itself, the second decouples them, so a method cannot look correct
merely by recovering the mixing weight.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class Scenario:
    name: str
    dim: int
    alpha_star: float
    sample: callable  # (n, rng) -> (X, Y)

    def draw(self, n, m, rng):
        X = self.sample_one(n, rng, which="P")
        Y = self.sample_one(m, rng, which="Q")
        return X, Y

    def sample_one(self, n, rng, which):
        return self.sample(n, rng, which)


def _mixture_sampler(core_sampler, contam_sampler_P, contam_sampler_Q, a):
    def sample(n, rng, which):
        k = rng.binomial(n, a)
        base = core_sampler(n - k, rng)
        contam = (contam_sampler_P if which == "P" else contam_sampler_Q)(k, rng)
        out = np.vstack([base, contam]) if k > 0 else base
        rng.shuffle(out)
        return out
    return sample


def disjoint_contamination(dim=2, a=0.2, sep=8.0, width=1.0) -> Scenario:
    """Common Gaussian core; contaminations uniform on disjoint boxes.

    Because the two contaminating laws are mutually singular,
    TV(P, Q) = a exactly -- no approximation, at any separation.
    """
    def core(k, rng):
        return rng.standard_normal((k, dim))

    def cP(k, rng):
        return rng.uniform(-sep - width, -sep, size=(k, dim))

    def cQ(k, rng):
        return rng.uniform(sep, sep + width, size=(k, dim))

    return Scenario(
        name=f"disjoint(d={dim},a={a})",
        dim=dim,
        alpha_star=a,
        sample=_mixture_sampler(core, cP, cQ, a),
    )


def gaussian_contamination(dim=2, a=0.3, mu_norm=1.0) -> Scenario:
    """Common core; Gaussian contaminations with closed-form TV.

    TV(P0, Q0) = 2 Phi(||mu||) - 1 for P0 = N(mu, I), Q0 = N(-mu, I), since
    the two differ only along one direction at Mahalanobis separation
    2||mu||. Hence alpha* = a * (2 Phi(||mu||) - 1) < a.
    """
    mu = np.zeros(dim)
    mu[0] = mu_norm
    tv0 = 2.0 * norm.cdf(mu_norm) - 1.0

    def core(k, rng):
        return rng.standard_normal((k, dim))

    def cP(k, rng):
        return rng.standard_normal((k, dim)) + mu

    def cQ(k, rng):
        return rng.standard_normal((k, dim)) - mu

    return Scenario(
        name=f"gaussian(d={dim},a={a},|mu|={mu_norm})",
        dim=dim,
        alpha_star=a * tv0,
        sample=_mixture_sampler(core, cP, cQ, a),
    )


@dataclass
class SobolevBumpPair:
    """CLAIMS.md n14 (cor:marginderived): P, Q with a controllable, verified
    Sobolev smoothness t on p-q.

    p = (f0 + eps*bump)/Z_p, q = (f0 - eps*bump)/Z_p on a fine 1-D grid,
    f0 uniform on [-L, L], bump a degree-m B-spline (an (m+1)-fold
    self-convolution of a box, via FFT). A cardinal B-spline of degree m
    has known critical Sobolev regularity m + 1/2 (standard approximation
    theory fact, not re-derived here) -- confirmed numerically for this
    exact construction, by direct evaluation of the defining Sobolev-norm
    integral's convergence/divergence on either side of that exponent, in
    the theory-stress-suite pass that added this (see CLAIMS.md's triage
    log for the check and why a naive FFT-slope fit was rejected first: the
    box's Fourier transform is a sinc power, whose oscillation biases a
    log-log OLS slope fit badly near its zero-crossings).
    """
    m: int
    t_used: float          # the Sobolev exponent actually targeted (< m+0.5)
    alpha_star: float       # TV(P,Q), computed numerically on the grid
    eps: float
    x: np.ndarray
    cdf_p: np.ndarray
    cdf_q: np.ndarray

    def sample(self, n, rng, which):
        u = rng.uniform(0.0, 1.0, size=n)
        cdf = self.cdf_p if which == "P" else self.cdf_q
        return np.interp(u, cdf, self.x).reshape(-1, 1)


def sobolev_bump_pair(m: int, eps: float = 0.01, L: float = 8.0,
                      N: int = 2 ** 16, box_width: float = 1.0,
                      margin_below_critical: float = 0.2) -> SobolevBumpPair:
    x = np.linspace(-L, L, N, endpoint=False)
    dx = x[1] - x[0]
    box = (np.abs(x) <= box_width / 2).astype(float) / box_width
    fbox = np.fft.rfft(box)
    fbump = fbox ** (m + 1)
    bump = np.fft.fftshift(np.fft.irfft(fbump, n=N)) * dx ** m
    bump = np.maximum(bump, 0.0)  # (m+1)-fold convolution of a nonneg
    # function is exactly nonneg; clip only float noise at the numerical floor

    f0 = np.full(N, 1.0 / (2.0 * L))
    p = f0 + eps * bump
    q = f0 - eps * bump
    if q.min() < 0:
        raise ValueError(f"eps={eps} too large for m={m}: q goes negative "
                         f"(min {q.min():.4g}); reduce eps or widen L")
    Zp, Zq = float(np.sum(p) * dx), float(np.sum(q) * dx)
    p, q = p / Zp, q / Zq

    alpha_star = 0.5 * float(np.sum(np.abs(p - q)) * dx)

    cdf_p = np.cumsum(p) * dx
    cdf_p = cdf_p / cdf_p[-1]
    cdf_q = np.cumsum(q) * dx
    cdf_q = cdf_q / cdf_q[-1]

    return SobolevBumpPair(m=m, t_used=m + 0.5 - margin_below_critical,
                           alpha_star=alpha_star, eps=eps, x=x,
                           cdf_p=cdf_p, cdf_q=cdf_q)


def mean_matched_disjoint(dim=1, a=0.25, sep=8.0, width=1.0, inner=1.0) -> Scenario:
    """Contaminations that are mutually singular but share every first moment.

    P0 is uniform on two far boxes symmetric about the origin, Q0 uniform on
    a central box. Supports are disjoint, so TV(P, Q) = a exactly, yet P and
    Q have identical means. A kernel that only sees means -- i.e. a
    non-characteristic one -- must therefore report zero discrepancy at every
    trimming level, which is what makes this the sharp test of
    ass:regularity(A3).
    """
    def core(k, rng):
        return rng.standard_normal((k, dim))

    def cP(k, rng):
        half = rng.integers(0, 2, size=(k, 1))
        lo = np.where(half == 1, sep, -sep - width)
        return lo + rng.uniform(0.0, width, size=(k, dim))

    def cQ(k, rng):
        return rng.uniform(-inner, inner, size=(k, dim))

    return Scenario(
        name=f"mean_matched(d={dim},a={a})",
        dim=dim,
        alpha_star=a,
        sample=_mixture_sampler(core, cP, cQ, a),
    )


def functional_contamination(n_grid=100, a=0.20, n_basis=6, offset=8.0) -> Scenario:
    """Contaminated random curves: the sample space is a function space.

    Nothing in the theory refers to a dimension -- ass:regularity asks only for
    a bounded, measurable, characteristic kernel on *some* sample space, and
    the concentration radius contains no d. This scenario exercises that:
    observations are curves sampled on a grid, compared through a Gaussian
    kernel on the (discretized) L2 distance, so the ambient dimension is the
    grid size and the natural object is a point of a function space.

    The common core is a smooth Gaussian random curve. Both contaminations
    are built from *bounded* coefficients and pushed to disjoint levels, so
    their laws are mutually singular as laws on curves and TV(P,Q) = a
    exactly, by the same identity used for the finite-dimensional families.
    """
    t = np.linspace(0.0, 1.0, n_grid)
    basis = np.stack([np.sin((j + 1) * np.pi * t) for j in range(n_basis)])

    def core(k, rng):
        coef = rng.standard_normal((k, n_basis))
        return coef @ basis

    def _bounded(k, rng, shift):
        coef = rng.uniform(-1.0, 1.0, size=(k, n_basis)) / n_basis
        return coef @ basis + shift

    def cP(k, rng):
        return _bounded(k, rng, +offset)

    def cQ(k, rng):
        return _bounded(k, rng, -offset)

    return Scenario(
        name=f"functional(grid={n_grid},a={a})",
        dim=n_grid,
        alpha_star=a,
        sample=_mixture_sampler(core, cP, cQ, a),
    )


def identical(dim=2) -> Scenario:
    """P = Q, so alpha* = 0. The degenerate corner of `thm:consistency`(i)."""
    def sample(n, rng, which):
        return rng.standard_normal((n, dim))

    return Scenario(name=f"identical(d={dim})", dim=dim, alpha_star=0.0,
                    sample=sample)


CATALOGUE = {
    "disjoint_d2_a20": lambda: disjoint_contamination(dim=2, a=0.20),
    "disjoint_d5_a20": lambda: disjoint_contamination(dim=5, a=0.20),
    "disjoint_d2_a35": lambda: disjoint_contamination(dim=2, a=0.35),
    "gaussian_d2_a30": lambda: gaussian_contamination(dim=2, a=0.30, mu_norm=1.0),
    "identical_d2": lambda: identical(dim=2),
    "mean_matched_d1_a25": lambda: mean_matched_disjoint(dim=1, a=0.25),
    "functional_a20": lambda: functional_contamination(n_grid=100, a=0.20),
}


def make(key: str) -> Scenario:
    """Build a scenario by catalogue key (picklable entry point for workers)."""
    return CATALOGUE[key]()

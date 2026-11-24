"""Shared helpers: null designs with a known common trimming, the plug-in
decomposition of Theorem P, and the QP statistic."""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

from trimmedmmd import dgp                                   # noqa: E402
from trimmedmmd.kernels import gaussian_kernel, median_heuristic  # noqa: E402
from trimmedmmd.solver import solve_gram, solve_path          # noqa: E402


def dhat(Kxx, Kyy, Kxy, alpha, rng=None):
    r = solve_gram(Kxx, Kyy, Kxy, alpha, rng=rng or np.random.default_rng(0),
                   max_iter=6000, tol=1e-12)
    return r.discrepancy, r.discrepancy_lower, r.converged


# --------------------------------------------------------------------------
# null designs at the boundary TV = alpha*, with the common trimming R known
# --------------------------------------------------------------------------
#
# Each design draws (X, Y) and returns, for the plug-in check, the densities
# a = dR/dP at the X's, b = dR/dQ at the Y's, mu_R evaluated at the X's and
# Y's, and ||mu_R||^2, all under the kernel actually used.

class Design:
    name: str
    alpha_star: float
    plugin: bool = True

    def draw(self, n, m, rng):
        raise NotImplementedError

    def sigma(self, X, Y, rng):
        raise NotImplementedError


class Disjoint(Design):
    """The paper's disjoint family: core N(0, I_d), contaminations uniform on
    the boxes [-sep-w, -sep]^d (P) and [sep, sep+w]^d (Q). R = core."""

    def __init__(self, d=2, a=0.2, bw="median", sep=8.0, width=1.0, bw_scale=1.0,
                 fixed_sigma=None, seed_offset=0):
        self.d, self.a, self.bw, self.sep, self.w = d, a, bw, sep, width
        self.bw_scale, self.fixed_sigma = bw_scale, fixed_sigma
        self.alpha_star = a
        self.sc = dgp.disjoint_contamination(dim=d, a=a, sep=sep, width=width)
        self.name = f"disjoint_d{d}_a{a}_{bw}" + (f"x{bw_scale}" if bw_scale != 1 else "")

    def draw(self, n, m, rng):
        return self.sc.sample(n, rng, "P"), self.sc.sample(m, rng, "Q")

    def sigma(self, X, Y, rng):
        if self.bw == "median":
            return self.bw_scale * median_heuristic(X, Y, rng=rng)
        return self.bw_scale * self.fixed_sigma

    def _box_density(self, Z, lo):
        inside = np.all((Z >= lo) & (Z <= lo + self.w), axis=1)
        return inside / self.w ** self.d

    def densities(self, X, Y):
        a = self.a
        f0 = lambda Z: np.exp(-0.5 * np.sum(Z * Z, 1)) / (2 * np.pi) ** (self.d / 2)
        fx = f0(X)
        p = (1 - a) * fx + a * self._box_density(X, -self.sep - self.w)
        fy = f0(Y)
        q = (1 - a) * fy + a * self._box_density(Y, self.sep)
        # dR/dP = f0/p, dR/dQ = f0/q  (R = core, (1-a) R = P ^ Q)
        return fx / p, fy / q

    def mu_R(self, Z, sigma):
        s2 = sigma * sigma
        return (s2 / (s2 + 1)) ** (self.d / 2) * np.exp(-np.sum(Z * Z, 1) / (2 * (s2 + 1)))

    def mu_R_sq(self, sigma):
        s2 = sigma * sigma
        return (s2 / (s2 + 2)) ** (self.d / 2)


class Atoms(Design):
    """Finitely many atoms, pairwise far apart (features orthonormal under a
    unit-bandwidth Gaussian kernel). P = (1-a) R + a delta_p, Q = (1-a) R + a
    delta_q with R uniform on `k_core` atoms (a = alpha*). With k_core = 1
    this is the three-point worst case for the normalization repair; with a
    large k_core the core is 'high-dimensional' (tr Cov_R ~ 1), the worst
    case for the Hilbert part. a = 0 gives P = Q = R."""

    def __init__(self, k_core=1, a=0.2, spacing=100.0):
        self.k, self.a, self.spacing = k_core, a, spacing
        self.alpha_star = a
        self.name = f"atoms_k{k_core}_a{a}"

    def _pts(self, idx):
        return (idx * self.spacing).reshape(-1, 1).astype(float)

    def draw(self, n, m, rng):
        def one(N, extra):
            contam = rng.random(N) < self.a
            core = rng.integers(0, self.k, size=N)
            idx = np.where(contam, extra, core)
            return idx
        self._ix = one(n, self.k)          # atom k   = P's contamination
        self._iy = one(m, self.k + 1)      # atom k+1 = Q's contamination
        return self._pts(self._ix), self._pts(self._iy)

    def sigma(self, X, Y, rng):
        return 1.0

    def densities(self, X, Y):
        beta = 1 / (1 - self.a)
        ix = np.rint(X[:, 0] / self.spacing).astype(int)
        iy = np.rint(Y[:, 0] / self.spacing).astype(int)
        return np.where(ix < self.k, beta, 0.0), np.where(iy < self.k, beta, 0.0)

    def mu_R(self, Z, sigma):
        iz = np.rint(Z[:, 0] / self.spacing).astype(int)
        return np.where(iz < self.k, 1.0 / self.k, 0.0)

    def mu_R_sq(self, sigma):
        return 1.0 / self.k


class GaussContam(Design):
    """The paper's Gaussian family (closed-form TV); a(x) is continuous here.
    mu_R has no closed form, so only the type-I error is checked."""
    plugin = False

    def __init__(self, d=2, a=0.3, mu_norm=1.0):
        self.sc = dgp.gaussian_contamination(dim=d, a=a, mu_norm=mu_norm)
        self.alpha_star = self.sc.alpha_star
        self.name = f"gauss_d{d}_a{a}_mu{mu_norm}"

    def draw(self, n, m, rng):
        return self.sc.sample(n, rng, "P"), self.sc.sample(m, rng, "Q")

    def sigma(self, X, Y, rng):
        return median_heuristic(X, Y, rng=rng)


# --------------------------------------------------------------------------
# the plug-in decomposition of Theorem P
# --------------------------------------------------------------------------

def plugin_decomposition(Kxx, Kyy, Kxy, a, b, mux, muy, musq, alpha):
    """Return the quantities of the deterministic inequality
        D-hat <= ||sum w phi(X) - sum v phi(Y)||  (the constructed pair)
              <= M + rho (c_X + c_Y),
    with A = (1/n) sum a_i (phi(X_i) - mu_R), B likewise, M = max(||A-B||,
    ||A||, ||B||), c = (1 - S)_+, and rho_emp = max over sample points of
    ||phi(x) - mu_R|| (the rho actually needed by the repair)."""
    n, m = len(a), len(b)
    beta = 1 / (1 - alpha)
    # Gram of centred features phi - mu_R
    Cxx = Kxx - mux[:, None] - mux[None, :] + musq
    Cyy = Kyy - muy[:, None] - muy[None, :] + musq
    Cxy = Kxy - mux[:, None] - muy[None, :] + musq
    ua, ub = a / n, b / m
    AA = float(ua @ Cxx @ ua)
    BB = float(ub @ Cyy @ ub)
    AB = float(ua @ Cxy @ ub)
    nA, nB = math.sqrt(max(AA, 0)), math.sqrt(max(BB, 0))
    nAmB = math.sqrt(max(AA + BB - 2 * AB, 0))
    SX, SY = float(a.mean()), float(b.mean())
    rho_emp = math.sqrt(max(float(np.max(np.r_[np.diag(Cxx), np.diag(Cyy)])), 0))

    def construct(wt, S, cap, K_c_diag):
        w0 = wt.copy()
        if S >= 1:
            return w0 / S
        room = cap - w0
        need = 1 - S
        # fill the deficit greedily on the points with the smallest
        # ||phi(x) - mu_R||, which the theorem allows (any feasible u works)
        order = np.argsort(K_c_diag)
        u = np.zeros_like(w0)
        for i in order:
            take = min(room[i], need)
            u[i] = take
            need -= take
            if need <= 1e-15:
                break
        return w0 + u

    w = construct(ua, SX, beta / n, np.diag(Cxx))
    v = construct(ub, SY, beta / m, np.diag(Cyy))
    val = float(w @ Kxx @ w + v @ Kyy @ v - 2 * w @ Kxy @ v)
    feas = (abs(w.sum() - 1) < 1e-9 and abs(v.sum() - 1) < 1e-9
            and w.max() <= beta / n + 1e-12 and v.max() <= beta / m + 1e-12)
    return dict(nAmB=nAmB, nA=nA, nB=nB, M=max(nAmB, nA, nB),
                cX=max(1 - SX, 0.0), cY=max(1 - SY, 0.0), SX=SX, SY=SY,
                rho_emp=rho_emp, plugin_value=math.sqrt(max(val, 0)),
                plugin_feasible=bool(feas))

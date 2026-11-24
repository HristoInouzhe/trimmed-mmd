"""Harder synthetic designs with exactly known total variation.

Every design is a pair (P, Q) of laws on R^d with TV(P, Q) = alpha_star known
in closed form (or by a 1-d sum/integral evaluated to machine precision).
Most use the contamination construction

    P = (1 - a) R + a P0,   Q = (1 - a) R + a Q0   =>   TV(P, Q) = a TV(P0, Q0),

and, when P0 and Q0 differ only in a block of coordinates B and agree (as a
product factor) on the rest, TV(P0, Q0) = TV of the B-marginals. See DESIGN.md
for the derivations; `check_tv` verifies each value by Monte Carlo with the
exact densities of the differing block.

Samplers draw the number of contaminated points as Binomial(n, a), i.e. they
are exact i.i.d. draws from the mixtures, and shuffle the rows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import integrate, optimize
from scipy.stats import norm, t as student_t

Z90 = float(norm.ppf(0.9))   # ||mu|| giving TV(N(mu,I), N(-mu,I)) = 0.8
Z70 = float(norm.ppf(0.7))   # ... = 0.4


def _mix(n, rng, a, core, contam):
    k = int(rng.binomial(n, a))
    parts = [core(n - k, rng)]
    if k > 0:
        parts.append(contam(k, rng))
    out = np.vstack(parts)
    rng.shuffle(out)
    return out


@dataclass
class Design:
    key: str
    family: str
    d: int
    alpha_star: float
    m_ratio: int = 1                 # m = m_ratio * n
    ns: tuple = (100, 200, 400, 800)
    tier: int = 1
    note: str = ""
    params: dict = field(default_factory=dict)

    # set by subclasses
    def sample(self, n, rng, which):
        raise NotImplementedError

    def scale(self):
        """Fixed coordinate scaling applied before any kernel (None = identity)."""
        return None

    def levels(self):
        """Grid alpha_j = j alpha*/8, j = 0..10; tests reported at j = 0, 4, 6, 8."""
        return [j * self.alpha_star / 8.0 for j in range(11)]


# ---------------------------------------------------------------------------
# Gaussian location contaminations (D1, D2, D3a, D4, D5a)
# ---------------------------------------------------------------------------

class GaussLoc(Design):
    """R = N(0, I_d), P0 = N(mu, s^2 I), Q0 = N(-mu, s^2 I).

    TV(P0, Q0) = 2 Phi(||mu|| / s) - 1 (the likelihood ratio depends on x only
    through <mu, x>, which is N(+-||mu||^2, s^2 ||mu||^2)), so
    TV(P, Q) = a (2 Phi(||mu||/s) - 1).
    """

    def __init__(self, key, family, d, a, mu, s=1.0, **kw):
        self.a, self.mu, self.s = a, np.asarray(mu, float), s
        tv0 = 2 * norm.cdf(np.linalg.norm(self.mu) / s) - 1
        super().__init__(key=key, family=family, d=d, alpha_star=float(a * tv0), **kw)
        self.tv0 = float(tv0)

    def sample(self, n, rng, which):
        sgn = 1.0 if which == "P" else -1.0
        return _mix(n, rng, self.a,
                    lambda k, r: r.standard_normal((k, self.d)),
                    lambda k, r: sgn * self.mu + self.s * r.standard_normal((k, self.d)))

    # log density ratio of the differing block, for check_tv
    def mc_tv(self, n_mc, rng):
        # TV(P0,Q0) = E_{P0}[(1 - q0/p0)_+]; q0/p0 = exp(-2 <mu,x>/s^2)
        x = self.mu + self.s * rng.standard_normal((n_mc, self.d))
        r = np.exp(-2.0 * (x @ self.mu) / self.s ** 2)
        return self.a * np.maximum(1 - r, 0.0)


# ---------------------------------------------------------------------------
# D3b: sparse correlation flip
# ---------------------------------------------------------------------------

class CorrFlip(Design):
    """R = N(0, I_d); P0, Q0: coordinates (1,2) ~ N(0, [[1, +-rho], [+-rho, 1]]),
    coordinates 3..d ~ N(0, I) in both.

    log p0/q0 = 2 rho x1 x2 / (1 - rho^2), so p0 > q0 iff x1 x2 > 0 and
    TV(P0, Q0) = P_{+rho}(x1 x2 > 0) - P_{-rho}(x1 x2 > 0) = 2 arcsin(rho) / pi
    (orthant probability 1/4 + arcsin(rho)/(2 pi)). rho = sin(0.4 pi) gives 0.8.
    """

    def __init__(self, key, family, d, a, rho, **kw):
        self.a, self.rho = a, rho
        tv0 = 2 * math.asin(rho) / math.pi
        super().__init__(key=key, family=family, d=d, alpha_star=float(a * tv0), **kw)
        self.tv0 = tv0

    def _contam(self, sign):
        rho = sign * self.rho

        def f(k, r):
            z = r.standard_normal((k, self.d))
            z[:, 1] = rho * z[:, 0] + math.sqrt(1 - rho ** 2) * z[:, 1]
            return z
        return f

    def sample(self, n, rng, which):
        return _mix(n, rng, self.a, lambda k, r: r.standard_normal((k, self.d)),
                    self._contam(1.0 if which == "P" else -1.0))

    def mc_tv(self, n_mc, rng):
        x = self._contam(1.0)(n_mc, rng)
        # q0/p0 = exp(-2 rho x1 x2 / (1 - rho^2))
        lr = np.exp(-2 * self.rho * x[:, 0] * x[:, 1] / (1 - self.rho ** 2))
        return self.a * np.maximum(1 - lr, 0.0)


# ---------------------------------------------------------------------------
# D5b: mixed discrete/continuous with ties, m = 5n
# ---------------------------------------------------------------------------

class Mixed(Design):
    """Five coordinates, RHC/Adult-like types:
      x1 integer ("age"): round(c + 1.5 Z);  x2 binary;  x3 = 0 w.p. 0.85 else
      exp(N(0,1)) ("capital gain", atom at 0);  x4 categorical {0,1,2,3} with
      probs (.4,.3,.2,.1), ordinal code;  x5 ~ Bern(0.4).
    R: c = 0, x2 ~ Bern(.5). P0: c = +1.5, x2 ~ Bern(.75). Q0: c = -1.5,
    x2 ~ Bern(.25). x3, x4, x5 have the same law under R, P0, Q0 and are
    independent of (x1, x2), so TV(P0, Q0) = TV of the (x1, x2) marginals
    = 1/2 sum_{k in Z, b in {0,1}} |p_k pi_b - q_k pi'_b|, an exact finite sum
    (terms beyond |k| = 40 are < 1e-80). a is set so that TV(P, Q) = 0.2.
    """

    C, SD = 1.5, 1.5

    @staticmethod
    def _pk(c, sd, ks):
        return norm.cdf((ks + 0.5 - c) / sd) - norm.cdf((ks - 0.5 - c) / sd)

    def __init__(self, key, family, target=0.2, **kw):
        ks = np.arange(-40, 41)
        p = np.outer(self._pk(self.C, self.SD, ks), [0.25, 0.75])
        q = np.outer(self._pk(-self.C, self.SD, ks), [0.75, 0.25])
        tv0 = 0.5 * np.abs(p - q).sum()
        self.tv0 = float(tv0)
        self.a = target / tv0
        super().__init__(key=key, family=family, d=5, alpha_star=float(self.a * tv0), **kw)
        self._scale = None

    def _block(self, k, r, c, p2):
        x1 = np.round(c + self.SD * r.standard_normal(k))
        x2 = (r.random(k) < p2).astype(float)
        x3 = np.where(r.random(k) < 0.85, 0.0, np.exp(r.standard_normal(k)))
        x4 = r.choice(4, size=k, p=[0.4, 0.3, 0.2, 0.1]).astype(float)
        x5 = (r.random(k) < 0.4).astype(float)
        return np.column_stack([x1, x2, x3, x4, x5])

    def sample(self, n, rng, which):
        c, p2 = (self.C, 0.75) if which == "P" else (-self.C, 0.25)
        return _mix(n, rng, self.a, lambda k, r: self._block(k, r, 0.0, 0.5),
                    lambda k, r: self._block(k, r, c, p2))

    def scale(self):
        return self._scale          # set from the pilot (pooled SDs)

    def mc_tv(self, n_mc, rng):
        x = self._block(n_mc, rng, self.C, 0.75)
        k = x[:, 0]
        p = self._pk(self.C, self.SD, k) * np.where(x[:, 1] == 1, 0.75, 0.25)
        q = self._pk(-self.C, self.SD, k) * np.where(x[:, 1] == 1, 0.25, 0.75)
        return self.a * np.maximum(1 - q / p, 0.0)


# ---------------------------------------------------------------------------
# D6: heavy tails, scale contamination
# ---------------------------------------------------------------------------

class HeavyScale(Design):
    """R = t_3^{(x)d}; P0, Q0: coordinate 1 ~ s_P t_3 resp. s_Q t_3, the
    other coordinates t_3 in both. TV(P0, Q0) = TV(s_P t_3, s_Q t_3) on R:
    the density ratio is monotone in x^2, so the densities cross once at
    +-x*, and TV = 2 (F_{s_P}(x*) - F_{s_Q}(x*)), with x* by root finding
    (checked against direct quadrature of |p - q|/2)."""

    NU = 3.0

    def __init__(self, key, family, d, sP, sQ, target=0.2, **kw):
        self.sP, self.sQ = sP, sQ
        f = lambda x: student_t.pdf(x / sP, self.NU) / sP - student_t.pdf(x / sQ, self.NU) / sQ
        xs = optimize.brentq(f, 1e-6, 100.0, xtol=1e-14)
        tv0 = 2 * (student_t.cdf(xs / sP, self.NU) - student_t.cdf(xs / sQ, self.NU))
        quad = 0.5 * sum(integrate.quad(lambda x: abs(f(x)), lo, hi, limit=400, epsabs=1e-13)[0]
                         for lo, hi in [(-np.inf, -xs), (-xs, xs), (xs, np.inf)])
        assert abs(tv0 - quad) < 1e-9, (tv0, quad)
        self.xstar, self.tv0 = float(xs), float(tv0)
        self.a = target / tv0
        super().__init__(key=key, family=family, d=d, alpha_star=float(self.a * tv0), **kw)

    def _block(self, k, r, s1):
        z = r.standard_t(self.NU, size=(k, self.d))
        z[:, 0] *= s1
        return z

    def sample(self, n, rng, which):
        s1 = self.sP if which == "P" else self.sQ
        return _mix(n, rng, self.a, lambda k, r: r.standard_t(self.NU, size=(k, self.d)),
                    lambda k, r: self._block(k, r, s1))

    def mc_tv(self, n_mc, rng):
        x = self.sP * rng.standard_t(self.NU, size=n_mc)
        p = student_t.pdf(x / self.sP, self.NU) / self.sP
        q = student_t.pdf(x / self.sQ, self.NU) / self.sQ
        return self.a * np.maximum(1 - q / p, 0.0)


# ---------------------------------------------------------------------------
# D7: functional data, Brownian core, localized bump
# ---------------------------------------------------------------------------

class BMBump(Design):
    """Curves on t_i = i/G, i = 1..G. R = P0 - b = Q0 + b in law: Brownian
    motion on the grid (covariance K_ij = min(t_i, t_j)); P0 = BM + b,
    Q0 = BM - b, b(t) = h exp(-(t - 1/2)^2 / (2 w^2)).
    TV(N(b, K), N(-b, K)) = 2 Phi(||K^{-1/2} b||) - 1 and, since K = L L' with
    L = sqrt(dt) * lower-triangular ones, ||K^{-1/2} b||^2 = sum_i (b_i - b_{i-1})^2 / dt
    (b_0 = 0), the discrete Cameron-Martin norm. h is set so that this norm is
    Phi^{-1}(0.9), i.e. TV(P0, Q0) = 0.8; a = 0.25 gives TV(P, Q) = 0.2.
    """

    def __init__(self, key, family, G, w, a=0.25, **kw):
        self.G, self.w, self.a = G, w, a
        t = np.arange(1, G + 1) / G
        self.dt = 1.0 / G
        b0 = np.exp(-(t - 0.5) ** 2 / (2 * w ** 2))
        cm = math.sqrt(np.sum(np.diff(np.r_[0.0, b0]) ** 2) / self.dt)
        self.h = Z90 / cm
        self.b = self.h * b0
        self.cm = float(np.sqrt(np.sum(np.diff(np.r_[0.0, self.b]) ** 2) / self.dt))
        tv0 = 2 * norm.cdf(self.cm) - 1
        self.tv0 = float(tv0)
        super().__init__(key=key, family=family, d=G, alpha_star=float(a * tv0), **kw)
        self.l2_bump = float(np.sqrt(np.mean(self.b ** 2)))   # L2[0,1] norm of b

    def _bm(self, k, r):
        return np.cumsum(math.sqrt(self.dt) * r.standard_normal((k, self.G)), axis=1)

    def sample(self, n, rng, which):
        sgn = 1.0 if which == "P" else -1.0
        return _mix(n, rng, self.a, self._bm, lambda k, r: self._bm(k, r) + sgn * self.b)

    def mc_tv(self, n_mc, rng):
        x = self._bm(n_mc, rng) + self.b
        # log q0/p0 = -2 <K^{-1} b, x>; with increments: <Db, Dx>/dt
        db = np.diff(np.r_[0.0, self.b])
        dx = np.diff(np.c_[np.zeros(n_mc), x], axis=1)
        lr = np.exp(-2 * (dx @ db) / self.dt)
        return self.a * np.maximum(1 - lr, 0.0)


# ---------------------------------------------------------------------------
# D8: oscillating densities (kernel-blind)
# ---------------------------------------------------------------------------

class Oscillating(Design):
    """X ~ U[0,1]; Y has density q_N = 1 + s(N x), s the +-1 square wave of
    period 1 (+1 on [0, 1/2)). q_N = 2 on A_N = {frac(N x) < 1/2}, 0 elsewhere,
    |A_N| = 1/2, so TV = 1/2 int |s(Nx)| dx = 1/2 for every N (paper, App. B)."""

    def __init__(self, key, family, N, **kw):
        self.N = N
        super().__init__(key=key, family=family, d=1, alpha_star=0.5, **kw)
        self.tv0 = 0.5

    def sample(self, n, rng, which):
        if which == "P":
            return rng.random((n, 1))
        k = rng.integers(0, self.N, size=n)
        return ((k + 0.5 * rng.random(n)) / self.N)[:, None]

    def mc_tv(self, n_mc, rng):
        # E_Q[(1 - p/q)_+] with p = 1, q = 2 on the support of Q
        return np.full(n_mc, 0.5)


# ---------------------------------------------------------------------------
# catalogue (pre-registered; see DESIGN.md)
# ---------------------------------------------------------------------------

class RHCBox(Design):
    """POST HOC (revision plan v1.1, D1, 2026-11-17): the RHC-shaped design of the paper's
    Section 6.1 under this study's protocol, so that all designs share one table.
    Core: smoothed resample of the pooled, standardized RHC covariates (33 dims, jitter 0.25);
    P0, Q0: the core with its first four coordinates replaced by Uniform boxes at -(3..4) and
    +(3..4) pooled SDs (scripts/export_fig2_data.py:rhc_scenario, sep=3, width=1). P0 and Q0 are
    mutually singular, so TV(P, Q) = a exactly."""

    _Z = None

    def __init__(self, key, family, a, sep=3.0, width=1.0, jitter=0.25, k=4, **kw):
        super().__init__(key, family, 33, a, **kw)
        self.a, self.sep, self.width, self.jitter, self.k = a, sep, width, jitter, k

    @classmethod
    def _core_pool(cls):
        if cls._Z is None:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
            from trimmedmmd.realdata import load_rhc
            X, Y, _, _ = load_rhc()
            cls._Z = np.vstack([X, Y])
        return cls._Z

    def _core(self, k, rng):
        Z = self._core_pool()
        idx = rng.integers(0, Z.shape[0], size=k)
        return Z[idx] + self.jitter * rng.standard_normal((k, Z.shape[1]))

    def sample(self, n, rng, which):
        sign = -1.0 if which == "P" else 1.0

        def contam(k, rng):
            out = self._core(k, rng)
            out[:, :self.k] = sign * (self.sep + rng.uniform(0.0, self.width, size=(k, self.k)))
            return out
        return _mix(n, rng, self.a, self._core, contam)


def _e1(d, r):
    v = np.zeros(d)
    v[0] = r
    return v


def catalogue():
    C = {}

    def add(dz):
        C[dz.key] = dz

    # D1 overlapping Gaussian contamination, a = 0.25
    for d in (2, 10):
        add(GaussLoc(f"D1_d{d}_a10", "D1 overlap", d, 0.25, _e1(d, Z70),
                     note="||mu||=Phi^-1(0.7), mu along e1"))
        add(GaussLoc(f"D1_d{d}_a20", "D1 overlap", d, 0.25, _e1(d, Z90),
                     note="||mu||=Phi^-1(0.9), mu along e1"))
    # D2 dense high dimension, total signal fixed
    for d in (20, 50, 100):
        add(GaussLoc(f"D2_d{d}", "D2 dense", d, 0.25, np.full(d, Z90 / math.sqrt(d)),
                     note="mu = Phi^-1(0.9) * 1/sqrt(d)"))
    # D3 sparse, d = 50
    mu = np.zeros(50)
    mu[:2] = Z90 / math.sqrt(2)
    add(GaussLoc("D3a_loc", "D3 sparse", 50, 0.25, mu, note="mu on coords 1-2 only"))
    add(CorrFlip("D3b_corr", "D3 sparse", 50, 0.25, math.sin(0.4 * math.pi),
                 note="corr(x1,x2) = +-sin(0.4 pi) in the contamination"))
    # D4 small contamination
    add(GaussLoc("D4_small", "D4 small", 2, 0.05, _e1(2, 2.5), s=0.5,
                 note="a=0.05, contaminations N(+-2.5 e1, 0.25 I)"))
    # D5 imbalanced, mixed types
    add(Mixed("D5b_mixed", "D5 imbalanced", m_ratio=5, ns=(100, 200, 400)))
    add(GaussLoc("D5a_imb", "D5 imbalanced", 2, 0.25, _e1(2, Z90), m_ratio=5,
                 ns=(100, 200, 400), tier=2, note="D1_d2_a20 geometry, m = 5n"))
    # D6 heavy tails
    add(HeavyScale("D6_t3", "D6 heavy tails", 5, 0.5, 3.0))
    # D7 functional
    add(BMBump("D7_wide", "D7 functional", 50, 0.2))
    add(BMBump("D7_narrow", "D7 functional", 50, 0.05))
    # D8 oscillating
    for N in (1, 4, 16, 64):
        add(Oscillating(f"D8_N{N}", "D8 oscillating", N))
    # POST HOC (revision plan v1.1, D1): the RHC-shaped design under this protocol
    add(RHCBox("D9_rhc", "D9 RHC-shaped", 0.2, note="RHC core, boxes at +-(3..4) SD in 4 coords"))
    # POST HOC (DESIGN.md section 9): P = Q validity nulls in the hardest geometries
    for b in ("D5b_mixed", "D2_d100", "D6_t3", "D7_narrow"):
        add(SameLaw(f"N0_{b}", C[b], note=f"X, Y ~ P-law of {b}"))
    return C


class SameLaw(Design):
    """POST HOC validity null: X and Y both drawn from the P-law of a base
    design, so TV = 0 and H_{0,0} is tight for every [0,1]-valued g."""

    def __init__(self, key, base, **kw):
        self.base = base
        super().__init__(key=key, family="N0 null (P = Q)", d=base.d, alpha_star=0.0,
                         m_ratio=base.m_ratio, ns=base.ns, tier=3, **kw)
        self.tv0 = 0.0

    def sample(self, n, rng, which):
        return self.base.sample(n, rng, "P")

    def mc_tv(self, n_mc, rng):
        return np.zeros(n_mc)


def check_tv(n_mc=2_000_000, seed=12345):
    rng = np.random.default_rng(seed)
    out = {}
    for k, dz in catalogue().items():
        v = dz.mc_tv(n_mc, rng)
        out[k] = (dz.alpha_star, float(v.mean()), float(v.std() / math.sqrt(n_mc)))
    return out


if __name__ == "__main__":
    for k, (tv, mc, se) in check_tv().items():
        print(f"{k:12s} TV={tv:.10f}  MC={mc:.5f} +- {se:.5f}  z={(mc - tv) / max(se, 1e-12):+.2f}")

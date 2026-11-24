"""The draft's explicit lower and upper bounds, evaluated for the probe laws.  All values are for
u(alpha*-eps) = (1-alpha*+eps) D_{alpha*-eps} (the draft's propositions bound D; dividing by 1-alpha*+eps converts)."""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, brentq
from common import Laws, SQ2PI

# ---------------------------------------------------------------------------------------------------------------
# helpers on the laws
# ---------------------------------------------------------------------------------------------------------------
_TAILTAB = {}


def absmass_beyond(law, t):
    """|P-Q|((t, inf)) for t >= 0 (by symmetry also |P-Q|((-inf,-t))); tabulated (log-linear interpolation)."""
    key = (law.kind, law.mu)
    if key not in _TAILTAB:
        ts = np.linspace(0, 60, 6001)
        vals = np.array([quad(lambda x: abs(law.e(x)), a, a + 0.01, epsabs=1e-30, epsrel=1e-13)[0]
                         for a in ts[:-1]])
        tail = np.concatenate([np.cumsum(vals[::-1])[::-1], [0.0]])
        _TAILTAB[key] = (ts, np.log(np.maximum(tail, 1e-300)))
    ts, lt = _TAILTAB[key]
    t = float(t)
    if t >= ts[-1]:
        return 0.0
    if t <= 0:
        return float(np.exp(lt[0]))
    return float(np.exp(np.interp(t, ts, lt)))


def ell_sup(law, r0):
    """sup_{0<|x|<=r0} |p-q|(x)/|x| (odd e)."""
    xs = np.linspace(1e-6, r0, 20001)
    return max(law.l0, np.max(np.abs(law.e(xs)) / xs))


def ell_inf(law, r0):
    xs = np.linspace(1e-6, r0, 20001)
    return np.min(law.e(xs) / xs)


# ---------------------------------------------------------------------------------------------------------------
# Laplace kernel exp(-|x|): hat k0 = 2/(1+w^2); ||f||_H^2 = (||f||^2 + ||f'||^2)/2
# ---------------------------------------------------------------------------------------------------------------
def lap_err_flat(law, w, rho):
    """err(f) for f = 1_(0,rho) * (flat density of half-width w), the trapezoid; requires rho > 2w."""
    e = lambda x: abs(law.e(x))
    I1, _ = quad(lambda x: (x + w) / (2 * w) * e(x), -w, 0, epsabs=1e-17, epsrel=1e-12)
    I2, _ = quad(lambda x: (w - x) / (2 * w) * e(x), 0, w, epsabs=1e-17, epsrel=1e-12)
    I3, _ = quad(lambda x: (x - rho + w) / (2 * w) * e(x), rho - w, rho + w, epsabs=1e-17, epsrel=1e-12)
    I4 = absmass_beyond(law, rho + w)
    return I1 + I2 + I3 + I4


def lap_direct_lower(law, eps):
    """max over (w, rho) of (eps - err(f))/||f||_H for the flat-mollified indicator (Prop. newapproxgen(ii))."""
    best = (-np.inf, None)
    for rho in np.linspace(3.0, law.mu + 7.0, 25):
        def neg(lw):
            w = np.exp(lw)
            if 2 * w >= rho:
                return 1e9
            err = lap_err_flat(law, w, rho)
            nrm = np.sqrt((rho - 2 * w / 3 + 1 / w) / 2)
            return -(eps - err) / nrm
        r = minimize_scalar(neg, bounds=(np.log(1e-4), np.log(min(2.0, rho / 2.1))), method='bounded',
                            options=dict(xatol=1e-6))
        if -r.fun > best[0]:
            best = (-r.fun, (np.exp(r.x), rho))
    # refine rho
    w0, rho0 = best[1]
    def neg2(v):
        w, rho = np.exp(v[0]), v[1]
        if 2 * w >= rho:
            return 1e9
        return -(eps - lap_err_flat(law, w, rho)) / np.sqrt((rho - 2 * w / 3 + 1 / w) / 2)
    from scipy.optimize import minimize
    r = minimize(neg2, [np.log(w0), rho0], method='Nelder-Mead', options=dict(xatol=1e-7, fatol=1e-16, maxiter=400))
    if -r.fun > best[0]:
        best = (-r.fun, (np.exp(r.x[0]), r.x[1]))
    return best


# bump mollifier for the theorem's formula: b(t) = C exp(-1/(1-t^2)) on (-1,1)
def _bump_consts():
    f = lambda t: np.exp(-1 / (1 - t * t)) if abs(t) < 1 else 0.0
    Z, _ = quad(f, -1, 1, epsabs=1e-15)
    def b2(t):  # second derivative of exp(-1/(1-t^2))
        if abs(t) >= 1:
            return 0.0
        u = 1 - t * t
        g = np.exp(-1 / u)
        gp = g * (-2 * t / u ** 2)
        # d/dt of (-2t/u^2) = -2/u^2 - 8 t^2/u^3
        return gp * (-2 * t / u ** 2) + g * (-2 / u ** 2 - 8 * t * t / u ** 3)
    L1, _ = quad(lambda t: abs(b2(t)), -1, 1, limit=400, epsabs=1e-13)
    return Z, L1 / Z


BUMP_Z, BUMP_CB = None, None


def sob_theorem_lower(law, eps, s=1.0, c=2.0):
    """Theorem newsobolev / Corollary case 2 in d=1 (r=1, Pi=V=2, Leb(B_rho)=2 rho), Laplace: c=2, s=1, with the
    bump mollifier (C_b = ||b''||_1); maximised over w <= w0 <= 1 and rho >= 1."""
    global BUMP_Z, BUMP_CB
    if BUMP_CB is None:
        BUMP_Z, BUMP_CB = _bump_consts()
    c1 = 1 - np.sin(1.0)
    m = int(np.floor((2 * s - 1) / 4)) + 1
    Cdsb = 2 ** (3 * s) / (2 * c1) * (1 / (1 - 2 ** (1 - 2 * s)) + BUMP_CB ** 2 / (1 - 2 ** (-(4 * m + 1 - 2 * s))))
    best = (-np.inf, None)
    for w0 in [0.05, 0.1, 0.2, 0.4, 0.7, 1.0]:
        ell = ell_sup(law, w0)
        for rho in np.linspace(2.0, law.mu + 9.0, 30):
            T = 2 * absmass_beyond(law, rho - 1)
            ws = np.geomspace(1e-4, w0, 200)
            num = eps - ell * 2 * ws ** 2 - T
            den = np.sqrt(2 ** s * 2 * rho + Cdsb * 2 * ws ** (1 - 2 * s))
            val = np.sqrt(c) * np.maximum(num, 0) / den
            k = np.argmax(val)
            if val[k] > best[0]:
                best = (val[k], (ws[k], rho, w0, ell))
    return best[0], best[1], Cdsb


def lap_upperone(law, eps):
    """Prop. newupperone with s=1, C=2 (hat k0 = 2/(1+w^2) <= 2 w^{-2}), r=1: u <= 2 eps W^{1/2} (4/pi)^{1/2},
    W = (2 eps/l')^{1/2}, l' = inf_{0<x<r0} e(x)/x, eps <= l' r0^2/2; best r0."""
    best = np.inf
    for r0 in np.geomspace(0.05, 4.0, 200):
        lp = ell_inf(law, r0)
        if lp <= 0 or eps > lp * r0 ** 2 / 2:
            continue
        W = np.sqrt(2 * eps / lp)
        best = min(best, 2 * eps * np.sqrt(W) * np.sqrt(4 / np.pi))
    return best


def upperone_trimming(law, eps):
    """the explicit trimming of Prop. newupperone: nu = e 1_{(-wq, wp)} with int_0^wp e = eps (odd e: wq = wp)."""
    F = lambda w: quad(law.e, 0, w, epsabs=1e-17, epsrel=1e-13)[0] - eps
    wp = brentq(F, 1e-8, law.mu + 6)
    return wp


# ---------------------------------------------------------------------------------------------------------------
# Gaussian kernel: band-limited step with b(t) = 4 pi cos^2(t/2)/(pi^2-t^2)^2
# ---------------------------------------------------------------------------------------------------------------
def _hfun(t):
    t = np.asarray(t, float)
    out = np.empty_like(t)
    near = np.abs(np.abs(t) - np.pi) < 1e-5
    tt = t[~near]
    out[~near] = np.cos(tt / 2) / (np.pi ** 2 - tt ** 2)
    out[near] = 1 / (4 * np.pi)  # limit of cos(t/2)/(pi^2-t^2) at |t| = pi is 1/(4 pi)
    return out


def bdens(t):
    return 4 * np.pi * _hfun(t) ** 2


class Bbar:
    """bbar(t) = int_t^inf b for t >= 0, tabulated (Simpson) up to Tm, asymptotic tail beyond."""

    def __init__(self, Tm=3000.0, dt=0.002):
        t = np.arange(0, Tm + dt / 2, dt)
        bv = bdens(t)
        # cumulative integral from 0 (trapezoid with Richardson-like fine dt)
        cum = np.concatenate([[0], np.cumsum(0.5 * (bv[1:] + bv[:-1]) * dt)])
        tail = 2 * np.pi / (3 * Tm ** 3)  # int_Tm^inf of the mean 2 pi / s^4
        total = cum[-1] + tail
        self.t, self.cum, self.half, self.Tm = t, cum, total, Tm
        self.scale = 0.5 / total  # renormalise so that bbar(0) = 1/2 exactly

    def __call__(self, t):
        t = np.asarray(t, float)
        out = np.where(t >= self.Tm, 2 * np.pi / (3 * np.maximum(t, 1) ** 3) * 0.5 / self.half,
                       (self.half - np.interp(np.minimum(t, self.Tm), self.t, self.cum)) * self.scale)
        return out


BBAR = None


def bhat(xi):
    xi = np.abs(xi)
    return np.where(xi <= 1, (1 - xi) * np.cos(np.pi * xi) + np.sin(np.pi * xi) / np.pi, 0.0)


def gauss_step_norm2(Om, rho, sigma=1.0):
    w = np.linspace(1e-9, Om, 200001)
    g2 = 4 * np.sin(w * rho / 2) ** 2 / w ** 2
    integ = g2 * bhat(w / Om) ** 2 * np.exp(0.5 * (sigma * w) ** 2) / (SQ2PI * sigma)
    return 2 * np.trapezoid(integ, w) / (2 * np.pi)  # even integrand


def gauss_step_err(law, Om, rho):
    global BBAR
    if BBAR is None:
        BBAR = Bbar()
    e = lambda x: np.abs(law.e(x))
    # x<0: f = bbar(Om|x|) - bbar(Om(|x|+rho));  0<x<rho: 1-f = bbar(Om x)+bbar(Om(rho-x));  x>rho: 1-f
    xs = np.linspace(-(law.mu + 8), 0, 40001)
    f_neg = BBAR(Om * np.abs(xs)) - BBAR(Om * (np.abs(xs) + rho))
    I = np.trapezoid(f_neg * e(xs), xs)
    xs = np.linspace(0, rho, 40001)
    I += np.trapezoid((BBAR(Om * xs) + BBAR(Om * (rho - xs))) * e(xs), xs)
    xs = np.linspace(rho, rho + law.mu + 10, 40001)
    f_r = BBAR(Om * (xs - rho)) - BBAR(Om * xs)
    I += np.trapezoid((1 - f_r) * e(xs), xs)
    return I


def gauss_direct_lower(law, eps, sigma=1.0):
    """max over (Omega, rho) of (eps - err(f))/||f||_H for f = 1_(0,rho) * b_Omega."""
    from scipy.optimize import minimize
    best = (-np.inf, None)
    Om0 = np.sqrt(law.l0 * np.pi ** 2 / (2 * 0.8 * eps))
    for rho in [law.mu + 2, law.mu + 3, law.mu + 4]:
        for fac in [0.6, 0.8, 1.0, 1.2]:
            Om = Om0 * fac
            v = (eps - gauss_step_err(law, Om, rho)) / np.sqrt(gauss_step_norm2(Om, rho, sigma))
            if v > best[0]:
                best = (v, (Om, rho))
    def neg(v):
        Om, rho = np.exp(v[0]), v[1]
        if rho <= 0.5:
            return 1e9
        val = (eps - gauss_step_err(law, Om, rho)) / np.sqrt(gauss_step_norm2(Om, rho, sigma))
        return -np.sign(val) * np.log(abs(val) + 1e-300) if val > 0 else 1e3 - val
    r = minimize(neg, [np.log(best[1][0]), best[1][1]], method='Nelder-Mead',
                 options=dict(xatol=1e-5, fatol=1e-9, maxiter=200))
    Om, rho = np.exp(r.x[0]), r.x[1]
    v = (eps - gauss_step_err(law, Om, rho)) / np.sqrt(gauss_step_norm2(Om, rho, sigma))
    if v > best[0]:
        best = (v, (Om, rho))
    return best


def gauss_theorem_lower(law, eps, sigma=1.0):
    """Theorem newgauss: ((1-theta) eps/2) ((2pi)^{1/2} sigma/rho)^{1/2} exp(-sigma^2 l m2/(8 theta eps)) under
    2T+(rho/2) + 2 alpha* {bbar(Om r0) + bbar(Om rho/2)} <= (1-theta) eps/2, Om = (l m2/(2 theta eps))^{1/2}."""
    global BBAR
    if BBAR is None:
        BBAR = Bbar()
    m2 = np.pi ** 2
    best = (-np.inf, None)
    for r0 in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0, 4.0, 8.0, np.inf]:
        ell = ell_sup(law, r0 if np.isfinite(r0) else law.mu + 8)
        for th in np.linspace(0.3, 0.995, 71):
            Om = np.sqrt(ell * m2 / (2 * th * eps))
            bb0 = 0.0 if not np.isfinite(r0) else float(BBAR(Om * r0))
            # smallest admissible rho by bisection on the decreasing condition
            def cond(rho):
                return 2 * absmass_beyond(law, rho / 2) + 2 * law.alpha_star * (bb0 + float(BBAR(Om * rho / 2))) \
                    - (1 - th) * eps / 2
            if cond(4 * (law.mu + 10)) > 0:
                continue
            lo, hi = 1e-3, 4 * (law.mu + 10)
            if cond(lo) <= 0:
                rho = lo
            else:
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    if cond(mid) > 0:
                        lo = mid
                    else:
                        hi = mid
                rho = hi
            val = (1 - th) * eps / 2 * np.sqrt(SQ2PI * sigma / rho) * np.exp(-sigma ** 2 * ell * m2 / (8 * th * eps))
            if val > best[0]:
                best = (val, (r0, th, rho, ell))
    return best


def lap_err_flat2(law, w1, w2, rho):
    e = lambda x: abs(law.e(x))
    I1, _ = quad(lambda x: (x + w1) / (2 * w1) * e(x), -w1, 0, epsabs=1e-17, epsrel=1e-12)
    I2, _ = quad(lambda x: (w1 - x) / (2 * w1) * e(x), 0, w1, epsabs=1e-17, epsrel=1e-12)
    I3, _ = quad(lambda x: (x - rho + w2) / (2 * w2) * e(x), rho - w2, rho + w2, epsabs=1e-17, epsrel=1e-12)
    I4 = absmass_beyond(law, rho + w2)
    return I1 + I2 + I3 + I4


def lap_direct_lower2(law, eps, start=None):
    """as lap_direct_lower but with independent ramp half-widths w1 (at x0) and w2 (at x0+rho)."""
    from scipy.optimize import minimize
    def val(v):
        w1, w2, rho = np.exp(v[0]), np.exp(v[1]), v[2]
        if w1 + w2 >= rho:
            return -1e9
        nrm2 = (rho - w1 / 3 - w2 / 3 + 1 / (2 * w1) + 1 / (2 * w2)) / 2
        return (eps - lap_err_flat2(law, w1, w2, rho)) / np.sqrt(nrm2)
    best = (-np.inf, None)
    w1s = [start[0]] if start else [0.05, 0.2]
    for w1 in w1s:
        for w2 in [0.5, 1.5, 3.0]:
            for rho in [law.mu + 3, law.mu + 4.5]:
                r = minimize(lambda v: -val(v), [np.log(w1), np.log(w2), rho], method='Nelder-Mead',
                             options=dict(xatol=1e-7, fatol=1e-16, maxiter=600))
                if -r.fun > best[0]:
                    best = (-r.fun, (np.exp(r.x[0]), np.exp(r.x[1]), r.x[2]))
    return best

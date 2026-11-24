"""Common code for the margin probes (referee check of draft_under_review.tex).

Population path u(alpha) = (1-alpha) D_alpha(P,Q) = dist(K_alpha(P), K_alpha(Q)), where K_alpha(P) is the
set of embeddings of measures P' <= P of mass 1-alpha.  We discretise the line into cells with 0 = x0 a cell
boundary (so that the discretised laws have the same total variation as the continuous ones), solve the
min-norm-point problem between the two discretised trimmed regions with Wolfe's algorithm and a bathtub linear
oracle, and report

  * discrete certificates  lo_d <= u_disc <= up_d   (exact duality bracket of the discrete problem),
  * continuum certificates lo_c <= u_cont <= up_c   (lo_c: separation of the continuous trimmed regions in the
    direction of the discrete witness, computed by quadrature; up_c: an explicit continuous trimming).
"""
import numpy as np
from scipy.special import ndtr
from scipy.stats import norm as _norm

SQ2PI = np.sqrt(2 * np.pi)


def phi(x):
    return np.exp(-0.5 * x * x) / SQ2PI


# ----------------------------------------------------------------------------------------------------------------
# Laws
# ----------------------------------------------------------------------------------------------------------------
class Laws:
    """kind='mix':  P = 0.8 N(0,1) + 0.2 N(mu,1),  Q = 0.8 N(0,1) + 0.2 N(-mu,1)    (draft's numerics)
       kind='shift': P = N(mu,1), Q = N(-mu,1)."""

    def __init__(self, kind, mu):
        self.kind, self.mu = kind, float(mu)
        if kind == 'mix':
            self.wc, self.wp = 0.8, 0.2
        elif kind == 'shift':
            self.wc, self.wp = 0.0, 1.0
        else:
            raise ValueError(kind)
        mu = self.mu
        self.x0 = 0.0
        # (p-q)'(0) = wp * (phi'(-mu) - phi'(mu)) = 2 wp mu phi(mu)
        self.l0 = 2 * self.wp * mu * phi(mu)
        self.alpha_star = self.wp * (2 * ndtr(mu) - 1)

    def p(self, x):
        return self.wc * phi(x) + self.wp * phi(x - self.mu)

    def q(self, x):
        return self.wc * phi(x) + self.wp * phi(x + self.mu)

    def Pcdf(self, x):
        return self.wc * ndtr(x) + self.wp * ndtr(x - self.mu)

    def Qcdf(self, x):
        return self.wc * ndtr(x) + self.wp * ndtr(x + self.mu)

    def e(self, x):  # p - q, computed without cancellation
        mu = self.mu
        return self.wp * 2 * np.exp(-0.5 * (x * x + mu * mu)) * np.sinh(mu * x) / SQ2PI

    def L(self):
        return self.mu + 7.5


def make_grid(laws, h):
    L = np.ceil(laws.L() / h) * h
    edges = np.arange(-L, L + h / 2, h)
    edges[np.argmin(np.abs(edges))] = 0.0
    x = 0.5 * (edges[:-1] + edges[1:])
    # cell masses by 8-point Gauss-Legendre (relative accuracy in the tails, unlike CDF differences)
    gx, gw = np.polynomial.legendre.leggauss(8)
    mid, half = x, h / 2
    pts = mid[:, None] + half * gx[None, :]
    pm = (laws.p(pts) * gw[None, :]).sum(1) * half
    qm = (laws.q(pts) * gw[None, :]).sum(1) * half
    em = (laws.e(pts) * gw[None, :]).sum(1) * half
    keep = np.maximum(pm, qm) > 1e-18
    i0, i1 = np.nonzero(keep)[0][[0, -1]]
    sl = slice(i0, i1 + 1)
    return dict(edges=edges[i0:i1 + 2], x=x[sl], pm=pm[sl], qm=qm[sl], em=em[sl], h=h)


# ----------------------------------------------------------------------------------------------------------------
# Kernels.  Each kernel provides  z(c)  (feature vector of sum_i c_i phi(x_i), with ||z||^2 = ||.||_H^2 exactly or
# to ~1e-20 absolute) and  wit(z)  (the witness d(x_i) = <z, phi(x_i)> at the nodes).
# ----------------------------------------------------------------------------------------------------------------
class GaussFeatures:
    """Fourier features of k(x,y)=exp(-(x-y)^2/(2 sigma^2)) by the trapezoid rule in frequency with spacing
    2pi/Per: the feature kernel is the Per-periodisation of k (Poisson summation), equal to k up to
    exp(-(Per-diam)^2/(2 sigma^2)) for |x-y| <= diam, truncated at |omega| <= wmax (error exp(-sigma^2 wmax^2/2))."""

    def __init__(self, sigma, diam, wmax_sig=13.0):
        self.sigma = sigma
        Per = diam + 13 * sigma
        dw = 2 * np.pi / Per
        nw = int(np.ceil(wmax_sig / sigma / dw))
        self.w = dw * np.arange(nw + 1)
        W = dw * sigma * np.exp(-0.5 * (sigma * self.w) ** 2) / SQ2PI
        W[1:] *= 2
        self.sw = np.sqrt(W)
        self.D = 2 * len(self.w) - 1

    def feats(self, x):
        """M x D feature matrix."""
        ang = np.outer(x, self.w)
        C = np.cos(ang) * self.sw[None, :]
        S = np.sin(ang[:, 1:]) * self.sw[None, 1:]
        return np.hstack([C, S])

    def khat(self, w):
        s = self.sigma
        return SQ2PI * s * np.exp(-0.5 * (s * w) ** 2)


class GaussKernelOnGrid:
    def __init__(self, x, sigma):
        diam = x.max() - x.min() + 1.0
        self.F = GaussFeatures(sigma, diam)
        self.Phi = self.F.feats(x)  # M x D
        self.sigma = sigma

    def z(self, c):
        return self.Phi.T @ c

    def wit(self, z):
        return self.Phi @ z


class LaplaceKernelOnGrid:
    """k(x,y) = exp(-|x-y|/ell) on a uniform grid: K = L L^T with the AR(1) Cholesky factor, so z = L^T c and
    d = L z are O(M) recursions."""

    def __init__(self, x, ell=1.0):
        h = x[1] - x[0]
        assert np.allclose(np.diff(x), h)
        self.r = np.exp(-h / ell)
        self.s = np.sqrt(1 - self.r ** 2)
        self.M = len(x)

    def z(self, c):
        r, s = self.r, self.s
        # S_j = sum_{i>=j} r^{i-j} c_i  (backward recursion)
        from scipy.signal import lfilter
        S = lfilter([1.0], [1.0, -r], c[::-1])[::-1]
        z = S * s
        z[0] = S[0]
        return z

    def wit(self, z):
        from scipy.signal import lfilter
        r, s = self.r, self.s
        zz = z * s
        zz[0] = z[0]
        return lfilter([1.0], [1.0, -r], zz)


# ----------------------------------------------------------------------------------------------------------------
# Bathtub oracles and Wolfe's minimum-norm-point algorithm
# ----------------------------------------------------------------------------------------------------------------
def bathtub_min(d, cap, mass):
    """argmin_a d.a over 0 <= a <= cap, sum a = mass (fill smallest d first)."""
    idx = np.argsort(d, kind='stable')
    cs = np.cumsum(cap[idx])
    k = np.searchsorted(cs, mass)  # first index where cumulative >= mass
    a = np.zeros_like(cap)
    a[idx[:k]] = cap[idx[:k]]
    if k < len(idx):
        a[idx[k]] = mass - (cs[k - 1] if k > 0 else 0.0)
    return a


def oracle(kern, x, pm, qm, mass):
    d = kern.wit(x)
    a = bathtub_min(d, pm, mass)
    b = bathtub_min(-d, qm, mass)
    c = np.concatenate([a, b])  # stacked (a, b)
    return kern.z(a - b), c, d


def affine_min(Z):
    """min ||Z mu|| s.t. sum mu = 1.  Z is D x k."""
    k = Z.shape[1]
    if k == 1:
        return np.ones(1)
    # base point: column of smallest norm (better conditioning)
    j0 = int(np.argmin((Z * Z).sum(0)))
    others = [j for j in range(k) if j != j0]
    B = Z[:, others] - Z[:, [j0]]
    nu, *_ = np.linalg.lstsq(B, -Z[:, j0], rcond=1e-13)
    mu = np.empty(k)
    mu[others] = nu
    mu[j0] = 1 - nu.sum()
    return mu


def wolfe(kern, pm, qm, mass, maxit=20000, tol=1e-9, x_init=None, verbose=False):
    """Wolfe's min-norm point over Z = {z(a-b)}.  Returns dict with the best point, its coefficient vector c,
    upper certificate ||x||, lower certificate <x,q>/||x|| (q = oracle vertex at x), iterations."""
    M = len(pm)
    if x_init is None:
        x_init = kern.z(pm - qm)
    z0, c0, _ = oracle(kern, x_init, pm, qm, mass)
    Zs = [z0]
    Cs = [c0]
    lam = np.ones(1)
    x = z0.copy()
    best = None
    for it in range(maxit):
        zq, cq, d = oracle(kern, x, pm, qm, mass)
        xx = x @ x
        xq = x @ zq
        lo = xq / np.sqrt(xx) if xx > 0 else 0.0
        up = np.sqrt(xx)
        if best is None or up - max(lo, 0) < best['up'] - max(best['lo'], 0):
            best = dict(up=up, lo=lo, lam=lam.copy(), k=len(Zs))
        if verbose and it % 200 == 0:
            print(it, len(Zs), up, lo, 1 - lo / up)
        if xx - xq <= tol * xx:
            break
        Zs.append(zq)
        Cs.append(cq)
        lam = np.append(lam, 0.0)
        for _minor in range(1000):
            Z = np.array(Zs).T
            mu = affine_min(Z)
            if np.all(mu > 1e-15):
                lam = mu
                x = Z @ lam
                break
            mask = mu <= 1e-15
            th = np.min(lam[mask] / (lam[mask] - mu[mask]))
            lam = th * mu + (1 - th) * lam
            lam[lam < 1e-15] = 0.0
            keep = lam > 0
            if keep.sum() == len(lam):  # numerical: force removal of the smallest
                keep[np.argmin(lam)] = False
            Zs = [Zs[i] for i in range(len(Zs)) if keep[i]]
            Cs = [Cs[i] for i in range(len(Cs)) if keep[i]]
            lam = lam[keep]
            lam /= lam.sum()
            x = np.array(Zs).T @ lam
    AB = np.array(Cs).T @ lam
    a, b = AB[:M], AB[M:]
    C = a - b
    xfin = kern.z(C)
    zq, cq, d = oracle(kern, xfin, pm, qm, mass)
    up = np.sqrt(xfin @ xfin)
    lo = (xfin @ zq) / up
    return dict(c=C, a=a, b=b, up=up, lo=lo, it=it, k=len(Zs), d=d)


# ----------------------------------------------------------------------------------------------------------------
# Continuum certificates for the Gaussian kernel
# ----------------------------------------------------------------------------------------------------------------
_GX, _GW = np.polynomial.legendre.leggauss(12)


def gl_nodes(a, b, maxlen=0.05):
    """composite 12-point Gauss-Legendre nodes/weights on [a,b]."""
    n = max(1, int(np.ceil((b - a) / maxlen)))
    e = np.linspace(a, b, n + 1)
    mid = 0.5 * (e[:-1] + e[1:])
    half = 0.5 * (e[1:] - e[:-1])
    y = (mid[:, None] + half[:, None] * _GX[None, :]).ravel()
    w = (half[:, None] * _GW[None, :]).ravel()
    return y, w


class GaussWitness:
    """f(y) = sum_i c_i k(x_i, y) for the Gaussian kernel, evaluated exactly (chunked)."""

    def __init__(self, x, c, sigma):
        keep = c != 0
        self.x, self.c, self.s2 = x[keep], c[keep], 2 * sigma * sigma

    def __call__(self, y):
        y = np.atleast_1d(y)
        out = np.empty(len(y))
        for i in range(0, len(y), 2000):
            yy = y[i:i + 2000]
            out[i:i + 2000] = np.exp(-(yy[:, None] - self.x[None, :]) ** 2 / self.s2) @ self.c
        return out


def level_intervals(f, t, ygrid, fgrid, below=True):
    """intervals of {f<t} (below) or {f>t} on [ygrid[0], ygrid[-1]], roots refined by brentq."""
    from scipy.optimize import brentq
    g = (fgrid - t) if below else (t - fgrid)
    neg = g < 0
    ch = np.nonzero(neg[1:] != neg[:-1])[0]
    roots = []
    for j in ch:
        a, b = ygrid[j], ygrid[j + 1]
        fa = g[j]
        try:
            r = brentq(lambda y: (f(np.array([y]))[0] - t) * (1 if below else -1), a, b, xtol=1e-15, rtol=1e-15)
        except ValueError:
            r = 0.5 * (a + b)
        roots.append(r)
    pts = [ygrid[0]] + roots + [ygrid[-1]]
    ints = []
    state = neg[0]
    for k in range(len(pts) - 1):
        if state:
            ints.append((pts[k], pts[k + 1]))
        state = not state
    return ints


def mass_of(ints, cdf):
    return sum(cdf(b) - cdf(a) for a, b in ints)


def threshold_for_mass(f, ygrid, fgrid, cdf, mass, below=True):
    """t with law({f<t}) = mass (below) or law({f>t}) = mass."""
    lo, hi = fgrid.min() - 1e-300, fgrid.max() + 1e-300
    # also account for mass outside the grid (negligible): treat as inside nothing
    for _ in range(200):
        t = 0.5 * (lo + hi)
        ints = level_intervals(f, t, ygrid, fgrid, below)
        m = mass_of(ints, cdf)
        if below:
            if m < mass:
                lo = t
            else:
                hi = t
        else:
            if m < mass:
                hi = t
            else:
                lo = t
        if hi - lo <= 1e-15 * max(abs(lo), abs(hi), 1e-300):
            break
    t = 0.5 * (lo + hi)
    return t, level_intervals(f, t, ygrid, fgrid, below)


def integ_over(ints, fun, maxlen=0.05):
    tot = 0.0
    for a, b in ints:
        y, w = gl_nodes(a, b, maxlen)
        tot += (fun(y) * w).sum()
    return tot


def continuum_certs_gauss(laws, x, c, sigma, mass, A=None, dy=0.002):
    """lower and upper certificates for the continuum u(alpha) (mass = 1-alpha), Gaussian kernel."""
    f = GaussWitness(x, c, sigma)
    if A is None:
        A = laws.L() + 1.0
    ygrid = np.arange(-A, A + dy / 2, dy)
    fgrid = f(ygrid)
    # norm of f = || sum c_i phi(x_i) ||_H via Fourier features (accurate in the absolute sense)
    F = GaussFeatures(sigma, 2 * A + 2)
    zf = F.feats(x).T @ c
    fnorm = np.sqrt(zf @ zf)
    tP, SP = threshold_for_mass(f, ygrid, fgrid, laws.Pcdf, mass, below=True)
    tQ, SQ = threshold_for_mass(f, ygrid, fgrid, laws.Qcdf, mass, below=False)
    mP, mQ = mass_of(SP, laws.Pcdf), mass_of(SQ, laws.Qcdf)
    IP = integ_over(SP, lambda y: f(y) * laws.p(y))
    IQ = integ_over(SQ, lambda y: f(y) * laws.q(y))
    # exact-mass correction for the lower certificate (CVaR form: any t gives a valid bound):
    # inf_{P'} int f dP' >= tP*mass - int (tP - f)_+ dP = tP*(mass - mP) + IP
    lowP = tP * (mass - mP) + IP
    upQ = tQ * (mass - mQ) + IQ  # sup_{Q'} int f dQ' <= tQ*mass + int (f-tQ)_+ dQ
    lo = (lowP - upQ) / fnorm
    # upper certificate: nu = P|SP - Q|SQ (masses mP, mQ ~ mass); norm by Fourier features
    def nuhat():
        re = np.zeros(len(F.w)); im = np.zeros(len(F.w))
        for ints, dens, sgn in ((SP, laws.p, 1.0), (SQ, laws.q, -1.0)):
            for a, b in ints:
                y, w = gl_nodes(a, b, 0.05)
                wd = w * dens(y) * sgn
                ang = np.outer(F.w, y)
                re += np.cos(ang) @ wd
                im += np.sin(ang) @ wd
        return re, im
    re, im = nuhat()
    zup = np.concatenate([re * F.sw, im[1:] * F.sw[1:]])
    up = np.sqrt(zup @ zup)
    return dict(lo=lo, up=up, tP=tP, tQ=tQ, mP=mP, mQ=mQ, nP=len(SP), nQ=len(SQ), fnorm=fnorm)


# ----------------------------------------------------------------------------------------------------------------
# Gaussian kernel: continuum trimming by restriction to finite unions of intervals, endpoints optimised
# ----------------------------------------------------------------------------------------------------------------
def runs_to_intervals(wts, cap, edges, A, tiny=1e-13):
    """lift a discrete trimming (weights wts <= cap on cells) to a union of intervals with the same cell masses
    (to first order), fractional cells being cut on the side adjacent to the run."""
    on = wts > tiny * np.maximum(cap, 1e-300)
    ints = []
    i = 0
    M = len(wts)
    h = edges[1] - edges[0]
    while i < M:
        if not on[i]:
            i += 1
            continue
        j = i
        while j + 1 < M and on[j + 1]:
            j += 1
        th_l = wts[i] / cap[i]
        th_r = wts[j] / cap[j]
        if i == j:
            mid = 0.5 * (edges[i] + edges[i + 1])
            l, r = mid - th_l * h / 2, mid + th_l * h / 2
        else:
            l = edges[i + 1] - th_l * h
            r = edges[j] + th_r * h
        if i == 0:
            l = -A
        if j == M - 1:
            r = A
        ints.append([l, r])
        i = j + 1
    return ints


class IntervalTrim:
    def __init__(self, laws, F, A):
        self.laws, self.F, self.A = laws, F, A

    def z_and_mass(self, S, T):
        F = self.F
        re = np.zeros(len(F.w)); im = np.zeros(len(F.w))
        for ints, dens, sgn in ((S, self.laws.p, 1.0), (T, self.laws.q, -1.0)):
            for a, b in ints:
                if b <= a:
                    continue
                y, w = gl_nodes(a, b, 0.05)
                wd = w * dens(y) * sgn
                ang = np.outer(F.w, y)
                re += np.cos(ang) @ wd
                im += np.sin(ang) @ wd
        z = np.concatenate([re * F.sw, im[1:] * F.sw[1:]])
        mP = sum(self.laws.Pcdf(b) - self.laws.Pcdf(a) for a, b in S if b > a)
        mQ = sum(self.laws.Qcdf(b) - self.laws.Qcdf(a) for a, b in T if b > a)
        return z, mP, mQ


def optimise_endpoints(laws, F, S, T, mass, A, iters=60, verbose=False):
    """Gauss-Newton on the free endpoints (not at +-A) of S (for P) and T (for Q), masses held at `mass`."""
    it_ = IntervalTrim(laws, F, A)
    S = [list(I) for I in S]; T = [list(I) for I in T]

    def unpack():
        ends = []  # (which, k, side) with which 0 for S, 1 for T, side 0 left 1 right
        for which, ints in ((0, S), (1, T)):
            for k, (a, b) in enumerate(ints):
                if a > -A + 1e-12:
                    ends.append((which, k, 0))
                if b < A - 1e-12:
                    ends.append((which, k, 1))
        return ends

    def evalz():
        return it_.z_and_mass(S, T)

    def fix_mass():
        for _ in range(30):
            z, mP, mQ = evalz()
            gP, gQ = mP - mass, mQ - mass
            if abs(gP) < 1e-15 and abs(gQ) < 1e-15:
                break
            ends = unpack()
            for which, g, dens in ((0, gP, laws.p), (1, gQ, laws.q)):
                idx = [(k, side) for (wh, k, side) in ends if wh == which]
                ints = S if which == 0 else T
                if not idx:
                    continue
                vals = np.array([dens(ints[k][side]) * (1 if side == 1 else -1) for k, side in idx])
                # least-norm correction: delta = -g * vals / |vals|^2
                dl = -g * vals / (vals @ vals)
                for (k, side), dd in zip(idx, dl):
                    ints[k][side] += dd
        return evalz()

    z, mP, mQ = fix_mass()
    F2 = z @ z
    lam = 1e-3
    hist = []
    for it in range(iters):
        ends = unpack()
        n = len(ends)
        J = np.zeros((len(z), n)); G = np.zeros((2, n))
        for j, (which, k, side) in enumerate(ends):
            ints = S if which == 0 else T
            e = ints[k][side]
            sg = 1.0 if side == 1 else -1.0
            dens = laws.p(e) if which == 0 else laws.q(e)
            ph = F.feats(np.array([e]))[0]
            J[:, j] = (sg if which == 0 else -sg) * dens * ph
            G[which, j] = sg * dens
        H = 2 * (J.T @ J)
        gr = 2 * J.T @ z
        improved = False
        for _ls in range(30):
            Kk = np.zeros((n + 2, n + 2))
            Kk[:n, :n] = H + lam * np.trace(H) / n * np.eye(n)
            Kk[:n, n:] = G.T; Kk[n:, :n] = G
            rhs = np.concatenate([-gr, np.zeros(2)])
            sol = np.linalg.lstsq(Kk, rhs, rcond=1e-15)[0]
            dl = sol[:n]
            Sb = [I[:] for I in S]; Tb = [I[:] for I in T]
            for j, (which, k, side) in enumerate(ends):
                ints = S if which == 0 else T
                ints[k][side] += dl[j]
            ok = all(b > a for a, b in S) and all(b > a for a, b in T)
            if ok:
                z2, _, _ = fix_mass()
                if z2 @ z2 < F2:
                    z = z2; F2 = z2 @ z2; lam = max(lam / 3, 1e-12); improved = True
                    break
            S[:] = Sb; T[:] = Tb
            lam *= 4
        hist.append(np.sqrt(F2))
        if verbose:
            print(it, np.sqrt(F2), lam, n)
        if not improved or (len(hist) > 3 and hist[-4] - hist[-1] < 1e-10 * hist[-1]):
            break
    z, mP, mQ = evalz()
    return S, T, z, mP, mQ, hist


def certs_from_intervals(laws, F, S, T, mass, A, dy=0.002):
    """upper = ||z||; lower = separation of the continuum trimmed regions in the direction of the witness of
    nu = P|S - Q|T, computed by root finding and Gauss-Legendre quadrature."""
    it_ = IntervalTrim(laws, F, A)
    z, mP, mQ = it_.z_and_mass(S, T)
    up = np.sqrt(z @ z)
    f = lambda y: F.feats(np.atleast_1d(y)) @ z
    ygrid = np.arange(-A, A + dy / 2, dy)
    fgrid = f(ygrid)
    tP, SP = threshold_for_mass(f, ygrid, fgrid, laws.Pcdf, mass, below=True)
    tQ, SQ = threshold_for_mass(f, ygrid, fgrid, laws.Qcdf, mass, below=False)
    mP2, mQ2 = mass_of(SP, laws.Pcdf), mass_of(SQ, laws.Qcdf)
    IP = integ_over(SP, lambda y: f(y) * laws.p(y))
    IQ = integ_over(SQ, lambda y: f(y) * laws.q(y))
    lowP = tP * (mass - mP2) + IP
    upQ = tQ * (mass - mQ2) + IQ
    lo = (lowP - upQ) / up
    return dict(lo=lo, up=up, mPerr=mP - mass, mQerr=mQ - mass, nS=len(S), nT=len(T), nSP=len(SP), nSQ=len(SQ))

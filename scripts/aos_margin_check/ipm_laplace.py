"""Primal-dual interior point (Mehrotra) for the discrete trimmed-MMD problem with the Laplace kernel
k(x,y)=exp(-|x-y|/ell) on a uniform grid:  min (a-b)^T K (a-b)  s.t. 1'a = 1'b = mass, 0<=a<=p, 0<=b<=q.
K^{-1} = T is tridiagonal (Kac-Murdock-Szego), so each Newton step costs O(M) (three tridiagonal solves).
The returned (a,b) is only a candidate: certificates are computed independently (exact duality bracket)."""
import numpy as np
from scipy.linalg import solve_banded
from common import LaplaceKernelOnGrid, bathtub_min


def kms_inverse_banded(M, r):
    c = 1.0 / (1 - r * r)
    diag = np.full(M, (1 + r * r) * c)
    diag[0] = diag[-1] = c
    off = np.full(M - 1, -r * c)
    return diag, off


def ipm(x, pm, qm, mass, ell=1.0, tol=1e-11, maxit=300, verbose=False):
    """IPM in the cell fractions theta = (a/p, b/q) in [0,1]^{2M}."""
    M = len(x)
    h = x[1] - x[0]
    r = np.exp(-h / ell)
    kern = LaplaceKernelOnGrid(x, ell)
    Kmul = lambda c: kern.wit(kern.z(c))
    Td, To = kms_inverse_banded(M, r)
    th0 = mass / np.array([pm.sum(), qm.sum()])
    v = np.concatenate([np.full(M, th0[0]), np.full(M, th0[1])])
    s = 1 - v
    pq = np.concatenate([pm, -qm])

    def Hv(v):
        d = Kmul(pm * v[:M] - qm * v[M:])
        return 2 * pq * np.concatenate([d, d])

    def rhs_res(v, lam, z1, z2):
        hv = Hv(v)
        rd = hv - np.concatenate([pm * lam[0], qm * lam[1]]) - z1 + z2
        rp = np.array([pm @ v[:M] - mass, qm @ v[M:] - mass])
        return hv, rd, rp

    g0 = np.abs(Hv(v)).max()
    z1 = np.full(2 * M, g0 + 1e-12)
    z2 = np.full(2 * M, g0 + 1e-12)
    lam = np.zeros(2)

    def solve(D, g, rp):
        Da, Db = D[:M], D[M:]
        ga, gb = g[:M], g[M:]
        Lam = pm ** 2 / Da + qm ** 2 / Db
        ab = np.zeros((3, M))
        ab[0, 1:] = To / 2
        ab[1, :] = Td / 2 + Lam
        ab[2, :-1] = To / 2
        R = np.stack([pm * ga / Da - qm * gb / Db, pm ** 2 / Da, -qm ** 2 / Db], axis=1)
        W = solve_banded((1, 1), ab, R)
        w0, w1, w2 = W[:, 0], W[:, 1], W[:, 2]
        # dta = (ga - p w + p dl1)/Da ; dtb = (gb + q w + q dl2)/Db
        # sum p dta = -rp1 ; sum q dtb = -rp2
        A11 = np.sum(pm * (pm - pm * w1) / Da); A12 = -np.sum(pm * pm * w2 / Da)
        A21 = np.sum(qm * qm * w1 / Db); A22 = np.sum(qm * (qm + qm * w2) / Db)
        b1 = -rp[0] - np.sum(pm * (ga - pm * w0) / Da)
        b2 = -rp[1] - np.sum(qm * (gb + qm * w0) / Db)
        dl = np.linalg.solve(np.array([[A11, A12], [A21, A22]]), np.array([b1, b2]))
        w = w0 + w1 * dl[0] + w2 * dl[1]
        da = (ga - pm * w + pm * dl[0]) / Da
        db = (gb + qm * w + qm * dl[1]) / Db
        return np.concatenate([da, db]), dl

    def maxstep(x, dx):
        neg = dx < 0
        if not np.any(neg):
            return 1.0
        return min(1.0, np.min(-x[neg] / dx[neg]))

    best = (-np.inf, None)
    for it in range(maxit):
        if it >= 8:
            aa, bb = np.clip(v[:M], 0, 1) * pm, np.clip(v[M:], 0, 1) * qm
            lo_, up_, _, _, _ = certs_discrete(kern, pm, qm, aa, bb, mass)
            score = lo_ / up_
            if score > best[0]:
                best = (score, (aa, bb))
            if score > 1 - 1e-10:
                break
        hv, rd, rp = rhs_res(v, lam, z1, z2)
        mu = (v @ z1 + s @ z2) / (4 * M)
        obj = 0.5 * v @ hv
        if verbose and it % 5 == 0:
            print(it, obj, mu, np.abs(rd).max(), np.abs(rp).max(), flush=True)
        if mu * 4 * M < tol * max(obj, 1e-300) and np.abs(rp).max() < 1e-14:
            break
        if mu < 1e-28:
            break
        D = z1 / v + z2 / s
        rc1 = -v * z1
        rc2 = -s * z2
        g = -rd + rc1 / v - rc2 / s
        dv, dl = solve(D, g, rp)
        dz1 = (rc1 - z1 * dv) / v
        dz2 = (rc2 + z2 * dv) / s
        ap = min(maxstep(v, dv), maxstep(s, -dv))
        ad = min(maxstep(z1, dz1), maxstep(z2, dz2))
        mu_aff = ((v + ap * dv) @ (z1 + ad * dz1) + (s - ap * dv) @ (z2 + ad * dz2)) / (4 * M)
        sig = (mu_aff / mu) ** 3
        rc1 = -v * z1 - dv * dz1 + sig * mu
        rc2 = -s * z2 + dv * dz2 + sig * mu
        g = -rd + rc1 / v - rc2 / s
        dv, dl = solve(D, g, rp)
        dz1 = (rc1 - z1 * dv) / v
        dz2 = (rc2 + z2 * dv) / s
        ap = 0.995 * min(maxstep(v, dv), maxstep(s, -dv))
        ad = 0.995 * min(maxstep(z1, dz1), maxstep(z2, dz2))
        v = v + ap * dv
        s = s - ap * dv
        lam = lam + ad * dl
        z1 = z1 + ad * dz1
        z2 = z2 + ad * dz2
    a, b = np.clip(v[:M], 0, 1) * pm, np.clip(v[M:], 0, 1) * qm
    if best[1] is not None:
        lo_, up_, _, _, _ = certs_discrete(kern, pm, qm, a, b, mass)
        if lo_ / up_ < best[0]:
            a, b = best[1]
    return a, b, it


def repair_mass(w, cap, mass):
    w = w.copy()
    for _ in range(8):
        dm = mass - w.sum()
        if dm == 0:
            break
        room = (cap - w) if dm > 0 else w.copy()
        w += dm * room / room.sum()
        np.clip(w, 0, cap, out=w)
    return w


def certs_discrete(kern, pm, qm, a, b, mass):
    """exact duality bracket for the discrete problem from a candidate (a,b)."""
    a = repair_mass(a, pm, mass)
    b = repair_mass(b, qm, mass)
    c = a - b
    z = kern.z(c)
    up = np.sqrt(z @ z)
    d = kern.wit(z)
    aq = bathtub_min(d, pm, mass)
    bq = bathtub_min(-d, qm, mass)
    lo = (d @ (aq - bq)) / up
    return lo, up, a, b, d


# ----------------------------------------------------------------------------------------------------------------
# Continuum certificates for the Laplace kernel exp(-|x-y|) (ell = 1)
# ----------------------------------------------------------------------------------------------------------------
def lap_witness_eval(x, c, y):
    """f(y) = sum_i c_i exp(-|x_i - y|) for sorted uniform nodes x, any y (vectorised, O(M + len(y)))."""
    from scipy.signal import lfilter
    h = x[1] - x[0]
    r = np.exp(-h)
    Fl = lfilter([1.0], [1.0, -r], c)                 # sum_{i<=j} c_i r^{j-i}
    Fr = lfilter([1.0], [1.0, -r], c[::-1])[::-1]     # sum_{i>=j} c_i r^{i-j}
    j = np.clip(np.floor((y - x[0]) / h).astype(int), -1, len(x) - 1)
    out = np.zeros_like(y)
    left = j < 0
    right = j >= len(x) - 1
    mid = ~(left | right)
    jm = j[mid]
    out[mid] = np.exp(-(y[mid] - x[jm])) * Fl[jm] + np.exp(-(x[jm + 1] - y[mid])) * Fr[jm + 1]
    out[left] = np.exp(-(x[0] - y[left])) * Fr[0]
    out[right] = np.exp(-(y[right] - x[-1])) * Fl[-1]
    return out


def ou_quadform(y, nu, delta=None):
    """sum_{k,l} nu_k nu_l exp(-|y_k - y_l|) for sorted y (O(N)); if delta is given, the self terms are replaced by
    the exact mean of exp(-|s-t|) over a delta x delta square (midpoint cells of width delta)."""
    from scipy.signal import lfilter
    d = np.diff(y)
    # forward: G_k = nu_k + exp(-(y_k - y_{k-1})) G_{k-1}; non-uniform spacing -> explicit loop via cumulative trick
    # use log-sum: exp(-|y_k-y_l|) = exp(-(Y_k - Y_l)) with Y = y; G_k = sum_{l<=k} nu_l exp(-(y_k-y_l))
    if np.allclose(d, d[0]):
        r = np.exp(-d[0])
        G = lfilter([1.0], [1.0, -r], nu)
    else:
        G = np.empty_like(nu); acc = 0.0
        for k in range(len(nu)):
            acc = nu[k] + (np.exp(-d[k - 1]) * acc if k > 0 else 0.0)
            G[k] = acc
    tot = 2 * (nu @ G) - nu @ nu  # each pair twice, diagonal once
    if delta is not None:
        wself = 2 * (delta - 1 + np.exp(-delta)) / delta ** 2
        tot += (wself - 1) * (nu @ nu)
    return tot


def continuum_certs_laplace(laws, g, a, b, mass, nsub=20):
    x, edges, h = g['x'], g['edges'], g['h']
    pm, qm = g['pm'], g['qm']
    c = a - b
    from common import LaplaceKernelOnGrid
    kern = LaplaceKernelOnGrid(x)
    zc = kern.z(c)
    fnorm = np.sqrt(zc @ zc)
    # fine midpoint sub-grid over the cells (plus 3 units of margin on each side, where f decays like e^{-|y|})
    delta = h / nsub
    lo_e, hi_e = edges[0] - 3.0, edges[-1] + 3.0
    y = np.arange(lo_e + delta / 2, hi_e, delta)
    py, qy = laws.p(y) * delta, laws.q(y) * delta
    fy = lap_witness_eval(x, c, y)
    # lower certificate (CVaR form, valid for any thresholds): use the fine bathtub thresholds
    idx = np.argsort(fy); cs = np.cumsum(py[idx]); k = np.searchsorted(cs, mass); tP = fy[idx[min(k, len(y) - 1)]]
    idx = np.argsort(-fy); cs = np.cumsum(qy[idx]); k = np.searchsorted(cs, mass); tQ = fy[idx[min(k, len(y) - 1)]]
    LP = tP * mass - np.sum(np.maximum(tP - fy, 0) * py)
    UQ = tQ * mass + np.sum(np.maximum(fy - tQ, 0) * qy)
    lo = (LP - UQ) / fnorm
    # upper certificate: cell-ratio lift nu = sum_i (a_i/p_i) p 1_cell - (b_i/q_i) q 1_cell, exact sub-measures
    ycell = np.arange(edges[0] + delta / 2, edges[-1], delta)
    cell = np.clip(((ycell - edges[0]) / h).astype(int), 0, len(x) - 1)
    ta = np.where(pm > 0, a / np.where(pm > 0, pm, 1), 0)
    tb = np.where(qm > 0, b / np.where(qm > 0, qm, 1), 0)
    nu = (ta[cell] * laws.p(ycell) - tb[cell] * laws.q(ycell)) * delta
    # masses of the lift by the same quadrature (exact up to midpoint error); report
    up2 = ou_quadform(ycell, nu, delta)
    up = np.sqrt(max(up2, 0))
    return dict(lo=lo, up=up, fnorm=fnorm, massP=(ta[cell] * laws.p(ycell) * delta).sum() - mass,
                massQ=(tb[cell] * laws.q(ycell) * delta).sum() - mass)

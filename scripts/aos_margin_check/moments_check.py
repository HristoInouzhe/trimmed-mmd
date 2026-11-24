"""Check of Proposition newmoments: build the trimming with N vanishing moments exactly as in the proof, verify
its feasibility conditions (P' >= 0, P' <= P, Q' likewise, equal masses), the claimed inequalities
eps(w) >= l0 w^2/8, |nu| <= 2 l0 w^2, support in [-T_N w, T_N w], vanishing moments, and compare
its exact RKHS norm (Gaussian sigma=1 and Laplace) with the bound of part (b)/(a) and with the path."""
import sys, json
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from math import factorial
from common import Laws, GaussFeatures, gl_nodes, SQ2PI
from ipm_laplace import ou_quadform

law = Laws(sys.argv[1] if len(sys.argv) > 1 else 'shift', float(sys.argv[2]) if len(sys.argv) > 2 else 1.0)
l0 = law.l0
e = law.e


def gram(T, N):
    j = np.arange(N)
    s = j[:, None] + j[None, :] + 1
    return (T ** s - (-T) ** s) / s


def G_of(M, T):
    return np.linalg.solve(gram(T, len(M)), M)  # coefficients g_k of G(t) = sum g_k t^k


def polyval(c, t):
    return np.polynomial.polynomial.polyval(t, c)


def opnorm_G1(N):
    """sup over unit M of int_{-1}^1 |G_1[M]|."""
    tt, ww = np.polynomial.legendre.leggauss(400)
    def val(M):
        M = M / np.linalg.norm(M)
        return np.sum(np.abs(polyval(G_of(M, 1.0), tt)) * ww)
    best = 0
    rng = np.random.default_rng(0)
    for _ in range(60):
        r = minimize(lambda M: -val(M), rng.standard_normal(N), method='Nelder-Mead', options=dict(maxiter=2000))
        best = max(best, -r.fun)
    return best


def construct(N, w):
    cN = (2 / 3) * np.sqrt(N) * opnorm_G1(N)
    T = max(1.0, 8 * cN)
    # moments M_j(w) = - int_{-1}^1 t^j F_w(t) dt, F_w(t) = e(w t)/(l0 w)
    M = np.array([-quad(lambda t: t ** j * e(w * t) / (l0 * w), -1, 1, epsabs=1e-15, epsrel=1e-13)[0]
                  for j in range(N)])
    gc = G_of(M, T)
    M0 = np.array([-(1 - (-1) ** (j + 2)) / (j + 2) for j in range(N)])
    G0 = G_of(M0, T)
    tt = np.linspace(-T, T, 20001)
    B_N = np.max(np.abs(polyval(G0, tt))) + 1
    gfun = lambda x: np.where(np.abs(x) <= T * w, l0 * w * polyval(gc, x / w), 0.0)
    # conditions of the proof
    xs = np.linspace(-T * w, T * w, 20001)
    cond = dict(
        intG_le_quarter=float(np.trapezoid(np.abs(polyval(gc, xs / w)), xs / w)) <= 0.25,
        supG_le_BN=float(np.max(np.abs(polyval(gc, xs / w)))) <= B_N,
        minpq_ge_half=bool(np.min(np.minimum(law.p(xs), law.q(xs))) >= law.p(0) / 2),
        l0wBN_le_half=l0 * w * B_N <= law.p(0) / 2,
        e_pos=bool(np.all(e(np.linspace(1e-9, w, 1000)) > 0)),
    )
    IP = quad(e, 0, w, epsabs=1e-16)[0]
    cond['IP_in'] = 3 * l0 * w * w / 8 <= IP <= 5 * l0 * w * w / 8
    # feasibility: P' = R + (a_P - g_-), Q' = R + (a_Q - g_+) with R = min(p,q); g bounded by min(p,q)
    gx = gfun(xs)
    cond['abs_g_le_minpq'] = bool(np.all(np.abs(gx) <= np.minimum(law.p(xs), law.q(xs))))
    gneg = quad(lambda x: max(-gfun(np.array([x]))[0], 0), -T * w, T * w, limit=400, epsabs=1e-16)[0]
    gpos = quad(lambda x: max(gfun(np.array([x]))[0], 0), -T * w, T * w, limit=400, epsabs=1e-16)[0]
    IQ = -quad(e, -w, 0, epsabs=1e-16)[0]
    eps_w = IP - gneg
    eps_wQ = IQ - gpos
    nu_tv = IP + IQ + gneg + gpos
    mom = [quad(lambda x: x ** j * (e(x) * (abs(x) < w) + gfun(np.array([x]))[0]), -T * w, T * w, points=[-w, 0, w],
                limit=400, epsabs=1e-18)[0] for j in range(N + 1)]
    return dict(N=N, w=w, T=T, cN=cN, B_N=B_N, cond=cond, eps_w=eps_w, eps_wQ=eps_wQ, nu_tv=nu_tv,
                claim_eps=eps_w >= l0 * w * w / 8, claim_tv=nu_tv <= 2 * l0 * w * w, moments=mom, gfun=gfun)


def norm_gauss(con, sigma=1.0):
    T, w = con['T'], con['w']
    F = GaussFeatures(sigma, 2 * T * w + 2)
    re = np.zeros(len(F.w)); im = np.zeros(len(F.w))
    for a, b in [(-T * w, -w), (-w, 0), (0, w), (w, T * w)]:
        if b <= a:
            continue
        y, ww = gl_nodes(a, b, min(0.01, (b - a) / 4))
        dens = con['gfun'](y) + np.where(np.abs(y) < w, e(y), 0.0)
        ang = np.outer(F.w, y)
        re += np.cos(ang) @ (ww * dens)
        im += np.sin(ang) @ (ww * dens)
    z = np.concatenate([re * F.sw, im[1:] * F.sw[1:]])
    return np.sqrt(z @ z)


def norm_laplace(con):
    T, w = con['T'], con['w']
    delta = T * w / 20000
    y = np.arange(-T * w + delta / 2, T * w, delta)
    nu = (con['gfun'](y) + np.where(np.abs(y) < w, e(y), 0.0)) * delta
    return np.sqrt(max(ou_quadform(y, nu, delta), 0))


if __name__ == '__main__':
    res = []
    for N in [1, 2, 3, 4]:
        for w in [0.4, 0.2, 0.1, 0.05]:
            con = construct(N, w)
            ng = norm_gauss(con)
            nl = norm_laplace(con)
            M2N = factorial(2 * N) / (2 ** N * factorial(N))  # (2N-1)!! for sigma = 1
            bound_b = con['nu_tv'] * (con['T'] * w) ** N * np.sqrt(M2N) / factorial(N)
            bound_b_claim = 2 * l0 * w * w * (con['T'] * w) ** N * np.sqrt(M2N) / factorial(N)
            # Laplace part (a): s=1, C=2, I_{N,1} = int min(1, v^{2N}/N!^2) v^{-2} dv
            I = 2 * quad(lambda v: min(1, v ** (2 * N) / factorial(N) ** 2) / v ** 2, 0, np.inf, limit=400)[0]
            bound_a = 2 * l0 * w * w * (con['T'] * w) ** 0.5 * np.sqrt(2 * I / (2 * np.pi))
            row = dict(N=N, w=w, T=con['T'], cN=con['cN'], eps_w=con['eps_w'], eps_wQ=con['eps_wQ'],
                       l0w2_8=l0 * w * w / 8, nu_tv=con['nu_tv'], two_l0w2=2 * l0 * w * w,
                       cond=con['cond'], claim_eps=bool(con['claim_eps']), claim_tv=bool(con['claim_tv']),
                       moments=con['moments'], norm_gauss=ng, bound_b=bound_b, bound_b_claim=bound_b_claim,
                       norm_laplace=nl, bound_a_laplace=bound_a)
            res.append(row)
            print(f"N={N} w={w} T={con['T']:.2f} eps(w)={con['eps_w']:.4e} (Q side {con['eps_wQ']:.4e}, >= l0w^2/8="
                  f"{l0 * w * w / 8:.4e}: {con['claim_eps']}) |nu|={con['nu_tv']:.3e} (<=2l0w^2={2 * l0 * w * w:.3e}: "
                  f"{con['claim_tv']}) conds={all(con['cond'].values())} {con['cond']} mom={np.array(con['moments'])}"
                  f"\n    ||nu||_H gauss={ng:.4e} <= bound(b)={bound_b_claim:.4e}: {ng <= bound_b_claim}; laplace="
                  f"{nl:.4e} <= bound(a)={bound_a:.4e}: {nl <= bound_a}", flush=True)
    json.dump(res, open(f'out/moments_{law.kind}_{law.mu}.json', 'w'), indent=1, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o))

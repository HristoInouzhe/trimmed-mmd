"""R6/R5 check: the moment-cancelling trimmings of Proposition 5.2 on the discretized mu=3 mixture, Gaussian kernel
sigma=1.  For N vanishing moments, support half-width T*w, the pair (P',Q') is feasible (checked) and gives
u(alpha*-e(w)) <= ||mu_nu||_H; we print e(w), ||mu_nu|| and the exact (certified) path, interpolated in log-log."""
import numpy as np, re, sys, json
sys.path.insert(0, '.')
from kern import mixture_grid

def readlog(fn):
    rows = []
    for line in open(fn):
        m = re.match(r'e=(\S+) u=(\S+) lo=(\S+) u/e=\S+ D=(\S+)', line)
        if m: rows.append([float(v) for v in m.groups()])
    R = np.array(rows); ok = (R[:, 2] > 0) & ((R[:, 1]-R[:, 2]) <= 1e-3*R[:, 1]); return R[ok]

def hnorm_gauss(x, nu, sigma=1.0):
    w = np.linspace(0, 12/sigma, 6001)
    # nu^(w) = sum nu_i e^{-i w x_i}; |nu^|^2 even in w
    c = np.cos(np.outer(w, x)) @ nu; s = np.sin(np.outer(w, x)) @ nu
    integrand = (c**2+s**2)*np.sqrt(2*np.pi)*sigma*np.exp(-sigma**2*w**2/2)
    return np.sqrt(2*np.trapezoid(integrand, w)/(2*np.pi))

def construct(x, p, q, w, N, T):
    e = p - q; r = np.minimum(p, q)
    inn = np.abs(x) < w
    base = np.where(inn, e, 0.0)                      # a_P - a_Q (signed excess near the contact)
    if np.any((x > 0) & inn & (e < 0)) or np.any((x < 0) & inn & (e > 0)): return None
    sup = np.abs(x) <= T*w
    xs = x[sup]/w
    if N > 1:
        V = np.vander(xs, N, increasing=True)          # g_i = sum_k c_k (x_i/w)^k on the support
        M = np.vander(x[sup], N, increasing=True).T @ V
        rhs = -np.vander(x, N, increasing=True).T @ base
        c = np.linalg.solve(M, rhs)
        g = np.zeros_like(x); g[sup] = V @ c
    else:
        g = np.zeros_like(x); dm = -base.sum()          # N=1: fix the mass with a constant on the support
        g[sup] = dm/sup.sum()
    if np.any(np.abs(g) > r): return None
    gm = np.maximum(-g, 0)
    eps = np.maximum(base, 0).sum() - gm.sum()
    if eps <= 0: return None
    nu = base + g
    mom = np.vander(x, N, increasing=True).T @ nu
    return eps, nu, np.abs(nu).sum(), np.max(np.abs(mom))

x, p, q = mixture_grid(3.0, h=0.01)
path = readlog('../../gamma1_prior/mix_h0.01.log')
le, lu = np.log(path[:, 0]), np.log(path[:, 1])
out = []
for N in [1, 2, 3, 4, 6]:
    for w in [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        best = None
        for T in [1.0, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 16]:
            res = construct(x, p, q, w, N, T)
            if res is None: continue
            eps, nu, tvn, mres = res
            ub = hnorm_gauss(x, nu)
            lam0 = 0.2*2*3*np.exp(-4.5)/np.sqrt(2*np.pi)
            if eps < lam0*w*w/8: continue
            if best is None or ub < best[1]: best = (eps, ub, T, tvn, mres)
        if best is None: print(f'N={N} w={w}: no feasible T'); continue
        eps, ub, T, tvn, mres = best
        upath = np.exp(np.interp(np.log(eps), le[::-1], lu[::-1])) if path[:, 0].min() <= eps <= path[:, 0].max() else np.nan
        print(f'N={N} w={w:.2f} T={T}: e(w)={eps:.3e} |nu|={tvn:.2e} maxmom={mres:.1e}  UB={ub:.3e}  path u={upath:.3e}  UB>=u: {ub>=upath if upath==upath else "n/a"}', flush=True)
        out.append(dict(N=N, w=w, T=T, e=eps, ub=ub, u=upath))
json.dump(out, open('gauss_ub_mu3.json', 'w'), indent=1, default=float)

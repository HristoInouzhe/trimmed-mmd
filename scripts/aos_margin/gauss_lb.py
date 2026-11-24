"""R6 Gaussian kernel (sigma=1): lower bounds of RESULTS.md R4 for the discretized mixtures of the prior notes,
compared with the exact paths in gamma1_prior/*.log.  f = 1_(0,R) * K_Omega (Bohman band-limited density)."""
import numpy as np, re, sys, json
sys.path.insert(0, '.')
from kern import H, Khat, mixture_grid, bathtub_sigma, eta

def gnorm(Om, R, sigma=1.0, n=20001):
    w = np.linspace(-Om, Om, n)
    with np.errstate(invalid='ignore', divide='ignore'):
        g2 = np.where(np.abs(w) < 1e-12, R**2, 4*np.sin(w*R/2)**2/w**2)
    kh = np.sqrt(2*np.pi)*sigma*np.exp(-sigma**2*w**2/2)
    integrand = g2*Khat(w/Om)**2/kh
    return np.sqrt(np.trapezoid(integrand, w)/(2*np.pi))

def readlog(fn):
    rows = []
    for line in open(fn):
        m = re.match(r'e=(\S+) u=(\S+) lo=(\S+) u/e=\S+ D=(\S+)', line)
        if m: rows.append([float(v) for v in m.groups()])
    return np.array(rows)

def run(mu, h, logfn, out):
    x, p, q = mixture_grid(mu, h=h)
    tv = 0.5*np.abs(p-q).sum()
    R = readlog(logfn)
    Oms = np.r_[np.linspace(0.5, 4, 36), np.linspace(4.2, 16, 60)]
    Rs = np.array([3., 4., 5., 6., 7., 8.])
    norms = {(o, r): gnorm(o, r) for o in Oms for r in Rs}
    fvals = {(o, r): H(o*x) - H(o*(x-r)) for o in Oms for r in Rs}
    etas = {k: eta(v, x, p, q) for k, v in fvals.items()}
    res = []
    for e, u, lo, D in R:
        if lo <= 0 or (u-lo) > 1e-3*u: continue          # keep certified path values only
        best1 = (0.0, None); best2 = (0.0, None)
        for k in norms:
            lb1 = (e - etas[k])/norms[k]
            if lb1 > best1[0]: best1 = (lb1, k)
        # exact dual value for the best few parameter pairs of the crude bound and a neighbourhood
        for k in norms:
            if (e - etas[k]) < -5*e: continue
            s = bathtub_sigma(fvals[k], p, q, 1-(tv-e))/norms[k]
            if s > best2[0]: best2 = (s, k)
        res.append(dict(e=e, u=u, lb_crude=best1[0], par_crude=best1[1], lb_dual=best2[0], par_dual=best2[1]))
        print(f'e={e:.3e} u={u:.3e}  LBcrude={best1[0]:.3e} {best1[1]}  LBdual={best2[0]:.3e} {best2[1]}  '
              f'ratio u/LBcrude={u/max(best1[0],1e-300):.2e}', flush=True)
    json.dump(res, open(out, 'w'), default=float, indent=1)

if __name__ == '__main__':
    mu = float(sys.argv[1]); h = float(sys.argv[2]); run(mu, h, sys.argv[3], sys.argv[4])

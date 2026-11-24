"""Gaussian-kernel probe (sigma=1): certified continuum path u(alpha*-eps) vs the draft's bounds.
   python3 run_gauss.py mix 3.0 | python3 run_gauss.py shift 1.0"""
import sys, json, time
import numpy as np
from common import (Laws, make_grid, GaussKernelOnGrid, GaussFeatures, wolfe, runs_to_intervals,
                    optimise_endpoints, certs_from_intervals)
import bounds as B

kind, mu = sys.argv[1], float(sys.argv[2])
h = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
sigma = 1.0
law = Laws(kind, mu)
g = make_grid(law, h)
x = g['x']
kern = GaussKernelOnGrid(x, sigma)
A = law.L() + 1.0
F = GaussFeatures(sigma, 2 * A + 2)
astar, l0 = law.alpha_star, law.l0
facs = [20, 10, 5, 3, 2, 1.5, 1, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.16, 0.13, 0.11, 0.1, 0.09, 0.08, 0.07,
        0.06, 0.055, 0.05]
epss = [f * l0 for f in facs if f * l0 < 0.9 * astar]
if kind == 'mix' and abs(mu - 3) < 1e-9:
    epss = sorted(set(epss + [0.1, 3.2e-4]), reverse=True)
if len(sys.argv) > 4:
    epss = [float(v) * l0 for v in sys.argv[4].split(',')]
tag = sys.argv[5] if len(sys.argv) > 5 else ''
out = dict(kind=kind, mu=mu, h=h, l0=l0, alpha_star=astar, sigma=sigma, rows=[])
print('law', kind, mu, 'l0', l0, 'alpha*', astar, 'M', len(x), 'D', kern.F.D, flush=True)
for eps in epss:
    t0 = time.time()
    m = 1 - astar + eps
    r = wolfe(kern, g['pm'], g['qm'], m, tol=1e-11, maxit=20000)
    S = runs_to_intervals(r['a'], g['pm'], g['edges'], A)
    T = runs_to_intervals(r['b'], g['qm'], g['edges'], A)
    try:
        S2, T2, z, mP, mQ, hist = optimise_endpoints(law, F, S, T, m, A)
        cc = certs_from_intervals(law, F, S2, T2, m, A)
        ends = dict(S=S2, T=T2)
    except Exception as ex:  # keep going
        cc = dict(lo=np.nan, up=np.nan); ends = {}
        print('endpoint optimisation failed', ex, flush=True)
    t1 = time.time()
    dl, dpar = B.gauss_direct_lower(law, eps, sigma)
    tl, tpar = B.gauss_theorem_lower(law, eps, sigma)
    row = dict(eps=eps, eps_over_l0=eps / l0, lo_d=r['lo'], up_d=r['up'], wolfe_it=r['it'], corral=r['k'],
               lo_c=cc['lo'], up_c=cc['up'], direct_lower=dl, direct_par=dpar, thm_lower=tl, thm_par=tpar,
               D_factor=1 / m, ends=ends)
    out['rows'].append(row)
    print(f"eps={eps:.3e} (={eps / l0:.3f} l0) u_c in [{cc['lo']:.6e},{cc['up']:.6e}] u_d=[{r['lo']:.6e},"
          f"{r['up']:.6e}] it={r['it']} | direct_lo={dl:.4e} par={dpar} thm_lo={tl:.4e} par={tpar} "
          f"[{t1 - t0:.1f}s + {time.time() - t1:.1f}s]", flush=True)
    json.dump(out, open(f'out/gauss_{kind}_{mu}{tag}.json', 'w'), indent=1, default=float)

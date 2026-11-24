"""Laplace-kernel probe: certified continuum path u(alpha*-eps) vs the draft's bounds.  Usage:
   python3 run_laplace.py mix 3.0     |   python3 run_laplace.py shift 1.0"""
import sys, json, time
import numpy as np
from common import Laws, make_grid, LaplaceKernelOnGrid
from ipm_laplace import ipm, certs_discrete, continuum_certs_laplace, ou_quadform
import bounds as B

kind, mu = sys.argv[1], float(sys.argv[2])
h = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0025
law = Laws(kind, mu)
g = make_grid(law, h)
x = g['x']
kern = LaplaceKernelOnGrid(x)
astar = law.alpha_star
if kind == 'mix':
    epss = [0.1, 0.05, 0.02, 0.01, 5e-3, 2e-3, 1e-3, 5e-4, 3e-4, 2e-4, 1e-4, 5e-5]
else:
    epss = [0.3, 0.1, 0.05, 0.02, 0.01, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5]
if len(sys.argv) > 4:
    epss = [float(v) for v in sys.argv[4].split(',')]
tag = sys.argv[5] if len(sys.argv) > 5 else ''
out = dict(kind=kind, mu=mu, h=h, l0=law.l0, alpha_star=astar, rows=[])
print('law', kind, mu, 'l0', law.l0, 'alpha*', astar, 'M', len(x), flush=True)
for eps in epss:
    if eps >= astar:
        continue
    t0 = time.time()
    m = 1 - astar + eps
    a, b, it = ipm(x, g['pm'], g['qm'], m)
    lo_d, up_d, a, b, d = certs_discrete(kern, g['pm'], g['qm'], a, b, m)
    cc = continuum_certs_laplace(law, g, a, b, m, nsub=20)
    dl, dpar = B.lap_direct_lower(law, eps)
    dl2, dpar2 = B.lap_direct_lower2(law, eps, start=dpar)
    tl, tpar, Cdsb = B.sob_theorem_lower(law, eps)
    uo = B.lap_upperone(law, eps)
    # exact norm of the explicit trimming of Prop. newupperone
    wp = B.upperone_trimming(law, eps)
    delta = min(wp / 2000, 1e-4)
    yy = np.arange(-wp + delta / 2, wp, delta)
    nu = law.e(yy) * delta
    uo_exact = np.sqrt(ou_quadform(yy, nu, delta))
    row = dict(eps=eps, lo_d=lo_d, up_d=up_d, lo_c=cc['lo'], up_c=cc['up'], it=it,
               direct_lower=dl, direct_par=dpar, direct_lower_asym=dl2, direct_par_asym=dpar2, thm_lower=tl, thm_par=tpar, upperone_formula=uo,
               upperone_trim_exact=uo_exact, w_upperone=wp, D_factor=1 / m)
    out['rows'].append(row)
    print(f"eps={eps:.1e} u_c in [{cc['lo']:.6e},{cc['up']:.6e}] u_d=[{lo_d:.6e},{up_d:.6e}] it={it} | "
          f"direct_lo={dl:.4e} asym={dl2:.4e} {np.round(dpar2,3)} thm_lo={tl:.3e} | upperone={uo:.4e} trim_exact={uo_exact:.4e} "
          f"[{time.time() - t0:.1f}s]", flush=True)
out['Cdsb'] = Cdsb
json.dump(out, open(f'out/laplace_{kind}_{mu}{tag}.json', 'w'), indent=1, default=float)

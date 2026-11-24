"""R6: the closed-form lower bound of Theorem 4.2 (Bohman K, m2=pi^2, r0=1, L = sup_{0<|x|<=1}|p-q|/|x|,
T_+ exact for the mixture, theta optimised on a grid, R minimal), against the certified exact paths."""
import numpy as np, re, json
from scipy.stats import norm

def readlog(fn):
    rows = []
    for line in open(fn):
        m = re.match(r'e=(\S+) u=(\S+) lo=(\S+) u/e=\S+ D=(\S+)', line)
        if m: rows.append([float(v) for v in m.groups()])
    R = np.array(rows); ok = (R[:, 2] > 0) & ((R[:, 1]-R[:, 2]) <= 1e-3*R[:, 1]); return R[ok]

def bound(e, mu, a=0.2, sigma=1.0, r0=1.0):
    """r0=np.inf: global Lipschitz-type bound |p-q|(x) <= L|x| on all of R (no Kbar(Omega r0) term)."""
    xs = np.linspace(1e-6, min(r0, 40.0), 200001)
    d = a*(norm.pdf(xs-mu)-norm.pdf(xs+mu)); L = np.max(d/xs)
    tv = a*(2*norm.cdf(mu)-1)
    Tp = lambda r: a*(norm.sf(r-mu)-norm.sf(r+mu))
    m2 = np.pi**2; best = (0, None)
    for th in np.linspace(0.05, 0.95, 91):
        Om = np.sqrt(L*m2/(2*th*e))
        Kb = lambda t: np.where(t >= np.sqrt(2)*np.pi, 16*np.pi/(3*np.maximum(t, 1e-9)**3), 0.5)
        kb0 = 0.0 if np.isinf(r0) else Kb(Om*r0)
        Rs = np.linspace(2.0, 80, 7801)
        ok = 2*Tp(Rs/2) + 2*tv*(kb0 + Kb(Om*Rs/2)) <= (1-th)*e/2
        if not ok.any(): continue
        R = Rs[ok][0]
        val = ((1-th)*e/2)*np.sqrt(np.sqrt(2*np.pi)*sigma/R)*np.exp(-sigma**2*L*m2/(8*th*e))
        if val > best[0]: best = (val, (th, R, L))
    return best

out = {}
for mu, fn in [(3.0, 'mix_h0.01.log'), (2.0, 'var_gauss_mu2.log'), (1.5, 'var_gauss_mu1.5.log')]:
    Rr = readlog('../../gamma1_prior/'+fn); rows = []
    for e, u, lo, D in Rr:
        b, par = bound(e, mu); b2, par2 = bound(e, mu, r0=np.inf)
        rows.append((e, u, b, b2)); print(f'mu={mu} e={e:.3e} u={u:.3e} Thm4.2(r0=1)={b:.3e} Thm4.2(global L)={b2:.3e} L={par2[2] if par2 else None} below={max(b,b2)<=u}')
    out[mu] = rows
json.dump({str(k): v for k, v in out.items()}, open('thm42_formula.json', 'w'), indent=1)

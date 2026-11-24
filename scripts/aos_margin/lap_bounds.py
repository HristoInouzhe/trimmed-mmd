"""R6 Laplace kernel exp(-|x-y|) (RKHS norm ||f||^2 = (||f||^2_L2 + ||f'||^2_L2)/2): lower bounds (R2/R3, trapezoid
f) and the explicit-trimming upper bound (R5, N=1) for the discretized mixtures of the prior notes, against the
exact paths of gamma1_prior/lap_*.log.  Lower bounds use u >= (e-eta(f))/||f|| and u >= sigma(f;a*-e)/||f||."""
import numpy as np, re, sys, json
sys.path.insert(0, '.')
from kern import lap_grid, bathtub_sigma, eta

def trap(x, w, R, W):
    """0 below -w, linear to 1 on [-w,w], 1 on [w,R], linear to 0 on [R,R+W], 0 after; ||f||_H exact."""
    f = np.clip((x+w)/(2*w), 0, 1)*np.clip((R+W-x)/W, 0, 1)
    nrm2 = 0.5*(2*w/3 + (R-w) + W/3 + 1/(2*w) + 1/W)
    return f, np.sqrt(nrm2)

def readlog(fn):
    rows = []
    for line in open(fn):
        m = re.match(r'e=(\S+) u=(\S+) lo=(\S+) u/e=\S+ D=(\S+)', line)
        if m: rows.append([float(v) for v in m.groups()])
    return np.array(rows)

def ub_trim(x, p, q, e):
    """nu = a_P - a_Q, a_P = (p-q)_+ on the cells right of 0 nearest to 0 (mass e, last cell partial), a_Q likewise."""
    d = p - q
    nu = np.zeros_like(p)
    for side in (+1, -1):
        idx = np.where(side*x > 0)[0]
        idx = idx[np.argsort(np.abs(x[idx]))]
        mass = np.maximum(side*d[idx], 0)
        c = np.cumsum(mass); k = np.searchsorted(c, e)
        take = mass.copy(); take[k+1:] = 0; take[k] = e - (c[k-1] if k > 0 else 0)
        nu[idx] += side*take
        wside = np.abs(x[idx[k]])
    Kx = np.exp(-np.abs(x[:, None]-x[None, :]))
    return np.sqrt(max(nu @ Kx @ nu, 0)), np.max(np.abs(x[nu != 0]))

def run(mu, grid, logfn, out):
    x, p, q = lap_grid(mu, *grid)
    tv = 0.5*np.abs(p-q).sum()
    R = readlog(logfn)
    ws = np.geomspace(0.02, 3, 60); Rs = np.arange(3, 9.01, 0.5); Ws = np.array([0.5, 1, 2, 3, 4])
    res = []
    for e, u, lo, D in R:
        if lo <= 0 or (u-lo) > 1e-3*u: continue
        best_sym = (0, None); best_asym = (0, None); best_dual = (0, None)
        for w in ws:
            for Rr in Rs:
                f, nf = trap(x, w, Rr, 2*w)              # proof's f: 1_(0,R) * uniform[-w,w] (both ramps width 2w)
                v = (e - eta(f, x, p, q))/nf
                if v > best_sym[0]: best_sym = (v, (w, Rr))
                for W in Ws:
                    f, nf = trap(x, w, Rr, W)
                    v = (e - eta(f, x, p, q))/nf
                    if v > best_asym[0]: best_asym = (v, (w, Rr, W))
        w, Rr, W = best_asym[1]
        for w2 in w*np.geomspace(0.5, 2, 9):
            f, nf = trap(x, w2, Rr, W)
            s = bathtub_sigma(f, p, q, 1-(tv-e))/nf
            if s > best_dual[0]: best_dual = (s, (w2, Rr, W))
        ub, wid = ub_trim(x, p, q, e)
        res.append(dict(e=e, u=u, lb_sym=best_sym[0], lb_asym=best_asym[0], lb_dual=best_dual[0], ub=ub,
                        par=best_asym[1], width=wid))
        print(f'e={e:.3e} u={u:.4e}  LBsym={best_sym[0]:.4e} LBasym={best_asym[0]:.4e} LBdual={best_dual[0]:.4e} '
              f'UB={ub:.4e} (width {wid:.3f})  ok={best_sym[0]<=u and best_dual[0]<=u<=ub}', flush=True)
    json.dump(res, open(out, 'w'), default=float, indent=1)

if __name__ == '__main__':
    mu = float(sys.argv[1]); grid = tuple(float(v) for v in sys.argv[2].split(','))
    run(mu, grid, sys.argv[3], sys.argv[4])

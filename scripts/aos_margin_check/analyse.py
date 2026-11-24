"""Summaries of the probe runs: tables (markdown) of certified continuum brackets against the draft's bounds,
local log-log slopes (Laplace) and exp(-c/eps) fits (Gaussian)."""
import json, glob, sys
import numpy as np


def lap_table(fn):
    o = json.load(open(fn))
    rows = o['rows']
    print(f"\n### Laplace, {o['kind']} mu={o['mu']}  (l0={o['l0']:.5g}, alpha*={o['alpha_star']:.5g}, h={o['h']})\n")
    print("| eps | u_c lower | u_c upper | slope | u/eps^1.25 | thm formula lower | flat sym | flat asym | "
          "upperone trimming (exact) | upperone formula | checks |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    prev = None
    for r in rows:
        u = r['up_c']
        sl = '' if prev is None else f"{np.log(prev[1] / u) / np.log(prev[0] / r['eps']):.3f}"
        ok = (r['thm_lower'] <= r['up_c'] and r['direct_lower'] <= r['up_c'] and r['direct_lower_asym'] <= r['up_c']
              and r['upperone_trim_exact'] >= r['lo_c'] and r['upperone_formula'] >= r['lo_c'])
        print(f"| {r['eps']:.1e} | {r['lo_c']:.5e} | {r['up_c']:.5e} | {sl} | {u / r['eps'] ** 1.25:.4f} | "
              f"{r['thm_lower']:.3e} | {r['direct_lower']:.4e} | {r['direct_lower_asym']:.4e} | "
              f"{r['upperone_trim_exact']:.4e} | {r['upperone_formula']:.4e} | {'ok' if ok else 'VIOLATION'} |")
        prev = (r['eps'], u)


def gauss_table(fn, nfit=6):
    o = json.load(open(fn))
    rows = o['rows']
    ext = fn.replace('.json', '_ext.json')
    import os
    if os.path.exists(ext):
        rows = rows + json.load(open(ext))['rows']
    rows = sorted(rows, key=lambda r: -r['eps'])
    bad = [r for r in rows if not (r['lo_c'] > 0 and r['up_c'] < 1.1 * r['lo_c'])]
    for r in bad:
        print(f"(continuum certificate failed at eps={r['eps']:.3e}: lo={r['lo_c']:.3e} up={r['up_c']:.3e}; discrete "
              f"bracket [{r['lo_d']:.4e},{r['up_d']:.4e}] kept for reference)")
    rows = [r for r in rows if r not in bad]
    l0 = o['l0']
    print(f"\n### Gaussian sigma={o['sigma']}, {o['kind']} mu={o['mu']}  (l0={o['l0']:.5g}, alpha*={o['alpha_star']:.5g})\n")
    print("| eps | eps/l0 | u_c lower | u_c upper | u_disc (h) | thm formula lower | band-limited step lower | checks |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        lo = min(r['lo_c'], r['up_c']); up = max(r['lo_c'], r['up_c'])
        ok = r['thm_lower'] <= up and r['direct_lower'] <= up
        print(f"| {r['eps']:.3e} | {r['eps_over_l0']:.3f} | {lo:.6e} | {up:.6e} | {r['up_d']:.6e} | "
              f"{r['thm_lower']:.3e} | {r['direct_lower']:.4e} | {'ok' if ok else 'VIOLATION'} |")
    # fits log u = A + k log eps - c/eps on the smallest-eps points
    def fit(vals):
        e = np.array([r['eps'] for r in rows])[-nfit:]
        y = np.log(np.array(vals)[-nfit:])
        X = np.stack([np.ones_like(e), np.log(e), -1 / e], 1)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        # also two-parameter fit with k fixed at 1 (prefactor eps)
        X2 = np.stack([np.ones_like(e), -1 / e], 1)
        coef2, *_ = np.linalg.lstsq(X2, y - np.log(e), rcond=None)
        return coef[2] / l0, coef[1], coef2[1] / l0
    # window fit: all rows with 1e-8 <= u <= 1e-3 (three-parameter)
    sel = [r for r in rows if 1e-8 <= r['up_c'] <= 1e-3]
    if len(sel) >= 4:
        e_ = np.array([r['eps'] for r in sel]); y_ = np.log([r['up_c'] for r in sel])
        X = np.stack([np.ones_like(e_), np.log(e_), -1 / e_], 1)
        cw, *_ = np.linalg.lstsq(X, y_, rcond=None)
        yd = np.log([r['direct_lower'] for r in sel]); cwd, *_ = np.linalg.lstsq(X, yd, rcond=None)
        print(f"window 1e-8<=u<=1e-3 ({len(sel)} pts, eps/l0 in [{sel[-1]['eps_over_l0']:.3f},{sel[0]['eps_over_l0']:.3f}]):"
              f" path c'/l0={cw[2] / l0:.3f} (power {cw[1]:.2f}); band-limited bound c'/l0={cwd[2] / l0:.3f} (power {cwd[1]:.2f})")
    cpath = fit([max(r['up_c'], r['lo_c']) for r in rows])
    cdir = fit([r['direct_lower'] for r in rows])
    cthm = fit([r['thm_lower'] for r in rows])
    print(f"\nfit on the {nfit} smallest eps: path c'/l0 = {cpath[0]:.3f} (power {cpath[1]:.2f}; with power 1 fixed "
          f"{cpath[2]:.3f}); band-limited-step bound c'/l0 = {cdir[0]:.3f} (power {cdir[1]:.2f}; fixed {cdir[2]:.3f}); "
          f"theorem formula c'/l0 = {cthm[0]:.3f} (fixed {cthm[2]:.3f}); pi^2/8 = {np.pi ** 2 / 8:.4f}")
    # local slopes d log u / d log eps
    e = np.array([r['eps'] for r in rows]); u = np.array([max(r['up_c'], r['lo_c']) for r in rows])
    sl = np.diff(np.log(u)) / np.diff(np.log(e))
    print('local log-log slopes:', ' '.join(f"{s:.2f}" for s in sl))


if __name__ == '__main__':
    for fn in sorted(glob.glob('out/laplace_*.json')):
        lap_table(fn)
    for fn in sorted(glob.glob('out/gauss_*.json')):
        if fn.endswith('_ext.json'):
            continue
        gauss_table(fn)

"""Adult and ACS with ONE witness per split (fitted at alpha = 0) and level-specific cut points.

The per-level random-feature QP of fair_split_certificate.py costs about 9 min per level on Adult (2,000 FISTA
iterations on 22.6k x 512 features, single thread) and roughly 3x that on ACS, i.e. hours per split; at alpha = 0 the
constraint set is a single point and the fit is immediate. Same split RNG stream as split_test at level 0
(rng = default_rng(1000 + s); permutation; fold-1 median bandwidth; RFF with 512 features).

For each grid level (0, 0.05, ..., 0.8) on the SAME fold-2 values:
  boot_dir0   percentile bootstrap (500) of the unclipped trimmed-mean difference of this witness (Thm 16 rule)
  ramp_bet    Theorem 1 (ramp at self-excluded fold-1 quantiles, betting, c = 1/2)
  ramp_b2s, ramp_hoef, ramp_clt, prop17 (kappa = sqrt 2 for phased RFF), tlm
The published per-level-witness bootstrap numbers are in results/fair/<name>_split_cert.csv.

    python3 -I explore/split_adaptive/real_dir0.py adult acs --splits 8 --workers 4
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from trimmedmmd.kernels import RFF, median_heuristic  # noqa: E402
from trimmedmmd.splitting import evaluate_direction, trimmed_mean, witness_direction  # noqa: E402

import sacert as S  # noqa: E402
from fold1 import cut_points, pair_var  # noqa: E402
from real import dataset, grid_for  # noqa: E402

DELTA = 0.05
METHODS = ["boot_dir0", "ramp_bet", "ramp_b2s", "ramp_hoef", "ramp_clt", "prop17", "tlm"]


def one(args):
    name, i, n_boot = args
    X, Y = dataset(name)
    rng = np.random.default_rng(1000 + i)
    n, m = len(X), len(Y)
    ix, iy = rng.permutation(n), rng.permutation(m)
    n1, m1 = int(n * 0.5), int(m * 0.5)
    X1, X2 = X[ix[:n1]], X[ix[n1:]]
    Y1, Y2 = Y[iy[:m1]], Y[iy[m1:]]
    sigma = median_heuristic(X1, Y1, rng=rng)
    rff = RFF(dim=X.shape[1], n_features=512, sigma=sigma, rng=rng)
    t0 = time.time()
    w, v, norm = witness_direction(X1, Y1, 0.0, sigma, rng=rng, rff=rff)
    fit_s = time.time() - t0
    u2 = evaluate_direction(w, v, X1, Y1, X2, sigma, norm, rff=rff)
    v2 = evaluate_direction(w, v, X1, Y1, Y2, sigma, norm, rff=rff)
    zx, zy = rff.transform(X1), rff.transform(Y1)
    u1 = evaluate_direction(w, v, X1, Y1, X1, sigma, norm, rff=rff) - w * (zx ** 2).sum(1) / norm
    v1 = evaluate_direction(w, v, X1, Y1, Y1, sigma, norm, rff=rff) + v * (zy ** 2).sum(1) / norm
    kappa = math.sqrt(2.0)
    rows = []
    for a in grid_for(name):
        beta = 1.0 / (1.0 - a)
        r = {"alpha": float(a)}
        stat = lambda av, bv: trimmed_mean(av, a, upper=False) - trimmed_mean(bv, a, upper=True)
        sep = stat(u2, v2)
        ia = rng.integers(0, u2.size, size=(n_boot, u2.size))
        ib = rng.integers(0, v2.size, size=(n_boot, v2.size))
        boots = np.array([stat(u2[p], v2[q]) for p, q in zip(ia, ib)])
        r["T_hat"] = float(sep)
        r["boot_dir0"] = bool(np.quantile(boots, DELTA) > 0)
        e = lambda k: math.sqrt(math.log(4.0 / DELTA) / (2.0 * k))
        r["prop17"] = bool(sep > 2 * kappa * beta * (e(u2.size) + e(v2.size)))
        r["tlm"] = bool(S.dkw_lower_trimmed_lcb(u2, a, -kappa, DELTA / 2)
                        + S.dkw_lower_trimmed_lcb(-v2, a, -kappa, DELTA / 2) > 0)
        s, sp = cut_points(u1, v1, a)
        r["s"], r["sp"] = s, sp
        if s > sp:
            G, H = S.ramp(u2, s, sp), S.ramp(v2, s, sp)
            Gf, Hf = S.ramp(u1, s, sp), S.ramp(v1, s, sp)
            pv = max(pair_var(Gf.var(), Hf.var(), G.size, H.size), 1e-4)
            r["Delta_hat"] = float(G.mean() - H.mean())
            r["ramp_bet"] = S.betting_test(S.pair_up(G, -H) - a, -(1 + a), DELTA, c=0.5, prior_var=pv)[0]
            r["ramp_b2s"] = bool(S.lcb_bern2s(G, 1.0, -H, 1.0, DELTA) > a)
            r["ramp_hoef"] = bool(S.hoeffding2s_lcb(G, 1.0, -H, 1.0, DELTA) > a)
            r["ramp_clt"] = bool(r["Delta_hat"] - 1.6448536269514722
                                 * math.sqrt(G.var(ddof=1) / G.size + H.var(ddof=1) / H.size) > a)
            r["ramp_b2s_lcb"] = S.lcb_bern2s(G, 1.0, -H, 1.0, DELTA)
        else:
            for k in ("ramp_bet", "ramp_b2s", "ramp_hoef", "ramp_clt"):
                r[k] = False
        rows.append(r)
        if not any(r[k] for k in METHODS):
            break
    lo = {}
    for mth in METHODS:
        best = -1.0
        for r in rows:
            if r[mth]:
                best = r["alpha"]
            else:
                break
        lo[mth] = best
    return {"dataset": name, "split": i, "lower": lo, "levels": rows, "n": n, "m": m, "fit_seconds": fit_s,
            "norm": float(norm)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--splits", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out = HERE / "out" / "real_dir0.jsonl"
    jobs = [(nm, i, 500) for nm in a.names for i in range(a.splits)]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex, open(out, "a") as fh:
        for rec in ex.map(one, jobs):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"  {rec['dataset']} {rec['split']}: {rec['lower']} fit {rec['fit_seconds']:.0f}s "
                  f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

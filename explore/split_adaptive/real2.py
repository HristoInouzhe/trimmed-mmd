"""Second real-data pass (exploratory): store the fold-1 and fold-2 witness values at every
level, so that cut-point rules can be compared after the fact.

Same splits and witnesses as real.py (same RNG stream as splitting.split_test). Saves
out/real2/<dataset>_<split>.npz with, per level, the fold-1 values with and without the
self-term and the fold-2 values. The rules are evaluated by analyze_real2.py.

NOTE: comparing rules on the same data is a sensitivity analysis. Only the pre-specified
rule (self-excluded quantiles, as in real.py) is reported as the certificate.

    python3 -I explore/split_adaptive/real2.py rhc nhanes_child nhanes_age nhanes_bmi --workers 4
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
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

from trimmedmmd.kernels import median_heuristic  # noqa: E402
from trimmedmmd.splitting import evaluate_direction, witness_direction  # noqa: E402

from real import dataset, grid_for  # noqa: E402
import sacert as S  # noqa: E402
from fold1 import pair_var  # noqa: E402

DELTA = 0.05


def ramp_bet_rule(u1, v1, u2, v2, a, swap):
    s, sp = float(np.quantile(u1, 1 - a)), float(np.quantile(v1, a))
    if not s > sp:
        if not swap or s == sp:
            return False
        s, sp = sp, s           # crossed fold-1 cut points: ramp between them, other order
    G, H = S.ramp(u2, s, sp), S.ramp(v2, s, sp)
    Gf, Hf = S.ramp(u1, s, sp), S.ramp(v1, s, sp)
    pv = max(pair_var(Gf.var(), Hf.var(), G.size, H.size), 1e-4)
    return bool(S.betting_test(S.pair_up(G, -H) - a, -(1 + a), DELTA, c=0.5, prior_var=pv)[0])

OUT = HERE / "out" / "real2"


def one(args):
    name, i, top = args
    X, Y = dataset(name)
    split_seed = int(np.random.default_rng(1000 + i).integers(0, 2**31 - 1))
    store = {}
    alive = {"A_selfexcl", "B_insample", "C_selfexcl_swap"}
    for a in grid_for(name):
        if a > top + 1e-9:
            break
        rng = np.random.default_rng(split_seed)
        n, m = len(X), len(Y)
        ix, iy = rng.permutation(n), rng.permutation(m)
        n1, m1 = int(n * 0.5), int(m * 0.5)
        X1, X2 = X[ix[:n1]], X[ix[n1:]]
        Y1, Y2 = Y[iy[:m1]], Y[iy[m1:]]
        sigma = median_heuristic(X1, Y1, rng=rng)
        w, v, norm = witness_direction(X1, Y1, a, sigma, rng=rng)
        key = f"{a:.3f}"
        if norm <= 0:
            store[key + "_norm"] = np.array([0.0])
            continue
        u1 = evaluate_direction(w, v, X1, Y1, X1, sigma, norm)
        v1 = evaluate_direction(w, v, X1, Y1, Y1, sigma, norm)
        store[key + "_u1in"] = u1.astype(np.float32)
        store[key + "_v1in"] = v1.astype(np.float32)
        store[key + "_wn"] = (w / norm).astype(np.float32)
        store[key + "_vn"] = (v / norm).astype(np.float32)
        store[key + "_u2"] = evaluate_direction(w, v, X1, Y1, X2, sigma, norm).astype(np.float32)
        store[key + "_v2"] = evaluate_direction(w, v, X1, Y1, Y2, sigma, norm).astype(np.float32)
        store[key + "_norm"] = np.array([norm])
        u2, v2 = store[key + "_u2"].astype(float), store[key + "_v2"].astype(float)
        u1x, v1x = u1 - w / norm, v1 + v / norm
        res = {"A_selfexcl": ramp_bet_rule(u1x, v1x, u2, v2, a, False),
               "B_insample": ramp_bet_rule(u1, v1, u2, v2, a, False),
               "C_selfexcl_swap": ramp_bet_rule(u1x, v1x, u2, v2, a, True)}
        store[key + "_res"] = np.array([res["A_selfexcl"], res["B_insample"], res["C_selfexcl_swap"]])
        alive = {k for k in alive if res[k]}
        if not alive:
            break
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{name}_{i}.npz", **store)
    return name, i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--splits", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    tops = {"rhc": 0.5, "nhanes_child": 0.95, "nhanes_age": 0.55, "nhanes_bmi": 0.3}
    jobs = [(nm, i, tops.get(nm, 0.75)) for nm in a.names for i in range(a.splits)
            if not (OUT / f"{nm}_{i}.npz").exists()]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for nm, i in ex.map(one, jobs):
            print(f"  {nm} {i} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

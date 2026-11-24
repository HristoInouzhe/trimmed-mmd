"""Power on the paper's synthetic design (disjoint_d2_a20, alpha* = 0.2).

Same draws as Table tab:calibrations and the split classifier baseline
(scripts/exp_classifier_baseline.py): replicate i uses seed 70_000 + i,
X, Y = sc.sample(n, rng, "P"), sc.sample(n, rng, "Q"), then the pooled
median bandwidth from the same rng. For every replicate we store the whole
path D-hat_alpha on ALPHAS (exact Gram, warm starts) and, on the same data,
one random split with the bootstrap certificate (Theorem thm:split) and the
finite-sample split certificate (Proposition prop:splitfs) at every level.
Thresholds are applied afterwards (analyze.py), so nothing here depends on
which threshold is being judged.

    python3 -I explore/sharper_threshold/power.py --workers 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool

import numpy as np

import common as C
from trimmedmmd.splitting import split_test

ALPHAS = [0.0, 0.05, 0.1, 0.15, 0.2]
SPLIT_ALPHAS = [0.0, 0.05, 0.1, 0.15, 0.2]
DELTA = 0.05
SEED0 = 70_000


def work(args):
    n, seed = args
    sc = C.dgp.make("disjoint_d2_a20")
    rng = np.random.default_rng(seed)
    X = sc.sample(n, rng, "P")
    Y = sc.sample(n, rng, "Q")
    sigma = C.median_heuristic(X, Y, rng=rng)
    t0 = time.time()
    Kxx = C.gaussian_kernel(X, X, sigma)
    Kyy = C.gaussian_kernel(Y, Y, sigma)
    Kxy = C.gaussian_kernel(X, Y, sigma)

    def one(a, w0, v0):
        return C.solve_gram(Kxx, Kyy, Kxy, a, w0=w0, v0=v0, rng=rng,
                            max_iter=4000, tol=1e-11)

    res = C.solve_path(one, np.array(ALPHAS))
    del Kxx, Kyy, Kxy
    t1 = time.time()
    split_seed = seed + 500_000
    split = []
    for a in SPLIT_ALPHAS:
        r = split_test(X, Y, a, delta=DELTA, rng=np.random.default_rng(split_seed),
                       max_iter=4000, tol=1e-11)
        split.append(dict(alpha=a, sep=r.separation, lower=r.lower_ci,
                          certified=bool(r.certified)))
    return dict(n=n, seed=seed, sigma=float(sigma),
                dhat=[r.discrepancy for r in res],
                dhat_lower=[r.discrepancy_lower for r in res],
                converged=[bool(r.converged) for r in res],
                split=split, n2=n - n // 2, secs_path=t1 - t0,
                secs_split=time.time() - t1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sizes", type=int, nargs="*", default=[200, 400, 800, 1600, 3200])
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--reps_mid", type=int, default=50)    # n = 1600
    ap.add_argument("--reps_big", type=int, default=24)    # n = 3200
    a = ap.parse_args()
    out = C.OUT / "power_reps.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["n"], r["seed"]))
    jobs = [(n, SEED0 + i) for n in a.sizes
            for i in range(a.reps_big if n >= 3200 else a.reps_mid if n >= 1600 else a.reps)
            if (n, SEED0 + i) not in done]
    print(f"{len(jobs)} jobs", flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool, out.open("a") as fh:
        for i, row in enumerate(pool.imap_unordered(work, jobs, chunksize=1)):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if i % 20 == 0:
                print(i, row["n"], round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

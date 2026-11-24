"""Recompute the classifier baselines on exactly the folds of synth.py.

(The first synth run used a fold-2 variance as the betting prior for gbm_bet, which is not
predictable; this recomputes all gbm_* columns with a fixed prior pv = 0.5.)

    python3 -I explore/split_adaptive/gbm_rerun.py --ns 100 200 400 800 1600 --reps 200 --workers 12
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src")); sys.path.insert(0, str(HERE))
import numpy as np
from trimmedmmd import dgp
from synth import gbm_tests, MAIN_ALPHAS, GRID

def one(args):
    scen, n, i, seed0 = args
    sc = dgp.make(scen)
    seed = seed0 + i
    rng = np.random.default_rng(seed)
    X, Y = sc.sample(n, rng, "P"), sc.sample(n, rng, "Q")
    ix, iy = rng.permutation(n), rng.permutation(n)
    n1 = n // 2
    X1, X2 = X[ix[:n1]], X[ix[n1:]]
    Y1, Y2 = Y[iy[:n1]], Y[iy[n1:]]
    return {"n": n, "rep": i, "gbm": gbm_tests(X1, Y1, X2, Y2, seed, sorted(set(MAIN_ALPHAS + GRID)))}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="disjoint_d2_a20")
    ap.add_argument("--ns", type=int, nargs="+", default=[100, 200, 400, 800, 1600])
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    jobs = [(a.scenario, n, i, 70_000) for n in a.ns for i in range(a.reps)]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, jobs, chunksize=8))
    (HERE / "out" / f"gbm_{a.scenario}.jsonl").write_text("\n".join(json.dumps(r) for r in res) + "\n")
    print("done", len(res))

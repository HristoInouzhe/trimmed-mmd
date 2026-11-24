"""Sequential (multi-batch) certificates vs single splits, same data as synth.py.

Schemes (fractions of each sample, rows in the random order of synth.py):
  s50  : fit [0, .5)            bet [.5, 1)                  (= single split)
  s30  : fit [0, .3)            bet [.3, 1)
  seq3 : fit [0,.25) bet [.25,.5); refit [0,.5) bet [.5,1)
  seq4 : fit [0,.125) bet ..; refit [0,.25) bet ..; refit [0,.5) bet [.5,1)
The bandwidth is the median heuristic on the first fit batch only.

    python3 -I explore/split_adaptive/synth_seq.py --ns 100 200 400 800 --reps 200 --workers 12
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from trimmedmmd import dgp  # noqa: E402
from trimmedmmd.kernels import median_heuristic  # noqa: E402

from seqcert import seq_ramp_bet  # noqa: E402

SCHEMES = {"s30": [0, .3, 1], "seq3": [0, .25, .5, 1],
           "seq4": [0, .125, .25, .5, 1]}
LEVELS = [0.0, 0.1, 0.15, 0.2]


def one_rep(args):
    scen, n, i, seed0 = args
    sc = dgp.make(scen)
    rng = np.random.default_rng(seed0 + i)
    X, Y = sc.sample(n, rng, "P"), sc.sample(n, rng, "Q")
    ix, iy = rng.permutation(n), rng.permutation(n)
    X, Y = X[ix], Y[iy]
    rec = {"scenario": scen, "n": n, "rep": i, "res": {}}
    for name, edges in SCHEMES.items():
        b1 = int(round(edges[1] * n))
        sigma = median_heuristic(X[:b1], Y[:b1], rng=np.random.default_rng(seed0 + i + 1))
        for a in LEVELS:
            rej, logK = seq_ramp_bet(X, Y, a, edges, sigma, np.random.default_rng(seed0 + i + 2))
            rec["res"][f"{name}@{a}"] = [bool(rej), float(logK)]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="disjoint_d2_a20")
    ap.add_argument("--ns", type=int, nargs="+", default=[100, 200, 400, 800])
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed0", type=int, default=70_000)
    a = ap.parse_args()
    out = HERE / "out" / f"seq_{a.scenario}.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["n"], r["rep"]))
    jobs = [(a.scenario, n, i, a.seed0) for n in a.ns for i in range(a.reps) if (n, i) not in done]
    jobs.sort(key=lambda j: -j[1])
    t0 = time.time()
    print(f"{len(jobs)} jobs", flush=True)
    with ProcessPoolExecutor(a.workers) as ex, open(out, "a") as fh:
        futs = [ex.submit(one_rep, j) for j in jobs]
        for k, f in enumerate(as_completed(futs)):
            fh.write(json.dumps(f.result()) + "\n")
            fh.flush()
            if k % 100 == 0:
                print(f"  {k+1}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

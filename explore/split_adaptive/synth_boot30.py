"""Fairness check for the sequential/30-70 results: the bootstrap of Theorem 16 with the SAME 30/70 split.

Same data and row order as synth_seq.py; witness fitted on the first 30% of each sample (fold-1 median bandwidth),
percentile bootstrap (1,000) of the unclipped trimmed-mean difference on the remaining 70%, and Theorem 1 (ramp +
betting) on the same split for a matched pair.

    python3 -I explore/split_adaptive/synth_boot30.py --ns 200 400 800 --reps 200 --workers 8
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from trimmedmmd import dgp  # noqa: E402
from trimmedmmd.kernels import median_heuristic  # noqa: E402

import sacert as S  # noqa: E402
from fold1 import Witness, cut_points, priors  # noqa: E402

LEVELS = [0.1, 0.15, 0.2]


def one(args):
    scen, n, i, seed0, frac = args
    sc = dgp.make(scen)
    rng = np.random.default_rng(seed0 + i)
    X, Y = sc.sample(n, rng, "P"), sc.sample(n, rng, "Q")
    ix, iy = rng.permutation(n), rng.permutation(n)
    X, Y = X[ix], Y[iy]
    b = int(round(frac * n))
    X1, X2, Y1, Y2 = X[:b], X[b:], Y[:b], Y[b:]
    sigma = median_heuristic(X1, Y1, rng=np.random.default_rng(seed0 + i + 1))
    r2 = np.random.default_rng(seed0 + i + 2)
    out = {"n": n, "rep": i, "frac": frac, "res": {}}
    for a in LEVELS:
        wit = Witness(X1, Y1, a, sigma, r2)
        u2, v2 = wit(X2), wit(Y2)
        ia = r2.integers(0, u2.size, size=(1000, u2.size))
        ib = r2.integers(0, v2.size, size=(1000, v2.size))
        boots = np.array([S.trimmed_sep(u2[p], v2[q], a) for p, q in zip(ia, ib)])
        u1, v1 = wit.fold1_values()
        s, sp = cut_points(u1, v1, a)
        bet = False
        if s > sp:
            pr = priors(u1, v1, s, sp, a, u2.size, v2.size)
            Z = S.pair_up(S.ramp(u2, s, sp), -S.ramp(v2, s, sp)) - a
            bet = S.betting_test(Z, -(1 + a), 0.05, c=0.5, prior_var=pr.get("ramp"))[0]
        out["res"][str(a)] = {"boot": bool(np.quantile(boots, 0.05) > 0), "ramp_bet": bool(bet)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[200, 400, 800])
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--frac", type=float, default=0.3)
    a = ap.parse_args()
    jobs = [("disjoint_d2_a20", n, i, 70_000, a.frac) for n in a.ns for i in range(a.reps)]
    jobs.sort(key=lambda j: -j[1])
    outf = HERE / "out" / f"boot{int(a.frac * 100)}_disjoint_d2_a20.jsonl"
    with ProcessPoolExecutor(a.workers) as ex, open(outf, "a") as fh:
        for f in as_completed([ex.submit(one, j) for j in jobs]):
            fh.write(json.dumps(f.result()) + "\n")
            fh.flush()
    rows = [json.loads(l) for l in outf.read_text().splitlines()]
    import pandas as pd
    df = pd.DataFrame([{"n": r["n"], "alpha": float(k), **v} for r in rows for k, v in r["res"].items()])
    print(df.groupby(["alpha", "n"]).mean(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    main()

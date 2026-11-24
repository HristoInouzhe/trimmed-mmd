"""Monte Carlo validity check of the candidate thresholds at boundary nulls.

For every design (TV(P,Q) = alpha* exactly) and replicate we record the QP
statistic D-hat at alpha*, and, where the common trimming R is known in
closed form, the pieces of the deterministic inequality of Theorem P
(D-hat <= plug-in pair <= M + rho (c_X + c_Y)) so that both the inequality
and the frequency of the deviation events {M > t1}, {c_X + c_Y > t2} can be
checked directly.

    python3 -I explore/sharper_threshold/validity.py --workers 14 --reps 400
"""

from __future__ import annotations

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import time
from multiprocessing import Pool

import numpy as np

import common as C
from common import Atoms, Disjoint, GaussContam
import thresholds as T

N = 300
DELTA = 0.05


def fixed_sigma_disjoint(a):
    rng = np.random.default_rng(5)
    sc = C.dgp.disjoint_contamination(dim=2, a=a)
    X, Y = sc.sample(3000, rng, "P"), sc.sample(3000, rng, "Q")
    return float(C.median_heuristic(X, Y, rng=rng))


def designs():
    s20 = fixed_sigma_disjoint(0.2)
    return {
        "e1_median": Disjoint(a=0.2, bw="median"),
        "e1_fixed": Disjoint(a=0.2, bw="fixed", fixed_sigma=s20),
        "e1_bw0.1": Disjoint(a=0.2, bw="median", bw_scale=0.1),
        "e1_bw5": Disjoint(a=0.2, bw="median", bw_scale=5.0),
        "disjoint_a0.5": Disjoint(a=0.5, bw="median"),
        "same_gauss": Disjoint(a=0.0, bw="median"),
        "disjoint_d10": Disjoint(d=10, a=0.2, bw="median"),
        "atoms3_a0.2": Atoms(k_core=1, a=0.2),
        "atoms2_P=Q": Atoms(k_core=2, a=0.0),
        "atoms200_a0.2": Atoms(k_core=200, a=0.2),
        "atoms200_P=Q": Atoms(k_core=200, a=0.0),
        "atoms3_a0.5": Atoms(k_core=1, a=0.5),
        "gauss_contam": GaussContam(a=0.3, mu_norm=1.0),
        # second wave: adversarial pairs for the data-free floor
        "atoms2_a0.2": Atoms(k_core=2, a=0.2),
        "atoms2_a0.05": Atoms(k_core=2, a=0.05),
        "atoms3_a0.05": Atoms(k_core=1, a=0.05),
        "atoms200_a0.05": Atoms(k_core=200, a=0.05),
        "atoms5_a0.2": Atoms(k_core=5, a=0.2),
        "atoms2_a0.5": Atoms(k_core=2, a=0.5),
    }


_D = None


def work(args):
    global _D
    key, seed, n = args
    if _D is None:
        _D = designs()
    des = _D[key]
    rng = np.random.default_rng(seed)
    X, Y = des.draw(n, n, rng)
    sig = des.sigma(X, Y, rng)
    Kxx = C.gaussian_kernel(X, X, sig)
    Kyy = C.gaussian_kernel(Y, Y, sig)
    Kxy = C.gaussian_kernel(X, Y, sig)
    al = float(des.alpha_star)
    d, dlo, conv = C.dhat(Kxx, Kyy, Kxy, al, rng=np.random.default_rng(seed + 1))
    row = dict(design=key, seed=seed, n=n, alpha=al, sigma=float(sig),
               dhat=d, dhat_lower=dlo, converged=bool(conv))
    if getattr(des, "plugin", True):
        a, b = des.densities(X, Y)
        mux, muy = des.mu_R(X, sig), des.mu_R(Y, sig)
        row.update(C.plugin_decomposition(Kxx, Kyy, Kxy, a, b, mux, muy,
                                          des.mu_R_sq(sig), al))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    keys = a.only or list(designs())
    out = C.OUT / "validity_reps.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["design"], r["seed"], r["n"]))
    jobs = [(k, 10_000 + i, N) for k in keys for i in range(a.reps)
            if (k, 10_000 + i, N) not in done]
    print(f"{len(jobs)} jobs", flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool, out.open("a") as fh:
        for i, row in enumerate(pool.imap_unordered(work, jobs, chunksize=2)):
            fh.write(json.dumps(row) + "\n")
            if i % 200 == 0:
                fh.flush()
                print(i, round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

"""AoS revision: small validity probes of the new tests' implementations (not part of the study's replicates).

Writes results/aos/baselines/probes.json (read by scripts/aos_baselines_tables.py, which turns it into macros):

1. permutation MMD test: p-values under P = Q (Gaussian, d = 3, n = m = 100, B = 999), rejection rate at delta and a
   KS test of uniformity;
2. HPLB lambda_adapt: rate of lambda_hat > 0 under P = Q (uniform scores) and of lambda_hat > TV at a tight boundary
   (TV = 0.2, oracle score with point masses), m = n = 400;
3. MMDAgg on the mixed-type P = Q null (N0_D5b_mixed's law) at n = 100, m = 500, fresh seeds (rep 500000+): the
   rejection rate with the MMDAgg* bandwidths (inter-sample distances, as run) and with pooled-distance bandwidths
   (a permutation-invariant rule), to tell Monte Carlo slack from a defect.

    ~/.venvs/llm_book/bin/python -B scripts/aos_baselines_probes.py --workers 3
"""

from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aos_baselines_run as A  # noqa: E402

OUT = A.OUT / "probes.json"
R_PERM, R_HPLB, R_AGG = 600, 400, 300


def perm_one(s):
    rng = np.random.default_rng([20261124, s])
    n = m = 100
    Z = rng.standard_normal((n + m, 3))
    D2 = A.sqdist(Z, Z)
    K = np.exp(-D2 / 2.0)
    zl = np.r_[np.full(n, 1 / n), np.full(m, -1 / m)]
    Zp = rng.permuted(np.tile(zl, (A.B_PERM, 1)), axis=1).T
    T = zl @ K @ zl
    Tb = np.einsum("ib,ib->b", Zp, K @ Zp)
    return float((1 + np.sum(Tb >= T - 1e-12 * abs(T))) / (A.B_PERM + 1))


def hplb_one(s):
    rng = np.random.default_rng([20261125, s])
    m = n = 400
    grid = A.hplb_lambda_grid([j * 0.2 / 8 for j in range(11)])
    null = A.hplb_adapt(rng.random(m), rng.random(n), grid, rng) > 0
    cx, cy = rng.random(m) < 0.2, rng.random(n) < 0.2
    rx, ry = np.where(cx, 0.0, rng.random(m)), np.where(cy, 1.0, rng.random(n))
    bound = A.hplb_adapt(rx, ry, grid, rng) > 0.2 + 1e-12
    return int(null), int(bound)


def _pooled(D2, n, L=A.AGG_L):
    d = np.sqrt(D2[np.triu_indices(D2.shape[0], 1)])
    lmin = float(d.min())
    if lmin < 0.1:
        lmin = max(float(np.quantile(d, 0.05)), 0.1)
    return np.geomspace(lmin / 2.0, 2.0 * max(float(d.max()), 0.3), L)


def agg_one(rep):
    dz, pil, ss, X, Y = A.draw("N0_D5b_mixed", 100, 500000 + rep)
    Z = np.vstack([X, Y])
    D2 = A.sqdist(Z, Z)
    out = []
    orig = A.agg_bandwidths
    for rule in ("interSample", "pooled"):
        A.agg_bandwidths = orig if rule == "interSample" else _pooled
        try:
            out.append(int(A.mmdagg(D2, len(X), len(Y), np.random.default_rng([20261126, rep]))[0]))
        finally:
            A.agg_bandwidths = orig
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    assert a.workers <= 6
    from scipy.stats import kstest
    with ProcessPoolExecutor(a.workers) as ex:
        ps = np.array(list(ex.map(perm_one, range(R_PERM), chunksize=20)))
        hp = np.array(list(ex.map(hplb_one, range(R_HPLB), chunksize=20)))
        ag = np.array(list(ex.map(agg_one, range(R_AGG), chunksize=5)))
    res = {"perm": {"R": R_PERM, "rate": float(np.mean(ps <= A.DELTA)), "ks_p": float(kstest(ps, "uniform").pvalue)},
           "hplb": {"R": R_HPLB, "null_rate": float(hp[:, 0].mean()), "boundary_rate": float(hp[:, 1].mean())},
           "mmdagg": {"R": R_AGG, "design": "N0_D5b_mixed law, n = 100, m = 500",
                      "rate_interSample": float(ag[:, 0].mean()), "rate_pooled": float(ag[:, 1].mean()),
                      "se": float(np.sqrt(0.05 * 0.95 / R_AGG))}}
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1) + "\n")
    os.replace(tmp, OUT)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

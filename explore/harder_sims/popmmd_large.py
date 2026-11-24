"""POST HOC diagnostic (DESIGN.md section 9, 2026-11-10): population MMD at the pilot bandwidth from a larger sample.

popmmd.py estimated MMD(P, Q) by the unbiased U-statistic on 4000 + 4000 draws. Against the alpha = 0 extension
(mean of D_hat_0^2 over 200 replicates at n = m = 12800, whose upward bias is at most 1/n + 1/m) those estimates are off
by a few percent in most designs and by 25-30% in D4_small and D8_N4. This averages the U-statistic over 8
independent blocks of 4000 + 4000 fresh draws (m = 5n designs: 1600 + 8000), 32000 + 32000 in all, with blocked kernel
sums (alpha0.kernel_sums), seeds SeedSequence([crc32(design), 888888, block]), disjoint from every run; the standard
error is that of the 8 block values. Output: out/popmmd_large.csv. Descriptive only; nothing in the study or the paper reads
it (the paper's design table still prints popmmd.csv).

    PYTHONPATH=src python3 -I explore/harder_sims/popmmd_large.py --workers 15
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import math
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import alpha0 as A  # noqa: E402
import designs as DS  # noqa: E402

N_EQ, N_UNEQ = 32000, 12800
BLOCKS = 8


def one(args):
    key, b = args
    dz = DS.catalogue()[key]
    pil = json.loads((HERE / "out" / "pilot.json").read_text())[key]
    n = (N_EQ if dz.m_ratio == 1 else N_UNEQ) // BLOCKS
    m = dz.m_ratio * n
    rng = np.random.default_rng(np.random.SeedSequence([zlib.crc32(key.encode()), 888888, b]))
    X, Y = dz.sample(n, rng, "P"), dz.sample(m, rng, "Q")
    if pil["scale"] is not None:
        s = np.array(pil["scale"])
        X, Y = X / s, Y / s
    g = 1.0 / (2 * pil["sigma_fix"] ** 2)
    Z = np.ascontiguousarray(np.vstack([X, Y]))
    # all points in "fold one", no batch 0: Tm holds the sums over all ordered pairs
    Tm = A.kernel_sums(Z, n, m, 0, 0, 0, 0, g, g, g)[0]
    u = (Tm[0, 0] - n) / (n * (n - 1)) + (Tm[1, 1] - m) / (m * (m - 1)) - 2 * Tm[0, 1] / (n * m)
    return key, b, float(u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=15)
    a = ap.parse_args()
    keys = [k for k in DS.catalogue() if not k.startswith("N0") and k != "D5a_imb"]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, [(k, b) for k in keys for b in range(BLOCKS)]))
    old = {ln.split(",")[0]: ln.split(",") for ln in (HERE / "out" / "popmmd.csv").read_text().splitlines()[1:]}
    rows = ["design,n_total,m_total,mmd2_large,mmd2_se,mmd_large,mmd_popmmd_csv"]
    for k in keys:
        u = np.array([r[2] for r in res if r[0] == k])
        dz = DS.catalogue()[k]
        nt = N_EQ if dz.m_ratio == 1 else N_UNEQ
        m2, se = float(u.mean()), float(u.std(ddof=1) / math.sqrt(BLOCKS))
        rows.append(f"{k},{nt},{nt * dz.m_ratio},{m2:.6f},{se:.6f},{math.sqrt(max(m2, 0)):.4f},{old[k][3]}")
        print(rows[-1], flush=True)
    (HERE / "out" / "popmmd_large.csv").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()

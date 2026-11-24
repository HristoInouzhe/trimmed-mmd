"""Population MMD(P, Q) at each design's fixed (pilot) bandwidth, by the
unbiased U-statistic on 4000 + 4000 fresh draws (seed disjoint from all runs),
next to tau_new(0) and tau_old(0): the analytic test of H_{0,0} can only have
power once D_hat_0 ~ MMD exceeds the threshold. Output: out/popmmd.csv."""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import json
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
import designs as DS  # noqa: E402
from trimmedmmd.kernels import sqdist  # noqa: E402

pil = json.loads((HERE / "out" / "pilot.json").read_text())
thr = json.loads((HERE / "out" / "thr_cache.json").read_text())
rows = ["design,sigma_fix,mmd2_pop,mmd_pop,n_max_tier1,tau_new0_nmax,tau_old0_nmax,tau_new0_n1600"]
N = 4000
for k, dz in DS.catalogue().items():
    if k.startswith("N0"):
        continue
    rng = np.random.default_rng(np.random.SeedSequence([zlib.crc32(k.encode()), 777777]))
    X, Y = dz.sample(N, rng, "P"), dz.sample(N, rng, "Q")
    if pil[k]["scale"] is not None:
        s = np.array(pil[k]["scale"])
        X, Y = X / s, Y / s
    sig = pil[k]["sigma_fix"]
    kxx = np.exp(-sqdist(X, X) / (2 * sig ** 2))
    kyy = np.exp(-sqdist(Y, Y) / (2 * sig ** 2))
    kxy = np.exp(-sqdist(X, Y) / (2 * sig ** 2))
    np.fill_diagonal(kxx, 0)
    np.fill_diagonal(kyy, 0)
    # per-row h-values for a rough s.e. of the U-statistic
    h = kxx.sum(1) / (N - 1) + kyy.sum(1) / (N - 1) - kxy.sum(1) / N - kxy.sum(0) / N
    m2 = h.mean()
    se = 2 * h.std() / np.sqrt(N)
    mmd = float(np.sqrt(max(m2, 0)))
    m = dz.m_ratio
    nmax = max(dz.ns)
    t8 = thr[f"{nmax},{nmax * m},0"]
    t16 = thr[f"1600,{1600 * m},0"]
    rows.append(f"{k},{sig:.4f},{m2:.5f},{mmd:.4f},{nmax},{t8['new']:.4f},{t8['old']:.4f},{t16['new']:.4f}")
    print(rows[-1], flush=True)
(HERE / "out" / "popmmd.csv").write_text("\n".join(rows) + "\n")

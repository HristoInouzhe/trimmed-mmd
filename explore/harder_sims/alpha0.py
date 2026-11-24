"""POST HOC (DESIGN.md section 9, 2026-11-10): the alpha = 0 tests only, for sample sizes beyond the study's grid.

The author asked for the sample size at which each method reaches power 0.8 against H_{0,0} (P = Q) where the
pre-registered grid (n = m <= 800, 1600 in two designs) ends below it. At alpha = 0 no method needs the QP:

* analytic: D_hat_0 = MMD between the empirical kernel mean embeddings (V-statistic, the alpha = 0 value of the
  study's certified QP, whose feasible set is the single point w = 1/n, v = 1/m), at the pilot bandwidth sigma_fix,
  compared with tau_{n,m}(0, delta) = trimmedmmd.threshold.tau_new with the Gaussian constants the study used
  (kappa = 1, D_phi = rho = sqrt 2, v^2 = 1); also r_n + r_m (an_old), the closed form (an_newcf) and the study's
  own minimum min(tau_A, tau_P', tau_P'^cpl) (an_new_study; tau_new also takes the closed form into the minimum,
  which lowers it by the 1e-6 root margin at n <= 3200 and changes nothing at 6400 and 12800);
* split kernel certificates: the alpha = 0 witness is the difference of the empirical mean embeddings of the fit
  block, w = 1/n1, v = 1/m1 (fold one, median bandwidth of fold one) and 1/q_n, 1/q_m (batch 0 of the sequential
  certificate, median bandwidth of batch 0); cut points, ramp, priors and bets exactly as methods.run_replicate;
* classifiers: trained and evaluated exactly as methods.run_replicate (same folds, same seed).

Data, permutation, bandwidth subsamples, bootstrap stream and classifier seed are drawn from
SeedSequence([crc32(design), n, rep]) in the same order as methods.run_replicate, so on a stored replicate this
module reproduces the stored alpha = 0 decisions (checked by `python3 alpha0.py check`). The extension uses
replicate indices rep >= 200000, which no earlier run used.

Kernel sums are computed in blocks of at most B x B entries (B = 512), so memory stays bounded at n = m = 12800 (or
m = 5n = 64000): the pilot-bandwidth sums over all ordered pairs, the fold-one witness on every point (columns: fold
one) and the batch-0 witness on fold one (columns: batch 0). Nothing of size (n + m)^2 is formed.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import math
import sys
import time
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (HERE, ROOT / "src", ROOT / "explore" / "sharper_threshold", ROOT / "explore" / "split_adaptive"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402

import methods as M  # noqa: E402
import sacert as S  # noqa: E402
import thresholds as T  # noqa: E402
from fold1 import cut_points, priors  # noqa: E402
from trimmedmmd import threshold as TH  # noqa: E402
from trimmedmmd.kernels import median_heuristic  # noqa: E402

DELTA = M.DELTA
BLOCK = 512      # 512 x 512 blocks stay in cache; 2048 was 1.45x slower (timing pilot)
AN = ("old", "newcf", "new", "new_study")
METHODS0 = (tuple(f"an_{t}" for t in AN) + M.SPLIT_K
            + tuple(f"{c}_{r}" for c in M.CLFS for r in ("hoef", "bet")))


@lru_cache(maxsize=None)
def taus0(n, m):
    """Thresholds at alpha = 0 (the study's constants)."""
    G = T.GAUSS
    return {"old": float(T.tau_old(n, m, 0.0, DELTA)),
            "newcf": float(T.tau_P_closed(n, m, 0.0, DELTA, **G)),
            "new": float(TH.tau_new(n, m, 0.0, DELTA, TH.GAUSSIAN)),
            "new_study": float(min(T.tau_A(n, m, 0.0, DELTA, **G), T.tau_P(n, m, 0.0, DELTA, **G),
                                   T.tau_P_coupled(n, m, 0.0, DELTA, kappa=1.0)))}


def _chunks(start, stop, size):
    for a in range(start, stop, size):
        yield a, min(a + size, stop)


def kernel_sums(Z, n1, m1, n2, m2, q_n, q_m, g_fix, g1, g0, block=BLOCK):
    """Blocked Gaussian kernel sums. Rows of Z are ordered [X1, Y1, X2, Y2]; g = 1/(2 sigma^2).

    Returns
      Tm  2x2 sums over ordered pairs at sigma_fix, class 0 = X, 1 = Y (Tm[0,1] = sum_{x,y} k(x, y));
      A1, B1  for every row, the sums over the fold-one X resp. Y points at sigma1;
      A0, B0  for every fold-one row, the sums over the batch-0 X resp. Y points at sigma0.
    """
    N1 = n1 + m1
    N = N1 + n2 + m2
    sq = np.einsum("ij,ij->i", Z, Z)
    Zm = -2.0 * Z                       # exact (power of two): -2 x.y in one product
    rows = [(0, n1, 0, 1), (n1, N1, 1, 1), (N1, N1 + n2, 0, 2), (N1 + n2, N, 1, 2)]   # (start, stop, class, fold)
    cols = [(0, q_n, 0, True), (q_n, n1, 0, False), (n1, n1 + q_m, 1, True), (n1 + q_m, N1, 1, False)]
    Tm = np.zeros((2, 2))
    A1, B1 = np.zeros(N), np.zeros(N)
    A0, B0 = np.zeros(N1), np.zeros(N1)
    G = np.empty((block, block))
    E = np.empty((block, block))

    def d2(r0, r1, c0, c1):
        g = G[: r1 - r0, : c1 - c0]
        np.matmul(Zm[r0:r1], Z[c0:c1].T, out=g)
        g += sq[r0:r1, None]
        g += sq[None, c0:c1]
        np.maximum(g, 0.0, out=g)
        return g

    def kexp(g, gam):
        e = E[: g.shape[0], : g.shape[1]]
        np.multiply(g, -gam, out=e)
        np.exp(e, out=e)
        return e

    # columns in fold one, rows everywhere
    for cs, ce, ccls, cb0 in cols:
        for c0, c1 in _chunks(cs, ce, block):
            for rs, re_, rcls, rfold in rows:
                for r0, r1 in _chunks(rs, re_, block):
                    g = d2(r0, r1, c0, c1)
                    s = float(kexp(g, g_fix).sum())
                    Tm[rcls, ccls] += s
                    if rfold == 2:          # the mirrored (fold one, fold two) block, not computed
                        Tm[ccls, rcls] += s
                    (A1 if ccls == 0 else B1)[r0:r1] += kexp(g, g1).sum(1)
                    if cb0 and rfold == 1:
                        (A0 if ccls == 0 else B0)[r0:r1] += kexp(g, g0).sum(1)
    # fold two x fold two at sigma_fix only, upper triangle of class-homogeneous chunks
    ch = [(a, b, 0) for a, b in _chunks(N1, N1 + n2, block)] + [(a, b, 1) for a, b in _chunks(N1 + n2, N, block)]
    for i, (r0, r1, rc) in enumerate(ch):
        for c0, c1, cc in ch[i:]:
            s = float(kexp(d2(r0, r1, c0, c1), g_fix).sum())
            Tm[rc, cc] += s
            if (c0, c1) != (r0, r1):
                Tm[cc, rc] += s
    return Tm, A1, B1, A0, B0


def run_replicate(design, n, rep, seed_key, sigma_fix, scale=None, n_boot=1000, block=BLOCK):
    """All tests of H_{0,0} on one replicate (decisions 0/1 per method, plus the statistics)."""
    m = design.m_ratio * n
    tt = {}
    t_all = time.perf_counter()
    # ---- data, folds, bandwidths, streams: same draws in the same order as methods.run_replicate
    ss = np.random.SeedSequence([seed_key, n, rep])
    rng = np.random.default_rng(ss)
    X = design.sample(n, rng, "P")
    Y = design.sample(m, rng, "Q")
    if scale is not None:
        X, Y = X / scale, Y / scale
    px, py = rng.permutation(n), rng.permutation(m)
    X, Y = X[px], Y[py]
    n1, m1 = n // 2, m // 2
    n2, m2 = n - n1, m - m1
    sig1 = median_heuristic(X[:n1], Y[:m1], rng=np.random.default_rng(ss.spawn(1)[0]))
    q_n, q_m = int(round(n / 4)), int(round(m / 4))
    nb1x, nb1y = n1 - q_n, m1 - q_m
    sig0 = median_heuristic(X[:q_n], Y[:q_m], rng=np.random.default_rng(ss.spawn(2)[1]))
    brng = np.random.default_rng(ss.spawn(3)[2])
    cseed = int(ss.generate_state(1)[0] % (2 ** 31 - 1))
    tt["setup"] = time.perf_counter() - t_all

    # ---- kernel sums (blocked)
    t0 = time.perf_counter()
    Z = np.ascontiguousarray(np.vstack([X[:n1], Y[:m1], X[n1:], Y[m1:]]))
    Tm, A1, B1, A0, B0 = kernel_sums(Z, n1, m1, n2, m2, q_n, q_m, 1.0 / (2 * sigma_fix ** 2),
                                     1.0 / (2 * sig1 ** 2), 1.0 / (2 * sig0 ** 2), block)
    del Z
    tt["kernel"] = time.perf_counter() - t0

    rec = {"design": design.key, "n": n, "m": m, "rep": rep, "mode": "alpha0", "sigma_fix": sigma_fix,
           "sigma1": sig1, "sigma0": sig0, "dec0": {}, "stat": {}, "time": tt}
    dec = rec["dec0"]
    st = rec["stat"]

    # ---- analytic: D_hat_0^2 = z'Kz with w = 1/n, v = 1/m
    mmd2 = Tm[0, 0] / n ** 2 + Tm[1, 1] / m ** 2 - 2.0 * Tm[0, 1] / (n * m)
    val = max(mmd2, 0.0)
    tau = taus0(n, m)
    for k in AN:
        dec[f"an_{k}"] = int(val > tau[k] ** 2)
    st.update(mmd2=float(mmd2), mmd=math.sqrt(val), tau=tau)

    # ---- split kernel: fold-one witness (mean-embedding difference at sigma1)
    t0 = time.perf_counter()
    N1 = n1 + m1
    Kz = A1 / n1 - B1 / m1
    val1 = max(float(Kz[:n1].sum() / n1 - Kz[n1:N1].sum() / m1), 0.0)
    norm = math.sqrt(val1)
    res = {k: False for k in M.SPLIT_K}
    Z2, pv2 = None, None
    st["norm1"] = norm
    if norm > 1e-12:
        u1 = Kz[:n1] / norm - (1.0 / n1) / norm
        v1 = Kz[n1:N1] / norm + (1.0 / m1) / norm
        u2, v2 = Kz[N1:N1 + n2] / norm, Kz[N1 + n2:] / norm
        T_hat = S.trimmed_sep(u2, v2, 0.0)
        e = lambda k: math.sqrt(math.log(4.0 / DELTA) / (2.0 * k))
        res["prop17"] = bool(T_hat > 2 * 1.0 * 1.0 * (e(n2) + e(m2)))
        tb = time.perf_counter()
        res["boot"] = M.boot_reject(u2, v2, 0.0, brng, n_boot=n_boot)
        tt["boot"] = time.perf_counter() - tb
        s, sp = cut_points(u1, v1, 0.0)
        st.update(T_hat=float(T_hat), s=s, sp=sp)
        if s > sp:
            Gr, Hr = S.ramp(u2, s, sp), S.ramp(v2, s, sp)
            res["ramp_b2s"] = bool(S.lcb_bern2s(Gr, 1.0, -Hr, 1.0, DELTA) > 0.0)
            pv2 = priors(u1, v1, s, sp, 0.0, n2, m2)["ramp"]
            Z2 = S.pair_up(Gr, -Hr) - 0.0
            res["ramp_bet"] = S.betting_test(Z2, -(1.0 + 0.0), DELTA, c=0.5, prior_var=pv2)[0]
            st["Delta_hat"] = float(Gr.mean() - Hr.mean())
    # ---- sequential: batch-1 witness fitted on batch 0 (sigma0), batch 2 = the fold-one ramp
    Kz0 = A0 / q_n - B0 / q_m
    val0 = max(float(Kz0[:q_n].sum() / q_n - Kz0[n1:n1 + q_m].sum() / q_m), 0.0)
    norm0 = math.sqrt(val0)
    st["norm0"] = norm0
    Z1, pv1 = None, None
    if norm0 > 1e-12:
        u0 = Kz0[:q_n] / norm0 - (1.0 / q_n) / norm0
        v0 = Kz0[n1:n1 + q_m] / norm0 + (1.0 / q_m) / norm0
        s0, sp0 = cut_points(u0, v0, 0.0)
        if s0 > sp0:
            G1, H1 = S.ramp(Kz0[q_n:n1] / norm0, s0, sp0), S.ramp(Kz0[n1 + q_m:N1] / norm0, s0, sp0)
            Z1 = S.pair_up(G1, -H1) - 0.0
            pv1 = priors(u0, v0, s0, sp0, 0.0, nb1x, nb1y)["ramp"]
    N_seq = M.n_pairs(nb1x, nb1y) + M.n_pairs(n2, m2)
    res["seq_bet"] = M.bet_stream([Z1, Z2], [pv1, pv2], 0.0, N_seq)[0]
    for k in M.SPLIT_K:
        dec[k] = int(res[k])
    tt["split"] = time.perf_counter() - t0
    del A1, B1, A0, B0, Kz, Kz0

    # ---- classifiers (as methods.run_replicate)
    X1, X2, Y1, Y2 = X[:n1], X[n1:], Y[:m1], Y[m1:]
    for c in M.CLFS:
        t0 = time.perf_counter()
        Gc, Hc = M.classifier_scores(c, X1, Y1, X2, Y2, cseed)
        n2c, m2c = Gc.size, Hc.size
        slack = (math.sqrt(math.log(2 / DELTA) / (2 * n2c)) + math.sqrt(math.log(2 / DELTA) / (2 * m2c)))
        diff = float(Gc.mean() - Hc.mean())
        dec[f"{c}_hoef"] = int(diff - slack > 0.0)
        Zb = S.pair_up(Gc, -Hc)
        dec[f"{c}_bet"] = int(S.betting_test(Zb - 0.0, -(1.0 + 0.0), DELTA, c=0.5, prior_var=0.5)[0])
        st[f"{c}_diff"] = diff
        tt[c] = time.perf_counter() - t0
    tt["total"] = time.perf_counter() - t_all
    return rec


# ---------------------------------------------------------------------------
# reproduction check on stored replicates (decisions at alpha = 0 must match)
# ---------------------------------------------------------------------------

def _check_one(args):
    import json
    import designs as DS
    import run as R
    key, n, rep = args
    dz = DS.catalogue()[key]
    pil = json.loads(R.PILOT.read_text())[key]
    scale = None if pil["scale"] is None else np.array(pil["scale"])
    rec = run_replicate(dz, n, rep, R.seed_key(key), pil["sigma_fix"], scale=scale)
    return key, n, rep, rec


def check(cells, reps, workers=16, block=None):
    """Recompute stored replicates in alpha = 0 mode and compare every decision with dec[...][0]."""
    import json
    from concurrent.futures import ProcessPoolExecutor
    import run as R
    stored = {}
    for key, n in cells:
        for line in (R.OUT / f"{key}.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["n"] == n and r["rep"] < reps:
                stored[(key, n, r["rep"])] = r
    jobs = sorted(stored)
    mism, tot, mmd_err = [], 0, 0.0
    with ProcessPoolExecutor(workers) as ex:
        for key, n, rep, rec in ex.map(_check_one, jobs):
            old = stored[(key, n, rep)]
            for mth in METHODS0:
                ref = "an_new" if mth == "an_new_study" else mth
                tot += 1
                if rec["dec0"][mth] != old["dec"][ref][0]:
                    mism.append((key, n, rep, mth, rec["dec0"][mth], old["dec"][ref][0]))
            j0 = old["diag"].get("an_val", {}).get("0")
            if j0 is not None:
                mmd_err = max(mmd_err, abs(rec["stat"]["mmd"] - j0[0]))
            sx = old["diag"]["split_extra"].get("0")
            if sx is not None and "norm1" in rec["stat"]:
                mmd_err = max(mmd_err, abs(rec["stat"]["norm1"] - sx["norm"]))
    return {"replicates": len(jobs), "decisions": tot, "mismatches": mism, "max_abs_stat_diff": mmd_err}


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "check":
    import json
    cells = [tuple(c.split(":")) for c in sys.argv[2].split(",")]
    cells = [(k, int(n)) for k, n in cells]
    out = check(cells, int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 16)
    print(json.dumps(out, indent=1))

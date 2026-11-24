"""Unit checks of methods.py against the reference implementations (run once
before the main runs; output in out/checks.txt)."""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import methods as M  # noqa: E402  (sets the other paths)
import designs as DS  # noqa: E402

import numpy as np  # noqa: E402

from trimmedmmd.kernels import gaussian_kernel, sqdist  # noqa: E402
from trimmedmmd.solver import solve_gram  # noqa: E402
import sacert as S  # noqa: E402
import seqcert  # noqa: E402
from fold1 import cut_points, priors  # noqa: E402


def check_qp():
    rng = np.random.default_rng(1)
    dz = DS.catalogue()["D1_d2_a20"]
    X, Y = dz.sample(150, rng, "P"), dz.sample(150, rng, "Q")
    sig = 1.8
    Z = np.vstack([X, Y])
    K = np.exp(-sqdist(Z, Z) / (2 * sig ** 2))
    qp = M.GramQP(K, 150, 150)
    out = []
    for a in (0.0, 0.05, 0.1, 0.2, 0.3):
        ref = solve_gram(gaussian_kernel(X, X, sig), gaussian_kernel(Y, Y, sig),
                         gaussian_kernel(X, Y, sig), a, max_iter=20000, tol=1e-14)
        w, v, val, _, _, it = qp.solve(a, max_iter=20000, tol=1e-14)
        # decision mode at thresholds bracketing the reference value
        d = ref.discrepancy
        _, _, _, _, dec, it2 = qp.solve(a, thr2=np.array([(0.9 * d) ** 2, (1.1 * d) ** 2]),
                                        max_iter=20000)
        out.append(f"alpha={a}: ref D={d:.6f}  mine D={math.sqrt(val):.6f}  "
                   f"decisions at 0.9D/1.1D = {dec.tolist()} (want [1, -1]) after {it2} its")
    return out


def check_boot():
    rng = np.random.default_rng(2)
    u, v = rng.normal(size=77), rng.normal(size=91)
    a = 0.15
    wu = M.ltm_weights(77, a)
    vals = [abs(np.sort(u) @ wu - S.lower_trimmed_mean(u, a))]
    r1 = np.random.default_rng(5)
    ia = r1.integers(0, 77, size=(200, 77))
    ib = r1.integers(0, 91, size=(200, 91))
    mine = np.sort(u[ia], axis=1) @ wu + np.sort(-v[ib], axis=1) @ M.ltm_weights(91, a)
    ref = np.array([S.trimmed_sep(u[i], v[j], a) for i, j in zip(ia, ib)])
    vals.append(float(np.max(np.abs(mine - ref))))
    return [f"ltm weight check {vals[0]:.2e}; bootstrap rows max |diff| {vals[1]:.2e}"]


def check_seq():
    """bet_stream == seqcert.seq_ramp_bet when the horizon agrees (no skipped batch)."""
    out = []
    dz = DS.catalogue()["D1_d2_a20"]
    agree = 0
    tot = 0
    for seed in range(20):
        rng = np.random.default_rng(100 + seed)
        n = 200
        X, Y = dz.sample(n, rng, "P"), dz.sample(n, rng, "Q")
        sig = 1.8
        a = 0.05
        edges = (0, 0.25, 0.5, 1.0)
        from fold1 import Witness
        rej_ref, logK_ref = seqcert.seq_ramp_bet(X, Y, a, edges, sig, np.random.default_rng(0))
        # mine, same witnesses (fold1.Witness through the fit callback is what seqcert uses)
        bx = [int(round(e * n)) for e in edges]
        Zs, pvs = [], []
        for b in (1, 2):
            w = Witness(X[:bx[b]], Y[:bx[b]], a, sig, np.random.default_rng(0))
            u1, v1 = w.fold1_values()
            s, sp = cut_points(u1, v1, a)
            Xb, Yb = X[bx[b]:bx[b + 1]], Y[bx[b]:bx[b + 1]]
            if s > sp:
                G, H = S.ramp(w(Xb), s, sp), S.ramp(w(Yb), s, sp)
                Zs.append(S.pair_up(G, -H) - a)
                Gf, Hf = S.ramp(u1, s, sp), S.ramp(v1, s, sp)
                from fold1 import pair_var
                pvs.append(max(pair_var(Gf.var(), Hf.var(), len(Xb), len(Yb)), 1e-4))
            else:
                Zs.append(None)
                pvs.append(None)
        N = (bx[2] - bx[1]) + (bx[3] - bx[2])
        rej, lmax = M.bet_stream(Zs, pvs, a, N)
        tot += 1
        agree += int(rej == rej_ref)
    out.append(f"seq betting decisions agree with seqcert.seq_ramp_bet: {agree}/{tot}")
    return out


if __name__ == "__main__":
    lines = check_qp() + check_boot() + check_seq()
    txt = "\n".join(lines)
    print(txt)
    (HERE / "out" / "checks.txt").write_text(txt + "\n")

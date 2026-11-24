"""Adversarial validity checks of the SECOND-FOLD step, at the exact null boundary.

Validity of every finite-sample certificate depends only on the laws of the
fold-2 scalars given fold 1. So the hardest nulls are scalar laws at which the
tested mean sits exactly on the boundary:

* ramp family: G = g(X) in [0,1], H = g(Y) in [0,1] with E G - E H = alpha exactly;
* psi family: U, V in [-1,1] and fixed cut points (s, sp) with
  E psi_s(U) + E psi_{-sp}(-V) = 0 exactly (atoms at the cut points, skew).

Each cell: R replicates, fixed seeds. Output: out/null_scalar.csv.

    python3 -I explore/split_adaptive/null_scalar.py --reps 20000 --workers 8
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import sacert as S  # noqa: E402

DELTA = 0.05


# ---------------------------------------------------------------- ramp laws
def law_ramp(name, alpha):
    """Return samplers (G, H) on [0,1] with E G - E H = alpha exactly."""
    if name == "bern_mid":          # Bernoulli(0.5+a/2) vs Bernoulli(0.5-a/2)
        p, q = 0.5 + alpha / 2, 0.5 - alpha / 2
        return (lambda r, k: (r.random(k) < p).astype(float),
                lambda r, k: (r.random(k) < q).astype(float))
    if name == "bern_edge":         # G = 1 a.s., H ~ Bernoulli(1 - alpha): Var G = 0
        return (lambda r, k: np.ones(k),
                lambda r, k: (r.random(k) < 1 - alpha).astype(float))
    if name == "bern_rare":         # rare successes: Bernoulli(alpha+0.01) vs Bernoulli(0.01)
        p, q = alpha + 0.01, 0.01
        return (lambda r, k: (r.random(k) < p).astype(float),
                lambda r, k: (r.random(k) < q).astype(float))
    if name == "skew_beta":         # skewed continuous G, H = shifted mixture
        # G ~ Beta(0.2, 2) (mean 1/11), H = 0 w.p. pi else Beta(0.2,2), so E H = (1-pi)/11
        mG = 0.2 / 2.2
        target = mG - alpha
        if target < 0:
            # put G on the high side instead: G = 1 - Beta(0.2,2), H = mixture with zeros
            mG = 1 - 0.2 / 2.2
            pi = alpha / mG
            return (lambda r, k: 1 - r.beta(0.2, 2.0, k),
                    lambda r, k: np.where(r.random(k) < pi, 0.0, 1 - r.beta(0.2, 2.0, k)))
        pi = alpha / mG
        return (lambda r, k: r.beta(0.2, 2.0, k),
                lambda r, k: np.where(r.random(k) < pi, 0.0, r.beta(0.2, 2.0, k)))
    if name == "three_point":       # atoms at 0, 1/2, 1; H has the extreme low atom
        # G: 1 w.p. 0.5, 0.5 w.p. 0.5 (mean 0.75); H: 0 w.p. w, 0.75... choose H = 0 w.p. w, 1 w.p. 1-w
        mG = 0.75
        mH = mG - alpha
        return (lambda r, k: np.where(r.random(k) < 0.5, 1.0, 0.5),
                lambda r, k: (r.random(k) < mH).astype(float))
    if name == "rare_bad":          # Z = G - H mostly positive, rare -1 shocks (stresses lambda cap)
        # G = 1 w.p. 1; H = 1 w.p. 1 - alpha... that is bern_edge; instead:
        # G = 1 - e, e ~ 1 w.p. 0.02 else 0; H = 0 w.p. 1-h else 1 with E G - E H = alpha
        pe = 0.02
        mH = (1 - pe) - alpha
        return (lambda r, k: (r.random(k) >= pe).astype(float),
                lambda r, k: (r.random(k) < mH).astype(float))
    raise ValueError(name)


RAMP_LAWS = ["bern_mid", "bern_edge", "bern_rare", "skew_beta", "three_point", "rare_bad"]


def ramp_tests(G, H, alpha, prior_var=None):
    out = {}
    out["ramp_hoef"] = S.hoeffding2s_lcb(G, 1.0, -H, 1.0, DELTA) > alpha
    out["ramp_mpu"] = S.lcb_mp_union(G, 1.0, -H, 1.0, DELTA) > alpha
    out["ramp_b2s"] = S.lcb_bern2s(G, 1.0, -H, 1.0, DELTA) > alpha
    if min(G.size, H.size) >= 2 and S.equal_blocks(G, H).size >= 2:
        out["ramp_mp_pair"] = S.mp_pair_lcb(G, H, DELTA) > alpha
    else:
        out["ramp_mp_pair"] = False
    Z = S.pair_up(G, -H) - alpha
    out["ramp_bet"] = S.betting_test(Z, -(1 + alpha), DELTA, c=0.5, prior_var=prior_var)[0]
    out["ramp_bet75"] = S.betting_test(Z, -(1 + alpha), DELTA, c=0.75, prior_var=prior_var)[0]
    out["ramp_bet100"] = S.betting_test(Z, -(1 + alpha), DELTA, c=1.0, prior_var=prior_var)[0]
    sd = math.sqrt(G.var(ddof=1) / G.size + H.var(ddof=1) / H.size) if min(G.size, H.size) > 1 else 0
    out["ramp_clt"] = (G.mean() - H.mean() - 1.6448536269514722 * sd) > alpha
    return out


# ---------------------------------------------------------------- psi laws
def law_psi(name, alpha):
    """U, V samplers on [-1,1] and cut points with E A + E B = 0 exactly."""
    beta = 1 / (1 - alpha)
    if name == "atoms_at_cut":
        # U: atom at s = 0.3 with mass p0, else uniform on [-1, 0.3]; V symmetric (V = -U law)
        # then E A + E B = 2 E psi_s(U); shift s so that E psi_s(U) = 0 -> solve numerically
        p0 = 0.4

        def EA(s):
            # E[s - beta (s-U)_+], U = s w.p. p0, else Unif[-1, s]
            return s - beta * (1 - p0) * (s + 1) / 2
        lo, hi = -1.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if EA(mid) < 0:
                lo = mid
            else:
                hi = mid
        s = (lo + hi) / 2
        su = lambda r, k: np.where(r.random(k) < p0, s, r.uniform(-1, s, k))
        sv = lambda r, k: -su(r, k)
        return su, sv, s, -s
    if name == "skewed":
        # U = -1 + 2 Beta(5, 0.3) (mass near 1, long left tail); V = -U; cut s with E psi_s(U)=0
        rr = np.random.default_rng(12345)
        ref = -1 + 2 * rr.beta(5, 0.3, 4_000_000)
        lo, hi = -1.0, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2
            val = mid - beta * np.maximum(mid - ref, 0).mean()
            if val < 0:
                lo = mid
            else:
                hi = mid
        s = (lo + hi) / 2
        su = lambda r, k: -1 + 2 * r.beta(5, 0.3, k)
        sv = lambda r, k: 1 - 2 * r.beta(5, 0.3, k)
        return su, sv, s, -s
    raise ValueError(name)


PSI_LAWS = ["atoms_at_cut", "skewed"]


def psi_tests(U, V, s, sp, alpha):
    beta = 1 / (1 - alpha)
    A, B = S.psi_vars(U, V, s, sp, alpha)
    bA, bB = beta * (s + 1), beta * (1 - sp)
    out = {"psi_mpu": S.lcb_mp_union(A, bA, B, bB, DELTA) > 0,
           "psi_b2s": S.lcb_bern2s(A, bA, B, bB, DELTA) > 0}
    zlo = (s - beta * (s + 1)) + (-sp - beta * (1 - sp))
    out["psi_bet"] = S.betting_test(S.pair_up(A, B), zlo, DELTA, c=0.5)[0]
    out["psi_bet75"] = S.betting_test(S.pair_up(A, B), zlo, DELTA, c=0.75)[0]
    return out


def cell(args):
    family, law, alpha, n2, m2, reps, seed = args
    rng = np.random.default_rng(seed)
    acc = None
    if family == "ramp":
        sg, sh = law_ramp(law, alpha)
        for _ in range(reps):
            G, H = sg(rng, n2), sh(rng, m2)
            r = ramp_tests(G, H, alpha)
            acc = {k: 0 for k in r} if acc is None else acc
            for k, v in r.items():
                acc[k] += bool(v)
    else:
        su, sv, s, sp = law_psi(law, alpha)
        for _ in range(reps):
            U, V = su(rng, n2), sv(rng, m2)
            r = psi_tests(U, V, s, sp, alpha)
            acc = {k: 0 for k in r} if acc is None else acc
            for k, v in r.items():
                acc[k] += bool(v)
    return [dict(family=family, law=law, alpha=alpha, n2=n2, m2=m2, reps=reps, method=k,
                 rate=v / reps) for k, v in acc.items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    sizes = [(5, 5), (10, 10), (20, 20), (50, 50), (100, 100), (400, 400), (30, 70), (70, 30)]
    jobs = []
    k = 0
    for alpha in (0.0, 0.1, 0.3):
        for law in RAMP_LAWS:
            for n2, m2 in sizes:
                jobs.append(("ramp", law, alpha, n2, m2, a.reps, 500_000 + k)); k += 1
    for alpha in (0.1, 0.3):
        for law in PSI_LAWS:
            for n2, m2 in sizes:
                jobs.append(("psi", law, alpha, n2, m2, a.reps // 2, 600_000 + k)); k += 1
    with ProcessPoolExecutor(a.workers) as ex:
        rows = [r for res in ex.map(cell, jobs) for r in res]
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "out" / "null_scalar.csv", index=False)
    piv = df.pivot_table(index=["family", "method"], values="rate", aggfunc="max")
    print(piv.to_string())
    print(df.sort_values("rate", ascending=False).head(25).to_string(index=False))


if __name__ == "__main__":
    main()

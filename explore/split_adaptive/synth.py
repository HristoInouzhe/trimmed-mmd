"""Power and boundary type-I on the paper's synthetic designs, all certificates
on the SAME splits and the SAME fold-1 witnesses (matched comparison).

Data: replicate i uses numpy default_rng(SEED0 + i) and X, Y = sc.sample(n, rng, 'P'),
sc.sample(n, rng, 'Q') -- the draws of e6/e6b for disjoint_d2_a20 (SEED0 = 70000).
The same rng then permutes the rows into two halves (as splitting.split_test does),
and gives the fold-1 median bandwidth.

Persists one JSON line per (n, replicate) in out/synth_<scenario>.jsonl; resumable.

    OMP_NUM_THREADS=1 python3 -I explore/split_adaptive/synth.py --ns 100 200 400 800 1600 --reps 200 --workers 12
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import math
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

import sacert as S  # noqa: E402
from fold1 import Witness, cut_points, priors  # noqa: E402

DELTA = 0.05
MAIN_ALPHAS = [0.0, 0.1, 0.15, 0.2]
GRID = [0.05, 0.08, 0.11, 0.14, 0.17, 0.20, 0.23, 0.26]   # as in run.e7_split_bounds
N_REF = 20_000
DIAG_REPS = 40


def fold_tests(wit, X2, Y2, alpha, rng, prior_from_fold1=True, n_boot=1000):
    u1, v1 = wit.fold1_values()
    s, sp = cut_points(u1, v1, alpha)
    u2, v2 = wit(X2), wit(Y2)
    pr = priors(u1, v1, s, sp, alpha, len(X2), len(Y2)) if prior_from_fold1 else None
    res = S.evaluate_all(u2, v2, s, sp, alpha, DELTA, n_boot=n_boot, rng=rng, prior=pr)
    res["s"], res["sp"] = s, sp
    return res, (u2, v2, s, sp, pr)


def stopped_wealth(Z, lo, delta, c, pv):
    """Wealth stopped at the first crossing of 1/delta (an e-value by optional stopping)."""
    Y = np.asarray(Z, float)
    N = Y.size
    L = math.log(1.0 / delta)
    lam_max = c / abs(lo)
    prior_w = 20.0
    pv = (abs(lo) / 2) ** 2 if pv is None else pv
    cs = np.concatenate([[0.0], np.cumsum(Y)[:-1]])
    cnt = np.arange(N, dtype=float)
    mu_prev = cs / (prior_w + cnt)
    dev2 = (Y - mu_prev) ** 2
    cdev = np.concatenate([[0.0], np.cumsum(dev2)[:-1]])
    sig2 = np.maximum((prior_w * pv + cdev) / (prior_w + cnt), 1e-8)
    lam = np.minimum(lam_max, np.sqrt(2.0 * L / (N * sig2)))
    logK = np.cumsum(np.log1p(lam * Y))
    # final wealth K_N: an e-value (E K_N <= 1 under H0). (Run A used the wealth stopped at
    # the first crossing, also an e-value but capped near 1/delta, which wastes evidence
    # when two e-values are averaged.)
    return float(logK[-1])


def gbm_tests(X1, Y1, X2, Y2, seed, alphas):
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(random_state=seed)
    Z1 = np.vstack([X1, Y1])
    lab = np.r_[np.ones(len(X1), int), np.zeros(len(Y1), int)]
    clf.fit(Z1, lab)
    G = (clf.predict(X2) == 1).astype(float)
    H = (clf.predict(Y2) == 1).astype(float)
    n2, m2 = len(X2), len(Y2)
    slack = math.sqrt(math.log(2 / DELTA) / (2 * n2)) + math.sqrt(math.log(2 / DELTA) / (2 * m2))
    # fold-1 prior for the bet: in-sample rates are optimistic; use a neutral prior
    out = {"gbm_diff": float(G.mean() - H.mean())}
    for a in alphas:
        out[f"gbm_hoef@{a}"] = bool(G.mean() - H.mean() - slack > a)
        Z = S.pair_up(G, -H) - a
        # neutral fold-1-free prior (max variance of a difference of two indicators);
        # NOT the fold-2 variance, which would make the bets non-predictable
        pv = 0.5
        out[f"gbm_bet@{a}"] = S.betting_test(Z, -(1 + a), DELTA, c=0.5, prior_var=pv)[0]
        out[f"gbm_b2s@{a}"] = bool(S.lcb_bern2s(G, 1.0, -H, 1.0, DELTA) > a)
    return out


def population_diag(wit, sc, alpha, s, sp, seed):
    """Population quantities of the fold-1 witness from a large fresh sample."""
    rr = np.random.default_rng(10_000_000 + seed)
    Xr, Yr = sc.sample(N_REF, rr, "P"), sc.sample(N_REF, rr, "Q")
    u, v = wit(Xr), wit(Yr)
    beta = 1 / (1 - alpha)
    T = S.trimmed_sep(u, v, alpha)                       # T(f), population (approx.)
    q = float(np.quantile(u, 1 - alpha)); qp = float(np.quantile(v, alpha))
    Aq, Bq = S.psi_vars(u, v, q, qp, alpha)
    sig_boot = beta * 0 + math.sqrt(Aq.var() + Bq.var())   # per-pair sd at the true quantiles
    A, B = S.psi_vars(u, v, s, sp, alpha)
    Ts = float(A.mean() + B.mean())                      # T at fold-1 cut points (<= T)
    d = {"T_pop": T, "Ts_pop": Ts, "sd_psi_q": sig_boot,
         "sd_psi_s": math.sqrt(A.var() + B.var())}
    if s > sp:
        G, H = S.ramp(u, s, sp), S.ramp(v, s, sp)
        Dl = float(G.mean() - H.mean())
        d.update({"Delta_pop": Dl,
                  "Tclip_pop": beta * (s - sp) * (Dl - alpha),   # = E A' + E B' (clipped)
                  "sd_ramp": math.sqrt(G.var() + H.var()),
                  "sd_ramp_T": beta * (s - sp) * math.sqrt(G.var() + H.var())})
    return d


def one_rep(args):
    scen, n, i, seed0, levels, crossfit, gbm, diag = args
    sc = dgp.make(scen)
    seed = seed0 + i
    rng = np.random.default_rng(seed)
    X, Y = sc.sample(n, rng, "P"), sc.sample(n, rng, "Q")
    ix, iy = rng.permutation(n), rng.permutation(n)
    n1 = n // 2
    X1, X2 = X[ix[:n1]], X[ix[n1:]]
    Y1, Y2 = Y[iy[:n1]], Y[iy[n1:]]
    sigma = median_heuristic(X1, Y1, rng=rng)
    rec = {"scenario": scen, "n": n, "rep": i, "seed": seed, "sigma": sigma, "levels": {}}
    wit0 = None
    for a in levels:
        wit = Witness(X1, Y1, a, sigma, rng)
        if a == 0.0:
            wit0 = wit
        res, (u2, v2, s, sp, pr) = fold_tests(wit, X2, Y2, a, rng)
        res["norm"] = wit.norm
        if diag:
            res.update(population_diag(wit, sc, a, s, sp, seed))
        if crossfit and (a in MAIN_ALPHAS):
            # swap the folds: witness and cut points on fold 2, tested on fold 1
            sigma2 = median_heuristic(X2, Y2, rng=rng)
            wit2 = Witness(X2, Y2, a, sigma2, rng)
            res2, (u1b, v1b, s2, sp2, pr2) = fold_tests(wit2, X1, Y1, a, rng, n_boot=200)
            res["cf_union_bet"] = bool(
                S.betting_test(S.pair_up(S.ramp(u2, s, sp), -S.ramp(v2, s, sp)) - a, -(1 + a), DELTA / 2,
                               c=0.5, prior_var=(pr or {}).get("ramp"))[0] if s > sp else False) or bool(
                S.betting_test(S.pair_up(S.ramp(u1b, s2, sp2), -S.ramp(v1b, s2, sp2)) - a, -(1 + a), DELTA / 2,
                               c=0.5, prior_var=(pr2 or {}).get("ramp"))[0] if s2 > sp2 else False)
            k1 = (stopped_wealth(S.pair_up(S.ramp(u2, s, sp), -S.ramp(v2, s, sp)) - a, -(1 + a), DELTA, 0.5,
                                 (pr or {}).get("ramp")) if s > sp else -np.inf)
            k2 = (stopped_wealth(S.pair_up(S.ramp(u1b, s2, sp2), -S.ramp(v1b, s2, sp2)) - a, -(1 + a), DELTA,
                                 0.5, (pr2 or {}).get("ramp")) if s2 > sp2 else -np.inf)
            # average of two e-values is an e-value
            avg = np.logaddexp(k1, k2) - math.log(2)
            res["cf_avg_bet"] = bool(avg >= math.log(1 / DELTA))
            res["cf_second_ramp_bet"] = res2["ramp_bet"]
        # single direction (alpha = 0 witness) with level-specific cut points
        if wit0 is not None and a > 0:
            u1, v1 = wit0.fold1_values()
            s0, sp0 = cut_points(u1, v1, a)
            if s0 > sp0:
                pr0 = priors(u1, v1, s0, sp0, a, len(X2), len(Y2))
                uu, vv = wit0(X2), wit0(Y2)
                Z = S.pair_up(S.ramp(uu, s0, sp0), -S.ramp(vv, s0, sp0)) - a
                res["dir0_ramp_bet"] = S.betting_test(Z, -(1 + a), DELTA, c=0.5, prior_var=pr0.get("ramp"))[0]
            else:
                res["dir0_ramp_bet"] = False
        elif a == 0.0:
            res["dir0_ramp_bet"] = res["ramp_bet"]
        rec["levels"][str(a)] = res
    if gbm:
        rec["gbm"] = gbm_tests(X1, Y1, X2, Y2, seed, sorted(set(levels)))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="disjoint_d2_a20")
    ap.add_argument("--ns", type=int, nargs="+", default=[100, 200, 400, 800, 1600])
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed0", type=int, default=70_000)
    ap.add_argument("--no-grid", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--levels", type=float, nargs="*", default=None)
    ap.add_argument("--no-crossfit", action="store_true")
    ap.add_argument("--no-gbm", action="store_true")
    ap.add_argument("--diag-reps", type=int, default=DIAG_REPS)
    ap.add_argument("--boundary", action="store_true", help="test only alpha = alpha_star")
    a = ap.parse_args()
    if a.boundary:
        a.levels = [float(dgp.make(a.scenario).alpha_star)]
    levels = sorted(set(MAIN_ALPHAS + ([] if a.no_grid else GRID))) if a.levels is None else sorted(a.levels)
    out = HERE / "out" / f"synth_{a.scenario}{a.tag}.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["n"], r["rep"]))
    jobs = [(a.scenario, n, i, a.seed0, levels, not a.no_crossfit, not a.no_gbm, i < a.diag_reps)
            for n in a.ns for i in range(a.reps) if (n, i) not in done]
    # big n first so the pool drains evenly
    jobs.sort(key=lambda j: -j[1])
    print(f"{len(jobs)} jobs, {len(done)} already done -> {out}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex, open(out, "a") as fh:
        futs = [ex.submit(one_rep, j) for j in jobs]
        for k, f in enumerate(as_completed(futs)):
            rec = f.result()
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if k % 50 == 0:
                print(f"  {k+1}/{len(jobs)}  n={rec['n']}  {time.time()-t0:.0f}s", flush=True)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

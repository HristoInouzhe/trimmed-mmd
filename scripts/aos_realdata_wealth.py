"""AoS revision: per-split stopped wealths of the sequential betting certificate on the four real-data contrasts.

The paper (aos/sections/07_numerical.tex, Table tab:realdata) reports, for each contrast, the minimum and the median
over eight random splits of the split-sample lower bound of Corollary cor:lower (sequential betting certificate of
Theorem thm:betting, batches (0, 1/4, 1/2, 1), per-level witnesses, grid 0:0.05:0.75, 0:0.05:0.95 for the child
contrast, delta = 0.05, split seeds 1000-1007). A referee notes that the average of the eight stopped wealths is
itself an e-value. This script reruns every split, level by level, with the same seeds and protocol, and stores for
each (contrast, split, level)

    certified        max_t K_t >= 1/delta (the published decision; checked against the stored outputs)
    log_final        log K_N  (as stored by the published runs)
    log_max          log max_t K_t
    log_stopped      log K_{tau ^ N}, tau = first t with K_t >= 1/delta (the stopped wealth; 0 if no pair was bet on)

Protocol per contrast: RHC uses the raw covariates and standardizes by the moments of batch 0 of each split
(standardize_batch0=True, as scripts/v3_rhc_batch0_lower.py); the NHANES profiles are used unstandardized, as in the
published run (realdata_run._dataset). Bandwidth: median heuristic on batch 0 (splitting.sequential_order).

The betting loop below is a copy of splitting.sequential_betting_certificate that also records the wealth path; the
witness fits are the package's (splitting._sequential_fit) and consume the random stream in the same order, so the
decisions and final wealths reproduce the published ones.

Levels are computed upward, wave by wave per contrast: level k+1 is computed for all eight splits while some split
has not yet reached its first failure or one of the averaged tests is still rejecting. Every finished job is appended
to results/aos/realdata/wealth/<contrast>.jsonl, so a killed run resumes.

    PYTHONPATH=src python3 -I scripts/aos_realdata_wealth.py --workers 3 --threads 2
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path


def _arg(name, default):
    for i, a in enumerate(sys.argv):
        if a == name and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


_T = _arg("--threads", "2")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _T

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trimmedmmd.experiments.realdata_run import ALPHA_GRID, DELTA, _dataset  # noqa: E402
from trimmedmmd.realdata import load_rhc  # noqa: E402
from trimmedmmd.splitting import (SEQ_EDGES, _sequential_fit, pair_statistics, ramp,  # noqa: E402
                                  ramp_cut_points, ramp_prior_var, sequential_horizon,
                                  sequential_order)

OUT = ROOT / "results" / "aos" / "realdata" / "wealth"
SEEDS = [1000 + i for i in range(8)]
CONTRASTS = ["rhc", "nhanes_child", "nhanes_age", "nhanes_bmi"]
GRID = {"nhanes_child": np.round(np.arange(0.0, 0.95 + 1e-9, 0.05), 3)}
C_BET, PRIOR_W = 0.5, 20.0


def grid_for(name):
    return GRID.get(name, ALPHA_GRID)


_DATA = {}


def data(name):
    if name not in _DATA:
        if name == "rhc":
            X, Y, _, _ = load_rhc(standardize=False)
            _DATA[name] = (np.asarray(X, float), np.asarray(Y, float), True)
        else:
            X, Y = _dataset(name)
            _DATA[name] = (np.asarray(X, float), np.asarray(Y, float), False)
    return _DATA[name]


def wealth_path_certificate(order, alpha, delta=DELTA, edges=SEQ_EDGES, c=C_BET, prior_w=PRIOR_W):
    """splitting.sequential_betting_certificate with the package fit, recording the wealth path."""
    fit = _sequential_fit(order)
    X, Y = order.X, order.Y
    n, m = len(X), len(Y)
    bx = [int(round(e * n)) for e in edges]
    by = [int(round(e * m)) for e in edges]
    N = sequential_horizon(n, m, edges)
    Zs, pri, skipped, cuts = [], [], [], []
    for b in range(1, len(edges) - 1):
        Xf, Yf = X[:bx[b]], Y[:by[b]]
        Xb, Yb = X[bx[b]:bx[b + 1]], Y[by[b]:by[b + 1]]
        wit, u1, v1 = fit(Xf, Yf, alpha)
        s, sp = ramp_cut_points(u1, v1, alpha)
        cuts.append([float(s), float(sp)])
        if len(Xb) == 0 or len(Yb) == 0:
            continue
        if not s > sp:
            skipped.append(b)
            continue
        G, H = ramp(wit(Xb), s, sp), ramp(wit(Yb), s, sp)
        Zs.append(pair_statistics(G, H) - alpha)
        pri.append(ramp_prior_var(u1, v1, s, sp, len(Xb), len(Yb)))
    L = math.log(1.0 / delta)
    rec = {"horizon": int(N), "pairs_bet": int(sum(z.size for z in Zs)), "skipped": skipped, "cuts": cuts,
           "sigma": float(order.sigma)}
    if not Zs or N == 0:
        # no pair bet on: the wealth stays K_0 = 1 (the package reports log K_N = -inf as a sentinel)
        rec.update(certified=False, log_final=0.0, log_max=0.0, log_stopped=0.0, stop_index=None,
                   package_log_final=float("-inf") if not Zs else 0.0)
        return rec
    lam_max = c / (1.0 + alpha)
    logK, best = 0.0, float("-inf")
    ssd, cnt, mu_sum = 0.0, 0, 0.0
    stopped, stop_index = None, None
    for z, pv in zip(Zs, pri):
        for y in z:
            mu = mu_sum / (prior_w + cnt)
            sig2 = max((prior_w * pv + ssd) / (prior_w + cnt), 1e-8)
            lam = min(lam_max, math.sqrt(2.0 * L / (N * sig2)))
            logK += math.log1p(lam * y)
            best = max(best, logK)
            if stopped is None and logK >= L:
                stopped, stop_index = logK, cnt + 1
            ssd += (y - mu) ** 2
            mu_sum += y
            cnt += 1
    rec.update(certified=bool(best >= L), log_final=float(logK), log_max=float(best),
               log_stopped=float(stopped if stopped is not None else logK), stop_index=stop_index,
               package_log_final=float(logK))
    return rec


def job(args):
    name, seed, alpha = args
    t0 = time.time()
    X, Y, std0 = data(name)
    split_seed = int(np.random.default_rng(seed).integers(0, 2**31 - 1))   # as lower_bound_betting
    order = sequential_order(X, Y, np.random.default_rng(split_seed), edges=SEQ_EDGES, sigma=None,
                             standardize_batch0=std0)
    rec = wealth_path_certificate(order, float(alpha))
    rec.update(contrast=name, seed=int(seed), split_seed=split_seed, alpha=float(alpha),
               standardize_batch0=bool(std0), seconds=round(time.time() - t0, 1))
    return rec


def load_done(name):
    f = OUT / f"{name}.jsonl"
    done = {}
    if f.exists():
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["seed"], round(r["alpha"], 6))] = r
    return done


def avg_wealth(recs, key):
    return float(np.mean([math.exp(r[key]) for r in recs]))


def need_next(name, done, k):
    """Is level index k+1 needed, given levels 0..k are complete for all splits?"""
    grid = grid_for(name)
    if k + 1 >= len(grid):
        return False
    alive_split = False
    for s in SEEDS:
        recs = [done[(s, round(float(a), 6))] for a in grid[:k + 1]]
        if all(r["certified"] for r in recs):
            alive_split = True
    alive_avg = {"log_stopped": True, "log_final": True}
    for a in grid[:k + 1]:
        recs = [done[(s, round(float(a), 6))] for s in SEEDS]
        for key in alive_avg:
            if avg_wealth(recs, key) < 1.0 / DELTA:
                alive_avg[key] = False
    return alive_split or any(alive_avg.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--contrasts", nargs="*", default=CONTRASTS)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    done = {nm: load_done(nm) for nm in a.contrasts}
    level = {nm: 0 for nm in a.contrasts}
    active = set(a.contrasts)
    pending = {}
    log = open(OUT / "run.log", "a")
    print(f"start {time.strftime('%F %T')} workers={a.workers} threads={a.threads}", file=log, flush=True)

    def submit_level(ex, nm):
        alpha = float(grid_for(nm)[level[nm]])
        for s in SEEDS:
            if (s, round(alpha, 6)) not in done[nm]:
                fut = ex.submit(job, (nm, s, alpha))
                pending[fut] = (nm, s, alpha)

    def level_complete(nm):
        alpha = float(grid_for(nm)[level[nm]])
        return all((s, round(alpha, 6)) in done[nm] for s in SEEDS)

    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for nm in list(active):
            submit_level(ex, nm)
        while active:
            # advance every contrast whose current level is complete
            progressed = True
            while progressed:
                progressed = False
                for nm in list(active):
                    if level_complete(nm):
                        if need_next(nm, done[nm], level[nm]):
                            level[nm] += 1
                            submit_level(ex, nm)
                            progressed = True
                        else:
                            active.discard(nm)
                            print(f"{nm}: done at level {grid_for(nm)[level[nm]]}", file=log, flush=True)
            if not pending:
                break
            fin, _ = wait(list(pending), return_when=FIRST_COMPLETED)
            for fut in fin:
                nm, s, alpha = pending.pop(fut)
                rec = fut.result()
                with open(OUT / f"{nm}.jsonl", "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                done[nm][(s, round(alpha, 6))] = rec
                print(f"{time.strftime('%T')} {nm} seed={s} a={alpha:.2f} cert={rec['certified']} "
                      f"logK_N={rec['log_final']:.3f} logK_stop={rec['log_stopped']:.3f} "
                      f"t={rec['seconds']}s", file=log, flush=True)
    print(f"end {time.strftime('%F %T')}", file=log, flush=True)


if __name__ == "__main__":
    main()

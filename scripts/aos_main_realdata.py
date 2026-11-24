"""AoS restructure (plan section 4.3): the split-sample bound with the Laplace kernel and genuine empirical quantile cut
points on the four real-data contrasts of Table 2. Protocol: results/aos/main/NOTES.md.

Same data, split seeds (1000-1007), batches (0, 1/4, 1/2, 1), batch-0 standardization for RHC, bandwidth h = the median
distance on batch 0 (splitting.sequential_order; the Laplace kernel exp(-|x - y| / h) takes the Gaussian's h), per-level
witnesses (package solver, trimmedmmd.aos_fallback.fit_batch with the stream of splitting._sequential_fit), grids
0:0.05:0.75 (child contrast 0:0.05:0.95) and delta = 0.05 as scripts/aos_realdata_wealth.py and
scripts/aos_fallback_realdata.py. At each (split, level) the two batch witnesses are fitted once and two cut rules are
evaluated on the own-term-removed fit values (batch skipped if the cuts cross): "order" (genuine empirical quantiles,
the paper's method) and "linear" (numpy.quantile; = the OWN rule of results/aos/fallback/raw/realdata_laplace, a check
of the fits). Per split the grid is read upward until both rules have failed.

Contrast "rhc_full" (AoS referee runs, 2026-11-22; run only when named, `--contrasts rhc_full`): RHC with the 45
covariates of realdata.load_rhc(covariates="full"), batch-0 standardization as "rhc"; grid 0:0.05:0.75.

One chain (contrast, split) per task; each level is appended to $AOS_MAIN_WORK/realdata_laplace/<contrast>_<seed>.jsonl
(local disk), `sync` copies to results/aos/main/raw/. A killed run resumes.

    PYTHONPATH=src python -B scripts/aos_main_realdata.py run --workers 3 --threads 2
"""
from __future__ import annotations

import os
import sys


def _arg(name, default):
    for i, a in enumerate(sys.argv):
        if a == name and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


_T = _arg("--threads", "2")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _T
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trimmedmmd.aos_fallback import bet, choose_cuts, fit_batch  # noqa: E402
from trimmedmmd.experiments.realdata_run import ALPHA_GRID, DELTA, _dataset  # noqa: E402
from trimmedmmd.realdata import load_rhc  # noqa: E402
from trimmedmmd.splitting import (SEQ_EDGES, pair_statistics, ramp, ramp_prior_var,  # noqa: E402
                                  sequential_horizon, sequential_order)

WORK = Path(os.environ.get("AOS_MAIN_WORK", "/home/hristouam/aos_main_work/raw"))
OUT = ROOT / "results" / "aos" / "main" / "raw"
SEEDS = [1000 + i for i in range(8)]
CONTRASTS = ["rhc", "nhanes_child", "nhanes_age", "nhanes_bmi"]
GRID = {"nhanes_child": np.round(np.arange(0.0, 0.95 + 1e-9, 0.05), 3)}
CUTS = ("order", "linear")
# AOS_REALDATA_KERNEL=gaussian (2026-11-24, cold read B2): the same runs with the Gaussian kernel, e.g. on rhc_full,
# so that the kernel comparison of Section 7.2 covers the 45 RHC covariates; outputs go to realdata_gaussian/.
KERNEL = os.environ.get("AOS_REALDATA_KERNEL", "laplace")
_DATA = {}


def grid_for(name):
    return GRID.get(name, ALPHA_GRID)


def data(name):
    if name not in _DATA:
        if name in ("rhc", "rhc_full"):
            # "rhc_full" (AoS referee runs, 2026-11-22; not in the default CONTRASTS): the 45 covariates of
            # realdata.load_rhc(covariates="full"), the 33 of "rhc" plus the 12 string-coded binaries
            X, Y, _, _ = load_rhc(standardize=False, covariates="full" if name == "rhc_full" else "paper")
            _DATA[name] = (np.asarray(X, float), np.asarray(Y, float), True)
        else:
            X, Y = _dataset(name)
            _DATA[name] = (np.asarray(X, float), np.asarray(Y, float), False)
    return _DATA[name]


def level_record(name, seed, alpha):
    t0 = time.time()
    X, Y, std0 = data(name)
    split_seed = int(np.random.default_rng(seed).integers(0, 2**31 - 1))   # as lower_bound_betting
    order = sequential_order(X, Y, np.random.default_rng(split_seed), edges=SEQ_EDGES, sigma=None,
                             standardize_batch0=std0)
    Xo, Yo = order.X, order.Y
    n, m = len(Xo), len(Yo)
    edges = SEQ_EDGES
    bx = [int(round(e * n)) for e in edges]
    by = [int(round(e * m)) for e in edges]
    N = sequential_horizon(n, m, edges)
    fits = []
    for b in range(1, len(edges) - 1):
        Xf, Yf = Xo[:bx[b]], Yo[:by[b]]
        Xb, Yb = Xo[bx[b]:bx[b + 1]], Yo[by[b]:by[b + 1]]
        fits.append((fit_batch(Xf, Yf, Xb, Yb, float(alpha), order.sigma, order.rng, kernel=KERNEL), len(Xb), len(Yb)))
    rec = {"contrast": name, "seed": int(seed), "split_seed": split_seed, "alpha": float(alpha), "kernel": KERNEL,
           "standardize_batch0": bool(std0), "horizon": int(N), "sigma": float(order.sigma),
           "norms": [float(bf.norm) for bf, _, _ in fits], "n_fit": [[bx[b], by[b]] for b in (1, 2)]}
    for c in CUTS:
        Zs, pri, src, cuts = [], [], [], []
        for bf, nb, mb in fits:
            s, sp, u, v, pick, _ = choose_cuts(bf.u_own, bf.v_own, bf.u_plain, bf.v_plain, float(alpha), "own",
                                               cut_rule=c)
            src.append(pick)
            if pick == "skip":
                cuts.append(None)
                continue
            cuts.append([float(s), float(sp)])
            G, H = ramp(bf.fXb, s, sp), ramp(bf.fYb, s, sp)
            Zs.append(pair_statistics(G, H) - float(alpha))
            pri.append(ramp_prior_var(u, v, s, sp, nb, mb))
        r = bet(Zs, pri, float(alpha), N, delta=DELTA)
        r.update(source=src, cuts=cuts)
        rec[c] = r
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def chain_path(name, seed):
    return WORK / f"realdata_{KERNEL}" / f"{name}_{seed}.jsonl"


def read_chain(path):
    recs = {}
    if path.exists():
        for line in path.read_bytes().split(b"\n"):
            if not line.strip() or b"\x00" in line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            recs[round(r["alpha"], 6)] = r
    return recs


def alive_after(recs, grid, k):
    alive = set(CUTS)
    for a in grid[:k + 1]:
        r = recs[round(float(a), 6)]
        alive = {c for c in alive if r[c]["certified"]}
    return alive


def run_chain(args):
    name, seed = args
    path = chain_path(name, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    recs = read_chain(path)
    grid = grid_for(name)
    for k, a in enumerate(grid):
        key = round(float(a), 6)
        if key not in recs:
            rec = level_record(name, seed, a)
            with open(path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            recs[key] = rec
        if not alive_after(recs, grid, k):
            break
    return name, seed, len(recs)


def sync(log=print):
    OUT.mkdir(parents=True, exist_ok=True)
    nf = nr = 0
    for d in sorted(WORK.glob("realdata_*")):
        od = OUT / d.name
        od.mkdir(parents=True, exist_ok=True)
        for f in sorted(d.glob("*.jsonl")):
            recs = read_chain(f)
            data_ = "".join(json.dumps(recs[k]) + "\n" for k in sorted(recs))
            tmp = od / (f.name + ".tmp")
            tmp.write_text(data_)
            os.replace(tmp, od / f.name)
            if (od / f.name).read_text() != data_:
                log(f"SYNC MISMATCH {f}")
            nf += 1
            nr += len(recs)
    log(f"synced {nf} files ({nr} level records) to {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "sync"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--contrasts", nargs="*", default=CONTRASTS, choices=CONTRASTS + ["rhc_full"])
    a = ap.parse_args()
    if a.cmd == "sync":
        sync()
        return
    from concurrent.futures import ProcessPoolExecutor, as_completed

    def log(s):
        print(time.strftime("%H:%M:%S"), s, flush=True)
    jobs = [(nm, s) for nm in a.contrasts for s in SEEDS]
    log(f"{len(jobs)} chains, kernel {KERNEL}, workers {a.workers} x threads {a.threads}")
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(run_chain, j): j for j in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                nm, s, nl = fu.result()
                log(f"[{i}/{len(jobs)}] {nm} seed={s}: {nl} levels ({time.time() - t0:.0f}s)")
            except Exception as e:
                log(f"FAILED {futs[fu]} {e!r}")
    sync(log)
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

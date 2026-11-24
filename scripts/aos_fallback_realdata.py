"""Exploratory (AoS): cut-point rules OWN / FALLBACK / PLAIN (and the Gaussian vs Laplace kernel) for the sequential
betting lower bound on the four real-data contrasts. Protocol: results/aos/fallback/NOTES.md.

Same data, split seeds (1000-1007), batches (0, 1/4, 1/2, 1), batch-0 bandwidth, per-level witnesses (package
solver), grids (0:0.05:0.75; child 0:0.05:0.95) and delta = 0.05 as scripts/aos_realdata_wealth.py. For each
(contrast, split) the grid is read upward until every rule has failed; at each level the witnesses are fitted once and
the three rules are evaluated on them (trimmedmmd.aos_fallback.sequential_all_rules).

One chain (contrast, split, kernel) per task; each level is appended (one writer per file) to
$AOS_FALLBACK_WORK/realdata_<kernel>/<contrast>_<seed>.jsonl on local disk, and `sync` copies the files atomically to
results/aos/fallback/raw/. A killed run resumes from the last stored level of each chain.

    PYTHONPATH=src python -B scripts/aos_fallback_realdata.py run --kernel gaussian --workers 3 --threads 2
    PYTHONPATH=src python -B scripts/aos_fallback_realdata.py check     # OWN vs the stored wealth records
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

from trimmedmmd.aos_fallback import RULES, sequential_all_rules  # noqa: E402
from trimmedmmd.experiments.realdata_run import ALPHA_GRID, DELTA, _dataset  # noqa: E402
from trimmedmmd.realdata import load_rhc  # noqa: E402
from trimmedmmd.splitting import SEQ_EDGES, sequential_order  # noqa: E402

WORK = Path(os.environ.get("AOS_FALLBACK_WORK", "/home/hristouam/aos_fallback_work/raw"))
OUT = ROOT / "results" / "aos" / "fallback" / "raw"
STORED = ROOT / "results" / "aos" / "realdata" / "wealth"
SEEDS = [1000 + i for i in range(8)]
CONTRASTS = ["rhc", "nhanes_child", "nhanes_age", "nhanes_bmi"]
GRID = {"nhanes_child": np.round(np.arange(0.0, 0.95 + 1e-9, 0.05), 3)}


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


def level_record(name, seed, alpha, kernel):
    t0 = time.time()
    X, Y, std0 = data(name)
    split_seed = int(np.random.default_rng(seed).integers(0, 2**31 - 1))   # as lower_bound_betting
    order = sequential_order(X, Y, np.random.default_rng(split_seed), edges=SEQ_EDGES, sigma=None,
                             standardize_batch0=std0)
    rec = sequential_all_rules(order, float(alpha), kernel=kernel, delta=DELTA, edges=SEQ_EDGES)
    rec.update(contrast=name, seed=int(seed), split_seed=split_seed, alpha=float(alpha),
               standardize_batch0=bool(std0), seconds=round(time.time() - t0, 2))
    return rec


def chain_path(kernel, name, seed):
    return WORK / f"realdata_{kernel}" / f"{name}_{seed}.jsonl"


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
    """Rules still alive after levels grid[0..k] (all certified)."""
    alive = set(RULES)
    for a in grid[:k + 1]:
        r = recs[round(float(a), 6)]
        alive = {rl for rl in alive if r[rl]["certified"]}
    return alive


def run_chain(args):
    name, seed, kernel = args
    path = chain_path(kernel, name, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    recs = read_chain(path)
    grid = grid_for(name)
    for k, a in enumerate(grid):
        key = round(float(a), 6)
        if key not in recs:
            rec = level_record(name, seed, a, kernel)
            with open(path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            recs[key] = rec
        if not alive_after(recs, grid, k):
            break
    return name, seed, kernel, len(recs)


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


def check(kernel="gaussian"):
    """OWN against the stored per-(split, level) records of scripts/aos_realdata_wealth.py."""
    tot = agree = 0
    dmax = 0.0
    bad = []
    for name in CONTRASTS:
        stored = {}
        for line in (STORED / f"{name}.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                stored[(r["seed"], round(r["alpha"], 6))] = r
        for seed in SEEDS:
            recs = read_chain(chain_path(kernel, name, seed))
            for key, r in recs.items():
                s = stored.get((seed, key))
                if s is None:
                    continue
                tot += 1
                ok = (s["certified"] == r["own"]["certified"]
                      and s["skipped"] == [b + 1 for b, src in enumerate(r["own"]["source"]) if src == "skip"])
                agree += ok
                if not ok:
                    bad.append((name, seed, key))
                if s["pairs_bet"] > 0:
                    dmax = max(dmax, abs(s["log_final"] - r["own"]["log_final"]))
    print(f"OWN vs stored: {agree}/{tot} (split, level) decisions and skip patterns agree; "
          f"max |log K_N difference| = {dmax:.3g}; disagreements: {bad[:10]}")
    return agree, tot, dmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "sync", "check"])
    ap.add_argument("--kernel", default="gaussian", choices=["gaussian", "laplace"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--contrasts", nargs="*", default=CONTRASTS)
    a = ap.parse_args()
    if a.cmd == "sync":
        sync()
        return
    if a.cmd == "check":
        check(a.kernel)
        return
    from concurrent.futures import ProcessPoolExecutor, as_completed

    def log(s):
        print(time.strftime("%H:%M:%S"), s, flush=True)
    jobs = [(nm, s, a.kernel) for nm in a.contrasts for s in SEEDS]
    log(f"{len(jobs)} chains, kernel {a.kernel}, workers {a.workers} x threads {a.threads}")
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(run_chain, j): j for j in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                nm, s, k, nl = fu.result()
                log(f"[{i}/{len(jobs)}] {nm} seed={s} {k}: {nl} levels ({time.time() - t0:.0f}s)")
            except Exception as e:
                log(f"FAILED {futs[fu]} {e!r}")
            if i % 8 == 0:
                try:
                    sync(log)
                except Exception as e:
                    log(f"SYNC FAILED {e!r}")
    sync(log)
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

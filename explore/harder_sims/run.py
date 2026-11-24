"""Runner: pilot bandwidths, threshold cache, resumable replicate runs.

    cd /nvme0n1-disk/trimmed-mmd
    PYTHONPATH=src python3 -I explore/harder_sims/run.py prep
    PYTHONPATH=src python3 -I explore/harder_sims/run.py run --designs all --ns 100 200 400 800 --reps 200 --workers 16

Each finished replicate is appended (one JSON line, flushed) to
out/<design>.jsonl; on restart, (n, rep) pairs already present are skipped.
Seeds: replicate (design, n, rep) uses SeedSequence([crc32(design), n, rep]);
pilot draws use SeedSequence([crc32(design), 424242]) and are never reused.

POST HOC (DESIGN.md section 9, 2026-11-10): `run --alpha0` runs only the tests of H_{0,0} (alpha0.py; no QP,
blocked kernel sums) at any n, writing out/<design>_n0ext.jsonl; the extension uses --rep0 200000. Without
--alpha0 nothing below changes. The size ladder is driven by ladder_alpha0.py.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
for p in (ROOT / "src", ROOT / "explore" / "sharper_threshold"):
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402

import designs as DS  # noqa: E402

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
PILOT = OUT / "pilot.json"
THR = OUT / "thr_cache.json"
DELTA = 0.05


def seed_key(key):
    return zlib.crc32(key.encode())


# ---------------------------------------------------------------------------
# pilot: fixed bandwidth (and coordinate scaling) per design
# ---------------------------------------------------------------------------

def pilot_one(key):
    from trimmedmmd.kernels import median_heuristic
    dz = DS.catalogue()[key]
    rng = np.random.default_rng(np.random.SeedSequence([seed_key(key), 424242]))
    n_p = 1000 if dz.m_ratio == 1 else 400
    Xp, Yp = dz.sample(n_p, rng, "P"), dz.sample(n_p * dz.m_ratio, rng, "Q")
    scale = None
    if "D5b" in key:
        scale = np.vstack([Xp, Yp]).std(0).tolist()
        Xp, Yp = Xp / np.array(scale), Yp / np.array(scale)
    sig = median_heuristic(Xp, Yp, rng=np.random.default_rng(0))
    return key, {"sigma_fix": float(sig), "scale": scale, "n_pilot": n_p,
                 "m_pilot": n_p * dz.m_ratio}


def thr_one(args):
    import thresholds as T
    n, m, a = args
    G = T.GAUSS
    old = T.tau_old(n, m, a, DELTA)
    cf = T.tau_P_closed(n, m, a, DELTA, **G)
    new = min(T.tau_A(n, m, a, DELTA, **G), T.tau_P(n, m, a, DELTA, **G),
              T.tau_P_coupled(n, m, a, DELTA, kappa=1.0))
    return f"{n},{m},{a:.12g}", {"old": old, "newcf": cf, "new": new}


def prep(workers, extra_ns=(1600,)):
    C = DS.catalogue()
    pil = json.loads(PILOT.read_text()) if PILOT.exists() else {}
    todo = [k for k in C if k not in pil]
    with ProcessPoolExecutor(workers) as ex:
        for k, v in ex.map(pilot_one, todo):
            pil[k] = v
    PILOT.write_text(json.dumps(pil, indent=1))
    thr = json.loads(THR.read_text()) if THR.exists() else {}
    jobs = set()
    for k, dz in C.items():
        for n in tuple(dz.ns) + tuple(extra_ns):
            m = dz.m_ratio * n
            for a in dz.levels():
                key = f"{n},{m},{a:.12g}"
                if key not in thr:
                    jobs.add((n, m, a))
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for key, v in ex.map(thr_one, sorted(jobs), chunksize=4):
            thr[key] = v
    THR.write_text(json.dumps(thr, indent=1))
    print(f"pilot {len(pil)} designs; thresholds {len(thr)} entries ({time.time() - t0:.0f}s)")


def thr_table(thr, n, m, levels):
    rows = [thr[f"{n},{m},{a:.12g}"] for a in levels]
    return {k: [r[k] for r in rows] for k in ("old", "newcf", "new")}


# ---------------------------------------------------------------------------
# replicate jobs
# ---------------------------------------------------------------------------

_CACHE = {}


def job(args):
    key, n, rep, do_analytic = args
    import methods as M
    if "C" not in _CACHE:
        _CACHE["C"] = DS.catalogue()
        _CACHE["pil"] = json.loads(PILOT.read_text())
        _CACHE["thr"] = json.loads(THR.read_text())
    dz = _CACHE["C"][key]
    pil = _CACHE["pil"][key]
    scale = None if pil["scale"] is None else np.array(pil["scale"])
    tab = thr_table(_CACHE["thr"], n, dz.m_ratio * n, dz.levels())
    t0 = time.perf_counter()
    rec = M.run_replicate(dz, n, rep, seed_key(key), pil["sigma_fix"], tab, scale=scale,
                          do_analytic=do_analytic)
    rec["time"]["total"] = time.perf_counter() - t0
    rec["alpha_star"] = dz.alpha_star
    rec["levels"] = dz.levels()
    return rec


def job_alpha0(args):
    """POST HOC alpha = 0 mode (alpha0.py): same data, folds and seeds as `job`, tests of H_{0,0} only."""
    key, n, rep = args[:3]
    import alpha0 as A0
    if "C" not in _CACHE:
        _CACHE["C"] = DS.catalogue()
        _CACHE["pil"] = json.loads(PILOT.read_text())
    dz = _CACHE["C"][key]
    pil = _CACHE["pil"][key]
    scale = None if pil["scale"] is None else np.array(pil["scale"])
    rec = A0.run_replicate(dz, n, rep, seed_key(key), pil["sigma_fix"], scale=scale)
    rec["alpha_star"] = dz.alpha_star
    return rec


def done_set(key):
    f = OUT / f"{key}.jsonl"
    s = set()
    if f.exists():
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
                s.add((r["n"], r["rep"]))
            except Exception:
                pass
    return s


def run(keys, ns, reps, workers, rep0=0, tag="", do_analytic=True, order="n", alpha0=False):
    C = DS.catalogue()
    jobs = []
    for key in keys:
        dz = C[key]
        done = done_set(key + tag)
        for n in ns:
            if not alpha0 and n not in dz.ns and n not in (1600,):
                continue
            for r in range(rep0, rep0 + reps):
                if (n, r) not in done:
                    jobs.append((key, n, r, do_analytic))
    if order == "n":
        jobs.sort(key=lambda j: (j[1], j[2], j[0]))
    elif order == "cost":             # alpha0 mode: most expensive first, for load balance
        jobs.sort(key=lambda j: (-(j[1] * (1 + C[j[0]].m_ratio)) ** 2 * (C[j[0]].d + 8), j[0], j[2]))
    print(f"{len(jobs)} jobs", flush=True)
    t0 = time.time()
    handles = {}
    done = 0
    fn = job_alpha0 if alpha0 else job
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(fn, j): j for j in jobs}
        for fu in as_completed(futs):
            j = futs[fu]
            try:
                rec = fu.result()
            except Exception as e:      # record and continue
                print("FAILED", j, repr(e), flush=True)
                continue
            f = handles.get(j[0])
            if f is None:
                f = handles[j[0]] = open(OUT / f"{j[0]}{tag}.jsonl", "a")
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
            done += 1
            if done % 50 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f"{done}/{len(jobs)} {el:.0f}s eta {el / done * (len(jobs) - done):.0f}s",
                      flush=True)
    for f in handles.values():
        f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "run"])
    ap.add_argument("--designs", nargs="+", default=["all"])
    ap.add_argument("--ns", type=int, nargs="+", default=[100, 200, 400, 800])
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--rep0", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-analytic", action="store_true")
    ap.add_argument("--alpha0", action="store_true",
                    help="POST HOC: tests of H_{0,0} only (alpha0.py), any n; default tag _n0ext")
    a = ap.parse_args()
    if a.cmd == "prep":
        prep(a.workers)
        return
    keys = list(DS.catalogue()) if a.designs == ["all"] else a.designs
    if a.alpha0:
        run(keys, a.ns, a.reps, a.workers, rep0=a.rep0, tag=a.tag or "_n0ext", order="cost", alpha0=True)
        return
    run(keys, a.ns, a.reps, a.workers, rep0=a.rep0, tag=a.tag, do_analytic=not a.no_analytic)


if __name__ == "__main__":
    main()

"""POST HOC (DESIGN.md section 9, 2026-11-10): the size ladder of the alpha = 0 extension.

For every (design, method) cell of the power designs whose power against H_{0,0} at the largest pre-registered size is
below 0.8, run n = m in {1600, 3200, 6400, 12800} (m = 5n and also n = 800 for the mixed-type design, whose grid
ends at 400, so the ladder keeps doubling) above that size, 200 replicates per size, and stop the cell's ladder once
its power is >= 0.8 at two consecutive new sizes or at the cap 12800. A design is run at a size when at least one of
its cells still needs that size; every method's alpha = 0 decision is recorded on every replicate run (alpha0.py).

Driving methods: the four of the 50-page version's Table 1 (an_new, seq_bet, gbm_bet, logit_bet) and the other four
columns of the long version's n80 table (an_old, ramp_bet, boot, knn_bet), so that every column of both tables
follows the same rule. Replicates: SeedSequence([crc32(design), n, 200000 + r]), r = 0..199 (no earlier run used
rep >= 200000 at any size; the timing pilot of this extension used rep >= 300000 and is not reported).
Raw records: out/<design>_n0ext.jsonl. Log: out/run_n0ext.log; ladder state: out/ladder_n0ext.json.

    cd /nvme0n1-disk/trimmed-mmd
    PYTHONPATH=src python3 -I explore/harder_sims/ladder_alpha0.py --workers 15            # power designs
    PYTHONPATH=src python3 -I explore/harder_sims/ladder_alpha0.py --nulls --workers 15    # optional P = Q nulls
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import designs as DS  # noqa: E402
import run as R  # noqa: E402

OUT = HERE / "out"
TAG = "_n0ext"
REP0 = 200000
REPS = 200
CAP = 12800
LADDER = (800, 1600, 3200, 6400, 12800)
DRIVE = ("an_new", "seq_bet", "gbm_bet", "logit_bet", "an_old", "ramp_bet", "boot", "knn_bet")
POWER = ("D1_d2_a10", "D1_d2_a20", "D1_d10_a10", "D1_d10_a20", "D2_d20", "D2_d50", "D2_d100", "D3a_loc", "D3b_corr",
         "D4_small", "D5b_mixed", "D6_t3", "D7_wide", "D7_narrow", "D8_N1", "D8_N4", "D8_N16", "D8_N64")
# optional P = Q nulls at alpha = 0 (type-I of H_{0,0}); the mixed-type null (m = 5n) stops at 3200
NULLS = {"N0_D2_d100": (1600, 3200, 6400, 12800), "N0_D6_t3": (1600, 3200, 6400, 12800),
         "N0_D7_narrow": (1600, 3200, 6400, 12800), "N0_D5b_mixed": (800, 1600, 3200)}
LOG = OUT / "run_n0ext.log"


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC  {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def main_power():
    """(design, method) -> {n: power} from the pre-registered runs (complete cells only)."""
    out = {}
    with open(OUT / "summary_long.csv") as f:
        for r in csv.DictReader(f):
            if int(r["R"]) >= REPS:
                out.setdefault((r["design"], r["method"]), {})[int(r["n"])] = float(r["pow0"])
    return out


def ext_records(key):
    f = OUT / f"{key}{TAG}.jsonl"
    recs = {}
    if f.exists():
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if REP0 <= r["rep"] < REP0 + REPS:
                recs[(r["n"], r["rep"])] = r
    return recs


def ext_power(key):
    """method -> {n: power} over complete extension cells (all REPS replicates present)."""
    by = {}
    for (n, _), r in ext_records(key).items():
        by.setdefault(n, []).append(r)
    out = {}
    for n, rs in by.items():
        if len(rs) < REPS:
            continue
        for mth in rs[0]["dec0"]:
            out.setdefault(mth, {})[n] = sum(r["dec0"][mth] for r in rs) / len(rs)
    return out


def ladder_state(key, mp, ep):
    """For each driving method: ('closed', why) or ('next', size)."""
    st = {}
    for mth in DRIVE:
        base = mp[(key, mth)]
        nmax = max(base)
        if base[nmax] >= 0.8:
            st[mth] = ("closed", f"power {base[nmax]:.3f} >= 0.8 at the largest pre-registered size {nmax}")
            continue
        sizes = [n for n in LADDER if n > nmax]
        prev_hit = False
        for n in sizes:
            p = ep.get(mth, {}).get(n)
            if p is None:
                st[mth] = ("next", n)
                break
            hit = p >= 0.8
            if hit and prev_hit:
                st[mth] = ("closed", f"power >= 0.8 at two consecutive sizes (to {n})")
                break
            if n == CAP:
                st[mth] = ("closed", f"cap {CAP} reached (power {p:.3f})")
                break
            prev_hit = hit
    return st


def job_list(key, n, done):
    return [(key, n, REP0 + r) for r in range(REPS) if (n, REP0 + r) not in done]


def run_jobs(jobs, workers):
    C = DS.catalogue()
    jobs = sorted(jobs, key=lambda j: (-(j[1] * (1 + C[j[0]].m_ratio)) ** 2, j[0], j[2]))
    handles, done, t0 = {}, 0, time.time()
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(R.job_alpha0, j): j for j in jobs}
        for fu in as_completed(futs):
            j = futs[fu]
            try:
                rec = fu.result()
            except Exception as e:  # record and continue
                log(f"FAILED {j} {e!r}")
                continue
            f = handles.get(j[0])
            if f is None:
                f = handles[j[0]] = open(OUT / f"{j[0]}{TAG}.jsonl", "a")
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
            done += 1
            if done % 200 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f"  {done}/{len(jobs)} {el:.0f}s eta {el / done * (len(jobs) - done):.0f}s", flush=True)
    for f in handles.values():
        f.close()


def power_ladder(workers):
    mp = main_power()
    rnd = 0
    while True:
        need, states = {}, {}
        for key in POWER:
            ep = ext_power(key)
            st = ladder_state(key, mp, ep)
            states[key] = {m: list(v) for m, v in st.items()}
            nxt = sorted({v[1] for v in st.values() if v[0] == "next"})
            if nxt:
                need[key] = nxt[0]
        (OUT / "ladder_n0ext.json").write_text(json.dumps(states, indent=1))
        if not need:
            log("power ladder complete")
            return
        rnd += 1
        jobs = []
        for key, n in need.items():
            jobs += job_list(key, n, set(ext_records(key)))
        log(f"round {rnd}: " + ", ".join(f"{k}@{n}" for k, n in need.items()) + f" ({len(jobs)} replicates)")
        t0 = time.time()
        run_jobs(jobs, workers)
        msg = []
        for key, n in need.items():
            ep = ext_power(key)
            msg.append(f"{key}@{n}: " + " ".join(f"{m}={ep.get(m, {}).get(n, float('nan')):.3f}" for m in DRIVE))
        log(f"round {rnd} done in {time.time() - t0:.0f}s\n    " + "\n    ".join(msg))


def null_runs(workers):
    for key, ns in NULLS.items():
        for n in ns:
            jobs = job_list(key, n, set(ext_records(key)))
            if not jobs:
                continue
            log(f"null {key}@{n} ({len(jobs)} replicates)")
            run_jobs(jobs, workers)
            ep = ext_power(key)
            log(f"null {key}@{n}: " + " ".join(f"{m}={ep.get(m, {}).get(n, float('nan')):.3f}"
                                              for m in ("an_new", "seq_bet", "ramp_bet", "boot", "gbm_bet",
                                                        "logit_bet", "knn_bet")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--nulls", action="store_true")
    a = ap.parse_args()
    if a.nulls:
        null_runs(a.workers)
    else:
        power_ladder(a.workers)


if __name__ == "__main__":
    main()

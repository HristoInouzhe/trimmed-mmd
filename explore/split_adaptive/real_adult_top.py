"""Adult, per-level witnesses (exactly as fair_split_certificate / split_test, 2,000 FISTA iterations) at the top
levels only, to compare the finite-sample certificate with the published per-level bootstrap where it matters.
Exploratory: below these levels the alpha = 0 witness run (real_dir0.py) already certifies on every split.

    python3 -I explore/split_adaptive/real_adult_top.py --splits 4 --workers 4
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from real import dataset, level  # noqa: E402

def one(args):
    name, i = args
    X, Y = dataset(name)
    rows = []
    for a in (0.40, 0.45, 0.50):
        t = time.time()
        r = level(X, Y, a, 1000 + i, True, 512, 500)
        r.pop("_arrays", None)
        r["seconds"] = time.time() - t
        rows.append(r)
        if not (r["boot"] or r.get("ramp_bet", False)):
            break
    return {"dataset": name, "split": i, "levels": rows}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out = HERE / "out" / "real_adult_top.jsonl"
    with ProcessPoolExecutor(a.workers) as ex, open(out, "a") as fh:
        for rec in ex.map(one, [("adult", i) for i in range(a.splits)]):
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            print(rec["split"], [(r["alpha"], r["boot"], r.get("ramp_bet"), r.get("ramp_clt"), round(r.get("Delta_hat", float("nan")), 3)) for r in rec["levels"]], flush=True)

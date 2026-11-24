"""Timing pilot on seeds disjoint from the main run (rep >= 100000); only runtimes are inspected."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, json, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as R

if __name__ == "__main__":
    cfg = [(k, n) for k in sys.argv[2].split(",") for n in map(int, sys.argv[3].split(","))]
    jobs = [(k, n, 100000 + r, True) for k, n in cfg for r in range(int(sys.argv[1]))]
    t0 = time.time()
    with ProcessPoolExecutor(int(sys.argv[4])) as ex:
        recs = list(ex.map(R.job, jobs))
    agg = {}
    for r in recs:
        a = agg.setdefault((r["design"], r["n"]), [])
        a.append(r)
    for (k, n), rs in agg.items():
        t = {kk: sum(x["time"].get(kk, 0) for x in rs) / len(rs) for kk in rs[0]["time"]}
        it = sum(x["diag"]["an_iters"] for x in rs) / len(rs)
        und = sum(x["diag"]["an_undecided"] for x in rs)
        print(k, n, {kk: round(v, 2) for kk, v in t.items()}, "an_iters", round(it), "undecided", und, flush=True)
    print("wall", round(time.time() - t0, 1))

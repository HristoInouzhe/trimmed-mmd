"""JMLR v3: the RHC split-sample lower bound with the covariates standardized by batch-0 moments.

The published RHC bound (results/v2/tables/r2_lower_bounds.csv, realdata_run.py, certificate "all") standardizes the
33 covariates by the moments of the whole study (realdata.load_rhc), so the ramps depend, through two vectors of
moments, on the batches bet on, which Theorem thm:betting does not cover. Here the raw covariates go in and
splitting.sequential_order standardizes them by the moments of batch 0 only (standardize_batch0=True); everything else
is the published run: the sequential betting certificate with per-level witnesses, batches (0, 1/4, 1/2, 1), the grid
0:0.05:0.75, delta = 0.05, and the eight split seeds 1000-1007 of realdata_run.lower_bounds.

    PYTHONPATH=src python3 -I scripts/v3_rhc_batch0_lower.py --workers 8
writes results/v3/rhc_lower_batch0.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trimmedmmd.realdata import load_rhc                      # noqa: E402
from trimmedmmd.splitting import lower_bound_betting          # noqa: E402
from trimmedmmd.experiments.realdata_run import ALPHA_GRID, DELTA  # noqa: E402

SEEDS = [1000 + i for i in range(8)]


def one(seed):
    t0 = time.time()
    X, Y, _, _ = load_rhc(standardize=False)
    b, rows = lower_bound_betting(np.asarray(X, float), np.asarray(Y, float), ALPHA_GRID, delta=DELTA,
                                  rng=np.random.default_rng(seed), methods=("sequential",),
                                  none_value=-1.0, standardize_batch0=True)
    return {"seed": seed, "lower": float(b["sequential"]),
            "levels": [{k: r[k] for k in ("alpha", "sequential", "seq_log_final_wealth", "seq_horizon",
                                          "seq_pairs_bet", "seq_skipped") if k in r} for r in rows],
            "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        out = list(ex.map(one, SEEDS))
    lows = [max(r["lower"], 0.0) for r in out]
    res = {"script": "scripts/v3_rhc_batch0_lower.py", "dataset": "rhc", "standardization": "batch 0 moments",
           "certificate": "sequential", "witness": "per_level", "edges": [0.0, 0.25, 0.5, 1.0],
           "grid": ALPHA_GRID.tolist(), "delta": DELTA, "seeds": SEEDS,
           "min_lower": min(lows), "median_lower": float(np.median(lows)), "max_lower": max(lows),
           "lowers": lows, "splits": out}
    od = ROOT / "results" / "v3"
    od.mkdir(parents=True, exist_ok=True)
    (od / "rhc_lower_batch0.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("min_lower", "median_lower", "max_lower", "lowers")}))


if __name__ == "__main__":
    main()

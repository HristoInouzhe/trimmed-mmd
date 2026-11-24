"""Apply the candidate thresholds to the published real-data paths.

The paths D-hat_alpha (exact Gaussian Gram, pooled median bandwidth, grid
0, 0.05, ..., 0.75) are read from results/tables/r1_paths.csv, so nothing is
recomputed; only the threshold changes. alpha-hat is the first grid level
with D-hat <= tau (eq:alphaest on the grid, as in trimmedmmd.estimate), and
alpha_last the largest grid level rejected (all smaller levels are rejected
too, since D-hat is nonincreasing and every tau here is nondecreasing).

    python3 -I explore/sharper_threshold/realdata_apply.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import common as C
import thresholds as T

DELTA = 0.05
LABEL = {"rhc": "RHC, treated vs control", "nhanes_child": "NHANES, age 6-11 vs 60-79",
         "nhanes_age": "NHANES, age 20-39 vs 60-79", "nhanes_bmi": "NHANES, BMI<25 vs >=30"}


def main():
    paths = pd.read_csv(C.ROOT / "results" / "tables" / "r1_paths.csv")
    summ = pd.read_csv(C.ROOT / "results" / "tables" / "r1_summary.csv").set_index("dataset")
    G = T.GAUSS
    fns = {
        "old": lambda n, m, a: T.tau_old(n, m, a, DELTA),
        "A": lambda n, m, a: T.tau_A(n, m, a, DELTA, **G),
        "P_closed": lambda n, m, a: T.tau_P_closed(n, m, a, DELTA, **G),
        "P": lambda n, m, a: T.tau_P(n, m, a, DELTA, **G),
        "P_coupled": lambda n, m, a: T.tau_P_coupled(n, m, a, DELTA, kappa=1.0),
    }
    rows, detail = [], []
    for ds, g in paths.groupby("dataset", sort=False):
        n, m = int(summ.loc[ds, "n"]), int(summ.loc[ds, "m"])
        al = g.alpha.to_numpy()
        d = g.d_hat.to_numpy()
        conv = g.converged.to_numpy()
        row = dict(dataset=ds, label=LABEL[ds], n=n, m=m, D0=float(d[0]))
        for name, fn in fns.items():
            thr = np.array([fn(n, m, a) for a in al])
            assert np.allclose(thr[0:1], thr[0:1])
            cross = np.nonzero(d <= thr)[0]
            ahat = float(al[cross[0]]) if cross.size else float(al[-1])
            rej = np.nonzero(d > thr)[0]
            last = float(al[rej].max()) if rej.size else None
            row[f"tau0_{name}"] = float(thr[0])
            row[f"alpha_hat_{name}"] = ahat
            row[f"alpha_last_{name}"] = last
            row[f"conv_at_last_{name}"] = (bool(conv[rej].all()) if rej.size else None)
            for a, dd, t, c in zip(al, d, thr, conv):
                detail.append(dict(dataset=ds, threshold=name, alpha=a, d_hat=dd,
                                   tau=t, rejects=bool(dd > t), converged=bool(c)))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(C.OUT / "realdata_summary.csv", index=False)
    pd.DataFrame(detail).to_csv(C.OUT / "realdata_detail.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(df.T)


if __name__ == "__main__":
    main()

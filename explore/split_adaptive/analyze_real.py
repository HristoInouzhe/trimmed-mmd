"""Summaries of out/real.jsonl: certified lower bounds per method (min over splits)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
METHODS = ["boot", "ramp_bet", "dir0_ramp_bet", "ramp_b2s", "ramp_mp_pair", "ramp_mpu", "ramp_hoef", "prop17", "tlm", "ramp_clt"]
ORDER = ["rhc", "nhanes_child", "nhanes_age", "nhanes_bmi", "adult", "acs"]


def main():
    recs = [json.loads(l) for l in (HERE / "out" / "real.jsonl").read_text().splitlines()]
    rows = []
    for r in recs:
        for m in METHODS:
            rows.append({"dataset": r["dataset"], "split": r["split"], "method": m,
                         "lower": r["lower"].get(m, np.nan), "n": r["n"], "m": r["m"]})
    df = pd.DataFrame(rows)
    summ = (df.groupby(["dataset", "method"])["lower"]
              .agg(["min", "median", "max", "count"]).reset_index())
    summ["dataset"] = pd.Categorical(summ.dataset, ORDER, ordered=True)
    summ["method"] = pd.Categorical(summ.method, METHODS, ordered=True)
    summ = summ.sort_values(["dataset", "method"])
    summ.to_csv(HERE / "out" / "real_summary.csv", index=False)
    wide = summ.pivot_table(index="dataset", columns="method", values="min", observed=True)
    wide.to_csv(HERE / "out" / "real_min.csv")
    print(wide.round(3).to_string())
    print()
    print(summ.pivot_table(index="dataset", columns="method", values="median", observed=True)
          .round(3).to_string())
    # reproduction check against the published bootstrap numbers
    pub = pd.read_csv(ROOT / "results" / "tables" / "r2_lower_bounds.csv").set_index("dataset")
    print("\npublished (r2_lower_bounds.csv, 8 splits): min / median / max")
    for d in pub.index:
        print(f"  {d}: {pub.loc[d,'min_lower']:.2f} / {pub.loc[d,'median_lower']:.2f} / {pub.loc[d,'max_lower']:.2f}")
    for nm in ("adult", "acs"):
        f = ROOT / "results" / "fair" / f"{nm}_split_cert.csv"
        if f.exists():
            p = pd.read_csv(f)
            p = p[p.method == "original"]
            print(f"  {nm} (fair_split_certificate, {len(p)} splits): "
                  f"{p.certified_level.min():.2f} / {p.certified_level.median():.2f} / "
                  f"{p.certified_level.max():.2f}; per split {p.sort_values('split').certified_level.tolist()}")
    # per-split detail for boot vs ramp_bet
    det = df[df.method.isin(["boot", "ramp_bet", "ramp_b2s"])].pivot_table(
        index=["dataset", "split"], columns="method", values="lower", observed=True)
    print(det.to_string())


if __name__ == "__main__":
    main()

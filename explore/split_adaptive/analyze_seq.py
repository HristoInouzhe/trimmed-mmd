"""Rates from out/seq_<scenario>.jsonl, joined with the single-split ramp_bet of synth.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def main(scen="disjoint_d2_a20"):
    recs = [json.loads(l) for l in (HERE / "out" / f"seq_{scen}.jsonl").read_text().splitlines()]
    rows = []
    for r in recs:
        for k, (rej, logk) in r["res"].items():
            name, a = k.split("@")
            rows.append({"n": r["n"], "rep": r["rep"], "alpha": float(a), "scheme": name, "rej": rej})
    df = pd.DataFrame(rows)
    w = df.pivot_table(index=["alpha", "n"], columns="scheme", values="rej", aggfunc="mean").reset_index()
    cnt = df.groupby(["alpha", "n"]).rep.nunique().rename("reps").reset_index()
    w = w.merge(cnt, on=["alpha", "n"])
    f = HERE / "out" / f"rates_{scen}_all.csv"
    if f.exists():
        r = pd.read_csv(f)[["alpha", "n", "ramp_bet", "dir0_ramp_bet", "boot"]]
        w = w.merge(r, on=["alpha", "n"], how="left")
    w.to_csv(HERE / "out" / f"seq_rates_{scen}.csv", index=False)
    print(w.round(3).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:])

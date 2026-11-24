"""Cut-point-rule sensitivity on RHC/NHANES from out/real2/*.npz (exploratory)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RULES = ["A_selfexcl", "B_insample", "C_selfexcl_swap"]
LAB = {"A_selfexcl": "A: self-excluded quantiles (pre-specified)", "B_insample": "B: in-sample quantiles",
       "C_selfexcl_swap": "C: A, swapped when crossed"}


def main():
    rows = []
    for f in sorted((HERE / "out" / "real2").glob("*.npz")):
        name, split = f.stem.rsplit("_", 1)
        z = np.load(f)
        levels = sorted({float(k.split("_")[0]) for k in z.files if k.endswith("_res")})
        best = {r: 0.0 for r in RULES}
        alive = set(RULES)
        for a in levels:
            res = z[f"{a:.3f}_res"]
            for j, r in enumerate(RULES):
                if r in alive and res[j]:
                    best[r] = a
                elif r in alive:
                    alive.discard(r)
        for r in RULES:
            rows.append({"dataset": name, "split": int(split), "rule": r, "lower": best[r]})
    df = pd.DataFrame(rows)
    if df.empty:
        print("no results")
        return None
    s = df.groupby(["dataset", "rule"]).lower.agg(["min", "median", "max", "count"]).reset_index()
    s.to_csv(HERE / "out" / "real2_summary.csv", index=False)
    print(s.to_string(index=False))
    return s


def table_md():
    s = pd.read_csv(HERE / "out" / "real2_summary.csv")
    names = {"rhc": "RHC", "nhanes_child": "NHANES 6–11 vs 60–79", "nhanes_age": "NHANES 20–39 vs 60–79",
             "nhanes_bmi": "NHANES BMI"}
    lines = ["| contrast | " + " | ".join(LAB[r] for r in RULES) + " |", "|---|" + "---|" * len(RULES)]
    for d, lab in names.items():
        cells = []
        for r in RULES:
            x = s[(s.dataset == d) & (s.rule == r)]
            cells.append("" if x.empty else f"{x['min'].iloc[0]:.2f} [{x['median'].iloc[0]:.2f}, "
                                              f"{x['max'].iloc[0]:.2f}] ({int(x['count'].iloc[0])})")
        lines.append(f"| {lab} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    if main() is not None:
        (HERE / "tables" / "real2.md").write_text(table_md() + "\n")

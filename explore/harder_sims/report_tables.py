"""Generate the report's tables (markdown) from out/summary_long.csv,
out/runtime.csv, out/wtl.csv and the post hoc files. Writes
report_text/5x_*.md (numbers are never hand-typed into REPORT.md).

    python3 -I explore/harder_sims/report_tables.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze as A  # noqa: E402

OUT = HERE / "out"
RT = HERE / "report_text"
HEAD = ["an_old", "an_new", "prop17", "boot", "ramp_b2s", "ramp_bet", "seq_bet", "gbm_hoef", "gbm_bet",
        "logit_bet", "knn_bet"]
ORDER = list(A.TITLES)


def f2(v):
    return "–" if pd.isna(v) else (f"{v:.2f}" if v < 0.995 else "1")


def table(rows, cols):
    s = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows:
        s += "| " + " | ".join(str(x) for x in r) + " |\n"
    return s


def main():
    df = pd.read_csv(OUT / "summary_long.csv")
    full = df[df.R >= A.REPS]
    pre = full[~full.design.str.startswith("N0")]
    designs = [d for d in ORDER if d in set(pre.design)]
    soft = A.load_posthoc()
    md = {}

    # ---------------- validity
    rows = []
    for m in A.METHODS:
        s = pre[pre.method == m]
        if s.empty:
            continue
        worst = s.sort_values(["typeI", "n"], ascending=[False, True]).iloc[0]
        k_tot, R_tot = int(s.typeI_k.sum()), int(s.R.sum())
        rows.append([m, len(s), int((s.typeI > 0.05).sum()), int((s.typeI_k >= 19).sum()),
                     f"{int(worst.typeI_k)}/{int(worst.R)} = {s.typeI.max():.3f} ({worst.design}, n={int(worst.n)})",
                     f"{worst.typeI_cp:.3f}", f"{k_tot}/{R_tot} = {k_tot / R_tot:.4f}", f"{A.cp_upper(k_tot, R_tot):.4f}",
                     f"{s.alo_ge.max():.3f}", f"{s.alo_gt.max():.3f}"])
    md["51_validity"] = ("**Boundary type-I, pre-registered designs** (each cell: one design × n, 200 "
                         "replicates; a cell is flagged when k ≥ 19 of 200, binomial p < 0.01 against δ = 0.05).\n\n"
                         + table(rows, ["method", "cells", "cells > 0.05", "flagged", "worst cell k/R (where)",
                                        "CP95 upper, worst cell", "pooled k/R", "CP95 upper, pooled",
                                        "max P(α̂lo ≥ TV)", "max P(α̂lo > TV)"]))
    # per-design boundary rows with Clopper-Pearson limits (largest n of each design)
    VM = ["an_new", "prop17", "boot", "ramp_b2s", "ramp_bet", "seq_bet", "gbm_hoef", "gbm_bet", "logit_bet", "knn_bet"]
    rows = []
    for d in designs:
        s = pre[pre.design == d]
        nmax = int(s.n.max())
        t = s[s.n == nmax].set_index("method")
        rows.append([d, nmax] + [f"{int(t.loc[m, 'typeI_k'])} ({t.loc[m, 'typeI_cp']:.3f})" for m in VM])
    md["51_validity"] += ("\n**Boundary rejections at the largest n of each design: k of 200 (Clopper–Pearson 95% "
                          "upper limit).** 0/200 has upper limit 0.018. Pooled pre-registered boundary cells "
                          "and all n are in the table above.\n\n" + table(rows, ["design", "n"] + VM))
    n0 = full[full.design.str.startswith("N0")]
    if len(n0):
        rows = []
        nd = [d for d in sorted(n0.design.unique())]
        for m in A.METHODS:
            r = [m]
            for d in nd:
                s = n0[(n0.design == d) & (n0.method == m)].sort_values("n")
                r.append(" / ".join(f"{v:.3f}" for v in s.typeI) if len(s) else "–")
            rows.append(r)
        ns_ = {d: "/".join(str(int(x)) for x in sorted(n0[n0.design == d].n.unique())) for d in nd}
        md["52_nulls"] = ("**Post hoc P = Q nulls (TV = 0, tight for every [0,1]-valued g): rejection rate of "
                          "H_{0,0}**, by n (" + "; ".join(f"{d}: n = {ns_[d]}" for d in nd) + ").\n\n"
                          + table(rows, ["method"] + nd))
        # soft classifier on nulls
        if soft:
            rows = []
            for c in ("gbm", "logit", "knn"):
                r = [f"{c}_soft_bet (post hoc)"]
                for d in nd:
                    vals = []
                    for n in sorted(n0[n0.design == d].n.unique()):
                        ks = [k for k in soft if k[0] == d and k[1] == n and k[2] < A.REPS]
                        if len(ks) >= A.REPS:
                            vals.append(f"{np.mean([soft[k][f'{c}_soft_bet'][0] for k in ks]):.3f}")
                    r.append(" / ".join(vals) if vals else "–")
                rows.append(r)
            md["52_nulls"] += "\n" + table(rows, ["method"] + nd)

    # ---------------- n80 tables
    for col, lab, key in (("pow0", "H_{0,0}", "53_n80_0"), ("pow4", "H_{0,α*/2}", "54_n80_half")):
        rows = []
        for d in designs:
            rows.append([d] + [A.n80(df[~df.design.str.startswith("N0")], d, m, col) for m in HEAD])
        md[key] = (f"**Smallest n (= m; D5b: n with m = 5n) with power ≥ 0.8 against {lab}** "
                   "(grid 100, 200, 400, 800, and 1600 where the tier-2 cell finished; '>N' = not reached).\n\n"
                   + table(rows, ["design"] + HEAD))

    # ---------------- power at largest tier-1 n
    rows = []
    for d in designs:
        s = pre[(pre.design == d) & (pre.n <= 800)]
        nmax = int(s.n.max())
        t = s[s.n == nmax].set_index("method")
        for col, lab in (("pow0", "0"), ("pow4", "α*/2"), ("pow6", "0.75α*")):
            rows.append([d if lab == "0" else "", nmax if lab == "0" else "", lab] + [f2(t.loc[m, col]) if m in t.index else "–" for m in HEAD])
    md["55_power"] = ("**Power at the largest tier-1 n** (rejection rates of H_{0,α} at α = 0, α*/2, 0.75α*; "
                      "MC s.e. ≤ 0.035).\n\n" + table(rows, ["design", "n", "α"] + HEAD))

    # ---------------- alpha_lo
    rows = []
    for d in designs:
        s = pre[(pre.design == d) & (pre.n <= 800)]
        nmax = int(s.n.max())
        t = s[s.n == nmax].set_index("method")
        a = t.alpha_star.iloc[0]
        rows.append([d, f"{a:.3g}", nmax] + [f"{t.loc[m, 'alo_mean']:.3f}" if m in t.index else "–" for m in HEAD])
    md["56_alo"] = ("**Mean lower confidence bound α̂_lo at the largest tier-1 n** (down-closed rule on the grid "
                    "jα*/8; 0 when H_{0,0} is not rejected).\n\n" + table(rows, ["design", "TV", "n"] + HEAD))

    # ---------------- tier 2
    t2 = pre[pre.n == 1600]
    if len(t2):
        rows = []
        for d in [x for x in ORDER if x in set(t2.design)]:
            t = t2[t2.design == d].set_index("method")
            for col, lab in (("pow0", "0"), ("pow4", "α*/2"), ("pow6", "0.75α*"), ("typeI", "α* (type-I)")):
                rows.append([d if lab == "0" else "", lab] + [f2(t.loc[m, col]) if m in t.index else "–" for m in HEAD])
        md["57_tier2"] = ("**Tier 2, n = m = 1600** (200 replicates per cell; cells in pre-registered order, only "
                          "completed cells shown).\n\n" + table(rows, ["design", "α"] + HEAD))

    # ---------------- paired W/T/L
    if (OUT / "wtl.csv").exists():
        w = pd.read_csv(OUT / "wtl.csv")
        rows = [[f"{r.a} vs {r.b}", r.informative, r.win, r.tie, r.loss, f"{r.mean_diff:+.3f}",
                 f"{r.min_diff:+.3f} … {r.max_diff:+.3f}"] for r in w.itertuples()]
        md["55a_wtl"] = ("**Paired comparisons on the same replicates.** A cell is one design × n × level "
                        "(α = 0, α*/2, 0.75α*) on the tier-1 grid (n ≤ 800, pre-registered designs). It counts as informative if either method rejects in ≥ 10%. "
                        "Win/loss means the paired difference exceeds 2 paired s.e.\n\n"
                        + table(rows, ["a vs b", "informative cells", "a wins", "tie", "a loses",
                                       "mean diff", "range"]))

    # ---------------- post hoc soft (power)
    if soft:
        rows = []
        for d in designs:
            s = pre[(pre.design == d) & (pre.n <= 800)]
            nmax = int(s.n.max())
            t = s[s.n == nmax].set_index("method")
            r = [d, nmax]
            for c in ("gbm", "logit", "knn"):
                ks = [k for k in soft if k[0] == d and k[1] == nmax and k[2] < A.REPS]
                sp = np.mean([soft[k][f"{c}_soft_bet"][0] for k in ks]) if len(ks) >= A.REPS else np.nan
                r += [f2(t.loc[f"{c}_bet", "pow0"]), f2(sp)]
            r += [f2(t.loc["ramp_bet", "pow0"])]
            rows.append(r)
        md["59_soft"] = ("**Post hoc: hard-label vs soft-score classifier betting, power against H_{0,0} at the "
                         "largest tier-1 n** (same replicates, same betting test).\n\n"
                         + table(rows, ["design", "n", "gbm hard", "gbm soft", "logit hard", "logit soft",
                                        "knn hard", "knn soft", "ramp_bet"]))

    # ---------------- runtime
    tm = pd.read_csv(OUT / "runtime.csv")
    tm = tm[(tm.R >= A.REPS) & (~tm.design.str.startswith("N0"))]
    rows = []
    for n in sorted(tm.n.unique()):
        s = tm[tm.n == n]
        rows.append([n, len(s)] + [f"{s[c].mean():.2f}" for c in ("analytic", "split_kernel", "boot_extra",
                                                                  "seq_extra", "gbm", "logit", "knn", "total")]
                    + [f"{s.an_iters.mean():.0f}", int(s.an_undecided.sum())])
    md["60_runtime"] = ("**Mean runtime per replicate (s, one core), averaged over designs.** 'analytic' covers all "
                        "three thresholds (one certified QP path) plus the distance matrix. 'split kernel' is the "
                        "fold-one witness fits shared by prop17, boot, ramp_b2s and ramp_bet over the levels "
                        "visited. 'boot' and 'seq' are the extra cost of the bootstrap and of the sequential "
                        "variant.\n\n"
                        + table(rows, ["n", "designs", "analytic", "split kernel", "boot", "seq", "gbm", "logit",
                                       "knn", "total", "QP iters", "undecided QPs"]))
    for k, v in md.items():
        (RT / f"{k}.md").write_text(v)
    print("wrote", sorted(md))


if __name__ == "__main__":
    main()

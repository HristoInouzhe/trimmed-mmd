#!/usr/bin/env python3
"""Copy the summary-level results of the harder synthetic study into results/harder/.

The study itself (pre-registered designs D1-D8, 14 tests on the same data, 200 replicates per cell) was run in
explore/harder_sims/ (run.py, posthoc_soft.py, validity_logit.py, popmmd.py; summaries by analyze.py). This script
does no Monte Carlo work. It copies the summary files the paper uses and condenses two raw sources into small CSVs:

* soft_summary.csv  <- explore/harder_sims/out/posthoc_soft/*.jsonl (post hoc soft-score classifier bets: rejection
                       rates at the grid levels j = 0, 4, 6, 8 per design, n and method; hard-label reproduction count)
* thresholds.csv    <- explore/harder_sims/out/thr_cache.json (r_n+r_m, the closed form of Theorem thm:plugin and
                       tau_{n,m}, at n = m and the levels used)

The post hoc alpha = 0 extension (DESIGN.md section 9, 2026-11-10; alpha0.py, ladder_alpha0.py) is summarized by
`analyze.py ext` into out/summary_n0ext.csv and out/runtime_n0ext.csv, copied here with its ladder state
(ladder_n0ext.json), its reproduction check on stored replicates (alpha0_check.json) and the gradient-boosting
early-stopping diagnostic on the correlation flip (gbm_earlystop.py -> gbm_earlystop.json).

Usage (repository root):
    python3 -I explore/harder_sims/analyze.py ext     # only after new extension replicates
    python3 scripts/harder_collect.py
"""
from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "explore" / "harder_sims"
OUT = ROOT / "results" / "harder"

COPY = ["summary_long.csv", "paired.csv", "wtl.csv", "n0_pooled.txt", "validity_logit.json", "popmmd.csv",
        "runtime.csv", "checks.txt", "prereg_sha256.txt", "pilot.json",
        "summary_n0ext.csv", "runtime_n0ext.csv", "ladder_n0ext.json", "alpha0_check.json",  # alpha = 0 extension
        "gbm_earlystop.json"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in COPY:
        shutil.copy2(SRC / "out" / f, OUT / f)
    shutil.copy2(SRC / "DESIGN.md", OUT / "DESIGN.md")

    # soft-score classifier bets (post hoc), condensed per design x n x method
    acc = defaultdict(lambda: [0, 0, 0, 0, 0])  # rejections at j = 0, 4, 6, 8, replicates
    match = defaultdict(lambda: [0, 0])          # hard-label decisions reproduced, replicates
    for f in sorted((SRC / "out" / "posthoc_soft").glob("*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r["rep"] >= 200:  # main replicates only, as in the pre-registered tables
                continue
            for meth, dec in r["dec"].items():
                a = acc[(r["design"], r["n"], r["m"], meth)]
                for i in range(4):
                    a[i] += int(dec[i])
                a[4] += 1
            for clf, ok in r["hard_match"].items():
                mm = match[(r["design"], r["n"], clf)]
                mm[0] += int(bool(ok))
                mm[1] += 1
    with open(OUT / "soft_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["design", "n", "m", "method", "R", "k0", "k4", "k6", "k8", "pow0", "pow4", "pow6", "typeI",
                    "hard_match", "hard_R"])
        for (d, n, m, meth), a in sorted(acc.items()):
            R = a[4]
            mm = match[(d, n, meth.split("_")[0])]
            w.writerow([d, n, m, meth, R, *a[:4], *(f"{x / R:.4f}" for x in a[:4]), *mm])

    # thresholds at n = m
    thr = json.loads((SRC / "out" / "thr_cache.json").read_text())
    with open(OUT / "thresholds.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["n", "m", "alpha", "tau_rnrm", "tau_plugin_closed", "tau_nm"])
        for key, v in sorted(thr.items(), key=lambda kv: tuple(float(x) for x in kv[0].split(","))):
            n, m, a = key.split(",")
            w.writerow([n, m, a, f"{v['old']:.6f}", f"{v['newcf']:.6f}", f"{v['new']:.6f}"])
    print(f"wrote {OUT.relative_to(ROOT)}: " + ", ".join(sorted(p.name for p in OUT.iterdir())))


if __name__ == "__main__":
    main()

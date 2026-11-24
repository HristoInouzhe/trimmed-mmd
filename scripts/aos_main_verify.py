"""AoS restructure: checks of the new package options (results/aos/main/NOTES.md). Writes results/aos/main/verify.json.

1. ramp_cut_points(rule="linear") (the default) is numpy.quantile and fold1.cut_points bit for bit; rule="order" returns
   the order statistics u_(k_s), v_(k_u) of splitting.cut_ranks.
2. eq:cutrank (aos/supplement/S3_split.tex): iota^-(n, 1-alpha) <= k_s <= iota^+(n, 1-alpha) and
   iota^-(m, alpha) <= k_u <= iota^+(m, alpha), iota^-(n, p) = max(1, ceil(np) - 1), iota^+(n, p) = min(n, floor(np) + 2),
   for every fit size and level of the runs (synthetic: batch 0 (q_n, q_m) and fold one (n1, m1) at every stored size and
   level j alpha*/8; real data: the fit sizes of both batches at every grid level), evaluated in exact rational
   arithmetic both at the float level and at the decimal level j alpha*/8 it stands for.
3. The default cut rule of the package reproduces stored decisions: splitting.sequential_certificate (no cut_rule given)
   on real-data cells against results/aos/realdata/wealth (certified, log K_N, skipped batches).
4. tau_aos = min(tau_plugin_closed, tau_A); on the synthetic study's cells it equals the stored closed-form threshold
   (thr_cache "newcf").

    PYTHONPATH=src python -B scripts/aos_main_verify.py
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"

import json  # noqa: E402
import math  # noqa: E402
from fractions import Fraction  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

import aos_baselines_run as B  # noqa: E402
from fold1 import cut_points as fold1_cut_points  # noqa: E402
from trimmedmmd import threshold as TH  # noqa: E402
from trimmedmmd.experiments.realdata_run import ALPHA_GRID  # noqa: E402
from trimmedmmd.splitting import (SEQ_EDGES, cut_ranks, ramp_cut_points, sequential_certificate,  # noqa: E402
                                  sequential_order)

OUT = ROOT / "results" / "aos" / "main" / "verify.json"


def iota(n, p):
    """(iota^-, iota^+) in exact arithmetic; p a Fraction."""
    x = n * p
    fl = x.numerator // x.denominator
    ce = -((-x.numerator) // x.denominator)
    return max(1, ce - 1), min(n, fl + 2)


def cutrank_ok(n, m, a_float, a_exact):
    k_s, k_u = cut_ranks(n, m, a_float)
    res = []
    for a in (Fraction(a_float), a_exact):
        lo_s, hi_s = iota(n, 1 - a)
        lo_u, hi_u = iota(m, a)
        res.append(lo_s <= k_s <= hi_s and lo_u <= k_u <= hi_u)
    return all(res), (k_s, k_u)


def main():
    out = {}
    rng = np.random.default_rng(20261121)
    # 1. default = numpy.quantile = fold1.cut_points; order = order statistics
    bad_lin = bad_ord = 0
    for t in range(2000):
        n, m = int(rng.integers(1, 400)), int(rng.integers(1, 400))
        u, v = rng.standard_normal(n), rng.standard_normal(m)
        if t % 3 == 0:                              # ties
            u, v = np.round(u, 1), np.round(v, 1)
        a = float(rng.choice([0.0, rng.uniform(0, 0.95), round(rng.uniform(0, 0.95), 2)]))
        d = ramp_cut_points(u, v, a)
        if d != (float(np.quantile(u, 1 - a)), float(np.quantile(v, a))) or d != fold1_cut_points(u, v, a):
            bad_lin += 1
        k_s, k_u = cut_ranks(n, m, a)
        if ramp_cut_points(u, v, a, rule="order") != (float(np.sort(u)[k_s - 1]), float(np.sort(v)[k_u - 1])):
            bad_ord += 1
        if a == 0.0 and ramp_cut_points(u, v, a, rule="order") != d:
            bad_ord += 1
    out["default_equals_numpy_quantile_and_fold1"] = {"trials": 2000, "mismatches": bad_lin}
    out["order_rule_is_order_statistic"] = {"trials": 2000, "mismatches": bad_ord}

    # 2. eq:cutrank over every fit size and level used
    C = B.catalogue()
    cells = set()
    for key in B.DESIGNS + B.NULLS:
        dz = C[key]
        ast = dz.alpha_star
        ns = set(dz.ns)
        f = B.STORED / f"{key}_n0ext.jsonl"
        if f.exists():
            ns |= {json.loads(line)["n"] for line in f.read_text().splitlines() if line.strip()}
        for n in ns:
            m = dz.m_ratio * n
            n1, m1 = n // 2, m // 2
            q_n, q_m = int(round(n / 4)), int(round(m / 4))
            ast_exact = Fraction(repr(round(ast, 10))) if ast > 0 else Fraction(0)
            for j, a in enumerate(dz.levels()):
                for nf, mf in ((q_n, q_m), (n1, m1)):
                    cells.add((nf, mf, float(a), Fraction(j, 8) * ast_exact))
    rd = {"rhc": (2184, 3551), "nhanes_child": (1969, 2300), "nhanes_age": (2519, 2300), "nhanes_bmi": (1627, 2008)}
    for name, (n, m) in rd.items():
        grid = np.round(np.arange(0.0, 0.95 + 1e-9, 0.05), 3) if name == "nhanes_child" else ALPHA_GRID
        bx = [int(round(e * n)) for e in SEQ_EDGES]
        by = [int(round(e * m)) for e in SEQ_EDGES]
        for a in grid:
            for b in (1, 2):
                cells.add((bx[b], by[b], float(a), Fraction(repr(round(float(a), 10)))))
    fails = []
    for nf, mf, af, ae in sorted(cells):
        ok, ks = cutrank_ok(nf, mf, af, ae)
        if not ok:
            fails.append([nf, mf, af, str(ae), ks])
    out["eq_cutrank"] = {"cells": len(cells), "failures": fails[:20], "n_failures": len(fails)}
    ex = {}
    for nf, mf, a in ((200, 200, 0.025), (200, 200, 0.2), (400, 400, 0.05), (100, 500, 0.15), (1092, 1776, 0.3)):
        ex[f"n={nf},m={mf},alpha={a}"] = {"k_s": cut_ranks(nf, mf, a)[0], "k_u": cut_ranks(nf, mf, a)[1],
                                          "iota_s": iota(nf, 1 - Fraction(repr(a))),
                                          "iota_u": iota(mf, Fraction(repr(a)))}
    out["eq_cutrank_examples"] = ex

    # 3. package default reproduces stored real-data decisions
    from trimmedmmd.experiments.realdata_run import _dataset
    from trimmedmmd.realdata import load_rhc
    stored = {}
    for name in ("rhc", "nhanes_bmi", "nhanes_age"):
        for line in (ROOT / "results" / "aos" / "realdata" / "wealth" / f"{name}.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                stored[(name, r["seed"], round(r["alpha"], 6))] = r
    checks = []
    for name, seed, a in (("rhc", 1000, 0.30), ("rhc", 1001, 0.35), ("nhanes_bmi", 1000, 0.05),
                          ("nhanes_bmi", 1005, 0.10), ("nhanes_age", 1002, 0.35)):
        if name == "rhc":
            X, Y, _, _ = load_rhc(standardize=False)
            std0 = True
        else:
            X, Y = _dataset(name)
            std0 = False
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        split_seed = int(np.random.default_rng(seed).integers(0, 2**31 - 1))
        order = sequential_order(X, Y, np.random.default_rng(split_seed), edges=SEQ_EDGES, standardize_batch0=std0)
        r = sequential_certificate(X, Y, a, order=order)
        s = stored.get((name, seed, round(a, 6)))
        checks.append({"cell": [name, seed, a], "certified": r.certified, "stored_certified": s["certified"],
                       "log_final": r.log_final_wealth, "stored_log_final": s["log_final"],
                       "skipped": r.skipped_batches, "stored_skipped": s["skipped"],
                       "agree": bool(r.certified == s["certified"] and list(r.skipped_batches) == s["skipped"]
                                     and (s["pairs_bet"] == 0 or abs(r.log_final_wealth - s["log_final"]) < 0.02))})
    out["package_default_vs_stored_realdata"] = checks

    # 4. thresholds
    thr = json.loads(B.THR.read_text())
    md = 0.0
    nA = 0
    for k, v in thr.items():
        n, m, a = k.split(",")
        n, m, a = int(n), int(m), float(a)
        t = TH.tau_aos(n, m, a, 0.05)
        assert t == min(TH.tau_plugin_closed(n, m, a, 0.05), TH.tau_A(n, m, a, 0.05))
        md = max(md, abs(t - v["newcf"]) / v["newcf"])
        nA += int(TH.tau_A(n, m, a, 0.05) < TH.tau_plugin_closed(n, m, a, 0.05))
    out["tau_aos_vs_stored_closed_form"] = {"cells": len(thr), "max_rel_diff": md, "cells_tauA_below_tauP": nA}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=str))
    ok = (bad_lin == 0 and bad_ord == 0 and not fails and all(c["agree"] for c in checks) and md < 1e-9)
    print(json.dumps(out, indent=1, default=str)[:3000])
    print("ALL CHECKS PASSED" if ok else "SOME CHECK FAILED")


if __name__ == "__main__":
    main()

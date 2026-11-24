"""AoS revision: summaries, LaTeX macros and the cohort balance table from the real-data recomputation.

Reads
    results/aos/realdata/wealth/<contrast>.jsonl     (scripts/aos_realdata_wealth.py)
    results/aos/realdata/cohort/<label>.json/.npz    (scripts/aos_realdata_cohort.py)
and the published per-split outputs it is checked against
    results/v3/rhc_lower_batch0.json                 (RHC, batch-0 standardization)
    results/v2/checkpoints/r_lower_v2__*.jsonl       (NHANES, the run behind Table tab:realdata)
Writes
    results/aos/realdata/wealth_summary.json, wealth_levels.csv, cohort_summary.json, cohort_smd.csv
    results/aos/tables/aos_realdata_macros.tex       (\\newcommand lines; every number of REPORT.md)
    results/aos/tables/aos_cohort_balance.tex        (tabular), aos_realdata_evalues.tex (tabular)
    results/aos/realdata/REPORT.md                   (via scripts/aos_realdata_report.py)

    PYTHONPATH=src python3 -I scripts/aos_realdata_tables.py
"""
from __future__ import annotations

import csv
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

RD = ROOT / "results" / "aos" / "realdata"
TAB = ROOT / "results" / "aos" / "tables"
DELTA = 0.05
SEEDS = [1000 + i for i in range(8)]
CONTRASTS = ["rhc", "nhanes_child", "nhanes_age", "nhanes_bmi"]
TAG = {"rhc": "Rhc", "nhanes_child": "Child", "nhanes_age": "Age", "nhanes_bmi": "Bmi"}
LABEL = {"rhc": "RHC arms", "nhanes_child": r"ages 6--11 vs.\ 60--79", "nhanes_age": r"ages 20--39 vs.\ 60--79",
         "nhanes_bmi": r"BMI ${<}25$ vs.\ ${\ge}30$"}


# ----------------------------------------------------------------------------------------------------------------
# wealth
# ----------------------------------------------------------------------------------------------------------------

def load_wealth(name):
    recs = {}
    for line in (RD / "wealth" / f"{name}.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            recs[(r["seed"], round(r["alpha"], 6))] = r
    return recs


def published(name):
    """seed -> {alpha: (certified, log_final)} from the published runs."""
    out = {}
    if name == "rhc":
        d = json.loads((ROOT / "results" / "v3" / "rhc_lower_batch0.json").read_text())
        for s in d["splits"]:
            out[s["seed"]] = {round(l["alpha"], 6): (bool(l["sequential"]), float(l["seq_log_final_wealth"]))
                              for l in s["levels"]}
        return out, {s["seed"]: float(s["lower"]) for s in d["splits"]}
    for f in glob.glob(str(ROOT / "results" / "v2" / "checkpoints" / "r_lower_v2__*.jsonl")):
        meta = json.loads(Path(f.replace(".jsonl", ".meta.json")).read_text())
        if meta["dataset"] != name:
            continue
        lows = {}
        for line in Path(f).read_text().splitlines():
            r = json.loads(line)
            val = r.get("value", r)
            out[val["seed"]] = {round(l["alpha"], 6): (bool(l["sequential"]), float(l["seq_log_final_wealth"]))
                                for l in val["levels"] if "seq_log_final_wealth" in l}
            lows[val["seed"]] = float(val["lower"])
        return out, lows
    raise FileNotFoundError(name)


def down_closed(alphas, decisions):
    lo = -1.0
    for a, d in zip(alphas, decisions):
        if not d:
            break
        lo = a
    return lo


def wealth_summary(name):
    recs = load_wealth(name)
    alphas = sorted({a for (_, a) in recs})
    alphas = [a for a in alphas if all((s, a) in recs for s in SEEDS)]
    thr = 1.0 / DELTA
    per_split = {s: down_closed(alphas, [recs[(s, a)]["certified"] for a in alphas]) for s in SEEDS}
    lows = np.array([max(per_split[s], 0.0) for s in SEEDS])
    rows = []
    for a in alphas:
        rs = [recs[(s, a)] for s in SEEDS]
        rows.append({"contrast": name, "alpha": a,
                     "n_certified": int(sum(r["certified"] for r in rs)),
                     "n_no_bet": int(sum(r["pairs_bet"] == 0 for r in rs)),
                     "avg_stopped_wealth": float(np.mean([math.exp(r["log_stopped"]) for r in rs])),
                     "avg_final_wealth": float(np.mean([math.exp(r["log_final"]) for r in rs])),
                     "min_stopped_wealth_certified": float(min([math.exp(r["log_stopped"]) for r in rs
                                                                if r["certified"]], default=float("nan"))),
                     "max_stopped_wealth_certified": float(max([math.exp(r["log_stopped"]) for r in rs
                                                                if r["certified"]], default=float("nan"))),
                     "stopped_wealths": [math.exp(r["log_stopped"]) for r in rs],
                     "log_final_wealths": [r["log_final"] for r in rs]})
    avg_stop = down_closed(alphas, [r["avg_stopped_wealth"] >= thr for r in rows])
    avg_final = down_closed(alphas, [r["avg_final_wealth"] >= thr for r in rows])
    # reproduction check against the published per-split outputs
    pub, pub_lows = published(name)
    n_cmp = n_dec = 0
    max_dlog = 0.0
    for s in SEEDS:
        for a, (cert, lf) in pub.get(s, {}).items():
            if (s, a) not in recs:
                continue
            r = recs[(s, a)]
            n_cmp += 1
            n_dec += int(r["certified"] == cert)
            mine = r["package_log_final"]
            if math.isinf(lf) or math.isinf(mine):
                max_dlog = max(max_dlog, 0.0 if (math.isinf(lf) and math.isinf(mine)) else float("inf"))
            else:
                max_dlog = max(max_dlog, abs(mine - lf))
    lows_match = all(abs(per_split[s] - pub_lows[s]) < 1e-9 for s in SEEDS)

    def row_at(a):
        return next((r for r in rows if abs(r["alpha"] - a) < 1e-9), None)

    nxt_stop = next((r for r in rows if r["alpha"] > avg_stop + 1e-9), None)
    return {"contrast": name, "grid_computed": alphas, "per_split_lower": per_split,
            "min_lower": float(lows.min()), "median_lower": float(np.median(lows)),
            "min_lower_raw": float(min(per_split.values())),
            "avg_stopped_level": avg_stop, "avg_final_level": avg_final,
            "first_failure_avg_stopped": nxt_stop["alpha"] if nxt_stop else None,
            "avg_stopped_wealth_at_first_failure": nxt_stop["avg_stopped_wealth"] if nxt_stop else None,
            "n_certified_at_first_failure": nxt_stop["n_certified"] if nxt_stop else None,
            "n_no_bet_at_first_failure": nxt_stop["n_no_bet"] if nxt_stop else None,
            "avg_stopped_wealth_at_level": row_at(avg_stop)["avg_stopped_wealth"] if row_at(avg_stop) else None,
            "check": {"compared_split_levels": n_cmp, "same_decision": n_dec, "max_abs_diff_log_final": max_dlog,
                      "per_split_lowers_match_published": bool(lows_match)},
            "rows": rows}


# ----------------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------------

def f2(x):
    return f"{x:.2f}"


def lev(x):
    """A certified level for print: -1 (nothing rejected) as '--'."""
    return "--" if x < 0 else f"{x:.2f}"


def thousands(k):
    return f"{int(k):,}".replace(",", "{,}")


def sci(x, nd=1):
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    mant = x / 10 ** e
    if round(mant, nd) >= 10:
        mant, e = mant / 10, e + 1
    return rf"{mant:.{nd}f}\times10^{{{e}}}"


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    macros = ["% Generated by scripts/aos_realdata_tables.py from results/aos/realdata -- do not edit."]

    def M(name, val):
        if r"\times" in val:                      # scientific notation: usable in text and in math
            val = rf"\ensuremath{{{val}}}"
        macros.append(rf"\newcommand{{\{name}}}{{{val}}}")

    # -------------------------------------------------------------- wealth
    ws = {}
    with open(RD / "wealth_levels.csv", "w", newline="") as fh:
        wr = None
        for nm in CONTRASTS:
            if not (RD / "wealth" / f"{nm}.jsonl").exists():
                continue
            s = wealth_summary(nm)
            ws[nm] = s
            for r in s["rows"]:
                flat = {k: (";".join(f"{x:.6g}" for x in v) if isinstance(v, list) else v) for k, v in r.items()}
                if wr is None:
                    wr = csv.DictWriter(fh, fieldnames=list(flat))
                    wr.writeheader()
                wr.writerow(flat)
    (RD / "wealth_summary.json").write_text(json.dumps(ws, indent=1, default=float))
    M("rdDelta", f2(DELTA))
    M("rdSplits", str(len(SEEDS)))
    M("rdMedianLevel", f2(len(SEEDS) * DELTA / (len(SEEDS) // 2)))      # P(>= 4 of 8 err) <= 8 delta / 4
    M("rdFwerNhanes", f2(3 * DELTA))
    M("rdFwerNhanesMedian", f2(3 * len(SEEDS) * DELTA / (len(SEEDS) // 2)))
    M("rdFwerAll", f2(4 * DELTA))
    for nm, s in ws.items():
        t = TAG[nm]
        M(f"rdMin{t}", lev(s["min_lower_raw"]))
        M(f"rdMed{t}", f2(s["median_lower"]))
        M(f"rdAvgStop{t}", lev(s["avg_stopped_level"]))
        M(f"rdAvgFinal{t}", lev(s["avg_final_level"]))
        if s["first_failure_avg_stopped"] is not None:
            M(f"rdAvgStopNext{t}", f2(s["first_failure_avg_stopped"]))
            M(f"rdAvgStopWealthNext{t}", f"{s['avg_stopped_wealth_at_first_failure']:.1f}")
            M(f"rdCertNext{t}", str(s["n_certified_at_first_failure"]))
            M(f"rdNoBetNext{t}", str(s["n_no_bet_at_first_failure"]))
        M(f"rdCheckLevels{t}", str(s["check"]["compared_split_levels"]))
        M(f"rdCheckSame{t}", str(s["check"]["same_decision"]))
        allw = [w for r in s["rows"] for w in r["stopped_wealths"] if w >= 1 / DELTA]
        if allw:
            M(f"rdStopWealthMax{t}", f"{max(allw):.1f}")
        full = [r for r in s["rows"] if r["n_certified"] == len(SEEDS)]
        if full:
            last = full[-1]
            M(f"rdAllCertLevel{t}", f2(last["alpha"]))
            M(f"rdAllCertLogKmin{t}", f"{min(last['log_final_wealths']):.1f}")
            M(f"rdAllCertLogKmax{t}", f"{max(last['log_final_wealths']):.1f}")
    allw = [w for s in ws.values() for r in s["rows"] for w in r["stopped_wealths"] if w >= 1 / DELTA]
    if allw:
        M("rdStopWealthMin", f"{min(allw):.1f}")
        M("rdStopWealthMax", f"{max(allw):.1f}")
    if "nhanes_bmi" in ws:
        b = ws["nhanes_bmi"]
        r5 = next(r for r in b["rows"] if abs(r["alpha"] - 0.05) < 1e-9)
        M("rdBmiCertFive", str(r5["n_certified"]))
        M("rdBmiAvgStopWealthFive", f"{r5['avg_stopped_wealth']:.1f}")
        M("rdBmiAvgFinalWealthFive", f"{r5['avg_final_wealth']:.1f}")
        fails = [w for w in r5["stopped_wealths"] if w < 1 / DELTA]
        if fails:
            M("rdBmiFailWealthFive", f"{min(fails):.1f}")
        r10 = next((r for r in b["rows"] if abs(r["alpha"] - 0.10) < 1e-9), None)
        if r10:
            M("rdBmiCertTen", str(r10["n_certified"]))
            M("rdBmiAvgStopWealthTen", f"{r10['avg_stopped_wealth']:.1f}")
            M("rdBmiAvgFinalWealthTen", f"{r10['avg_final_wealth']:.1f}")

    # wealth table (tabular, for the report and, if wanted, the supplement)
    lines = [r"% Generated by scripts/aos_realdata_tables.py -- do not edit.",
             r"\begin{tabular}{lccccc}", r"\toprule",
             r"contrast & min & median & average, stopped & average, final & checked \\",
             r" & ($\delta$) & ($2\delta$) & ($\delta$) & ($\delta$) & levels \\", r"\midrule"]
    for nm in CONTRASTS:
        if nm in ws:
            s = ws[nm]
            lines.append(f"{LABEL[nm]} & {lev(s['min_lower_raw'])} & {f2(s['median_lower'])} & "
                         f"{lev(s['avg_stopped_level'])} & {lev(s['avg_final_level'])} & "
                         f"{s['check']['same_decision']}/{s['check']['compared_split_levels']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "aos_realdata_evalues.tex").write_text("\n".join(lines) + "\n")

    # -------------------------------------------------------------- cohort
    coh = {}
    for f in sorted((RD / "cohort").glob("*.json")):
        coh[f.stem] = json.loads(f.read_text())
    csum = write_cohort(coh, M) if coh else {}
    (TAB / "aos_realdata_macros.tex").write_text("\n".join(macros) + "\n")
    from aos_realdata_report import write_report
    write_report(ws, coh, csum, macros)
    print("\n".join(macros))


def write_cohort(coh, M, cdir=None, P="rd", smd_csv=None, summary_json=None, balance_tex=None,
                 generator="scripts/aos_realdata_tables.py"):
    """Cohort summaries, macros (prefix P) and the balance table. The defaults are the stored Gaussian run
    (results/aos/realdata/cohort, prefix rd); scripts/aos_main_cohort.py calls it for the Laplace run (prefix lc)."""
    cdir = RD / "cohort" if cdir is None else Path(cdir)
    smd_csv = RD / "cohort_smd.csv" if smd_csv is None else Path(smd_csv)
    summary_json = RD / "cohort_summary.json" if summary_json is None else Path(summary_json)
    balance_tex = TAB / "aos_cohort_balance.tex" if balance_tex is None else Path(balance_tex)
    from trimmedmmd.realdata import load_rhc
    from v3_rhc_estimand import boot_ci, risk_difference, rhc_outcome
    from trimmedmmd.experiments.realdata_run import _logistic_fit, standardized_differences

    Xraw, Yraw, names, _ = load_rhc(standardize=False)
    n, m = len(Xraw), len(Yraw)
    W = {k: np.load(cdir / f"{k}.npz") for k in coh}
    prim = "batch0_seed1000"
    # Risk-difference intervals with the bootstrap stream of scripts/v3_rhc_estimand.py: one default_rng(20261117),
    # the unweighted arms' 2,000 resamples drawn first, then the trimming's; so the published row reproduces the
    # paper's interval, and every trimming row uses the same resamples (the draws do not depend on the weights).
    yx, yy = rhc_outcome()
    for k in coh:
        rng = np.random.default_rng(20261117)
        boot_ci(yx, yy, np.ones(n), np.ones(m), rng)
        lo_, hi_ = boot_ci(yx, yy, np.asarray(W[k]["w"], float), np.asarray(W[k]["v"], float), rng)
        coh[k]["describe"]["rd_lo"], coh[k]["describe"]["rd_hi"] = lo_, hi_
    summary = {}
    for k, c in coh.items():
        si, st, de = c["solver_info"], c["structure"], c["describe"]
        summary[k] = {"D_hat": si["D_hat"], "D_hat_lower": si["D_hat_lower"], "F": si["F"], "gap": si["gap"],
                      "gap_longdouble": si["gap_longdouble"], "gap_rel": si["gap"] / si["F"],
                      "tau": c["tau_plugin"], "sigma": c["sigma"],
                      **{kk: de[kk] for kk in de if kk != "smd"}, "cut_x": st["x"]["cut"], "cut_y": st["y"]["cut"],
                      "cut_x_normalized": st["x"]["cut_normalized"], "cut_y_normalized": st["y"]["cut_normalized"]}
    # overlaps of the retained sets with the primary cohort
    if prim in W:
        cx, cy = 1 / (n * (1 - coh[prim]["alpha"])), 1 / (m * (1 - coh[prim]["alpha"]))
        kx0 = W[prim]["w"] > 1e-9 * cx
        ky0 = W[prim]["v"] > 1e-9 * cy
        for k in W:
            kx = W[k]["w"] > 1e-9 * cx
            ky = W[k]["v"] > 1e-9 * cy
            summary[k]["jaccard_x_vs_primary"] = float((kx & kx0).sum() / (kx | kx0).sum())
            summary[k]["jaccard_y_vs_primary"] = float((ky & ky0).sum() / (ky | ky0).sum())
            summary[k]["only_here_x"] = int((kx & ~kx0).sum())
            summary[k]["only_primary_x"] = int((~kx & kx0).sum())
            summary[k]["only_here_y"] = int((ky & ~ky0).sum())
            summary[k]["only_primary_y"] = int((~ky & ky0).sum())
            summary[k]["l1_w_vs_primary"] = float(np.abs(W[k]["w"] - W[prim]["w"]).sum())
            summary[k]["l1_v_vs_primary"] = float(np.abs(W[k]["v"] - W[prim]["v"]).sum())
    if "old" in W and "pooled" in W:
        alpha0 = coh["old"]["alpha"]
        cx0, cy0 = 1 / (n * (1 - alpha0)), 1 / (m * (1 - alpha0))
        same = True
        for arm, key, c in (("x", "w", cx0), ("y", "v", cy0)):
            a_, b_ = W["old"][key], W["pooled"][key]
            sc = bool(np.array_equal(a_ >= c * (1 - 1e-9), b_ >= c * (1 - 1e-9))
                      and np.array_equal(a_ <= c * 1e-9, b_ <= c * 1e-9))
            same = same and sc
            summary["old"][f"l1_vs_pooled_{arm}"] = float(np.abs(a_ - b_).sum())
            summary["old"][f"maxdiff_over_cap_vs_pooled_{arm}"] = float(np.abs(a_ - b_).max() / c)
        summary["old"]["same_cap_zero_sets_as_pooled"] = same
        M(f"{P}OldPoolSameSets", "yes" if same else "no")
        M(f"{P}OldPoolLoneX", sci(summary["old"]["l1_vs_pooled_x"]))
        M(f"{P}OldPoolLoneY", sci(summary["old"]["l1_vs_pooled_y"]))
        M(f"{P}OldPoolMaxDiffX", f"{summary['old']['maxdiff_over_cap_vs_pooled_x']:.3f}")
        M(f"{P}OldPoolMaxDiffY", f"{summary['old']['maxdiff_over_cap_vs_pooled_y']:.3f}")
    # per-covariate SMD
    smd0 = standardized_differences(Xraw, Yraw)
    keys = [k for k in ("old", "pooled", prim) if k in coh] + sorted(k for k in coh if k.startswith("batch0_")
                                                                       and k != prim)
    with open(smd_csv, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["covariate", "all_patients"] + keys)
        for i, nmv in enumerate(names):
            wr.writerow([nmv, f"{smd0[i]:.6f}"] + [f"{coh[k]['describe']['smd'][i]:.6f}" for k in keys])

    def tag(k):
        return {"old": "Old", "pooled": "Pool", prim: "Coh"}.get(k)

    alpha = coh[prim]["alpha"] if prim in coh else coh["old"]["alpha"]
    M(f"{P}CohAlpha", f2(alpha))
    M(f"{P}CohNalphaX", thousands(int(np.floor(n * (1 - alpha) + 1e-12))))
    M(f"{P}CohNalphaY", thousands(int(np.floor(m * (1 - alpha) + 1e-12))))
    M(f"{P}CohNalphaPlusOneX", thousands(int(np.floor(n * (1 - alpha) + 1e-12)) + 1))
    M(f"{P}CohNalphaPlusOneY", thousands(int(np.floor(m * (1 - alpha) + 1e-12)) + 1))
    M(f"{P}UnwSmdMax", f"{np.abs(smd0).max():.3f}")
    M(f"{P}UnwSmdMean", f"{np.abs(smd0).mean():.3f}")
    for k in ("old", "pooled", prim):
        if k not in coh:
            continue
        t, s, c = tag(k), summary[k], coh[k]
        M(f"{P}{t}Dhat", f"{s['D_hat']:.4f}")
        M(f"{P}{t}DhatSix", f"{s['D_hat']:.6f}")
        M(f"{P}{t}DhatWidth", sci(max(s['D_hat'] - s['D_hat_lower'], 0.0)))
        M(f"{P}{t}DhatLower", f"{s['D_hat_lower']:.4f}")
        M(f"{P}{t}Gap", sci(max(s["gap"], s["gap_longdouble"])))
        M(f"{P}{t}GapRel", sci(max(s["gap"], s["gap_longdouble"]) / s["F"]))
        M(f"{P}{t}Tau", f"{s['tau']:.3f}")
        M(f"{P}{t}Sigma", f"{s['sigma']:.2f}")
        for arm, A in (("x", "X"), ("y", "Y")):
            M(f"{P}{t}Cap{A}", thousands(s[f"cap_{arm}"]))
            M(f"{P}{t}Zero{A}", thousands(s[f"zero_{arm}"]))
            M(f"{P}{t}Frac{A}", thousands(s[f"frac_{arm}"]))
            M(f"{P}{t}Kept{A}", thousands(s[f"kept_{arm}"]))
            M(f"{P}{t}Ess{A}", f2(s[f"ess_frac_{arm}"]))
            M(f"{P}{t}FracMass{A}", f"{s[f'frac_mass_{arm}']:.3f}")
            if "jaccard_x_vs_primary" in s and k != prim:
                M(f"{P}{t}Jaccard{A}", f"{s[f'jaccard_{arm}_vs_primary']:.3f}")
                M(f"{P}{t}OnlyHere{A}", thousands(s[f"only_here_{arm}"]))
                M(f"{P}{t}OnlyCoh{A}", thousands(s[f"only_primary_{arm}"]))
        M(f"{P}{t}SmdMax", f"{s['max_abs_smd']:.3f}")
        M(f"{P}{t}SmdMean", f"{s['mean_abs_smd']:.3f}")
        M(f"{P}{t}SmdAbove", str(s["n_abs_smd_above_0.1"]))
        M(f"{P}{t}Overlap", f2(s["prop_overlap"]))
        M(f"{P}{t}RD", f"{s['rd_dth30']:.3f}")
        M(f"{P}{t}RDlo", f"{s['rd_lo']:.3f}")
        M(f"{P}{t}RDhi", f"{s['rd_hi']:.3f}")
        if k != "old":
            si = c["solver_info"]
            M(f"{P}{t}SmoSteps", thousands(si["smo"]["steps"]))
            M(f"{P}{t}Seconds", f"{si['seconds_total']:.0f}")
            ce = c.get("certify", {})
            if ce:
                M(f"{P}{t}LamMin", sci(ce["lam_min_gram"]))
                M(f"{P}{t}LtwoBound", sci(ce["l2_bound_to_minimizer"]))
                M(f"{P}{t}WitnessBound", sci(ce["witness_sup_bound"]))
                M(f"{P}{t}CountsExact", "yes" if ce["counts_exact_for_minimizer"] else "no")
            st = c["structure"]
            M(f"{P}{t}CutX", f"{st['x']['cut_normalized']:.4f}")
            M(f"{P}{t}CutY", f"{st['y']['cut_normalized']:.4f}")
            for arm, A in (("x", "X"), ("y", "Y")):
                sp = st[arm].get("frac_witness_spread")
                if sp is not None:
                    M(f"{P}{t}FracSpread{A}", sci(sp / st[arm]["witness_sd"]))
    sens = [k for k in coh if k.startswith("batch0_")]
    for k in sens:                          # units whose cap/zero status the gap does not settle
        c, z = coh[k], W[k]
        ce, a0 = c["certify"], c["alpha"]
        tot = 0
        for arm, u, d, cap, low in (("x", z["w"], z["dx"], 1 / (n * (1 - a0)), True),
                                    ("y", z["v"], z["dy"], 1 / (m * (1 - a0)), False)):
            capm, zero = u >= cap * (1 - 1e-9), u <= cap * 1e-9
            fr = ~capm & ~zero
            t, sg, need = d[fr].mean(), (1.0 if low else -1.0), ce[arm]["window_needed"]
            tot += int(np.sum(sg * (t - d[capm]) <= need) + np.sum(sg * (d[zero] - t) <= need))
        summary[k]["n_units_status_unsettled"] = tot
    if len(sens) > 1:
        M(f"{P}SensUnsettledMax", str(max(summary[k]["n_units_status_unsettled"] for k in sens)))
        kx = [summary[k]["kept_x"] for k in sens]
        ky = [summary[k]["kept_y"] for k in sens]
        fx = [summary[k]["frac_x"] for k in sens]
        fy = [summary[k]["frac_y"] for k in sens]
        dh = [summary[k]["D_hat"] for k in sens]
        gp = [max(summary[k]["gap"], summary[k]["gap_longdouble"]) / summary[k]["F"] for k in sens]
        sm = [summary[k]["max_abs_smd"] for k in sens]
        jx = [summary[k]["jaccard_x_vs_primary"] for k in sens if k != prim]
        jy = [summary[k]["jaccard_y_vs_primary"] for k in sens if k != prim]
        M(f"{P}SensN", str(len(sens)))
        M(f"{P}SensExactN", str(sum(bool(coh[k].get("certify", {}).get("counts_exact_for_minimizer")) for k in sens)))
        M(f"{P}SensKeptXlo", thousands(min(kx)))
        M(f"{P}SensKeptXhi", thousands(max(kx)))
        M(f"{P}SensKeptYlo", thousands(min(ky)))
        M(f"{P}SensKeptYhi", thousands(max(ky)))
        M(f"{P}SensFracXlo", thousands(min(fx)))
        M(f"{P}SensFracXhi", thousands(max(fx)))
        M(f"{P}SensFracYlo", thousands(min(fy)))
        M(f"{P}SensFracYhi", thousands(max(fy)))
        M(f"{P}SensDhatLo", f"{min(dh):.4f}")
        M(f"{P}SensDhatHi", f"{max(dh):.4f}")
        M(f"{P}SensGapRelHi", sci(max(gp)))
        M(f"{P}SensSmdMaxLo", f"{min(sm):.3f}")
        M(f"{P}SensSmdMaxHi", f"{max(sm):.3f}")
        if jx:
            M(f"{P}SensJaccardXlo", f"{min(jx):.3f}")
            M(f"{P}SensJaccardYlo", f"{min(jy):.3f}")

    summary_json.write_text(json.dumps(summary, indent=1, default=float))
    # balance table
    rows = [("all patients", None)]
    if "old" in coh:
        rows.append((r"published (penalized, iteration limit)", "old"))
    if "pooled" in coh:
        rows.append((r"converged, pooled moments", "pooled"))
    if prim in coh:
        rows.append((r"converged, batch-0 moments", prim))
    L = [rf"% Generated by {generator} -- do not edit.",
         r"\begin{tabular}{lccccccccc}", r"\toprule",
         r"RHC, $\alpha=" + f2(alpha) + r"$ & \multicolumn{2}{c}{retained (fractional)} & "
         r"\multicolumn{2}{c}{effective fraction} & max $|$SMD$|$ & mean $|$SMD$|$ & overlap & "
         r"risk diff. & $\widehat D_\alpha$ \\",
         r" & cath. & others & cath. & others & & & & (95\% int.) & (gap) \\", r"\midrule"]
    from trimmedmmd.experiments.realdata_run import _overlap as ov
    Xp, Yp, _, _ = load_rhc()
    _, prop = _logistic_fit(np.vstack([Xp, Yp]), np.r_[np.ones(n), np.zeros(m)])
    rd0 = risk_difference(yx, yy, np.ones(n), np.ones(m))
    lo0, hi0 = boot_ci(yx, yy, np.ones(n), np.ones(m), np.random.default_rng(20261117))
    for lab, k in rows:
        if k is None:
            L.append(f"{lab} & {thousands(n)} & {thousands(m)} & 1.00 & 1.00 & {np.abs(smd0).max():.3f} & "
                     f"{np.abs(smd0).mean():.3f} & {ov(prop[:n], prop[n:]):.2f} & "
                     f"{rd0:.3f} ({lo0:.3f}, {hi0:.3f}) & -- \\\\")
            continue
        s = summary[k]
        g = max(s["gap"], s["gap_longdouble"])
        L.append(f"{lab} & {thousands(s['kept_x'])} ({s['frac_x']}) & {thousands(s['kept_y'])} ({s['frac_y']}) & "
                 f"{s['ess_frac_x']:.2f} & {s['ess_frac_y']:.2f} & {s['max_abs_smd']:.3f} & "
                 f"{s['mean_abs_smd']:.3f} & {s['prop_overlap']:.2f} & "
                 f"{s['rd_dth30']:.3f} ({s['rd_lo']:.3f}, {s['rd_hi']:.3f}) & "
                 f"{s['D_hat']:.4f} (${sci(g)}$) \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    balance_tex.write_text("\n".join(L) + "\n")
    M(f"{P}UnwRD", f"{rd0:.3f}")
    M(f"{P}UnwRDlo", f"{lo0:.3f}")
    M(f"{P}UnwRDhi", f"{hi0:.3f}")
    M(f"{P}UnwOverlap", f"{ov(prop[:n], prop[n:]):.2f}")
    return summary


if __name__ == "__main__":
    main()

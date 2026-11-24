"""AoS referees: tables, macros (prefix \\rr) and results/aos/main/REFEREE_RUNS.md for the real-data runs of
scripts/aos_ref_realdata.py, the 45-covariate RHC runs (scripts/aos_main_realdata.py --contrasts rhc_full,
scripts/exp_valid_bandwidth.py --datasets rhc_full, scripts/aos_realdata_cohort.py --covariates full) and their
33-covariate counterparts. No Monte Carlo work and no hand-typed number: everything is read from

    results/aos/main/raw/realdata_laplace/<contrast>_<seed>.jsonl     the bound (rhc = 33, rhc_full = 45 covariates)
    results/aos/main/raw/ref_realdata_ext/<contrast>_<seed>.jsonl     extension levels for the averaged e-values
    results/aos/main/raw/ref_realdata_cluster/<contrast>_<seed>.jsonl NHANES, sampling units to batches
    results/aos/main/raw/ref_realdata_clf/<contrast>.jsonl            RHC classifiers on the batches of the ramp
    results/aos/main/ref_realdata/propensity.json                     propensity-model estimate of alpha*
    results/aos/main/ref_realdata/rhc_estimand_full.csv               comparison rows of tab:cohort, 45 covariates
    results/aos/main/ref_realdata/check_generic.json, rhc_estimand_paper_check.json   reproduction checks
    results/aos/main/analytic_rd{,_full}/tables/r12_valid_bandwidth.csv              analytic baseline
    results/aos/main/cohort{,_full}/<label>.{json,npz}, cohort{,_full}_summary.json   the Laplace cohorts
    results/v3/tables/rhc_estimand.csv                                comparison rows of tab:cohort, 33 covariates

Writes results/aos/tables/aos_ref_realdata_macros.tex, aos_ref_realdata_rhc.tex (RHC, 33 vs 45 covariates),
aos_ref_realdata_evalues.tex, aos_ref_realdata_clf.tex, aos_ref_realdata_cluster.tex, aos_ref_realdata_cov.tex (the 12
added covariates), results/aos/main/ref_realdata/summary.json and results/aos/main/REFEREE_RUNS.md.

    PYTHONPATH=src python3 -B scripts/aos_ref_realdata_tables.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

MAIN = ROOT / "results" / "aos" / "main"
RAW = MAIN / "raw"
REF = MAIN / "ref_realdata"
TAB = ROOT / "results" / "aos" / "tables"
GEN = "scripts/aos_ref_realdata_tables.py"
SEEDS = [1000 + i for i in range(8)]
DELTA = 0.05
TAGS = {"rhc": "Rhc", "rhc_full": "RhcFull", "nhanes_child": "Child", "nhanes_age": "Age", "nhanes_bmi": "Bmi"}
LABEL = {"rhc": "RHC arms, 33 covariates", "rhc_full": "RHC arms, 45 covariates",
         "nhanes_child": r"ages 6--11 vs.\ 60--79", "nhanes_age": r"ages 20--39 vs.\ 60--79",
         "nhanes_bmi": r"BMI ${<}25$ vs.\ ${\ge}30$"}
GRID_TOP = {"nhanes_child": 0.95}
MACROS, MD = [], []


def mac(name, val):
    assert name.isalpha() and name.startswith("rr"), name
    assert name not in dict(MACROS), name
    val = str(val)
    if r"\times" in val:
        val = rf"\ensuremath{{{val}}}"
    MACROS.append((name, val))


def say(s=""):
    MD.append(s)


def jl(path):
    out = []
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


def f2(x):
    return f"{x:.2f}"


def f3(x):
    return f"{x:.3f}"


def thousands(k):
    return f"{int(k):,}".replace(",", "{,}")


def plain(s):
    return str(s).replace("{,}", ",")


def sci(x, nd=1):
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    mant = x / 10 ** e
    if round(mant, nd) >= 10:
        mant, e = mant / 10, e + 1
    return rf"{mant:.{nd}f}\times 10^{{{e}}}"


def sci_md(x):
    return f"{x:.1e}"


def grid(name):
    top = GRID_TOP.get(name, 0.75)
    return [round(0.05 * k, 6) for k in range(int(round(top / 0.05)) + 1)]


# ----------------------------------------------------------------------------------------------------------------
# chains
# ----------------------------------------------------------------------------------------------------------------

def chain(name, seed, ext=True):
    recs = {round(r["alpha"], 6): r for r in jl(RAW / "realdata_laplace" / f"{name}_{seed}.jsonl")}
    if ext:
        for r in jl(RAW / "ref_realdata_ext" / f"{name}_{seed}.jsonl"):
            k = round(r["alpha"], 6)
            if k in recs:
                assert recs[k]["order"] == r["order"], (name, seed, k)
            recs.setdefault(k, r)
    return recs


def down_closed(ch, rule="order"):
    """Table 2's reading: last level certified before the first failure (0.0 if none); also whether alpha = 0 was
    certified and the record at the first failure."""
    lv, stop, zero = 0.0, None, False
    for a in sorted(ch):
        if ch[a][rule]["certified"]:
            lv = a
            zero = zero or a == 0.0
        else:
            stop = ch[a]
            break
    return lv, stop, zero


def bound_summary(name, chains=None, rule="order"):
    chains = chains or {s: chain(name, s, ext=False) for s in SEEDS}
    lows, nobet, stops = [], 0, []
    for s in SEEDS:
        lv, stop, _ = down_closed(chains[s], rule)
        lows.append(lv)
        stops.append(None if stop is None else stop["alpha"])
        if stop is not None and all(x == "skip" for x in stop[rule]["source"]):
            nobet += 1
    return {"lows": lows, "min": min(lows), "median": float(np.median(lows)), "nobet": nobet, "stops": stops,
            "sigma": [chains[s][0.0]["sigma"] for s in SEEDS if 0.0 in chains[s] and "sigma" in chains[s][0.0]]}


def averaged(name):
    """The averaged e-value tests: per level, the average over the eight splits of exp(log_stopped) and of
    exp(log_final) ('order' rule), read upward to the first failure against 1/delta and 1/(2 delta)."""
    chains = {s: chain(name, s) for s in SEEDS}
    rows = []
    alive = {(k, t): True for k in ("log_stopped", "log_final") for t in (1 / DELTA, 1 / (2 * DELTA))}
    last = {key: None for key in alive}
    first_fail = {key: None for key in alive}
    for a in grid(name):
        if not all(a in chains[s] for s in SEEDS):
            break
        recs = [chains[s][a]["order"] for s in SEEDS]
        row = {"alpha": a, "n_certified": int(sum(r["certified"] for r in recs)),
               "stopped": [math.exp(r["log_stopped"]) for r in recs], "final": [math.exp(r["log_final"]) for r in recs],
               "no_bet": int(sum(all(x == "skip" for x in r["source"]) for r in recs))}
        row["avg_stopped"] = float(np.mean(row["stopped"]))
        row["avg_final"] = float(np.mean(row["final"]))
        rows.append(row)
        for (k, t) in alive:
            if not alive[(k, t)]:
                continue
            v = row["avg_stopped" if k == "log_stopped" else "avg_final"]
            if v >= t:
                last[(k, t)] = a
            else:
                alive[(k, t)] = False
                first_fail[(k, t)] = (a, v)
    unsettled = [key for key, v in alive.items() if v]
    return {"rows": rows, "last": last, "first_fail": first_fail, "unsettled": unsettled}


def lv_print(x):
    return "--" if x is None else f2(x)


# ----------------------------------------------------------------------------------------------------------------

def main():
    t_start = time.time()
    TAB.mkdir(parents=True, exist_ok=True)
    say("# AoS referee runs on the real data (generated by scripts/aos_ref_realdata_tables.py; no hand-typed numbers)")
    say()
    say(f"Generated {time.strftime('%Y-%m-%d %H:%M')}. Requests: aos/reviews/referee2_methods.md M6 (averaged "
        "e-values), M7 (propensity link, classifier bounds, covariate set; MISMATCH 3), M8 (NHANES dependence); "
        "aos/reviews/referee1_theory.md. Method throughout: the split-sample bound with the Laplace kernel, sigma = "
        "median distance on batch 0, genuine-quantile cuts (rule \"order\"), batches (0, 1/4, 1/2, 1), eight splits "
        "(seeds 1000-1007), grid mesh 0.05, delta = 0.05; the analytic baseline with tau = tau^P (pilot route). "
        "Every number below is also a macro of results/aos/tables/aos_ref_realdata_macros.tex (named in brackets) or "
        "is read from results/aos/main/ref_realdata/summary.json.")
    say()
    summary = {}

    # ============================================================================ 1. the 12 covariates
    from trimmedmmd.experiments.realdata_run import standardized_differences
    from trimmedmmd.realdata import RHC_STRING_BINARY, RHC_STRING_CODES, load_rhc, rhc_covariate_names
    Xf, Yf, names_f, meta_f = load_rhc(standardize=False, covariates="full")
    Xp, Yp, names_p, _ = load_rhc(standardize=False)
    assert names_f[:len(names_p)] == names_p and np.array_equal(Xf[:, :len(names_p)], Xp)
    assert meta_f["n_rows_complete"] == 5735
    n, m = len(Xf), len(Yf)
    Zs = np.vstack(load_rhc(covariates="full")[:2])
    smd0 = standardized_differences(Xf, Yf)
    new_idx = [names_f.index(c) for c in RHC_STRING_BINARY]
    mac("rrRhcNcovPaper", len(names_p))
    mac("rrRhcNcovFull", len(names_f))
    mac("rrRhcNcovAdded", len(RHC_STRING_BINARY))
    say("## 1. RHC with every binary covariate (M7.3, MISMATCH 3)")
    say()
    say(f"The data hold {len(RHC_STRING_BINARY)} binary baseline covariates stored as strings that no earlier run used "
        f"[rrRhcNcovAdded]: {', '.join(RHC_STRING_BINARY)}. They are coded 0/1 (realdata.load_rhc(covariates=\"full\"), "
        "RHC_STRING_CODES) and added to the "
        f"{len(names_p)} covariates [rrRhcNcovPaper], {len(names_f)} in all [rrRhcNcovFull]; standardization, "
        "bandwidth rule and everything else unchanged. No value is missing, so the 5,735 complete-case patients are "
        "the same.")
    say()
    say("| covariate | coded 1 / coded 0 | catheterized: share coded 1 | others | unweighted SMD | largest standardized value |")
    say("|---|---|---|---|---|---|")
    cov_rows = []
    for c, i in zip(RHC_STRING_BINARY, new_idx):
        one, zero = RHC_STRING_CODES[c]
        px, py = float(Xf[:, i].mean()), float(Yf[:, i].mean())
        zmax = float(np.abs(Zs[:, i]).max())
        cov_rows.append((c, one, zero, px, py, float(smd0[i]), zmax))
        say(f"| {c} | {one} / {zero} | {px:.3f} | {py:.3f} | {smd0[i]:+.3f} | {zmax:.1f} |")
    say()
    old_bin = [names_f.index(c) for c in names_p[21:]]
    zmax_old = float(np.abs(Zs[:, old_bin]).max())
    big = [r for r in cov_rows if abs(r[5]) > 0.1]
    mac("rrAddedSmdAbove", len(big))
    mac("rrAddedSmdMax", f3(max(abs(r[5]) for r in cov_rows)))
    mac("rrAddedSmdMaxName", max(cov_rows, key=lambda r: abs(r[5]))[0])
    mac("rrAddedZmax", f"{max(r[6] for r in cov_rows):.1f}")
    mac("rrAddedZmaxName", max(cov_rows, key=lambda r: r[6])[0])
    mac("rrOldBinaryZmax", f"{zmax_old:.1f}")
    nort = int(round(float(np.r_[Xf[:, names_f.index('ortho')], Yf[:, names_f.index('ortho')]].sum())))
    ntra = int(round(float(np.r_[Xf[:, names_f.index('trauma')], Yf[:, names_f.index('trauma')]].sum())))
    mac("rrOrthoN", nort)
    mac("rrTraumaN", ntra)
    say(f"Encoding. Eleven are Yes/No and coded Yes = 1; sex is Female/Male, coded Female = 1. The direction is a "
        "labelling choice with no effect: after the coordinatewise standardization a flipped code changes the sign "
        "of one coordinate, which leaves every Euclidean distance (so the Laplace kernel and the median bandwidth), "
        "the logistic propensity model and every |SMD| unchanged. No covariate is ambiguous in its values (two "
        "levels each, none missing). Two are rare: ortho has "
        f"{nort} patients coded 1 [rrOrthoN] and trauma {ntra} [rrTraumaN], so standardized they reach "
        f"{max(r[6] for r in cov_rows):.1f} SD [rrAddedZmax] ({max(cov_rows, key=lambda r: r[6])[0]}) against at most "
        f"{zmax_old:.1f} [rrOldBinaryZmax] for the twelve 0/1 covariates already used; under the Laplace kernel those "
        "few patients sit far from everyone, which is what standardizing a rare binary does (the paper standardizes "
        "its 0/1 covariates the same way). Not added, as before: the multi-category cat1, cat2, ca "
        "(No/Yes/Metastatic), ninsclas, race and income, which need a one-hot coding; wtkilo1 = 0 (the data set's "
        "missing-value code, M7.4) is kept as in every run. "
        f"{len(big)} of the 12 have an unweighted |SMD| above 0.1 [rrAddedSmdAbove]; the largest is "
        f"{max(abs(r[5]) for r in cov_rows):.3f} ({max(cov_rows, key=lambda r: abs(r[5]))[0]}) [rrAddedSmdMax, "
        "rrAddedSmdMaxName].")
    say()
    # table of the 12 covariates
    L = [f"% Generated by {GEN} -- do not edit.", r"\begin{tabular}{llcccc}", r"\toprule",
         r"covariate & coded $1$ & \multicolumn{2}{c}{share coded $1$} & unweighted & largest \\",
         r" & & catheterized & others & SMD & standardized value \\", r"\midrule"]
    for c, one, zero, px, py, sm, zm in cov_rows:
        L.append(rf"\texttt{{{c}}} & {one} & {px:.3f} & {py:.3f} & ${sm:+.3f}$ & {zm:.1f} \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "aos_ref_realdata_cov.tex").write_text("\n".join(L) + "\n")

    # ---------------------------------------------------------------- the bound, 33 vs 45
    B = {k: bound_summary(k) for k in ("rhc", "rhc_full")}
    BL = {k: bound_summary(k, rule="linear") for k in ("rhc", "rhc_full")}
    for k, t in (("rhc", "Paper"), ("rhc_full", "Full")):
        mac(f"rrRhc{t}Min", f2(B[k]["min"]))
        mac(f"rrRhc{t}Med", f2(B[k]["median"]))
        mac(f"rrRhc{t}Splits", " ".join(f2(x) for x in B[k]["lows"]))
        mac(f"rrRhc{t}NoBet", B[k]["nobet"])
        mac(f"rrRhc{t}LinMin", f2(BL[k]["min"]))
        mac(f"rrRhc{t}LinMed", f2(BL[k]["median"]))
        sg = [chain(k, s, ext=False)[0.0]["sigma"] for s in SEEDS]
        mac(f"rrRhc{t}SigmaLo", f2(min(sg)))
        mac(f"rrRhc{t}SigmaHi", f2(max(sg)))
        B[k]["sigma"] = sg
    summary["bound"] = {k: B[k] | {"linear": BL[k]} for k in B}

    # ---------------------------------------------------------------- analytic, 33 vs 45
    def pv_rows(path, ds):
        rows = [r for r in csv.DictReader(open(path)) if r["dataset"] == ds]
        out = {}
        for r in rows:
            out[r["route"]] = r
        return out
    AN = {"rhc": pv_rows(MAIN / "analytic_rd" / "tables" / "r12_valid_bandwidth.csv", "rhc"),
          "rhc_full": pv_rows(MAIN / "analytic_rd_full" / "tables" / "r12_valid_bandwidth.csv", "rhc_full")}
    c0 = 1.0 + math.sqrt(math.log(1 / DELTA))
    ANS = {}
    for k, t in (("rhc", "Paper"), ("rhc_full", "Full")):
        pv = AN[k]["pilot, strict (pilot standardization)"]
        pm = AN[k]["pooled median (paper)"]
        fired = pv["H00_rejected"] == "True"
        last = f2(float(pv["alpha_last"])) if fired else "--"
        fine = f"{float(pv['certified_fine']):.4f}".rstrip("0") if fired else "--"
        if fine.endswith("."):
            fine += "0"
        if fired and float(pv["certified_fine"]) == 0.0:
            fine = "0"
        fire_n = math.ceil(2 * (c0 / float(pv["D0"])) ** 2)
        ANS[k] = {"n": int(pm["n"]), "m": int(pm["m"]), "D0": float(pv["D0"]), "tau0": float(pv["tau0"]),
                  "fire": fire_n, "last": last, "fine": fine, "scale": pv["bandwidth"].split(" x ")[0],
                  "pooled_median": {"D0": float(pm["D0"]), "tau0": float(pm["tau0"]),
                                    "alpha_last": pm["alpha_last"], "fine": float(pm["certified_fine"])},
                  "maxD0": {r: AN[k][r]["certified_fine"] for r in AN[k] if "largest" in r}}
        mac(f"rrPv{t}Dzero", f3(ANS[k]["D0"]))
        mac(f"rrPv{t}Tauzero", f3(ANS[k]["tau0"]))
        mac(f"rrPv{t}Fire", thousands(fire_n))
        mac(f"rrPv{t}Last", last)
        mac(f"rrPv{t}Fine", fine)
        mac(f"rrPv{t}Scale", ANS[k]["scale"])
    summary["analytic"] = ANS

    # ---------------------------------------------------------------- cohorts, 33 vs 45
    def cohort(dirname, sumname):
        d = MAIN / dirname
        js = {f.stem: json.loads(f.read_text()) for f in sorted(d.glob("*.json")) if ".tmp" not in f.name}
        sm = json.loads((MAIN / sumname).read_text())
        return js, sm
    CJ = {"rhc": cohort("cohort", "cohort_summary.json"), "rhc_full": cohort("cohort_full", "cohort_full_summary.json")}
    COH = {}
    for k, t in (("rhc", "Paper"), ("rhc_full", "Full")):
        js, sm = CJ[k]
        p, si, ce = sm["pooled"], js["pooled"]["solver_info"], js["pooled"]["certify"]
        sens = [x for x in sm if x.startswith("batch0_")]
        COH[k] = {"alpha": js["pooled"]["alpha"], "sigma": p["sigma"], "tau": p["tau"], "D_hat": p["D_hat"],
                  "gap": max(p["gap"], p["gap_longdouble"]), "gap_rel": max(p["gap"], p["gap_longdouble"]) / p["F"],
                  "smo_steps": si["smo"]["steps"], "lam_min": ce["lam_min_gram"],
                  "counts_exact": ce["counts_exact_for_minimizer"], "seconds": si["seconds_total"],
                  **{x: p[x] for x in ("cap_x", "zero_x", "frac_x", "kept_x", "cap_y", "zero_y", "frac_y", "kept_y",
                                       "ess_frac_x", "ess_frac_y", "max_abs_smd", "mean_abs_smd",
                                       "n_abs_smd_above_0.1", "prop_overlap", "rd_dth30", "rd_lo", "rd_hi")},
                  "worst": None,
                  "sens": {"kept_x": [min(sm[x]["kept_x"] for x in sens), max(sm[x]["kept_x"] for x in sens)],
                           "kept_y": [min(sm[x]["kept_y"] for x in sens), max(sm[x]["kept_y"] for x in sens)],
                           "frac_x": [min(sm[x]["frac_x"] for x in sens), max(sm[x]["frac_x"] for x in sens)],
                           "frac_y": [min(sm[x]["frac_y"] for x in sens), max(sm[x]["frac_y"] for x in sens)],
                           "smd": [min(sm[x]["max_abs_smd"] for x in sens), max(sm[x]["max_abs_smd"] for x in sens)],
                           "rd": [min(sm[x]["rd_dth30"] for x in sens), max(sm[x]["rd_dth30"] for x in sens)],
                           "rd_lo": [min(sm[x]["rd_lo"] for x in sens), max(sm[x]["rd_lo"] for x in sens)],
                           "exact": int(sum(js[x]["certify"]["counts_exact_for_minimizer"] for x in sens)),
                           "gap_rel_max": max(max(sm[x]["gap"], sm[x]["gap_longdouble"]) / sm[x]["F"] for x in sens),
                           "n": len(sens)},
                  "alpha_seeds": sorted({js[x]["alpha"] for x in js})}
        smd = np.abs(np.array(js["pooled"]["describe"]["smd"]))
        nm = js["pooled"].get("names", names_p)
        COH[k]["worst"] = nm[int(smd.argmax())]
        COH[k]["top5"] = [(nm[i], float(smd[i])) for i in np.argsort(-smd)[:5]]
        assert len(smd) == (len(names_f) if k == "rhc_full" else len(names_p))
        c = COH[k]
        mac(f"rrCoh{t}Alpha", f2(c["alpha"]))
        mac(f"rrCoh{t}Sigma", f2(c["sigma"]))
        mac(f"rrCoh{t}Tau", f3(c["tau"]))
        mac(f"rrCoh{t}Dhat", f"{c['D_hat']:.4f}")
        mac(f"rrCoh{t}Gap", sci(c["gap"]))
        mac(f"rrCoh{t}GapRel", sci(c["gap_rel"]))
        mac(f"rrCoh{t}LamMin", sci(c["lam_min"]))
        mac(f"rrCoh{t}CountsExact", "yes" if c["counts_exact"] else "no")
        for arm, A in (("x", "X"), ("y", "Y")):
            mac(f"rrCoh{t}Kept{A}", thousands(c[f"kept_{arm}"]))
            mac(f"rrCoh{t}Frac{A}", thousands(c[f"frac_{arm}"]))
            mac(f"rrCoh{t}Cap{A}", thousands(c[f"cap_{arm}"]))
            mac(f"rrCoh{t}Ess{A}", f2(c[f"ess_frac_{arm}"]))
        mac(f"rrCoh{t}SmdMax", f3(c["max_abs_smd"]))
        mac(f"rrCoh{t}SmdMean", f3(c["mean_abs_smd"]))
        mac(f"rrCoh{t}SmdAbove", c["n_abs_smd_above_0.1"])
        mac(f"rrCoh{t}SmdWorst", c["worst"])
        mac(f"rrCoh{t}Overlap", f2(c["prop_overlap"]))
        mac(f"rrCoh{t}RD", f3(c["rd_dth30"]))
        mac(f"rrCoh{t}RDlo", f3(c["rd_lo"]))
        mac(f"rrCoh{t}RDhi", f3(c["rd_hi"]))
        mac(f"rrCoh{t}SensExact", f"{c['sens']['exact']}")
        mac(f"rrCoh{t}SensSmdLo", f3(c["sens"]["smd"][0]))
        mac(f"rrCoh{t}SensSmdHi", f3(c["sens"]["smd"][1]))
        mac(f"rrCoh{t}SensRDloLo", f3(c["sens"]["rd_lo"][0]))
        mac(f"rrCoh{t}SensRDloHi", f3(c["sens"]["rd_lo"][1]))
    # balance of the 33-covariate cohort on the 12 added covariates (M7: 'report balance on the omitted covariates')
    W = np.load(MAIN / "cohort" / "pooled.npz")
    smd_new = standardized_differences(Xf[:, new_idx], Yf[:, new_idx], W["w"] / W["w"].sum(), W["v"] / W["v"].sum())
    k_w = int(np.argmax(np.abs(smd_new)))
    mac("rrCohPaperOmitSmdMax", f3(float(np.abs(smd_new).max())))
    mac("rrCohPaperOmitSmdMaxName", RHC_STRING_BINARY[k_w])
    mac("rrCohPaperOmitSmdAbove", int(np.sum(np.abs(smd_new) > 0.1)))
    mac("rrCohPaperOmitSmdMean", f3(float(np.abs(smd_new).mean())))
    summary["cohort"] = COH
    summary["cohort_paper_on_added"] = dict(zip(RHC_STRING_BINARY, map(float, smd_new)))

    # comparison rows, 33 vs 45
    import pandas as pd
    E33 = pd.read_csv(ROOT / "results" / "v3" / "tables" / "rhc_estimand.csv").set_index("key")
    E45 = pd.read_csv(REF / "rhc_estimand_full.csv").set_index("key")
    ROWS = ["unweighted", "atm", "ato", "energy", "kernel"]
    RLAB = {"unweighted": "all patients", "atm": "matching weights", "ato": "overlap weights",
            "energy": "energy balancing", "kernel": "kernel balancing"}
    summary["comparison_rows"] = {k: {"paper": {c: float(E33.loc[k, c]) for c in ("max_abs_smd", "mean_abs_smd",
                                                                                   "ess_frac_x", "ess_frac_y",
                                                                                   "prop_overlap", "rd_dth30",
                                                                                   "rd_lo", "rd_hi")},
                                      "full": {c: float(E45.loc[k, c]) for c in ("max_abs_smd", "mean_abs_smd",
                                                                                  "ess_frac_x", "ess_frac_y",
                                                                                  "prop_overlap", "rd_dth30",
                                                                                  "rd_lo", "rd_hi")}}
                                  for k in ROWS}

    # ---------------------------------------------------------------- RHC section of the report
    say("### Side by side: 33 covariates (stored runs) and 45 covariates")
    say()
    o, nw = "rhc", "rhc_full"
    say("| quantity | 33 covariates | 45 covariates |")
    say("|---|---|---|")
    say(f"| split-sample bound, min over 8 splits (delta) | {f2(B[o]['min'])} | {f2(B[nw]['min'])} [rrRhcPaperMin, rrRhcFullMin] |")
    say(f"| median (2 delta) | {f2(B[o]['median'])} | {f2(B[nw]['median'])} [rrRhcPaperMed, rrRhcFullMed] |")
    say(f"| per split (seeds 1000-1007) | {' '.join(f2(x) for x in B[o]['lows'])} | {' '.join(f2(x) for x in B[nw]['lows'])} |")
    say(f"| first failing level, per split | {' '.join(f2(x) for x in B[o]['stops'])} | {' '.join(f2(x) for x in B[nw]['stops'])} |")
    say(f"| splits with no bet at the first failure (cuts crossed in both batches) | {B[o]['nobet']} | {B[nw]['nobet']} |")
    say(f"| linear cuts on the same fits, min / median | {f2(BL[o]['min'])} / {f2(BL[o]['median'])} | {f2(BL[nw]['min'])} / {f2(BL[nw]['median'])} |")
    say(f"| batch-0 bandwidth sigma, range over splits | {f2(min(B[o]['sigma']))}-{f2(max(B[o]['sigma']))} | {f2(min(B[nw]['sigma']))}-{f2(max(B[nw]['sigma']))} |")
    for lab, key in (("analytic: D-hat_0 (pilot, strict)", "D0"), ("analytic: tau(0)", "tau0")):
        say(f"| {lab} | {f3(ANS[o][key])} | {f3(ANS[nw][key])} |")
    say(f"| analytic: n_fire | {plain(thousands(ANS[o]['fire']))} | {plain(thousands(ANS[nw]['fire']))} |")
    say(f"| analytic: alpha_last / bisection / scale | {ANS[o]['last']} / {ANS[o]['fine']} / {ANS[o]['scale']} | {ANS[nw]['last']} / {ANS[nw]['fine']} / {ANS[nw]['scale']} |")
    say(f"| analytic, pooled-median reference: D-hat_0 / tau(0) / bisection | {f3(ANS[o]['pooled_median']['D0'])} / {f3(ANS[o]['pooled_median']['tau0'])} / {ANS[o]['pooled_median']['fine']:.4f} | {f3(ANS[nw]['pooled_median']['D0'])} / {f3(ANS[nw]['pooled_median']['tau0'])} / {ANS[nw]['pooled_median']['fine']:.4f} |")
    say(f"| cohort level alpha (= the min of the bound) | {f2(COH[o]['alpha'])} | {f2(COH[nw]['alpha'])} [rrCohFullAlpha] |")
    say(f"| cohort: pooled bandwidth / tau(alpha) | {f2(COH[o]['sigma'])} / {f3(COH[o]['tau'])} | {f2(COH[nw]['sigma'])} / {f3(COH[nw]['tau'])} |")
    say(f"| cohort: D-hat at the solution | {COH[o]['D_hat']:.6f} | {COH[nw]['D_hat']:.6f} |")
    say(f"| cohort: duality gap (relative to F) | {sci_md(COH[o]['gap'])} ({sci_md(COH[o]['gap_rel'])}) | {sci_md(COH[nw]['gap'])} ({sci_md(COH[nw]['gap_rel'])}) |")
    say(f"| cohort: smallest Gram eigenvalue | {sci_md(COH[o]['lam_min'])} | {sci_md(COH[nw]['lam_min'])} |")
    say(f"| cohort: SMO steps / counts certified exact | {COH[o]['smo_steps']:,} / {COH[o]['counts_exact']} | {COH[nw]['smo_steps']:,} / {COH[nw]['counts_exact']} |")
    for arm, A in (("x", "catheterized (n = 2,184)"), ("y", "others (n = 3,551)")):
        say(f"| cohort, {A}: at cap / fractional / retained | {COH[o][f'cap_{arm}']:,} / {COH[o][f'frac_{arm}']:,} / {COH[o][f'kept_{arm}']:,} | {COH[nw][f'cap_{arm}']:,} / {COH[nw][f'frac_{arm}']:,} / {COH[nw][f'kept_{arm}']:,} |")
        say(f"| cohort, {A}: effective fraction | {f2(COH[o][f'ess_frac_{arm}'])} | {f2(COH[nw][f'ess_frac_{arm}'])} |")
    say(f"| cohort: max / mean abs SMD (over its own covariates) | {f3(COH[o]['max_abs_smd'])} / {f3(COH[o]['mean_abs_smd'])} | {f3(COH[nw]['max_abs_smd'])} / {f3(COH[nw]['mean_abs_smd'])} |")
    say(f"| cohort: covariates with abs SMD > 0.1 / largest | {COH[o]['n_abs_smd_above_0.1']} / {COH[o]['worst']} | {COH[nw]['n_abs_smd_above_0.1']} / {COH[nw]['worst']} |")
    say(f"| cohort: propensity overlap (own propensity model) | {f2(COH[o]['prop_overlap'])} | {f2(COH[nw]['prop_overlap'])} |")
    say(f"| cohort: 30-day mortality difference (95% int.) | {f3(COH[o]['rd_dth30'])} ({f3(COH[o]['rd_lo'])}, {f3(COH[o]['rd_hi'])}) | {f3(COH[nw]['rd_dth30'])} ({f3(COH[nw]['rd_lo'])}, {f3(COH[nw]['rd_hi'])}) |")
    for k, A in (("kept_x", "retained catheterized"), ("kept_y", "retained others"), ("frac_x", "fractional catheterized"),
                 ("frac_y", "fractional others")):
        say(f"| eight batch-0 runs: {A} | {COH[o]['sens'][k][0]:,}-{COH[o]['sens'][k][1]:,} | {COH[nw]['sens'][k][0]:,}-{COH[nw]['sens'][k][1]:,} |")
    say(f"| eight batch-0 runs: max abs SMD | {f3(COH[o]['sens']['smd'][0])}-{f3(COH[o]['sens']['smd'][1])} | {f3(COH[nw]['sens']['smd'][0])}-{f3(COH[nw]['sens']['smd'][1])} |")
    say(f"| eight batch-0 runs: mortality difference / lower end of its interval | {f3(COH[o]['sens']['rd'][0])}-{f3(COH[o]['sens']['rd'][1])} / {f3(COH[o]['sens']['rd_lo'][0])} to {f3(COH[o]['sens']['rd_lo'][1])} | {f3(COH[nw]['sens']['rd'][0])}-{f3(COH[nw]['sens']['rd'][1])} / {f3(COH[nw]['sens']['rd_lo'][0])} to {f3(COH[nw]['sens']['rd_lo'][1])} |")
    say(f"| eight batch-0 runs: counts certified exact / largest relative gap | {COH[o]['sens']['exact']} of 8 / {sci_md(COH[o]['sens']['gap_rel_max'])} | {COH[nw]['sens']['exact']} of 8 / {sci_md(COH[nw]['sens']['gap_rel_max'])} |")
    say()
    say("Solver diagnostics, every 45-covariate run (the same solver, exit criterion and certificate as COHORT.md, "
        "section 8; 'units unsettled' as there):")
    say()
    say("| label | sigma | D-hat | gap float64 / extended | KKT violation x / y | FISTA it. | SMO steps | face solve "
        "(free, kept) | lambda_min | counts exact | units unsettled | seconds |")
    say("|---|---|---|---|---|---|---|---|---|---|---|---|")
    jsf = CJ["rhc_full"][0]
    diag = {}
    for k in sorted(jsf, key=lambda s_: (s_ != "pooled", s_)):
        c, si, ce = jsf[k], jsf[k]["solver_info"], jsf[k]["certify"]
        z = np.load(MAIN / "cohort_full" / f"{k}.npz")
        uns = 0
        for arm, key in (("x", "w"), ("y", "v")):
            d, u = z["d" + arm], z[key]
            cap = 1 / ((c["n"] if arm == "x" else c["m"]) * (1 - c["alpha"]))
            capm, zero = u >= cap * (1 - 1e-9), u <= cap * 1e-9
            fr = ~capm & ~zero
            t_, sg, need = d[fr].mean(), (1.0 if arm == "x" else -1.0), ce[arm]["window_needed"]
            uns += int(np.sum(sg * (t_ - d[capm]) <= need) + np.sum(sg * (d[zero] - t_) <= need))
        diag[k] = {"units_unsettled": uns, "counts_exact": ce["counts_exact_for_minimizer"],
                   "kkt": si["smo"]["viol_end"], "gap": [si["gap"], si["gap_longdouble"]]}
        say(f"| {k} | {c['sigma']:.4f} | {si['D_hat']:.6f} | {si['gap']:.1e} / {si['gap_longdouble']:.1e} | "
            f"{si['smo']['viol_end'][0]:.1e} / {si['smo']['viol_end'][1]:.1e} | {si['fista']['iters']} | "
            f"{si['smo']['steps']:,} | {si['polish']['free']}, {si['polish']['kept']} | {si['gram_eig_min']:.4f} | "
            f"{ce['counts_exact_for_minimizer']} | {uns} | {si['seconds_total']:.0f} |")
    say()
    COH["rhc_full"]["diagnostics"] = diag
    mac("rrCohFullUnsettledMax", max(v["units_unsettled"] for v in diag.values()))
    mac("rrCohFullKktMax", sci(max(max(v["kkt"]) for v in diag.values())))
    kk = max(max(v["kkt"]) for v in diag.values())
    say(f"{'Every' if kk < 1e-14 else 'NOT every'} 45-covariate run exits with a KKT violation below 1e-14 in both "
        f"arms (largest "
        f"{max(max(v['kkt']) for v in diag.values()):.1e} [rrCohFullKktMax]); the counts are certified exact for "
        f"{sum(v['counts_exact'] for v in diag.values())} of the {len(diag)} runs, and where they are not, at most "
        f"{max(v['units_unsettled'] for v in diag.values())} units [rrCohFullUnsettledMax] have a cap/zero status the "
        "gap leaves open.")
    say()
    say("Largest abs SMDs of the 45-covariate pooled cohort: " + ", ".join(f"{a} {b:.3f}" for a, b in COH[nw]["top5"])
        + "; of the 33-covariate one: " + ", ".join(f"{a} {b:.3f}" for a, b in COH[o]["top5"]) + ".")
    say()
    say(f"Balance of the stored (33-covariate) cohort on the 12 covariates it did not see: largest abs SMD "
        f"{float(np.abs(smd_new).max()):.3f} ({RHC_STRING_BINARY[k_w]}) [rrCohPaperOmitSmdMax, rrCohPaperOmitSmdMaxName], "
        f"mean {float(np.abs(smd_new).mean()):.3f}, {int(np.sum(np.abs(smd_new) > 0.1))} above 0.1 "
        "[rrCohPaperOmitSmdAbove]; per covariate: "
        + ", ".join(f"{c} {v:+.3f}" for c, v in zip(RHC_STRING_BINARY, smd_new)) + ".")
    say()
    say("The other rows of Table tab:cohort, recomputed on the 45 covariates (scripts/aos_ref_realdata.py estimand; "
        "propensity model, SMDs, energy and kernel balancing on the 45; the macros \\lcAtm..., \\lcAto..., "
        "\\lcEnergy..., \\lcKernel..., \\lcUnw... of aos_main_cohort_macros.tex now carry these values):")
    say()
    say("| weighting | max abs SMD 33 / 45 | effective fraction cath. 33 / 45 | others 33 / 45 | overlap 33 / 45 | risk difference (95% int.) 33 | 45 |")
    say("|---|---|---|---|---|---|---|")
    for k in ROWS:
        a_, b_ = summary["comparison_rows"][k]["paper"], summary["comparison_rows"][k]["full"]
        say(f"| {RLAB[k]} | {f3(a_['max_abs_smd'])} / {f3(b_['max_abs_smd'])} | {f2(a_['ess_frac_x'])} / {f2(b_['ess_frac_x'])} | "
            f"{f2(a_['ess_frac_y'])} / {f2(b_['ess_frac_y'])} | {f2(a_['prop_overlap'])} / {f2(b_['prop_overlap'])} | "
            f"{f3(a_['rd_dth30'])} ({f3(a_['rd_lo'])}, {f3(a_['rd_hi'])}) | {f3(b_['rd_dth30'])} ({f3(b_['rd_lo'])}, {f3(b_['rd_hi'])}) |")
    say()
    # RHC table
    L = [f"% Generated by {GEN} -- do not edit. RHC with the 33 covariates of the earlier runs and with all 45.",
         r"\begin{tabular}{lcc}", r"\toprule", r" & 33 covariates & 45 covariates \\", r"\midrule",
         rf"split-sample bound $\widehat\alpha$, min over splits ($\delta$) & {f2(B[o]['min'])} & {f2(B[nw]['min'])} \\",
         rf"median over splits ($2\delta$) & {f2(B[o]['median'])} & {f2(B[nw]['median'])} \\",
         rf"analytic: $\widehat D_0$ / $\tau(0)$ & {f3(ANS[o]['D0'])} / {f3(ANS[o]['tau0'])} & {f3(ANS[nw]['D0'])} / {f3(ANS[nw]['tau0'])} \\",
         rf"analytic: $\alpha_{{\mathrm{{last}}}}$ / bisection & {ANS[o]['last']} / {ANS[o]['fine']} & {ANS[nw]['last']} / {ANS[nw]['fine']} \\",
         r"\addlinespace",
         rf"cohort at $\alpha$ & {f2(COH[o]['alpha'])} & {f2(COH[nw]['alpha'])} \\",
         rf"retained, catheterized / others & {thousands(COH[o]['kept_x'])} / {thousands(COH[o]['kept_y'])} & {thousands(COH[nw]['kept_x'])} / {thousands(COH[nw]['kept_y'])} \\",
         rf"effective fraction, catheterized / others & {f2(COH[o]['ess_frac_x'])} / {f2(COH[o]['ess_frac_y'])} & {f2(COH[nw]['ess_frac_x'])} / {f2(COH[nw]['ess_frac_y'])} \\",
         rf"max / mean $|$SMD$|$ & {f3(COH[o]['max_abs_smd'])} / {f3(COH[o]['mean_abs_smd'])} & {f3(COH[nw]['max_abs_smd'])} / {f3(COH[nw]['mean_abs_smd'])} \\",
         rf"propensity overlap & {f2(COH[o]['prop_overlap'])} & {f2(COH[nw]['prop_overlap'])} \\",
         rf"mortality difference (95\% int.) & {f3(COH[o]['rd_dth30'])} ({f3(COH[o]['rd_lo'])}, {f3(COH[o]['rd_hi'])}) & {f3(COH[nw]['rd_dth30'])} ({f3(COH[nw]['rd_lo'])}, {f3(COH[nw]['rd_hi'])}) \\",
         r"\bottomrule", r"\end{tabular}"]
    (TAB / "aos_ref_realdata_rhc.tex").write_text("\n".join(L) + "\n")

    # ============================================================================ 2. averaged e-values
    say("## 2. Averaged e-values over the eight splits (M6)")
    say()
    say("At each grid level the stopped wealth of split s is K^(s) at the first time it reaches 1/delta, or at the "
        "horizon N if it never does (the 'order' rule, the method; log_stopped of the records); by optional stopping "
        "each has mean at most 1 under H_(0,alpha), so their average is an e-value and rejecting when it reaches "
        "1/delta is a level-delta test (Markov). If every split rejects, every K^(s) >= 1/delta, so the average "
        "rejects too: never less powerful than the minimum. Read upward to the first failure. The final wealth K_N "
        "is also an e-value and is averaged alongside; against 1/(2 delta) the averages give level-2delta tests, "
        "which reject whenever at least half the splits do (the median rule). Levels missing from a chain (a split "
        "stops at its first failure) are computed by scripts/aos_ref_realdata.py evalues "
        "(results/aos/main/raw/ref_realdata_ext).")
    say()
    say("| contrast | min (delta) | median (2 delta) | average, stopped (delta) | average, final (delta) | average, stopped (2 delta) | average, final (2 delta) | at the first failure of the stopped average: level, average, splits certified, stopped wealths |")
    say("|---|---|---|---|---|---|---|---|")
    EV = {}
    L = [f"% Generated by {GEN} -- do not edit. Averages over the eight splits of the stopped and of the final "
         r"wealth, Laplace kernel, genuine-quantile cuts; read upward to the first failure.",
         r"\begin{tabular}{lcccccc}", r"\toprule",
         r" & min & median & \multicolumn{2}{c}{average, stopped} & \multicolumn{2}{c}{average, final} \\",
         r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         r"contrast & ($\delta$) & ($2\delta$) & ($\delta$) & ($2\delta$) & ($\delta$) & ($2\delta$) \\", r"\midrule"]
    for name in ("rhc_full", "rhc", "nhanes_child", "nhanes_age", "nhanes_bmi"):
        bs = bound_summary(name)
        av = averaged(name)
        assert not av["unsettled"], (name, av["unsettled"])
        t = TAGS[name]
        D, D2 = 1 / DELTA, 1 / (2 * DELTA)
        vals = {"Min": bs["min"], "Med": bs["median"], "AvgStop": av["last"][("log_stopped", D)],
                "AvgFinal": av["last"][("log_final", D)], "AvgStopTwo": av["last"][("log_stopped", D2)],
                "AvgFinalTwo": av["last"][("log_final", D2)]}
        for k_, v in vals.items():
            mac(f"rrEv{t}{k_}", lv_print(v))
        ff = av["first_fail"][("log_stopped", D)]
        row_ff = next(r for r in av["rows"] if r["alpha"] == ff[0])
        mac(f"rrEv{t}AvgStopNext", f2(ff[0]))
        mac(f"rrEv{t}AvgStopWealthNext", f"{ff[1]:.1f}")
        mac(f"rrEv{t}CertNext", row_ff["n_certified"])
        EV[name] = {"min": bs["min"], "median": bs["median"], "lows": bs["lows"],
                    "last": {f"{k}@{'delta' if t_ == D else '2delta'}": v for (k, t_), v in av["last"].items()},
                    "first_fail_stopped_delta": ff, "rows": av["rows"]}
        say(f"| {name} | {f2(bs['min'])} | {f2(bs['median'])} | {lv_print(vals['AvgStop'])} | {lv_print(vals['AvgFinal'])} | "
            f"{lv_print(vals['AvgStopTwo'])} | {lv_print(vals['AvgFinalTwo'])} | {f2(ff[0])}, {ff[1]:.1f}, "
            f"{row_ff['n_certified']}/8, {' '.join(f'{w:.1f}' for w in row_ff['stopped'])} |")
        L.append(f"{LABEL[name]} & {f2(bs['min'])} & {f2(bs['median'])} & {lv_print(vals['AvgStop'])} & "
                 f"{lv_print(vals['AvgStopTwo'])} & {lv_print(vals['AvgFinal'])} & {lv_print(vals['AvgFinalTwo'])} \\\\")
        if name == "rhc":
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "aos_ref_realdata_evalues.tex").write_text("\n".join(L) + "\n")
    summary["evalues"] = EV
    say()
    allstop = [w for e in EV.values() for r in e["rows"] for w in r["stopped"] if w >= 1 / DELTA]
    mac("rrEvStopWealthMax", f"{max(allstop):.1f}")
    mac("rrEvStopWealthMin", f"{min(allstop):.1f}")
    gain = {nm: (e["last"]["log_stopped@delta"] or 0) - e["min"] for nm, e in EV.items()}
    say(f"A split that rejects stops just above 1/delta = 20: over every (contrast, split, level) the stopped wealths "
        f"that reached 20 lie between {min(allstop):.1f} and {max(allstop):.1f} [rrEvStopWealthMin, rrEvStopWealthMax]. "
        "So the stopped average reaches 20 only when nearly every split rejects, and it cannot exceed the minimum by "
        "much: gain over the minimum (level delta) " + ", ".join(f"{nm} {g:+.2f}" for nm, g in gain.items()) + ".")
    say()
    say("Per level (the order rule): number of the eight splits certified, average stopped wealth, average final wealth.")
    say()
    for name, e in EV.items():
        say(f"- {name}: " + "; ".join(f"{r['alpha']:.2f}: {r['n_certified']}/8, {r['avg_stopped']:.1f}, "
                                      f"{r['avg_final']:.3g}" for r in e["rows"]))
    say()

    # ============================================================================ 3. classifiers
    say("## 3. RHC: betting on classifier labels (M7.1)")
    say()
    say("Gradient boosting (scikit-learn HistGradientBoostingClassifier, defaults) and logistic regression (C = 1 after "
        "standardizing), fitted on the earlier batches of the ramp's schedule of each split (quarter 1 -> quarter 2, "
        "half 1 -> half 2, one wealth, horizon fixed in advance), on the split order and batch-0 standardization of "
        "the bound, every grid level; the hard label 1{p > training share of X} (the label of Section 7.1) and the "
        "prior-corrected probability (soft), with defaults and CV-tuned, exactly the code of the synthetic study "
        "(scripts/aos_baselines_run.py). Certified level read upward to the first failure; min over the eight splits "
        "(delta) and median (2 delta).")
    say()
    CL = {}
    meth = [("logit_m_hard", "LogitHard", "logistic, label"), ("gbm_m_hard", "GbmHard", "gradient boosting, label"),
            ("logit_m_soft", "LogitSoft", "logistic, probability"), ("gbm_m_soft", "GbmSoft",
                                                                       "gradient boosting, probability"),
            ("logit_mt_hard", "LogitTunedHard", "logistic tuned, label"),
            ("gbm_mt_hard", "GbmTunedHard", "gradient boosting tuned, label"),
            ("logit_mt_soft", "LogitTunedSoft", "logistic tuned, probability"),
            ("gbm_mt_soft", "GbmTunedSoft", "gradient boosting tuned, probability")]
    say("| method | 33 covariates: min / median | per split | 45 covariates: min / median | per split |")
    say("|---|---|---|---|---|")
    say(f"| split-sample bound (Laplace ramp) | {f2(B['rhc']['min'])} / {f2(B['rhc']['median'])} | "
        f"{' '.join(f2(x) for x in B['rhc']['lows'])} | {f2(B['rhc_full']['min'])} / {f2(B['rhc_full']['median'])} | "
        f"{' '.join(f2(x) for x in B['rhc_full']['lows'])} |")
    for nm in ("rhc", "rhc_full"):
        recs = sorted(jl(RAW / "ref_realdata_clf" / f"{nm}.jsonl"), key=lambda r: r["seed"])
        assert [r["seed"] for r in recs] == SEEDS, nm
        CL[nm] = {}
        for key, _, _ in meth:
            lows = [r["levels"][r["jlo"][key]] if r["jlo"][key] >= 0 else 0.0 for r in recs]
            none = sum(r["jlo"][key] < 0 for r in recs)
            CL[nm][key] = {"lows": lows, "min": min(lows), "median": float(np.median(lows)), "none": none}
        CL[nm]["_meandiff"] = {key: [r["diag"].get(f"{key}_meandiff") for r in recs] for key, _, _ in meth}
        CL[nm]["_tuned"] = {c: [r["diag"][f"{c}_tuned"] for r in recs] for c in ("gbm", "logit")}
    for key, t, lab in meth:
        a_, b_ = CL["rhc"][key], CL["rhc_full"][key]
        say(f"| {lab} ({key}) | {f2(a_['min'])} / {f2(a_['median'])} | {' '.join(f2(x) for x in a_['lows'])} | "
            f"{f2(b_['min'])} / {f2(b_['median'])} | {' '.join(f2(x) for x in b_['lows'])} |")
        for nm, P in (("rhc", "Paper"), ("rhc_full", "Full")):
            mac(f"rrClf{P}{t}Min", f2(CL[nm][key]["min"]))
            mac(f"rrClf{P}{t}Med", f2(CL[nm][key]["median"]))
    say()
    md_l = {nm: [np.mean(x) for x in CL[nm]["_meandiff"]["logit_m_hard"]] for nm in CL}
    say("Mean label difference E_P g - E_Q g on the betting batches (the quantity each label bet certifies; "
        "batch 1 / batch 2, mean over splits): " + "; ".join(
            f"{nm}: " + ", ".join(f"{key} {np.mean([d[0] for d in CL[nm]['_meandiff'][key]]):.3f} / "
                                  f"{np.mean([d[1] for d in CL[nm]['_meandiff'][key]]):.3f}"
                                  for key in ("logit_m_hard", "gbm_m_hard", "logit_mt_hard", "gbm_mt_hard")
                                  if CL[nm]["_meandiff"][key][0] is not None) for nm in CL) + ".")
    say()
    L = [f"% Generated by {GEN} -- do not edit. RHC, labels of classifiers fitted on the earlier batches of the ramp's "
         "schedule; min over eight splits ($\\delta$) / median ($2\\delta$).",
         r"\begin{tabular}{lcc}", r"\toprule", r" & 33 covariates & 45 covariates \\", r"\midrule",
         rf"split-sample bound $\widehat\alpha$ & {f2(B['rhc']['min'])} / {f2(B['rhc']['median'])} & "
         rf"{f2(B['rhc_full']['min'])} / {f2(B['rhc_full']['median'])} \\"]
    for key, t, lab in meth:
        L.append(f"{lab} & {f2(CL['rhc'][key]['min'])} / {f2(CL['rhc'][key]['median'])} & "
                 f"{f2(CL['rhc_full'][key]['min'])} / {f2(CL['rhc_full'][key]['median'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "aos_ref_realdata_clf.tex").write_text("\n".join(L) + "\n")
    summary["classifiers"] = {nm: {k: v for k, v in CL[nm].items()} for nm in CL}

    # ============================================================================ 4. propensity
    PR = json.loads((REF / "propensity.json").read_text())
    say("## 4. The propensity-score link (M7.1)")
    say()
    say("With e the propensity score and pi the catheterized share, p/q = {e/pi}/{(1-e)/(1-pi)}, so e(X) is a "
        "sufficient statistic for the two arms and alpha* = TV(P, Q) = TV(law of e(X) | catheterized, law of e(X) | "
        "others) = 1 - E_pooled min{e/pi, (1-e)/(1-pi)}. Replacing e by a fitted model e-hat gives a model-based "
        "estimate of alpha*. It rests entirely on the propensity model: it is the total variation along the one "
        "direction the logistic model can see, so a misspecified model understates alpha* (data processing, for a "
        "fixed function), while fitting e-hat in sample overstates it; it carries no finite-sample guarantee, unlike "
        "the bound. " + PR["model"] + ". Two estimators: " + "; ".join(f"{k} = {v}" for k, v in PR["estimands"].items())
        + ". Intervals: " + PR["bootstrap"] + ". Cross-fitted: " + PR["crossfit"] + ".")
    say()
    say("| model | 1 - OVL (Table S7) | 95% interval | 5% quantile | plug-in | 95% interval | 5% quantile | cross-fitted 1 - OVL / plug-in | share of e-hat outside [0.1, 0.9] |")
    say("|---|---|---|---|---|---|---|---|---|")
    for cs, P in (("paper", "Paper"), ("full", "Full")):
        r = PR["sets"][cs]
        say(f"| {r['d']} covariates | {f3(r['estimate']['one_minus_ovl'])} | ({f3(r['one_minus_ovl_ci95'][0])}, "
            f"{f3(r['one_minus_ovl_ci95'][1])}) | {f3(r['one_minus_ovl_lcl95'])} | {f3(r['estimate']['plugin'])} | "
            f"({f3(r['plugin_ci95'][0])}, {f3(r['plugin_ci95'][1])}) | {f3(r['plugin_lcl95'])} | "
            f"{f3(r['crossfit']['one_minus_ovl'])} / {f3(r['crossfit']['plugin'])} | {r['share_e_outside_0.1_0.9']:.3f} |")
        mac(f"rrProp{P}Ovl", f2(r["estimate"]["one_minus_ovl"]))
        mac(f"rrProp{P}OvlLo", f2(r["one_minus_ovl_ci95"][0]))
        mac(f"rrProp{P}OvlHi", f2(r["one_minus_ovl_ci95"][1]))
        mac(f"rrProp{P}OvlLcl", f2(r["one_minus_ovl_lcl95"]))
        mac(f"rrProp{P}Plug", f2(r["estimate"]["plugin"]))
        mac(f"rrProp{P}PlugLo", f2(r["plugin_ci95"][0]))
        mac(f"rrProp{P}PlugHi", f2(r["plugin_ci95"][1]))
        mac(f"rrProp{P}PlugLcl", f2(r["plugin_lcl95"]))
        mac(f"rrProp{P}OvlCf", f2(r["crossfit"]["one_minus_ovl"]))
        mac(f"rrProp{P}PlugCf", f2(r["crossfit"]["plugin"]))
    rp, rf_ = PR["sets"]["paper"], PR["sets"]["full"]
    ratio_p = B["rhc"]["min"] / rp["estimate"]["plugin"]
    ratio_f = B["rhc_full"]["min"] / rf_["estimate"]["plugin"]
    mac("rrPropRatioPaper", f2(ratio_p))
    mac("rrPropRatioFull", f2(ratio_f))
    say()
    say(f"Against the bound: 33 covariates, certified {f2(B['rhc']['min'])} vs plug-in {f3(rp['estimate']['plugin'])} "
        f"(ratio {ratio_p:.2f} [rrPropRatioPaper]) and 1 - OVL {f3(rp['estimate']['one_minus_ovl'])}; 45 covariates, "
        f"certified {f2(B['rhc_full']['min'])} vs plug-in {f3(rf_['estimate']['plugin'])} (ratio {ratio_f:.2f} "
        f"[rrPropRatioFull]) and 1 - OVL {f3(rf_['estimate']['one_minus_ovl'])}.")
    ovl_s7 = float(E33.loc["unweighted", "prop_overlap"])
    ovl_45 = float(E45.loc["unweighted", "prop_overlap"])
    say(f"Reproduction: 1 - OVL of the model refitted here is {f3(rp['estimate']['one_minus_ovl'])} on the 33 "
        f"covariates against 1 - {ovl_s7:.4f} = {1 - ovl_s7:.4f} from the overlap of the unweighted arms in "
        f"results/v3/tables/rhc_estimand.csv (Table S7), and {f3(rf_['estimate']['one_minus_ovl'])} on the 45 against "
        f"1 - {ovl_45:.4f} = {1 - ovl_45:.4f} from rhc_estimand_full.csv (difference "
        f"{abs(rp['estimate']['one_minus_ovl'] - (1 - ovl_s7)):.1e} and "
        f"{abs(rf_['estimate']['one_minus_ovl'] - (1 - ovl_45)):.1e}).")
    say()
    summary["propensity"] = PR

    # ============================================================================ 5. NHANES, units to batches
    say("## 5. NHANES: whole sampling units to batches (M8.1)")
    say()
    say("The public DEMO files carry no household identifier, so the clusters are the masked variance units, SDMVPSU "
        "within SDMVSTRA (strata 90-103 in 2011-12 and 104-118 in 2013-14, so the units of the two cycles are "
        "distinct). Per split seed the units are permuted and each goes, whole, to the batch in which the midpoint of "
        "its cumulative share of the contrast's participants falls (edges 1/4, 1/2), so no unit has participants in "
        "two batches; then the participants of each batch are permuted, sigma is the median distance on batch 0 and "
        "the rest is the bound's code (checked: on the participant-level order it reproduces stored records exactly). "
        "This keeps the fitting batches independent of the batches bet on across units; pairs within a betting batch "
        "can still share a unit, so within-unit dependence there is not removed (a bet on unit-level aggregates "
        "would be needed, with only about 30 units per betting batch).")
    say()
    say("| contrast | units | units per batch (range over splits) | batch-0 share of X / of Y (range) | participant split: min / median | unit split: min / median | per split (unit) | no bet at stop (unit) |")
    say("|---|---|---|---|---|---|---|---|")
    CLU = {}
    L = [f"% Generated by {GEN} -- do not edit. NHANES, split-sample bound with whole sampling units "
         r"(SDMVPSU within SDMVSTRA) assigned to batches, against the participant-level split of Table 2.",
         r"\begin{tabular}{lcccc}", r"\toprule",
         r" & \multicolumn{2}{c}{participants to batches} & \multicolumn{2}{c}{sampling units to batches} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"contrast & min ($\delta$) & median ($2\delta$) & min ($\delta$) & median ($2\delta$) \\", r"\midrule"]
    for name in ("nhanes_child", "nhanes_age", "nhanes_bmi"):
        ch = {}
        for s in SEEDS:
            ch[s] = {round(r["alpha"], 6): r for r in jl(RAW / "ref_realdata_cluster" / f"{name}_{s}.jsonl")}
        cb = bound_summary(name, chains=ch)
        pb = bound_summary(name)
        r0 = [ch[s][0.0] for s in SEEDS]
        upb = np.array([r["units_per_batch"] for r in r0])
        shx = [r["x_share_batch0"] for r in r0]
        shy = [r["y_share_batch0"] for r in r0]
        t = TAGS[name]
        CLU[name] = {"unit": cb, "participant": pb, "units": r0[0]["units"], "units_per_batch": upb.tolist(),
                     "x_share_batch0": shx, "y_share_batch0": shy, "sizes": [[r["n_batches"], r["m_batches"]]
                                                                              for r in r0],
                     "horizon": [r["horizon"] for r in r0]}
        mac(f"rrCl{t}Min", f2(cb["min"]))
        mac(f"rrCl{t}Med", f2(cb["median"]))
        mac(f"rrCl{t}Splits", " ".join(f2(x) for x in cb["lows"]))
        mac(f"rrCl{t}Units", r0[0]["units"])
        say(f"| {name} | {r0[0]['units']} | " + " / ".join(f"{upb[:, b].min()}-{upb[:, b].max()}" for b in range(3))
            + f" | {min(shx):.3f}-{max(shx):.3f} / {min(shy):.3f}-{max(shy):.3f} | {f2(pb['min'])} / {f2(pb['median'])} | "
            f"{f2(cb['min'])} / {f2(cb['median'])} | {' '.join(f2(x) for x in cb['lows'])} | {cb['nobet']} |")
        L.append(f"{LABEL[name]} & {f2(pb['min'])} & {f2(pb['median'])} & {f2(cb['min'])} & {f2(cb['median'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "aos_ref_realdata_cluster.tex").write_text("\n".join(L) + "\n")
    summary["cluster"] = CLU
    say()
    hz = {nm: (min(c["horizon"]), max(c["horizon"])) for nm, c in CLU.items()}
    say("Betting horizon N (pairs bet on), range over splits, unit split: " + "; ".join(
        f"{nm} {a:,}-{b:,}" for nm, (a, b) in hz.items()) + ".")
    say()

    # ============================================================================ observations (computed)
    say("## Observations and surprises (every statement computed from the numbers above)")
    say()
    o, nw = "rhc", "rhc_full"
    top = max(B[nw]["lows"])
    say(f"- RHC, 45 covariates: the minimum moves {f2(B[o]['min'])} -> {f2(B[nw]['min'])}, the median "
        f"{f2(B[o]['median'])} -> {f2(B[nw]['median'])}; {sum(x == top for x in B[nw]['lows'])}/8 splits certify "
        f"{f2(top)}, and at the next level {B[nw]['nobet']}/8 splits place no bet because the cut points cross in "
        "both batches: the bound stops where the genuine quantile cuts cross, not where the wealth runs out "
        f"(33 covariates: {B[o]['nobet']}/8).")
    an_same = (ANS[o]["last"], ANS[o]["fine"]) == (ANS[nw]["last"], ANS[nw]["fine"])
    say(f"- The analytic baseline {'does not change' if an_same else 'changes'}: alpha_last / bisection "
        f"{ANS[o]['last']} / {ANS[o]['fine']} -> {ANS[nw]['last']} / {ANS[nw]['fine']} (D-hat_0 "
        f"{f3(ANS[o]['D0'])} -> {f3(ANS[nw]['D0'])}, tau(0) {f3(ANS[o]['tau0'])} -> {f3(ANS[nw]['tau0'])}).")
    c_o, c_n = COH[o], COH[nw]
    say(f"- The cohort is now cut at alpha = {f2(c_n['alpha'])} instead of {f2(c_o['alpha'])}, yet it keeps "
        f"{c_n['kept_x']:,} / {c_n['kept_y']:,} patients against {c_o['kept_x']:,} / {c_o['kept_y']:,}, because "
        f"{c_n['frac_x']:,} / {c_n['frac_y']:,} of them carry fractional weights (was {c_o['frac_x']:,} / "
        f"{c_o['frac_y']:,}); the effective fractions are {f2(c_n['ess_frac_x'])} / {f2(c_n['ess_frac_y'])} (was "
        f"{f2(c_o['ess_frac_x'])} / {f2(c_o['ess_frac_y'])}). Largest abs SMD over its 45 covariates "
        f"{f3(c_n['max_abs_smd'])} ({c_n['worst']}), {c_n['n_abs_smd_above_0.1']} above 0.1; mortality difference "
        f"{f3(c_n['rd_dth30'])} ({f3(c_n['rd_lo'])}, {f3(c_n['rd_hi'])}), an interval that "
        f"{'contains' if c_n['rd_lo'] <= 0 <= c_n['rd_hi'] else 'excludes'} 0 (33 covariates: "
        f"{'contains' if c_o['rd_lo'] <= 0 <= c_o['rd_hi'] else 'excludes'} 0); counts certified exact for the "
        f"pooled run: {c_n['counts_exact']}, for {c_n['sens']['exact']} of the 8 batch-0 runs.")
    rd_o = {k: summary["comparison_rows"][k]["paper"]["rd_dth30"] for k in ROWS if k != "unweighted"}
    rd_n = {k: summary["comparison_rows"][k]["full"]["rd_dth30"] for k in ROWS if k != "unweighted"}
    rd_o["trimming"], rd_n["trimming"] = c_o["rd_dth30"], c_n["rd_dth30"]
    say(f"- Mortality difference over the five weightings: 33 covariates {min(rd_o.values()):.3f} "
        f"({min(rd_o, key=rd_o.get)}) to {max(rd_o.values()):.3f} ({max(rd_o, key=rd_o.get)}); 45 covariates "
        f"{min(rd_n.values()):.3f} ({min(rd_n, key=rd_n.get)}) to {max(rd_n.values()):.3f} "
        f"({max(rd_n, key=rd_n.get)}); unweighted {summary['comparison_rows']['unweighted']['paper']['rd_dth30']:.3f} "
        "under both (it does not depend on the covariates).")
    g = {nm: (e["last"]["log_stopped@delta"], e["min"]) for nm, e in EV.items()}
    up = [nm for nm, (a_, b_) in g.items() if a_ is not None and a_ > b_]
    gf = {nm: (e["last"]["log_final@delta"], e["min"]) for nm, e in EV.items()}
    upf = [nm for nm, (a_, b_) in gf.items() if a_ is not None and a_ > b_]
    say(f"- Averaging the stopped wealths certifies more than the minimum in {len(up)} of {len(EV)} contrasts"
        + (f" ({', '.join(up)})" if up else "") + "; averaging the final wealths, in " + str(len(upf))
        + (f" ({', '.join(f'{nm} {f2(gf[nm][1])} -> {f2(gf[nm][0])}' for nm in upf)})" if upf else "")
        + ". The stopped wealth of a rejecting split is capped just above 1/delta, so one split that fails pulls "
        "the average below 1/delta; the final wealth keeps growing after the crossing and is the more powerful "
        "e-value to average (both are valid; the choice must be fixed in advance).")
    best = {}
    for nm in CL:
        bm = max(CL[nm][k]["min"] for k, _, _ in meth)
        best[nm] = (bm, [k for k, _, _ in meth if CL[nm][k]["min"] == bm])
    say(f"- Classifier labels on the batches of the ramp: the best minimum over the eight variants is "
        f"{f2(best['rhc'][0])} ({', '.join(best['rhc'][1])}) on 33 covariates and {f2(best['rhc_full'][0])} "
        f"({', '.join(best['rhc_full'][1])}) on 45, against the split-sample bound's {f2(B['rhc']['min'])} and "
        f"{f2(B['rhc_full']['min'])}; betting the predicted probabilities instead of the labels certifies less "
        f"(best soft minimum {f2(max(CL['rhc'][k]['min'] for k, _, _ in meth if 'soft' in k))} and "
        f"{f2(max(CL['rhc_full'][k]['min'] for k, _, _ in meth if 'soft' in k))}).")
    say(f"- Propensity model (an estimate, not a bound): plug-in alpha* {f3(rp['estimate']['plugin'])} (95% interval "
        f"{f3(rp['plugin_ci95'][0])}-{f3(rp['plugin_ci95'][1])}) on 33 covariates, {f3(rf_['estimate']['plugin'])} "
        f"({f3(rf_['plugin_ci95'][0])}-{f3(rf_['plugin_ci95'][1])}) on 45; the certified bound is "
        f"{ratio_p:.2f} and {ratio_f:.2f} of it; cross-fitted {f3(rp['crossfit']['plugin'])} and "
        f"{f3(rf_['crossfit']['plugin'])}.")
    diffs = {nm: (c["participant"]["min"], c["unit"]["min"], c["participant"]["median"], c["unit"]["median"])
             for nm, c in CLU.items()}
    say("- NHANES with whole sampling units to batches, min / median, participant split -> unit split: "
        + "; ".join(f"{nm} {f2(a_)} / {f2(c_)} -> {f2(b_)} / {f2(d_)}" for nm, (a_, b_, c_, d_) in diffs.items())
        + ".")
    say()

    # ============================================================================ 6. checks
    say("## 6. Defaults and reproduction checks")
    say()
    chk = json.loads((REF / "check_generic.json").read_text())
    mac("rrCheckGeneric", f"{sum(c['identical'] for c in chk)}/{len(chk)}")
    say(f"- The batch-level code of the unit split (aos_ref_realdata.level_on_batches) on the participant-level order "
        f"reproduces stored records of results/aos/main/raw/realdata_laplace field for field in "
        f"{sum(c['identical'] for c in chk)}/{len(chk)} cells [rrCheckGeneric] ("
        + ", ".join(f"{c['contrast']} {c['seed']} {c['alpha']:.2f}" for c in chk) + ").")
    ext_n = sum(len(jl(f)) for f in (RAW / "ref_realdata_ext").glob("*.jsonl"))
    say(f"- Extension levels computed for the averaged e-values: {ext_n} (contrast, split, level) records; where a "
        "stored chain already holds a level the extension would have recomputed, it is read, not recomputed.")
    epc = REF / "rhc_estimand_paper_check.json"
    if epc.exists():
        e = json.loads(epc.read_text())
        d = e.get("check_vs_results_v3_rhc_estimand_csv_max_abs_diff", {})
        mac("rrCheckEstimandMaxDiff", sci(max(d.values())) if d else "--")
        say("- The comparison-row code (scripts/aos_ref_realdata.py estimand) on the 33 covariates against "
            "results/v3/tables/rhc_estimand.csv, largest absolute difference per row: "
            + ", ".join(f"{k} {v:.1e}" for k, v in d.items())
            + (" (kernel balancing not rerun: --skip-kernel)" if e.get("skip_kernel") else "") + ".")
    extra = REF / "default_checks.json"
    if extra.exists():
        for line in json.loads(extra.read_text()).get("lines", []):
            say(f"- {line}")
    say()

    # ============================================================================ files, macros
    lines = [f"% Generated by {GEN} -- do not edit."] + [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in MACROS]
    (TAB / "aos_ref_realdata_macros.tex").write_text("\n".join(lines) + "\n")
    REF.mkdir(parents=True, exist_ok=True)
    (REF / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    say("## 7. Files")
    say()
    for f in ("aos_ref_realdata_macros.tex", "aos_ref_realdata_rhc.tex", "aos_ref_realdata_cov.tex",
              "aos_ref_realdata_evalues.tex", "aos_ref_realdata_clf.tex", "aos_ref_realdata_cluster.tex"):
        say(f"- results/aos/tables/{f}")
    say("- results/aos/tables/aos_main_realdata.tex, aos_main_macros.tex (Table 2 and \\mn... macros; RHC row on the "
        "45 covariates: scripts/aos_main_tables.py, default --rhc-covariates full)")
    say("- results/aos/tables/aos_main_cohort.tex, aos_main_cohort_macros.tex, aos_main_cohort_balance.tex (tab:cohort "
        "and \\lc... macros on the 45 covariates: scripts/aos_main_cohort.py, default --covariates full)")
    say("- results/aos/main/ref_realdata/summary.json (every number here), propensity.json, rhc_estimand_full.csv, "
        "check_generic.json; raw records results/aos/main/raw/realdata_laplace/rhc_full_<seed>.jsonl, "
        "results/aos/main/raw/ref_realdata_{ext,cluster,clf}/; cohorts results/aos/main/cohort_full/; analytic "
        "results/aos/main/analytic_rd_full/")
    say()
    say(f"## 8. Macros (results/aos/tables/aos_ref_realdata_macros.tex, {len(MACROS)} macros)")
    say()
    say("```")
    for k, v in MACROS:
        say(f"\\{k} = {v}")
    say("```")
    print(f"{len(MACROS)} macros; {time.time() - t_start:.0f} s")
    return MD


COMMANDS = """## 9. Commands (each in `systemd-run --user --scope -q -p MemoryMax=3G -p MemorySwapMax=0 nice -n 10`, one at a time)

- PY=~/.venvs/llm_book/bin/python (as the stored real-data runs); SYS=/usr/bin/python3 (numba, for the cohort solver)
- AOS_MAIN_WORK=<local dir> PYTHONPATH=src $PY -B scripts/aos_main_realdata.py run --workers 3 --threads 2 --contrasts rhc_full
- PYTHONPATH=src $PY -I scripts/exp_valid_bandwidth.py --workers 2 --threads 2 --threshold aos --outdir <local dir> --pilot-only --datasets rhc_full (tables copied to results/aos/main/analytic_rd_full/tables)
- PYTHONPATH=src $SYS -B scripts/aos_realdata_cohort.py --kernel laplace --covariates full --run pooled batch0 --seed 1000 1001 1002 1003 1004 1005 1006 1007 --threads 4 --notes results/aos/main/ref_realdata/cohort_full.log
- PYTHONPATH=src $PY -B scripts/aos_ref_realdata.py check | propensity | classifiers | evalues --workers 3 | cluster --workers 3 | estimand --covariates full --threads 4 (and estimand --covariates paper, the check of section 6)
- PYTHONPATH=src $SYS -B scripts/aos_main_tables.py (default --rhc-covariates full; --rhc-covariates paper = the 33-covariate outputs)
- PYTHONPATH=src $SYS -B scripts/aos_main_cohort.py (default --covariates full; --covariates paper = the 33-covariate outputs and COHORT.md)
- PYTHONPATH=src $SYS -B scripts/aos_ref_realdata_tables.py (this file, the \\rr macros and the small tables)
- scripts/aos_ref_realdata_defaults.py: the default checks of section 6
- New options, defaults unchanged unless stated: realdata.load_rhc(covariates="paper"|"full", or the environment variable TRIMMEDMMD_RHC_COVARIATES); aos_main_realdata.py contrast rhc_full; exp_valid_bandwidth.py data set rhc_full; aos_realdata_cohort.py --covariates (default paper). The two table generators default to the full set, as asked: aos_main_tables.py --rhc-covariates (default full), aos_main_cohort.py --covariates (default full). These scripts write nothing under aos/ and do not run aos/tools/sync.sh. sync.sh copies aos_main_*.tex (Table 2, tab:cohort and their macros) but not the aos_ref_realdata_*.tex files; the paper needs a line there to read them.
"""


if __name__ == "__main__":
    md = main()
    (MAIN / "REFEREE_RUNS.md").write_text("\n".join(md) + "\n" + COMMANDS)
    print("wrote", MAIN / "REFEREE_RUNS.md")

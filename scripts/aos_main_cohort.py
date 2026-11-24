"""AoS restructure: the RHC cohort of Section sec:appcohort and Table tab:cohort with the Laplace kernel, the kernel of
the split-sample bound. Macros, tables and the Gaussian-vs-Laplace comparison behind results/aos/main/COHORT.md.

Inputs
    results/aos/main/cohort/<label>.{json,npz}     scripts/aos_realdata_cohort.py --kernel laplace (labels pooled,
                                                   batch0_seed1000 ... batch0_seed1007)
    results/aos/realdata/cohort/<label>.{json,npz} the stored Gaussian runs (same labels, plus old)
    results/v3/tables/rhc_estimand.csv             the other five rows of Table tab:cohort (scripts/v3_rhc_estimand.py);
                                                   none of them depends on the trimming, so they are reused (and the
                                                   unweighted, matching and overlap rows are recomputed as a check)
Outputs (Laplace)
    results/aos/tables/aos_main_cohort_macros.tex  \\lc... : the cohort macros of scripts/aos_realdata_tables.py with
                                                   the prefix rd -> lc (lcPool = pooled moments and pooled median
                                                   distance, lcCoh = batch 0 of split seed 1000, lcSens = the eight
                                                   seeds) and the macros of scripts/v3_rhc_estimand.py with est -> lc,
                                                   the trimming row lcTrim... being the run --trim-run (default pooled,
                                                   as the Gaussian text); plus lcRDmin / lcRDmax, the range of the
                                                   mortality difference over the five weightings
    results/aos/tables/aos_main_cohort.tex         Table tab:cohort, headers of aos/tables/estimand_aos.tex, Laplace
                                                   trimming row
    results/aos/tables/aos_main_cohort_balance.tex the table of aos_cohort_balance.tex for the Laplace runs
    results/aos/main/cohort_summary.json, cohort_smd.csv, cohort_compare.json, COHORT.md

Check of the Gaussian default (no stored file is written): --check-gaussian DIR runs the same code with the prefix rd
on the Gaussian cohort directory DIR and compares every generated cohort macro with results/aos/tables/
aos_realdata_macros.tex, the balance table with aos_cohort_balance.tex and the tab:cohort generator (Gaussian pooled
trimming row) with aos/tables/estimand_aos.tex; outputs go to --scratch.

    PYTHONPATH=src python3 -B scripts/aos_main_cohort.py
    PYTHONPATH=src python3 -B scripts/aos_main_cohort.py --check-gaussian results/aos/realdata/cohort --scratch /tmp/x

RHC covariate set (``--covariates``, AoS referee runs 2026-11-22). The default ``full`` builds the three tables above
(aos_main_cohort_macros.tex with the same \\lc... names, aos_main_cohort.tex, aos_main_cohort_balance.tex) from the
45-covariate Laplace cohort at its own certified level (results/aos/main/cohort_full/, scripts/aos_realdata_cohort.py
--kernel laplace --covariates full), with the five comparison rows recomputed on the 45 covariates
(results/aos/main/ref_realdata/rhc_estimand_full.csv, scripts/aos_ref_realdata.py estimand --covariates full) and the
balance over the 45; it writes cohort_full_summary.json and cohort_full_smd.csv and leaves COHORT.md,
cohort_compare.json, cohort_summary.json and cohort_smd.csv (the 33-covariate record, whose Gaussian comparison has no
45-covariate counterpart) as they are; the old-vs-new comparison is in results/aos/main/REFEREE_RUNS.md
(scripts/aos_ref_realdata_tables.py). ``--covariates paper`` is the code above, the 33-covariate outputs of 2026-11-22.

    PYTHONPATH=src python3 -B scripts/aos_main_cohort.py                       # 45 covariates
    PYTHONPATH=src python3 -B scripts/aos_main_cohort.py --covariates paper    # 33 covariates (and COHORT.md)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import aos_realdata_tables as RT  # noqa: E402

MAIN = ROOT / "results" / "aos" / "main"
LCDIR = MAIN / "cohort"
CHK = MAIN / "cohort_check"                  # check records (not runs: load_dir reads every *.json of a run dir)
GDIR = ROOT / "results" / "aos" / "realdata" / "cohort"
TAB = ROOT / "results" / "aos" / "tables"
EST_CSV = ROOT / "results" / "v3" / "tables" / "rhc_estimand.csv"
EST_AOS = ROOT / "aos" / "tables" / "estimand_aos.tex"
GEN = "scripts/aos_main_cohort.py"
ORDER = ["unweighted", "trimming", "atm", "ato", "energy", "kernel"]
CAP = {"unweighted": "Unw", "trimming": "Trim", "atm": "Atm", "ato": "Ato", "energy": "Energy", "kernel": "Kernel"}
TLABEL = {"unweighted": "all patients", "trimming": r"trimming, $\alpha=\cohAlpha$", "atm": "matching weights",
          "ato": "overlap weights", "energy": "energy balancing", "kernel": "kernel balancing"}
TTARGET = {"unweighted": "--", "trimming": "common core", "atm": "overlap (ATM)", "ato": "overlap (ATO)",
           "energy": "pooled (ATE)", "kernel": "pooled (ATE)"}
QCOLS = ["ess_frac_x", "ess_frac_y", "max_abs_smd", "mean_abs_smd", "prop_overlap", "rd_dth30", "rd_lo", "rd_hi"]


def fmt(x, nd=2):
    return f"{x:.{nd}f}"


def load_dir(d):
    return {f.stem: json.loads(f.read_text()) for f in sorted(Path(d).glob("*.json")) if ".tmp" not in f.name}


def macro_collector():
    lines = []

    def M(name, val):
        if r"\times" in val:
            val = rf"\ensuremath{{{val}}}"
        lines.append(rf"\newcommand{{\{name}}}{{{val}}}")
    return lines, M


def parse_macros(lines):
    out = {}
    for s in lines:
        mt = re.match(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}$", s)
        if mt:
            out[mt.group(1)] = mt.group(2)
    return out


# ------------------------------------------------------------------------------------------------------------------
# the five comparison rows of Table tab:cohort, and a recomputation check of the kernel-free ones
# ------------------------------------------------------------------------------------------------------------------

def comparison_rows(est_csv=None):
    est = pd.read_csv(EST_CSV if est_csv is None else est_csv).set_index("key")
    return {k: {c: float(est.loc[k, c]) for c in QCOLS + ["n_abs_smd_above_0.1"]} |
            {"worst_covariate_after": str(est.loc[k, "worst_covariate_after"])} for k in est.index}


def check_comparison_rows(rows):
    """Recompute the unweighted, matching and overlap rows (propensity only) with the bootstrap stream of
    scripts/v3_rhc_estimand.py (one default_rng(20261117), resamples drawn in the order unweighted, trimming, energy,
    kernel, ato, atm; the draws do not depend on the weights). Returns the largest absolute difference."""
    from trimmedmmd.experiments.realdata_run import _logistic_fit, _overlap, standardized_differences
    from trimmedmmd.realdata import load_rhc
    from v3_rhc_estimand import boot_ci, risk_difference, rhc_outcome
    X, Y, _, _ = load_rhc()
    n, m = len(X), len(Y)
    _, prop = _logistic_fit(np.vstack([X, Y]), np.r_[np.ones(n), np.zeros(m)])
    ex, ey = prop[:n], prop[n:]
    yx, yy = rhc_outcome()
    mx, my = np.minimum(ex, 1 - ex), np.minimum(ey, 1 - ey)
    W = {"unweighted": (np.ones(n), np.ones(m)), "ato": (1.0 - ex, ey), "atm": (mx / ex, my / (1 - ey))}
    brng = np.random.default_rng(20261117)
    diff = 0.0
    for k in ["unweighted", "trimming", "energy", "kernel", "ato", "atm"]:
        w, v = W.get(k, (np.ones(n), np.ones(m)))
        lo, hi = boot_ci(yx, yy, w, v, brng)
        if k not in W:
            continue
        smd = standardized_differences(X, Y, w / w.sum(), v / v.sum())
        mine = {"ess_frac_x": w.sum() ** 2 / (w ** 2).sum() / n,
                "ess_frac_y": v.sum() ** 2 / (v ** 2).sum() / m, "max_abs_smd": np.abs(smd).max(),
                "mean_abs_smd": np.abs(smd).mean(), "prop_overlap": _overlap(ex, ey, w, v),
                "rd_dth30": risk_difference(yx, yy, w, v), "rd_lo": lo, "rd_hi": hi}
        diff = max(diff, max(abs(float(mine[c]) - rows[k][c]) for c in QCOLS))
    return diff


# ------------------------------------------------------------------------------------------------------------------
# generation
# ------------------------------------------------------------------------------------------------------------------

def trim_row(summary, run):
    s = summary[run]
    return {c: float(s[c]) for c in QCOLS} | {"n_abs_smd_above_0.1": int(s["n_abs_smd_above_0.1"])}


def table_lines(rows):
    L = [f"% Generated by {GEN} -- do not edit.",
         r"\begin{tabular}{llcccccc}", r"\toprule",
         r"RHC & target & \multicolumn{2}{c}{effective fraction} & max $|$SMD$|$ & mean $|$SMD$|$ & overlap & "
         r"risk diff. \\",
         r" & population & cath. & others & & & & (95\% int.) \\", r"\midrule"]
    for k in ORDER:
        r = rows[k]
        L.append(f"{TLABEL[k]} & {TTARGET[k]} & {fmt(r['ess_frac_x'])} & {fmt(r['ess_frac_y'])} & "
                 f"{fmt(r['max_abs_smd'], 3)} & {fmt(r['mean_abs_smd'], 3)} & {fmt(r['prop_overlap'])} & "
                 f"{fmt(r['rd_dth30'], 3)} ({fmt(r['rd_lo'], 3)}, {fmt(r['rd_hi'], 3)}) \\\\")
        if k in ("trimming", "ato"):
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]
    return L


def estimand_macros(rows, M, P, have):
    """The macros of scripts/v3_rhc_estimand.py with est -> P; names already written (the Unw risk differences of
    the cohort block) are skipped after checking that the value agrees."""
    from v3_rhc_estimand import B, rhc_outcome
    yx, yy = rhc_outcome()
    n, m = len(yx), len(yy)
    vals = []
    for k in ORDER:
        r = rows[k]
        c = CAP[k]
        vals += [(f"{P}{c}RD", fmt(r["rd_dth30"], 3)), (f"{P}{c}RDlo", fmt(r["rd_lo"], 3)),
                 (f"{P}{c}RDhi", fmt(r["rd_hi"], 3)), (f"{P}{c}SMD", fmt(r["max_abs_smd"], 3)),
                 (f"{P}{c}EssX", fmt(r["ess_frac_x"])), (f"{P}{c}EssY", fmt(r["ess_frac_y"]))]
    vals += [(f"{P}PiTreated", fmt(n / (n + m))), (f"{P}Boot", f"{B:,}".replace(",", "{,}")),
             (f"{P}CrudeX", fmt(yx.mean(), 3)), (f"{P}CrudeY", fmt(yy.mean(), 3))]
    rds = {k: rows[k]["rd_dth30"] for k in ORDER if k != "unweighted"}
    kmin, kmax = min(rds, key=rds.get), max(rds, key=rds.get)
    vals += [(f"{P}RDmin", fmt(rds[kmin], 3)), (f"{P}RDmax", fmt(rds[kmax], 3)),
             (f"{P}RDminWeighting", TLABEL[kmin].split(",")[0]), (f"{P}RDmaxWeighting", TLABEL[kmax].split(",")[0])]
    for name, val in vals:
        if name in have:
            assert have[name] == val, (name, have[name], val)
            continue
        M(name, val)
    return kmin, kmax


def generate(cdir, P, out_macros, out_table, out_balance, smd_csv, summary_json, trim_run, est_csv=None):
    coh = load_dir(cdir)
    lines, M = macro_collector()
    lines.append(f"% Generated by {GEN} from {Path(cdir).relative_to(ROOT) if Path(cdir).is_relative_to(ROOT) else cdir}"
                 " -- do not edit.")
    summary = RT.write_cohort(coh, M, cdir=cdir, P=P, smd_csv=smd_csv, summary_json=summary_json,
                              balance_tex=out_balance, generator=GEN)
    rows = comparison_rows(est_csv)
    rows["trimming"] = trim_row(summary, trim_run)
    have = parse_macros(lines)
    kmin, kmax = estimand_macros(rows, M, P, have)
    Path(out_macros).write_text("\n".join(lines) + "\n")
    Path(out_table).write_text("\n".join(table_lines(rows)) + "\n")
    return coh, summary, rows, lines, (kmin, kmax)


# ------------------------------------------------------------------------------------------------------------------
# check of the Gaussian default
# ------------------------------------------------------------------------------------------------------------------

def check_gaussian(gdir, scratch):
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    coh, summary, rows, lines, _ = generate(gdir, "rd", scratch / "rd_macros.tex", scratch / "estimand_check.tex",
                                            scratch / "balance.tex", scratch / "cohort_smd.csv",
                                            scratch / "cohort_summary.json", "pooled")
    mine = parse_macros(lines)
    stored = parse_macros((TAB / "aos_realdata_macros.tex").read_text().splitlines())
    est_names = {f"rd{CAP[k]}{s}" for k in ORDER for s in ("RD", "RDlo", "RDhi", "SMD", "EssX", "EssY")} | {
        "rdPiTreated", "rdBoot", "rdCrudeX", "rdCrudeY", "rdRDmin", "rdRDmax", "rdRDminWeighting", "rdRDmaxWeighting"}
    est_names -= {"rdUnwRD", "rdUnwRDlo", "rdUnwRDhi"}           # these are also cohort macros (checked below)
    cohort_names = [k for k in mine if k not in est_names]
    same = [k for k in cohort_names if stored.get(k) == mine[k]]
    differ = {k: (stored.get(k), mine[k]) for k in cohort_names if stored.get(k) != mine[k]}
    missing = sorted(k for k in stored if k not in mine and re.match(r"rd(Old|Pool|Coh|Sens|Unw)", k))
    bal_stored = (TAB / "aos_cohort_balance.tex").read_text().splitlines()[1:]
    bal_mine = (scratch / "balance.tex").read_text().splitlines()[1:]
    est_stored = EST_AOS.read_text().splitlines()[2:]
    est_mine = (scratch / "estimand_check.tex").read_text().splitlines()[1:]
    est_aos = parse_macros((ROOT / "aos" / "tables" / "estimand_macros.tex").read_text().splitlines())
    est_differ = {k: (est_aos.get("est" + k[2:]), mine[k]) for k in est_names
                  if "est" + k[2:] in est_aos and est_aos["est" + k[2:]] != mine[k]}
    res = {"dir": str(gdir), "labels": sorted(coh), "cohort_macros": len(cohort_names), "equal": len(same),
           "differ": differ, "stored_cohort_macros_not_generated": missing,
           "balance_table_equal": bal_stored == bal_mine,
           "balance_rows_equal": [sum(r in bal_stored for r in bal_mine), len(bal_mine)],
           "balance_table_diff": [(a, b) for a, b in zip(bal_stored, bal_mine) if a != b],
           "estimand_table_equal_to_aos_estimand_aos_tex": est_stored == est_mine,
           "estimand_table_diff": [(a, b) for a, b in zip(est_stored, est_mine) if a != b],
           "estimand_macros_differ_from_aos_estimand_macros": est_differ,
           "comparison_rows_recomputed_max_abs_diff": check_comparison_rows(rows)}
    (scratch / "check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res


# ------------------------------------------------------------------------------------------------------------------
# Gaussian vs Laplace
# ------------------------------------------------------------------------------------------------------------------

def compare(gcoh, lcoh, gsum, lsum, grows, lrows):
    from trimmedmmd.realdata import load_rhc
    Xraw, Yraw, names, _ = load_rhc(standardize=False)
    n, m = len(Xraw), len(Yraw)
    out = {"labels": sorted(set(gcoh) & set(lcoh)), "per_label": {}}
    for k in out["labels"]:
        g, l_ = gcoh[k], lcoh[k]
        a0 = g["alpha"]
        cx, cy = 1 / (n * (1 - a0)), 1 / (m * (1 - a0))
        Wg, Wl = np.load(GDIR / f"{k}.npz"), np.load(LCDIR / f"{k}.npz")
        rec = {}
        for arm, key, c in (("x", "w", cx), ("y", "v", cy)):
            kg, kl = Wg[key] > 1e-9 * c, Wl[key] > 1e-9 * c
            capg, capl = Wg[key] >= c * (1 - 1e-9), Wl[key] >= c * (1 - 1e-9)
            rec[f"jaccard_kept_{arm}"] = float((kg & kl).sum() / (kg | kl).sum())
            rec[f"kept_both_{arm}"] = int((kg & kl).sum())
            rec[f"kept_only_gauss_{arm}"] = int((kg & ~kl).sum())
            rec[f"kept_only_laplace_{arm}"] = int((~kg & kl).sum())
            rec[f"jaccard_cap_{arm}"] = float((capg & capl).sum() / (capg | capl).sum())
            rec[f"l1_weights_{arm}"] = float(np.abs(Wg[key] - Wl[key]).sum())
            rec[f"tv_weights_{arm}"] = 0.5 * rec[f"l1_weights_{arm}"]
        smd_g, smd_l = np.abs(np.array(g["describe"]["smd"])), np.abs(np.array(l_["describe"]["smd"]))
        rec["worst_cov_gauss"], rec["worst_cov_laplace"] = names[int(smd_g.argmax())], names[int(smd_l.argmax())]
        rec["top5_laplace"] = [(names[i], float(smd_l[i]), float(smd_g[i])) for i in np.argsort(-smd_l)[:5]]
        for kk in ("max_abs_smd", "mean_abs_smd", "n_abs_smd_above_0.1", "kept_x", "kept_y", "frac_x", "frac_y",
                   "cap_x", "cap_y", "zero_x", "zero_y", "ess_frac_x", "ess_frac_y", "prop_overlap", "rd_dth30",
                   "rd_lo", "rd_hi", "D_hat", "gap", "gap_longdouble", "tau", "sigma"):
            rec[kk] = [gsum[k][kk], lsum[k][kk]]
        for kk in ("gram_eig_min", "gram_eig_max", "seconds_total"):
            rec[kk] = [g["solver_info"][kk], l_["solver_info"][kk]]
        rec["smo_steps"] = [g["solver_info"]["smo"]["steps"], l_["solver_info"]["smo"]["steps"]]
        rec["polish"] = [g["solver_info"]["polish"], l_["solver_info"]["polish"]]
        rec["fista_gap"] = [g["solver_info"]["fista"]["gap"], l_["solver_info"]["fista"]["gap"]]
        rec["viol_end"] = [g["solver_info"]["smo"]["viol_end"], l_["solver_info"]["smo"]["viol_end"]]
        rec["certify"] = [g["certify"], l_["certify"]]
        rec["cut_normalized"] = [[g["structure"]["x"]["cut_normalized"], g["structure"]["y"]["cut_normalized"]],
                                 [l_["structure"]["x"]["cut_normalized"], l_["structure"]["y"]["cut_normalized"]]]
        out["per_label"][k] = rec
    out["table_rows"] = {"gaussian_pooled_trimming": grows["trimming"], "laplace_trimming": lrows["trimming"],
                         "other": {k: grows[k] for k in ORDER if k != "trimming"}}
    return out


def tight_check(tdir, out):
    """Laplace runs at a tighter KKT tolerance (TDIR, --eps 1e-15) against the runs of results/aos/main/cohort
    (--eps 1e-14): gap, counts, every summary of the tables, and the largest weight difference in caps."""
    from trimmedmmd.realdata import load_rhc
    Xraw, Yraw, _, _ = load_rhc(standardize=False)
    n, m = len(Xraw), len(Yraw)
    res = {"dir": str(tdir), "labels": {}}
    keys = ["cap_x", "zero_x", "frac_x", "kept_x", "cap_y", "zero_y", "frac_y", "kept_y", "max_abs_smd",
            "mean_abs_smd", "n_abs_smd_above_0.1", "ess_frac_x", "ess_frac_y", "prop_overlap", "rd_dth30"]
    for f in sorted(Path(tdir).glob("*.json")):
        if ".tmp" in f.name or not (LCDIR / f.name).exists():
            continue
        t, b = json.loads(f.read_text()), json.loads((LCDIR / f.name).read_text())
        Wt, Wb = np.load(f.with_suffix(".npz")), np.load(LCDIR / f"{f.stem}.npz")
        a0 = b["alpha"]
        cx, cy = 1 / (n * (1 - a0)), 1 / (m * (1 - a0))
        r = {"eps": [b["solver_info"]["eps"], t["solver_info"]["eps"]],
             "gap": [max(b["solver_info"]["gap"], b["solver_info"]["gap_longdouble"]),
                     max(t["solver_info"]["gap"], t["solver_info"]["gap_longdouble"])],
             "gap_float64_minus_extended": [abs(b["solver_info"]["gap"] - b["solver_info"]["gap_longdouble"]),
                                            abs(t["solver_info"]["gap"] - t["solver_info"]["gap_longdouble"])],
             "smo_steps": [b["solver_info"]["smo"]["steps"], t["solver_info"]["smo"]["steps"]],
             "viol_end": [b["solver_info"]["smo"]["viol_end"], t["solver_info"]["smo"]["viol_end"]],
             "D_hat": [b["solver_info"]["D_hat"], t["solver_info"]["D_hat"]],
             "counts_exact": [b["certify"]["counts_exact_for_minimizer"], t["certify"]["counts_exact_for_minimizer"]],
             "max_weight_diff_over_cap": max(float(np.abs(Wt["w"] - Wb["w"]).max() / cx),
                                             float(np.abs(Wt["v"] - Wb["v"]).max() / cy)),
             "differing_summaries": {k: [b["describe"][k], t["describe"][k]] for k in keys
                                     if (round(b["describe"][k], 3) if isinstance(b["describe"][k], float)
                                         else b["describe"][k]) !=
                                     (round(t["describe"][k], 3) if isinstance(t["describe"][k], float)
                                      else t["describe"][k])}}
        res["labels"][f.stem] = r
    parts = []
    for k, r in res["labels"].items():
        parts.append(f"{k}: gap {r['gap'][0]:.1e} -> {r['gap'][1]:.1e} (float64 vs extended evaluations differ by "
                     f"{r['gap_float64_minus_extended'][1]:.1e}), SMO steps {r['smo_steps'][0]:,} -> "
                     f"{r['smo_steps'][1]:,}, largest weight change {r['max_weight_diff_over_cap']:.1e} caps, "
                     + ("counts and every tabulated summary (3 decimals) unchanged" if not r["differing_summaries"]
                        else f"changed: {r['differing_summaries']}"))
    res["text"] = ("Tighter tolerance (KKT violation 1e-15, the same solver otherwise; outputs in a scratch "
                   "directory, not used for the tables): " + "; ".join(parts) + ".")
    Path(out).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-gaussian", default=None)
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--trim-run", default="pooled")
    ap.add_argument("--tight", default=None, help="directory of Laplace runs at a tighter --eps")
    ap.add_argument("--full-kernel", choices=["gaussian", "laplace"], default="gaussian",
                    help="kernel of the 45-covariate cohort (default gaussian, with the Laplace program as comparison)")
    ap.add_argument("--covariates", choices=["full", "paper"], default="full",
                    help="RHC covariate set (default full = 45; paper = the 33-covariate outputs), see the docstring")
    a = ap.parse_args()
    if a.covariates == "full" and not (a.tight or a.check_gaussian):
        full_main(a)
        return
    if a.tight:
        CHK.mkdir(exist_ok=True)
        tight_check(Path(a.tight), CHK / "tight_eps.json")
        return
    if a.check_gaussian:
        CHK.mkdir(exist_ok=True)
        res = check_gaussian(Path(a.check_gaussian).resolve(), a.scratch or "/tmp/aos_main_cohort_check")
        if a.scratch is None or Path(a.check_gaussian).resolve() == GDIR.resolve():
            (CHK / "check_gaussian.json").write_text(json.dumps(res, indent=1))
        else:
            (CHK / "check_gaussian_rerun.json").write_text(json.dumps(res, indent=1))
        return
    t0 = time.time()
    lcoh, lsum, lrows, llines, (kmin, kmax) = generate(
        LCDIR, "lc", TAB / "aos_main_cohort_macros.tex", TAB / "aos_main_cohort.tex",
        TAB / "aos_main_cohort_balance.tex", MAIN / "cohort_smd.csv", MAIN / "cohort_summary.json", a.trim_run)
    for k, c in lcoh.items():
        assert c.get("kernel") == "laplace", k
    # the Gaussian side, recomputed from the stored runs into memory only (nothing stored is rewritten)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        gcoh, gsum, grows, glines, _ = generate(GDIR, "rd", Path(td) / "m.tex", Path(td) / "t.tex",
                                                Path(td) / "b.tex", Path(td) / "s.csv", Path(td) / "s.json",
                                                a.trim_run)
    cmp_ = compare(gcoh, lcoh, gsum, lsum, grows, lrows)
    cmp_["rd_range"] = {"laplace": [kmin, kmax]}
    cmp_["macros_lc"] = parse_macros(llines)
    cmp_["macros_rd"] = parse_macros(glines)
    (MAIN / "cohort_compare.json").write_text(json.dumps(cmp_, indent=1, default=float))
    from aos_main_cohort_report import write_report
    write_report(cmp_, lcoh, gcoh, MAIN / "COHORT.md")
    print(f"wrote {TAB / 'aos_main_cohort_macros.tex'}, {TAB / 'aos_main_cohort.tex'}, "
          f"{TAB / 'aos_main_cohort_balance.tex'}, {MAIN / 'COHORT.md'} in {time.time() - t0:.0f} s")


def full_main(a):
    """--covariates full: the tables and macros of the 45-covariate cohort (see the docstring).

    Since 2026-11-24 (author: "Laplace for the bound, Gaussian for the trimming") the cohort of Section sec:appcohort is
    the Gaussian trimming at the level of the Laplace bound (results/aos/main/cohort_full_gauss/, scripts/
    aos_realdata_cohort.py --kernel gaussian --covariates full), with the \\lc... names as before; the Laplace program at
    the same level and covariates (results/aos/main/cohort_full/) gives the comparison macros \\lp... (prefix lc -> lp),
    of which the Pool, Coh and Sens counts are appended to aos_main_cohort_macros.tex. --full-kernel laplace restores
    the earlier outputs (Laplace cohort, no comparison)."""
    import os
    import tempfile
    from trimmedmmd.realdata import RHC_COVARIATES_ENV, rhc_covariate_names
    os.environ[RHC_COVARIATES_ENV] = "full"            # load_rhc() of write_cohort and rhc_outcome: 45 covariates
    # the level of this cohort is its own macro (\\cohAlpha is the 0.30 of the earlier cohorts)
    TLABEL["trimming"] = r"trimming, $\alpha=\lcCohAlpha$"
    t0 = time.time()
    kern = a.full_kernel
    fdir = MAIN / ("cohort_full_gauss" if kern == "gaussian" else "cohort_full")
    est_full = MAIN / "ref_realdata" / "rhc_estimand_full.csv"
    lcoh, lsum, lrows, llines, (kmin, kmax) = generate(
        fdir, "lc", TAB / "aos_main_cohort_macros.tex", TAB / "aos_main_cohort.tex",
        TAB / "aos_main_cohort_balance.tex", MAIN / "cohort_full_smd.csv", MAIN / "cohort_full_summary.json",
        a.trim_run, est_csv=est_full)
    names = rhc_covariate_names("full")
    for k, c in lcoh.items():
        assert c.get("kernel") == kern and c.get("covariates") == "full" and c.get("names") == names, k
        assert len(c["describe"]["smd"]) == len(names), k
    est = pd.read_csv(est_full)
    assert int(est["n"].iloc[0]) + int(est["m"].iloc[0]) == 5735
    if kern == "gaussian":
        with tempfile.TemporaryDirectory() as td:
            pcoh, _, _, plines, _ = generate(MAIN / "cohort_full", "lp", Path(td) / "m.tex", Path(td) / "t.tex",
                                             Path(td) / "b.tex", Path(td) / "s.csv", Path(td) / "s.json",
                                             a.trim_run, est_csv=est_full)
        for k, c in pcoh.items():
            assert c.get("kernel") == "laplace" and c.get("covariates") == "full", k
            assert c["alpha"] == lcoh[k]["alpha"], k
        keep = [ln for ln in plines if re.match(r"\\newcommand\{\\lp(Pool|Coh|Sens)", ln)]
        mfile = TAB / "aos_main_cohort_macros.tex"
        txt = mfile.read_text().rstrip("\n") + "\n% Laplace program at the same level and covariates " \
            "(results/aos/main/cohort_full), for comparison\n" + "\n".join(keep) + "\n"
        mfile.write_text(txt)
        assert mfile.read_text() == txt
    print(f"wrote {TAB / 'aos_main_cohort_macros.tex'}, {TAB / 'aos_main_cohort.tex'}, "
          f"{TAB / 'aos_main_cohort_balance.tex'}, {MAIN / 'cohort_full_summary.json'} (45 covariates, {kern} kernel, "
          f"alpha = {lcoh['batch0_seed1000']['alpha']}) in {time.time() - t0:.0f} s")

if __name__ == "__main__":
    main()

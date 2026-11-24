"""AoS restructure (plan section 4.4): Table 1, Table 2, the Figure 2(a) data, the macros and REPORT.md of the main runs.
No Monte Carlo work here and no hand-typed number: everything is read from

    results/aos/main/raw/synth_laplace/<design>_n<n>.jsonl    scripts/aos_main_synth.py (Laplace ramp, all levels, n = 800)
    results/aos/main/raw/synth_gaussian/<design>_n<n>.jsonl   the same with the Gaussian kernel (reproduction check)
    results/aos/main/raw/alpha0_laplace/<design>_n<n>.jsonl   scripts/aos_main_alpha0.py (Laplace ramp at alpha = 0)
    results/aos/main/raw/alpha0_gaussian/...                  the same with the Gaussian kernel (reproduction check)
    results/aos/main/raw/analytic/<design>.jsonl              scripts/aos_main_analytic.py (analytic, tau = min{tau^P, tau^A})
    results/aos/main/raw/realdata_laplace/<contrast>_<seed>.jsonl  scripts/aos_main_realdata.py
    results/aos/main/analytic_rd/tables/r12_valid_bandwidth.csv    scripts/exp_valid_bandwidth.py --threshold aos
and, stored and read-only, explore/harder_sims/out (Gaussian ramp, classifiers; data of the study), results/aos/baselines/raw
(permutation MMD, MMDAgg, classifiers on the batches of the ramp), results/aos/fallback/raw (the 50-replicate kernel
comparison), results/aos/realdata/wealth (the Gaussian real-data runs), results/harder/popmmd.csv (population MMD).

Writes
    results/aos/tables/aos_main_n80.tex            Table 1 (Laplace ramp, analytic with the new threshold, GBM / logistic
                                                   + betting, permutation MMD, MMDAgg)
    results/aos/tables/aos_main_n80_heldout.tex    the same on replicates 50-199 (and 200050-200199 of the extension)
    results/aos/tables/aos_main_n80_gauss.tex      S6: Gaussian ramp (stored) next to the Laplace ramp and the analytic test
    results/aos/tables/aos_main_recovery_gauss.tex S6: recovery at n = 800, Laplace / Gaussian ramp, analytic, classifiers
    results/aos/tables/aos_main_realdata.tex       Table 2
    results/aos/tables/aos_main_macros.tex         macros (prefix \\mn)
    results/aos/main/fig_recovery_main.json        recovery figure data (Laplace ramp); fig_recovery_gauss.json (S6)
    results/aos/main/fig2_data_aos.json, fig2_small_aos.json   S6 panel (b) data, analytic under the new threshold
    results/aos/main/REPORT.md                     headline numbers, observations, checks, every file and macro

Column headers use the AoS notation: alpha-hat for the split-sample bound, alpha-hat_an for the analytic estimator.

RHC covariate set (``--rhc-covariates``, AoS referee runs 2026-11-22): the default ``full`` takes the RHC row of Table 2
and its macros (mnRdRhc..., mnPvRhc...) from the 45-covariate runs -- the split-sample bound of
results/aos/main/raw/realdata_laplace/rhc_full_<seed>.jsonl (scripts/aos_main_realdata.py --contrasts rhc_full) and the
analytic baseline of results/aos/main/analytic_rd_full/tables/r12_valid_bandwidth.csv (scripts/exp_valid_bandwidth.py
--threshold aos --datasets rhc_full) --; ``paper`` reproduces the earlier outputs on the 33 covariates exactly. Every
other number is the same under both. The Gaussian real-data runs and the stored tau_new column of REPORT.md exist on
the 33 covariates only and are quoted as such.

Table 2's split-sample block (2026-11-22, referee M6, orchestrator decision): alpha-hat (min), the minimum over the
eight splits, and e-value (avg), the level certified by the average over the splits of the final wealth K_N, read
upward to the first failure (ev_final_level; needs results/aos/main/raw/ref_realdata_ext from
scripts/aos_ref_realdata.py evalues); the median left the table and stays in REPORT.md (macros mnRd...Med unchanged);
new macros mnRd{Rhc,Child,Age,Bmi}EvAvg and mnRdRhcPaperEvAvg / mnRdRhcFullEvAvg (33 / 45 covariates, both modes).

    python -B scripts/aos_main_tables.py                          # RHC on the 45 covariates
    python -B scripts/aos_main_tables.py --rhc-covariates paper   # the 33-covariate outputs of 2026-11-22
"""
from __future__ import annotations

import csv
import json
import math
import os
import statistics
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from scipy.stats import binom  # noqa: E402

import aos_baselines_tables as T  # noqa: E402  (n80 rule, formatting, design labels; reads results/aos/baselines/raw)
from trimmedmmd import threshold as TH  # noqa: E402

MAIN = ROOT / "results" / "aos" / "main"
RAW = MAIN / "raw"
TAB = ROOT / "results" / "aos" / "tables"
STORED = ROOT / "explore" / "harder_sims" / "out"
BLRAW = ROOT / "results" / "aos" / "baselines" / "raw"
FBRAW = ROOT / "results" / "aos" / "fallback" / "raw"
DELTA = 0.05
DESIGNS = T.DESIGNS
NULLS = [k for k, _ in T.NULLS]
KEYS = [k for k, _, _ in DESIGNS]
HELD = 50            # replicates 0-49 were used in choosing the kernel (results/aos/fallback); held out: 50-199
MATCHED_DEF = ("gbm_m_hard", "gbm_m_soft", "logit_m_hard", "logit_m_soft")
MATCHED_ALL = MATCHED_DEF + ("gbm_mt_hard", "gbm_mt_soft", "logit_mt_hard", "logit_mt_soft")
HPLB = ("hplb_bayes_gbm", "hplb_adapt_gbm", "hplb_bayes_logit", "hplb_adapt_logit")
EVERY_CLF = MATCHED_ALL + HPLB + ("gbm_ss_soft", "logit_ss_soft", "gbm_ss_hard", "logit_ss_hard")
TAGS = {"rhc": "Rhc", "nhanes_child": "Child", "nhanes_age": "Age", "nhanes_bmi": "Bmi"}
CONTRASTS = list(TAGS)
LABEL_RD = {"rhc": "RHC arms", "nhanes_child": r"ages 6--11 vs.\ 60--79", "nhanes_age": r"ages 20--39 vs.\ 60--79",
            "nhanes_bmi": r"BMI ${<}25$ vs.\ ${\ge}30$"}
SEEDS = [1000 + i for i in range(8)]
RHC_COV = "full"     # --rhc-covariates; "paper" = the 33 covariates of the earlier runs
MACROS = []
REPORT = []
EXTRA = []           # results/aos/tables/aos_main_extra_macros.tex (see extra_summaries); not in MACROS or REPORT


def mac(name, val):
    assert name.isalpha(), name
    MACROS.append((name, str(val)))


def extra_mac(name, val):
    assert name.isalpha(), name
    EXTRA.append((name, str(val)))


def extra_summaries(R):
    """Cold read of the main text (2026-11-24, items I12 and B2), written to aos_main_extra_macros.tex so that every
    other output of this script stays byte-identical. R: the per-design recoveries of summaries("mn", ...).
    - mean recovery over the designs, and the median without the oscillating design nu = 4, for the Laplace ramp and
      the Gaussian ramp (next to mnRecRampMedian and mnRecGaussMedian);
    - power against H_{0,0} at the main size: the Laplace ramp against the best classifier refitted on its batches
      (default or tuned, labels or probabilities), counted as abBestMatchedAboveRampPow counts it for the Gaussian
      ramp (scripts/aos_baselines_tables.py: raw power, best of the eight)."""
    ramp = np.array([R[k]["ramp"][0] for k in KEYS])
    gauss = np.array([R[k]["ramp_gauss"][0] for k in KEYS])
    other = np.array([k != "D8_N4" for k in KEYS])
    extra_mac("mnRecRampMean", r2(np.mean(ramp)))
    extra_mac("mnRecGaussMean", r2(np.mean(gauss)))
    extra_mac("mnRecRampMedianNoNfour", r2(np.median(ramp[other])))
    extra_mac("mnRecGaussMedianNoNfour", r2(np.median(gauss[other])))
    above = below = 0
    for k in KEYS:
        lap, bl = synth("laplace", k), baselines_main(k)
        reps = sorted(r_ for r_ in lap if r_ in bl)
        if not reps:
            continue
        p_lap = float(np.mean([lap[r_]["cuts"]["order"]["jlo"] >= 0 for r_ in reps]))
        p_clf = max(float(np.mean([bl[r_]["jlo"][m_] >= 0 for r_ in reps])) for m_ in MATCHED_ALL)
        above += int(p_clf > p_lap + 1e-12)
        below += int(p_lap > p_clf + 1e-12)
    extra_mac("mnPowBestClfAboveLap", above)
    extra_mac("mnPowLapAboveBestClf", below)


def say(s=""):
    REPORT.append(s)


def held(rep):
    return (rep % 200000) >= HELD


def jl(path):
    out, seen = [], set()
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            k = (r.get("design"), r.get("n"), r.get("rep"), r.get("seed"), r.get("alpha"))
            if k in seen:
                continue
            seen.add(k)
            out.append(r)
    return out


def r2(x):
    return str(Decimal(repr(float(x))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def r3(x):
    return str(Decimal(repr(float(x))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def wtl(d):
    return T.wtl(d)


def binom_p(k, R):
    return float(binom.sf(k - 1, R, DELTA))


def frac(j, levels, ast):
    return 0.0 if j < 0 else levels[j] / ast


def jlo_of(dec):
    j = -1
    for x in dec:
        if not x:
            break
        j += 1
    return j


# ---------------------------------------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------------------------------------

C_N = {}


def main_n(key):
    if key not in C_N:
        ns = {r["n"] for r in jl(STORED / f"{key}.jsonl")}
        C_N[key] = 800 if 800 in ns else max(ns)
    return C_N[key]


def stored(key):
    return {(r["n"], r["rep"]): r for r in jl(STORED / f"{key}.jsonl")}


def stored_ext(key):
    return {(r["n"], r["rep"]): r for r in jl(STORED / f"{key}_n0ext.jsonl")}


def synth(kernel, key):
    return {r["rep"]: r for r in jl(RAW / f"synth_{kernel}" / f"{key}_n{main_n(key)}.jsonl")}


def alpha0(kernel, key):
    out = {}
    for f in sorted((RAW / f"alpha0_{kernel}").glob(f"{key}_n*.jsonl")):
        if f.stem[len(key) + 2:].isdigit():
            for r in jl(f):
                out[(r["n"], r["rep"])] = r
    return out


def analytic(key):
    return {(r["n"], r["rep"]): r for r in jl(RAW / "analytic" / f"{key}.jsonl")}


def baselines_main(key):
    return {r["rep"]: r for r in jl(BLRAW / f"{key}_n{main_n(key)}.jsonl")}


# ---------------------------------------------------------------------------------------------------------------------
# power curves against H_{0,0} and the n80 rule
# ---------------------------------------------------------------------------------------------------------------------

def curve(points, flt):
    """points: {(n, rep): decision}; flt: replicate filter -> {n: power}, {n: R}."""
    by = {}
    for (n, rep), d in points.items():
        if flt(rep):
            by.setdefault(n, []).append(int(d))
    return {n: float(np.mean(v)) for n, v in by.items()}, {n: len(v) for n, v in by.items()}


def points_ramp_lap(key):
    return {k: r["dec0"] for k, r in alpha0("laplace", key).items()}


def points_ramp_gauss(key):
    p = {k: r["dec"]["seq_bet"][0] for k, r in stored(key).items()}
    ns = {n for n, _ in p}          # the stored sizes, fixed before the extension is added (as points_stored)
    for k, r in stored_ext(key).items():
        if k[0] not in ns:
            p[k] = r["dec0"]["seq_bet"]
    return p


def points_stored(key, mth):
    p = {k: r["dec"][mth][0] for k, r in stored(key).items()}
    ns = {n for n, _ in p}
    for k, r in stored_ext(key).items():
        if k[0] not in ns:
            p[k] = r["dec0"][mth]
    return p


def points_analytic(key):
    p = {}
    for k, r in analytic(key).items():
        p[k] = r["dec0"] if r.get("mode") == "alpha0" else r["dec"][0]
    return p


def points_new(key, mth):
    p = {}
    for f in list(BLRAW.glob(f"{key}_n*_h00.jsonl")) + list(BLRAW.glob(f"{key}_n*_h00x.jsonl")) + [
            f for f in BLRAW.glob(f"{key}_n*.jsonl") if f.stem.split("_n")[-1].isdigit()]:
        for r in jl(f):
            p[(r["n"], r["rep"])] = r["dec_all"][mth][0]
    return p


# AoS notation (aos/restructure_plan.md, amendment 3): the split-sample bound is alpha-hat, the analytic estimator
# alpha-hat_an; every generated header uses them.
A_SPLIT = r"$\widehat\alpha$"
A_AN = r"$\widehat\alpha_{\mathrm{an}}$"
N80COLS = [("ramp", A_SPLIT + "|"), ("analytic", A_AN + "|"), ("gbm_bet", "GBM|+ betting"),
           ("logit_bet", "logistic|+ betting"), ("perm_pilot", "perm.|MMD"), ("mmdagg", "MMDAgg|")]


def n80_table(flt, with_gauss=False):
    """{(key, method): (tex, value, curve, R)}"""
    out = {}
    for key in KEYS:
        src = {"ramp": points_ramp_lap(key), "analytic": points_analytic(key),
               "gbm_bet": points_stored(key, "gbm_bet"), "logit_bet": points_stored(key, "logit_bet"),
               "perm_pilot": points_new(key, "perm_pilot"), "mmdagg": points_new(key, "mmdagg")}
        if with_gauss:
            src["ramp_gauss"] = points_ramp_gauss(key)
        for mth, pts in src.items():
            cv, R = curve(pts, flt)
            if not cv:
                out[(key, mth)] = ("--", None, cv, R)
                continue
            tex, v = T.n80(cv)
            out[(key, mth)] = (tex, v, cv, R)
    # every point of every curve rests on the same number of replicates (200, held out 150)
    reps = {R for (_, _, _, Rs) in out.values() for R in Rs.values()}
    assert len(reps) == 1, sorted(((k, m_, n, R) for (k, m_), (_, _, _, Rs) in out.items() for n, R in Rs.items()
                                   if R != max(reps)))[:20]
    return out


def n80val(tab, key, mth):
    v = tab[(key, mth)][1]
    return math.inf if v is None else v


def interpolated(tab, key, mth):
    t = tab[(key, mth)][0]
    return not (t.startswith("{\\le}") or t.startswith("{>}") or t == "--")


def write_n80(path, tab, note, cols=N80COLS, certify=4, head2=None):
    out = [f"% Generated by scripts/aos_main_tables.py -- do not edit. {note}",
           r"\begin{tabular}{l" + "c" * (len(cols) + 1) + "}", r"\toprule"]
    h1 = ["design", r"$\alpha^\star$"] + [c[1].split("|")[0] for c in cols]
    h2 = ["", ""] + [(c[1].split("|")[1] if "|" in c[1] else "") for c in cols]
    out.append(" & ".join(h1) + r" \\")
    if any(h2):
        out.append(" & ".join(h2) + r" \\")
    out.append(r"\midrule")
    for key, label, ast in DESIGNS:
        vals = [n80val(tab, key, m_) for m_, _ in cols]
        best_c = min(vals[:certify]) if certify else math.inf
        best_d = min(vals[certify:]) if len(cols) > certify else math.inf
        cells = [label, f"${ast}$"]
        for i, ((m_, _), v) in enumerate(zip(cols, vals)):
            t = tab[(key, m_)][0]
            if i < certify and v == best_c and best_c < math.inf:
                cells.append(rf"$\mathbf{{{t}}}$")
            elif i >= certify and v == best_d and best_d < math.inf:
                cells.append(rf"$\mathit{{{t}}}$")
            else:
                cells.append(f"${t}$")
        out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(out) + "\n")
    print("wrote", path)


def n80_macros(prefix, tab):
    """Counts and ratios of Table 1 (same rules as scripts/aos_baselines_tables.py and jmlr/v3/tools/derive.py)."""
    four = ("ramp", "analytic", "gbm_bet", "logit_bet")
    six = four + ("perm_pilot", "mmdagg")
    best4 = {k: min(n80val(tab, k, m_) for m_ in four) for k in KEYS}
    best6 = {k: min(n80val(tab, k, m_) for m_ in six) for k in KEYS}
    mac(f"{prefix}NeightyRampBest", sum(n80val(tab, k, "ramp") == best4[k] < math.inf for k in KEYS))
    mac(f"{prefix}NeightyRampBestAlone", sum(n80val(tab, k, "ramp") == best4[k] < math.inf and
                                             sum(n80val(tab, k, m_) == best4[k] for m_ in four) == 1 for k in KEYS))
    for m_, nm in (("perm_pilot", "Perm"), ("mmdagg", "Agg"), ("ramp", "RampAll")):
        mac(f"{prefix}Neighty{nm}Best", sum(n80val(tab, k, m_) == best6[k] < math.inf for k in KEYS))
    mac(f"{prefix}NeightyRampNotReached", sum(tab[(k, "ramp")][1] is None for k in KEYS))
    mac(f"{prefix}NeightyAnalyticNotReached", sum(tab[(k, "analytic")][1] is None for k in KEYS))
    rp = [n80val(tab, k, "ramp") / n80val(tab, k, "perm_pilot") for k in KEYS
          if interpolated(tab, k, "ramp") and interpolated(tab, k, "perm_pilot")]
    if rp:
        mac(f"{prefix}NeightyRampOverPermLo", r2(min(rp)))
        mac(f"{prefix}NeightyRampOverPermHi", r2(max(rp)))
        mac(f"{prefix}NeightyRampOverPermMed", r2(statistics.median(rp)))
        mac(f"{prefix}NeightyRampOverPermN", len(rp))
    ar = sorted(n80val(tab, k, "analytic") / n80val(tab, k, "ramp") for k in KEYS
                if interpolated(tab, k, "analytic") and interpolated(tab, k, "ramp"))
    if ar:
        mac(f"{prefix}AnaOverRampN", len(ar))
        mac(f"{prefix}AnaOverRampLo", f"{ar[0]:.1f}")
        mac(f"{prefix}AnaOverRampHi", f"{round(ar[-1])}")
        mac(f"{prefix}AnaOverRampMed", f"{round(statistics.median(ar))}")
    return rp, ar


# ---------------------------------------------------------------------------------------------------------------------
# tau(0) meets the population MMD (eq:samplesize check; jmlr/tools/make_harder_tables.py::n_tau with the new tau)
# ---------------------------------------------------------------------------------------------------------------------

def n_tau(key, mmd, ratio, sizes):
    c = [(n, TH.tau_aos(n, ratio * n, 0.0, DELTA)) for n in sorted(sizes)]
    if c[0][1] <= mmd:
        return 0
    for (n0, t0), (n1, t1) in zip(c, c[1:]):
        if t1 <= mmd:
            return math.exp(math.log(n0) + math.log(t0 / mmd) / math.log(t0 / t1) * math.log(n1 / n0))
    n1, t1 = c[-1]
    return n1 * (t1 / mmd) ** 2


# ---------------------------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------------------------

def main():
    TAB.mkdir(parents=True, exist_ok=True)
    say("# AoS restructure, main runs: report (generated by scripts/aos_main_tables.py; no hand-typed numbers)")
    say()
    say("Protocol and running log: results/aos/main/NOTES.md. Raw records: results/aos/main/raw.")
    say()
    # ------------------------------------------------------------------ reproduction checks
    say("## Reproduction checks")
    say()
    ver = MAIN / "verify.json"
    if ver.exists():
        v = json.loads(ver.read_text())
        say(f"- Package options (scripts/aos_main_verify.py): default cut rule = numpy.quantile = fold1.cut_points in "
            f"{v['default_equals_numpy_quantile_and_fold1']['trials'] - v['default_equals_numpy_quantile_and_fold1']['mismatches']}"
            f"/{v['default_equals_numpy_quantile_and_fold1']['trials']} random trials; rule \"order\" = the order "
            f"statistics of cut_ranks in {2000 - v['order_rule_is_order_statistic']['mismatches']}/2000; eq:cutrank "
            f"holds in {v['eq_cutrank']['cells'] - v['eq_cutrank']['n_failures']}/{v['eq_cutrank']['cells']} "
            f"(fit size, level) cells; the package default (splitting.sequential_certificate, no cut_rule) reproduces "
            f"{sum(c['agree'] for c in v['package_default_vs_stored_realdata'])}/"
            f"{len(v['package_default_vs_stored_realdata'])} stored real-data cells; tau_aos = stored closed form on "
            f"{v['tau_aos_vs_stored_closed_form']['cells']} threshold cells (max rel. diff "
            f"{v['tau_aos_vs_stored_closed_form']['max_rel_diff']:.1e}; tau^A < tau^P in "
            f"{v['tau_aos_vs_stored_closed_form']['cells_tauA_below_tauP']} of them).")
        mac("mnCutrankCells", v["eq_cutrank"]["cells"])
        mac("mnCutrankFailures", v["eq_cutrank"]["n_failures"])
    # Gaussian synth check: linear = stored seq_bet
    g_tot = g_jlo = g_dec = 0
    for key in KEYS:
        st = stored(key)
        for rep, r in synth("gaussian", key).items():
            s = st[(main_n(key), rep)]
            g_tot += 1
            g_jlo += int(r["cuts"]["linear"]["jlo"] == s["jlo"]["seq_bet"])
            g_dec += int([r["cuts"]["linear"]["dec"][j] for j in (0, 4, 6, 8)] == s["dec"]["seq_bet"])
    say(f"- Gaussian kernel, cut rule \"linear\" (the default), full warm-start chain: stored seq_bet jlo reproduced in "
        f"{g_jlo}/{g_tot} replicates, decisions at j = 0, 4, 6, 8 in {g_dec}/{g_tot} (19 designs at n = 800).")
    mac("mnReproGaussJlo", f"{g_jlo}/{g_tot}")
    # Laplace linear = fallback OWN (same fits)
    l_tot = l_ok = 0
    for key in KEYS:
        fb = {r["rep"]: r for r in jl(FBRAW / "synth_laplace" / f"{key}_n{main_n(key)}.jsonl")}
        for rep, r in synth("laplace", key).items():
            if rep in fb:
                l_tot += 1
                l_ok += int(r["cuts"]["linear"]["dec"] == fb[rep]["rules"]["own"]["dec"])
    say(f"- Laplace kernel, cut rule \"linear\" on the same fits: the decisions at all 11 levels equal the 50-replicate "
        f"kernel comparison (results/aos/fallback/raw/synth_laplace, OWN) in {l_ok}/{l_tot} replicates.")
    mac("mnReproLaplaceFallback", f"{l_ok}/{l_tot}")
    # alpha = 0 runs: Gaussian = stored; Laplace alpha0 = Laplace full run at j = 0
    a_tot = a_ok = 0
    for key in KEYS + NULLS:
        for k, r in alpha0("gaussian", key).items():
            if "stored_seq0" in r:
                a_tot += 1
                a_ok += int(r["dec0"] == r["stored_seq0"])
    say(f"- alpha = 0 blocked kernel sums, Gaussian kernel: stored seq_bet decision at alpha = 0 reproduced in "
        f"{a_ok}/{a_tot} replicates.")
    mac("mnReproAlphaZeroGauss", f"{a_ok}/{a_tot}")
    x_tot = x_ok = 0
    for key in KEYS:
        a0 = alpha0("laplace", key)
        for rep, r in synth("laplace", key).items():
            k = (main_n(key), rep)
            if k in a0:
                x_tot += 1
                x_ok += int(a0[k]["dec0"] == r["cuts"]["order"]["dec"][0])
    say(f"- Laplace kernel: the alpha = 0 decision of the blocked-sum run equals the j = 0 decision of the all-level run "
        f"in {x_ok}/{x_tot} replicates at n = 800.")
    mac("mnReproAlphaZeroLaplace", f"{x_ok}/{x_tot}")
    sig_cells = {}
    for key in KEYS + NULLS:
        for (n, rep), r in alpha0("laplace", key).items():
            if r.get("repro_sigma") is False:
                sig_cells[(key, n)] = sig_cells.get((key, n), 0) + 1
    for key in KEYS:
        for rep, r in synth("laplace", key).items():
            if r["repro"].get("sigma") is False:
                sig_cells[(key, main_n(key))] = sig_cells.get((key, main_n(key)), 0) + 1
    sig_bad = sum(sig_cells.values())
    g_sig = [r["dec0"] == r["stored_seq0"] for key in KEYS + NULLS for r in alpha0("gaussian", key).values()
             if r.get("repro_sigma") is False]
    say(f"- Fold bandwidths recomputed from the regenerated data differ from the stored ones (relative tolerance 1e-9) "
        f"in {sig_bad} replicates ({', '.join(f'{k} n={n}: {c}' for (k, n), c in sorted(sig_cells.items()))}); every "
        f"run uses the stored bandwidths, and the Gaussian alpha = 0 check reproduces the stored decision on "
        f"{sum(g_sig)}/{len(g_sig)} of the replicates of this kind it covers.")
    say()

    # ------------------------------------------------------------------ Table 1
    tab = n80_table(lambda rep: True, with_gauss=True)
    write_n80(TAB / "aos_main_n80.tex", tab,
              "Sample size n = m (mixed: n, m = 5n) for power 0.8 against H_{0,0} (first size with power >= 0.8, "
              "log-linear interpolation, two significant digits). Laplace ramp: results/aos/main (alpha = 0, Laplace "
              "kernel, all stored sizes and the alpha = 0 extension); analytic: tau = min{tau^P, tau^A} on the stored "
              "statistics; GBM, logistic: stored study; perm. MMD, MMDAgg: results/aos/baselines. Bold: smallest of the "
              "four tests that certify a level; italic: smaller of the two tests of P = Q.", cols=N80COLS)
    say("## Table 1: sample size for power 0.8 against H_{0,0}")
    say()
    say("| design | alpha* | Laplace ramp | Gaussian ramp (stored) | analytic (new tau) | analytic (stored tau_new) | GBM | logistic | perm. MMD | MMDAgg |")
    say("|---|---|---|---|---|---|---|---|---|---|")
    old_an = {key: T.n80(T.curve_stored(key, "an_new"))[0] for key in KEYS}
    for key, label, ast in DESIGNS:
        c = lambda m_: tab[(key, m_)][0].replace("{\\le}", "<=").replace("{>}", ">").replace("{,}", ",")
        say(f"| {key} | {ast} | {c('ramp')} | {c('ramp_gauss')} | {c('analytic')} | "
            f"{old_an[key].replace('{\\le}', '<=').replace('{>}', '>').replace('{,}', ',')} | {c('gbm_bet')} | "
            f"{c('logit_bet')} | {c('perm_pilot')} | {c('mmdagg')} |")
    say()
    rp, ar = n80_macros("mn", tab)
    Rs = sorted({R for key in KEYS for R in tab[(key, "ramp")][3].values()})
    mac("mnRepsPerSize", Rs[0] if len(Rs) == 1 else f"{Rs[0]}--{Rs[-1]}")
    sizes_ramp = sorted({n for key in KEYS for n in tab[(key, "ramp")][2]})
    mac("mnNmin", T.tex_n(sizes_ramp[0]))
    mac("mnNcap", T.tex_n(sizes_ramp[-1]))
    mac("mnNdesigns", len(KEYS))
    # cells not reached (computed sizes vs the stored study's sizes)
    missing = []
    for key in KEYS + NULLS:
        want = {k[0] for k in stored(key)} | {k[0] for k in stored_ext(key)}
        have = {k[0] for k in alpha0("laplace", key)}
        cnt = {}
        for k in alpha0("laplace", key):
            cnt[k[0]] = cnt.get(k[0], 0) + 1
        for n in sorted(want):
            if n not in have or cnt.get(n, 0) < 200:
                missing.append((key, n, cnt.get(n, 0)))
    say(f"Cells of the stored study (design x size) without all 200 Laplace alpha = 0 replicates: "
        f"{', '.join(f'{k} n={n} ({c})' for k, n, c in missing) if missing else 'none'}.")
    say()
    # Laplace vs Gaussian ramp, Table 1
    lg = [(n80val(tab, k, "ramp"), n80val(tab, k, "ramp_gauss")) for k in KEYS]
    mac("mnNeightyLapBelowGauss", sum(a < b for a, b in lg))
    mac("mnNeightyLapAboveGauss", sum(a > b for a, b in lg))
    mac("mnNeightyLapEqGauss", sum(a == b for a, b in lg))
    # the same on the 15 multivariate designs (all but the oscillating densities on the line; cold read B1, 2026-11-24)
    lgm = [(n80val(tab, k, "ramp"), n80val(tab, k, "ramp_gauss")) for k in KEYS if not k.startswith("D8_")]
    mac("mnMvDesigns", len(lgm))
    mac("mnNeightyLapBelowGaussMv", sum(a < b for a, b in lgm))
    mac("mnNeightyLapAboveGaussMv", sum(a > b for a, b in lgm))
    mac("mnNeightyLapEqGaussMv", sum(a == b for a, b in lgm))
    say(f"Laplace vs Gaussian ramp in Table 1: fewer observations in {sum(a < b for a, b in lg)}, more in "
        f"{sum(a > b for a, b in lg)}, the same tabulated size in {sum(a == b for a, b in lg)} designs.")
    say(f"Ramp / permutation MMD where both are interpolated: {', '.join(r2(x) for x in rp)}; "
        f"analytic / ramp: {', '.join(f'{x:.1f}' for x in ar)}.")
    say()
    # analytic: changes under the new threshold
    ch = [(k, old_an[k], tab[(k, 'analytic')][0]) for k in KEYS if old_an[k] != tab[(k, "analytic")][0]]
    say(f"Analytic column, stored tau_new vs the new tau: {len(ch)} designs change"
        + (": " + "; ".join(f"{k} {a} -> {b}" for k, a, b in ch) if ch else "") + ".")
    mac("mnNeightyAnalyticChanged", len(ch))
    say()

    # tau(0) meets the population MMD
    pop = {r["design"]: r for r in csv.DictReader(open(ROOT / "results" / "harder" / "popmmd.csv"))}
    fire = {}
    for key in KEYS:
        if key not in pop or float(pop[key]["mmd2_pop"]) <= 0:
            continue
        sizes = sorted(tab[(key, "analytic")][2])
        ratio = stored(key)[(sizes[0], 0)]["m"] // sizes[0]
        nt = n_tau(key, float(pop[key]["mmd_pop"]), ratio, sizes)
        v = tab[(key, "analytic")]
        if v[1] is not None and not v[0].startswith("{\\le}"):
            fire[key] = v[1] / nt
    if fire:
        mac("mnTauFireDesigns", sum(tab[(k, "analytic")][1] is not None for k in KEYS if k in pop))
        mac("mnTauFireRatioN", len(fire))
        mac("mnTauFireRatioLo", f"{min(fire.values()):.1f}")
        mac("mnTauFireRatioHi", f"{max(fire.values()):.1f}")
        say(f"eq:samplesize check (analytic n80 / size at which tau(0) = min(tau^P, tau^A)(0) meets the population MMD, "
            f"results/harder/popmmd.csv): {', '.join(f'{k} {v:.2f}' for k, v in fire.items())}; range "
            f"{min(fire.values()):.1f} to {max(fire.values()):.1f}.")
        say()

    # ------------------------------------------------------------------ held-out Table 1
    tab_h = n80_table(held, with_gauss=True)
    write_n80(TAB / "aos_main_n80_heldout.tex", tab_h,
              "As aos_main_n80.tex on replicates 50-199 (extension: 200050-200199), the ones not used in choosing the "
              "kernel.", cols=N80COLS)
    n80_macros("mnHo", tab_h)
    lgh = [(n80val(tab_h, k, "ramp"), n80val(tab_h, k, "ramp_gauss")) for k in KEYS]
    mac("mnHoNeightyLapBelowGauss", sum(a < b for a, b in lgh))
    mac("mnHoNeightyLapAboveGauss", sum(a > b for a, b in lgh))
    ho = sorted({R for key in KEYS for R in tab_h[(key, "ramp")][3].values()})
    mac("mnHoReps", ho[0] if len(ho) == 1 else f"{ho[0]}--{ho[-1]}")

    # ------------------------------------------------------------------ S6 Gaussian Table 1 (compact)
    write_n80(TAB / "aos_main_n80_gauss.tex", tab,
              "S6: sample size for power 0.8 against H_{0,0}: the ramp with the Gaussian kernel (stored study) and with "
              "the Laplace kernel (Table 1), and the analytic test with tau = min{tau^P, tau^A}. Bold: the smallest.",
              cols=[("ramp_gauss", A_SPLIT + ",|Gaussian"), ("ramp", A_SPLIT + ",|Laplace"), ("analytic", A_AN + "|")],
              certify=3)

    # ------------------------------------------------------------------ recovery at n = 800
    say("## Figure 2(a): recovery at n = m = 800 (mixed: n = 400, m = 5n)")
    say()
    rows = {}
    per_rep = {}
    for key, label, ast in DESIGNS:
        n = main_n(key)
        lap, st, an, bl = synth("laplace", key), stored(key), analytic(key), baselines_main(key)
        reps = sorted(lap)
        D = {m_: {} for m_ in ("ramp", "ramp_linear", "ramp_gauss", "analytic", "gbm_bet", "logit_bet")
             + EVERY_CLF}
        for rep in reps:
            r = lap[rep]
            lv, a_s = r["levels"], r["alpha_star"]
            D["ramp"][rep] = frac(r["cuts"]["order"]["jlo"], lv, a_s)
            D["ramp_linear"][rep] = frac(r["cuts"]["linear"]["jlo"], lv, a_s)
            s = st[(n, rep)]
            D["ramp_gauss"][rep] = frac(s["jlo"]["seq_bet"], lv, a_s)
            D["gbm_bet"][rep] = frac(s["jlo"]["gbm_bet"], lv, a_s)
            D["logit_bet"][rep] = frac(s["jlo"]["logit_bet"], lv, a_s)
            D["analytic"][rep] = frac(an[(n, rep)]["jlo"], lv, a_s)
            if rep in bl:
                for m_ in EVERY_CLF:
                    D[m_][rep] = frac(bl[rep]["jlo"][m_], lv, a_s)
        per_rep[key] = D
        rows[key] = {m_: (float(np.mean(list(v.values()))) if v else float("nan"), len(v)) for m_, v in D.items()}

    def summaries(prefix, flt):
        R = {key: {m_: (float(np.mean([x for rep, x in v.items() if flt(rep)])), sum(flt(rep) for rep in v))
                   for m_, v in per_rep[key].items() if v} for key in KEYS}
        ramp = np.array([R[k]["ramp"][0] for k in KEYS])
        gauss = np.array([R[k]["ramp_gauss"][0] for k in KEYS])
        lin = np.array([R[k]["ramp_linear"][0] for k in KEYS])
        ana = np.array([R[k]["analytic"][0] for k in KEYS])
        mac(f"{prefix}RecRampMedian", r2(np.median(ramp)))
        mac(f"{prefix}RecRampMax", r2(ramp.max()))
        mac(f"{prefix}RecRampMin", r2(ramp.min()))
        mac(f"{prefix}RecGaussMedian", r2(np.median(gauss)))
        mac(f"{prefix}RecLinearMedian", r2(np.median(lin)))
        mac(f"{prefix}RecAnalyticMedian", r2(np.median(ana)))
        mac(f"{prefix}RecAnalyticMax", r2(ana.max()))
        mac(f"{prefix}RecAnalyticZero", int(np.sum(ana == 0)))
        mac(f"{prefix}RecRampAboveAnalytic", int(np.sum(ramp > ana)))
        mac(f"{prefix}RecAnalyticAboveRamp", int(np.sum(ana > ramp)))
        mac(f"{prefix}RecRhcRamp", r2(R["D9_rhc"]["ramp"][0]))
        mac(f"{prefix}RecRhcGauss", r2(R["D9_rhc"]["ramp_gauss"][0]))
        mac(f"{prefix}RecRhcAnalytic", r2(R["D9_rhc"]["analytic"][0]))
        mac(f"{prefix}RecNfourLap", r2(R["D8_N4"]["ramp"][0]))
        mac(f"{prefix}RecNfourGauss", r2(R["D8_N4"]["ramp_gauss"][0]))
        # Laplace (order) vs Gaussian (stored), paired per design
        res = {}
        for k in KEYS:
            reps = [rep for rep in per_rep[k]["ramp"] if flt(rep)]
            d = [per_rep[k]["ramp"][r_] - per_rep[k]["ramp_gauss"][r_] for r_ in reps]
            res[k] = wtl(d)
        w = sum(v[2] == "win" for v in res.values())
        t = sum(v[2] == "tie" for v in res.values())
        lo = sum(v[2] == "loss" for v in res.values())
        mac(f"{prefix}LapGaussWins", w)
        mac(f"{prefix}LapGaussTies", t)
        mac(f"{prefix}LapGaussLosses", lo)
        mv = [v for k, v in res.items() if not k.startswith("D8_")]      # multivariate designs only
        mac(f"{prefix}LapGaussWinsMv", sum(v[2] == "win" for v in mv))
        mac(f"{prefix}LapGaussTiesMv", sum(v[2] == "tie" for v in mv))
        mac(f"{prefix}LapGaussLossesMv", sum(v[2] == "loss" for v in mv))
        mac(f"{prefix}LapGaussWinsLine", sum(v[2] == "win" for k, v in res.items() if k.startswith("D8_")))
        losses = [-v[0] for v in res.values() if v[2] == "loss"]
        gains = [v[0] for v in res.values() if v[2] == "win"]
        mac(f"{prefix}LapGaussLossMax", r3(max(losses)) if losses else "0")
        mac(f"{prefix}LapGaussGainMax", r3(max(gains)) if gains else "0")
        # order vs linear (same fits)
        res2 = {}
        for k in KEYS:
            reps = [rep for rep in per_rep[k]["ramp"] if flt(rep)]
            res2[k] = wtl([per_rep[k]["ramp"][r_] - per_rep[k]["ramp_linear"][r_] for r_ in reps])
        mac(f"{prefix}OrderLinearWins", sum(v[2] == "win" for v in res2.values()))
        mac(f"{prefix}OrderLinearTies", sum(v[2] == "tie" for v in res2.values()))
        mac(f"{prefix}OrderLinearLosses", sum(v[2] == "loss" for v in res2.values()))
        # classifiers on the batches of the ramp
        have = [k for k in KEYS if all(R[k].get(m_, (None, 0))[1] for m_ in MATCHED_ALL)]
        if have:
            rr = np.array([R[k]["ramp"][0] for k in have])
            mdef = np.array([max(R[k][m_][0] for m_ in MATCHED_DEF) for k in have])
            mall = np.array([max(R[k][m_][0] for m_ in MATCHED_ALL) for k in have])
            every = np.array([max(R[k][m_][0] for m_ in EVERY_CLF + ("gbm_bet", "logit_bet")) for k in have])
            lt = np.array([R[k]["logit_mt_hard"][0] for k in have])
            mac(f"{prefix}RampAboveBestMatched", int(np.sum(rr > mdef)))
            mac(f"{prefix}BestMatchedAboveRamp", int(np.sum(mdef > rr)))
            mac(f"{prefix}RampAboveBestMatchedTuned", int(np.sum(rr > mall)))
            mac(f"{prefix}RampAboveEveryClf", int(np.sum(rr > every)))
            mac(f"{prefix}LogitTunedAboveRamp", int(np.sum(lt > rr)))
            wl = []
            for k in have:
                reps = [rep for rep in per_rep[k]["ramp"] if flt(rep) and rep in per_rep[k]["logit_mt_hard"]]
                wl.append(wtl([per_rep[k]["ramp"][r_] - per_rep[k]["logit_mt_hard"][r_] for r_ in reps])[2])
            mac(f"{prefix}WtlRecLogitTuned", f"{wl.count('win')}/{wl.count('tie')}/{wl.count('loss')}")
            above_every = [k for k, a, b in zip(have, rr, every) if a > b]
        else:
            above_every = []
        return R, res, res2, above_every

    R, res, res2, above_every = summaries("mn", lambda rep: True)
    extra_summaries(R)
    Rh, resh, res2h, _ = summaries("mnHo", held)
    rec_reps = sorted({rows[k]["ramp"][1] for k in KEYS})
    rec_reps = rec_reps[0] if len(rec_reps) == 1 else f"{rec_reps[0]}--{rec_reps[-1]}"
    mac("mnRecReps", rec_reps)
    lab = {k: l for k, l, _ in DESIGNS}
    say("| design | Laplace ramp (order) | Laplace ramp (linear) | Gaussian ramp (stored) | Laplace - Gaussian (paired s.e.) | analytic (new tau) | GBM labels, batches of the ramp | logistic labels, batches | tuned logistic | held-out Laplace | held-out Gaussian |")
    say("|---|---|---|---|---|---|---|---|---|---|---|")
    for k in KEYS:
        say(f"| {k} | {R[k]['ramp'][0]:.3f} | {R[k]['ramp_linear'][0]:.3f} | {R[k]['ramp_gauss'][0]:.3f} | "
            f"{res[k][0]:+.3f} ({res[k][1]:.3f}) {res[k][2]} | {R[k]['analytic'][0]:.3f} | "
            f"{R[k].get('gbm_m_hard', (float('nan'),))[0]:.3f} | {R[k].get('logit_m_hard', (float('nan'),))[0]:.3f} | "
            f"{R[k].get('logit_mt_hard', (float('nan'),))[0]:.3f} | {Rh[k]['ramp'][0]:.3f} | "
            f"{Rh[k]['ramp_gauss'][0]:.3f} |")
    say()
    md = dict(MACROS)
    say(f"Median over the 19 designs of the mean recovered fraction: Laplace ramp (order) {md['mnRecRampMedian']}, "
        f"Laplace ramp (linear, same fits) {md['mnRecLinearMedian']}, Gaussian ramp (stored) {md['mnRecGaussMedian']}, "
        f"analytic (new tau) {md['mnRecAnalyticMedian']}; held-out replicates 50-199: Laplace {md['mnHoRecRampMedian']}, "
        f"Gaussian {md['mnHoRecGaussMedian']}.")
    say(f"Laplace vs Gaussian at two paired s.e.: {md['mnLapGaussWins']} wins, {md['mnLapGaussTies']} ties, "
        f"{md['mnLapGaussLosses']} losses (largest loss {md['mnLapGaussLossMax']}, largest gain {md['mnLapGaussGainMax']}); "
        f"held out: {md['mnHoLapGaussWins']}/{md['mnHoLapGaussTies']}/{md['mnHoLapGaussLosses']}. Order vs linear cuts "
        f"(Laplace, same fits): {md['mnOrderLinearWins']}/{md['mnOrderLinearTies']}/{md['mnOrderLinearLosses']}.")
    say(f"Ramp ahead of every classifier (matched, tuned, HPLB, single split): {', '.join(above_every) or 'none'}.")
    mac("mnRampAboveEveryClfList", ", ".join(lab[k].replace("$", "") for k in above_every) if above_every else "none")
    say()
    # the overlap with the 50-replicate kernel comparison of results/aos/fallback (n = 800, replicates 0-49)
    fb_rows, fb_eq = [], 0
    for k in KEYS:
        fb = {r["rep"]: r for r in jl(FBRAW / "synth_laplace" / f"{k}_n{main_n(k)}.jsonl")}
        fg = {r["rep"]: r for r in jl(FBRAW / "synth_gaussian" / f"{k}_n{main_n(k)}.jsonl")}
        if fb and fg:
            reps = sorted(fb)
            assert sorted(fg) == reps and all(rep in per_rep[k]["ramp"] for rep in reps), k
            fbl = float(np.mean([frac(fb[r_]["rules"]["own"]["jlo"], fb[r_]["levels"], fb[r_]["alpha_star"])
                                 for r_ in reps]))
            fbg = float(np.mean([frac(fg[r_]["rules"]["own"]["jlo"], fg[r_]["levels"], fg[r_]["alpha_star"])
                                 for r_ in reps]))
            lin = float(np.mean([per_rep[k]["ramp_linear"][r_] for r_ in reps]))
            ordr = float(np.mean([per_rep[k]["ramp"][r_] for r_ in reps]))
            gau = float(np.mean([per_rep[k]["ramp_gauss"][r_] for r_ in reps]))
            fb_eq += int(abs(lin - fbl) < 1e-12 and abs(gau - fbg) < 1e-12)
            fb_rows.append((k, len(reps), fbl, lin, ordr, fbg, gau))
    say(f"Overlap with the 50-replicate kernel comparison (results/aos/fallback, n = 800, replicates 0-49; OWN = linear "
        f"cuts): the Laplace linear-cut and the Gaussian (stored) recoveries equal the comparison's Laplace OWN and "
        f"Gaussian OWN in {fb_eq}/{len(fb_rows)} designs. Per design, fallback Laplace OWN / this run linear / this run "
        f"genuine quantile cuts | fallback Gaussian OWN / stored Gaussian: "
        + "; ".join(f"{k} {a:.3f} / {b:.3f} / {c:.3f} | {d:.3f} / {e:.3f}" for k, _, a, b, c, d, e in fb_rows) + ".")
    mac("mnFallbackOverlapEqual", f"{fb_eq}/{len(fb_rows)}")
    say()

    # figure data
    def fig_json(path, ramp_key, ramp_label):
        out = {"generated_by": "scripts/aos_main_tables.py", "ramp_label": ramp_label,
               "methods": ["analytic", "ramp", "gbm_m_hard", "logit_m_hard", "logit_mt_hard"],
               "rows": [{"design": k, "n": main_n(k), "rec": {"analytic": R[k]["analytic"], "ramp": R[k][ramp_key],
                                                              "gbm_m_hard": R[k]["gbm_m_hard"],
                                                              "logit_m_hard": R[k]["logit_m_hard"],
                                                              "logit_mt_hard": R[k]["logit_mt_hard"]}} for k in KEYS]}
        path.write_text(json.dumps(out, indent=1))
        print("wrote", path)
    if all("gbm_m_hard" in R[k] for k in KEYS):
        fig_json(MAIN / "fig_recovery_main.json", "ramp", "Laplace")
        fig_json(MAIN / "fig_recovery_gauss.json", "ramp_gauss", "Gaussian")

    # S6 recovery table
    out = ["% Generated by scripts/aos_main_tables.py -- do not edit. Mean certified level / alpha* at n = m = 800 "
           f"(mixed: n = 400, m = 5n), {rec_reps} replicates; bold: the largest of the row.",
           r"\begin{tabular}{lccccccc}", r"\toprule",
           rf"design & $\alpha^\star$ & {A_SPLIT}, & {A_SPLIT}, & {A_AN} & GBM & logistic & logistic \\",
           r" & & Laplace & Gaussian & & labels & labels & tuned \\", r"\midrule"]
    for key, label, ast in DESIGNS:
        ms = ("ramp", "ramp_gauss", "analytic", "gbm_m_hard", "logit_m_hard", "logit_mt_hard")
        vals = [r2(R[key][m_][0]) if m_ in R[key] else "--" for m_ in ms]
        best = max((v for v in vals if v != "--"), key=Decimal)
        cells = [label, f"${ast}$"] + [rf"$\mathbf{{{v}}}$" if v == best and Decimal(best) > 0 else f"${v}$"
                                       for v in vals]
        out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "aos_main_recovery_gauss.tex").write_text("\n".join(out) + "\n")
    print("wrote", TAB / "aos_main_recovery_gauss.tex")

    # ------------------------------------------------------------------ validity
    say("## Validity")
    say()
    bd = {}
    for k in KEYS:
        lap = synth("laplace", k)
        kk = sum(r["cuts"]["order"]["dec"][8] for r in lap.values())
        gt = sum(r["cuts"]["order"]["jlo"] >= 9 for r in lap.values())
        kl = sum(r["cuts"]["linear"]["dec"][8] for r in lap.values())
        bd[k] = (kk, len(lap), binom_p(kk, len(lap)) if lap else 1.0, gt, kl)
    worst = max(KEYS, key=lambda k: (bd[k][0], k))
    flags = [k for k in KEYS if bd[k][2] < 0.01]
    mac("mnBdCells", len(KEYS))
    mac("mnBdMax", f"{bd[worst][0]}/{bd[worst][1]}")
    mac("mnBdMaxCount", bd[worst][0])
    mac("mnBdMaxDesign", lab[worst].replace("$", ""))
    mac("mnBdMaxP", r3(bd[worst][2]))
    mac("mnBdFlags", len(flags))
    mac("mnBdTotal", sum(v[0] for v in bd.values()))
    mac("mnBdReps", sum(v[1] for v in bd.values()))
    mac("mnAloGtMax", r3(max(v[3] / v[1] for v in bd.values() if v[1])))
    say(f"Boundary alpha = alpha* (j = 8, a true null), Laplace ramp with genuine quantile cuts, n = 800: largest count "
        f"{bd[worst][0]}/{bd[worst][1]} ({worst}, one-sided binomial p {bd[worst][2]:.3f}); flagged (p < 0.01): "
        f"{len(flags)}; total {sum(v[0] for v in bd.values())} rejections in {sum(v[1] for v in bd.values())} "
        f"replicates; linear cuts total {sum(v[4] for v in bd.values())}. Largest share with alpha_lo > alpha* "
        f"(j_lo >= 9): {max(v[3] / v[1] for v in bd.values() if v[1]):.3f}.")
    say()
    say("| design | rejections at alpha* (order) | binomial p | alpha_lo > alpha* | rejections (linear) |")
    say("|---|---|---|---|---|")
    for k in KEYS:
        say(f"| {k} | {bd[k][0]}/{bd[k][1]} | {bd[k][2]:.3f} | {bd[k][3]} | {bd[k][4]} |")
    say()
    # P = Q nulls at alpha = 0, Laplace ramp, every size
    cells = []
    for k in NULLS:
        cv = {}
        for (n, rep), r in alpha0("laplace", k).items():
            cv.setdefault(n, []).append(r["dec0"])
        for n, v in sorted(cv.items()):
            cells.append((k, n, sum(v), len(v), binom_p(sum(v), len(v))))
    if cells:
        tot_k, tot_R = sum(c[2] for c in cells), sum(c[3] for c in cells)
        wc = max(cells, key=lambda c: c[2])
        mac("mnNullCells", len(cells))
        mac("mnNullMax", f"{wc[2]}/{wc[3]}")
        mac("mnNullPct", f"{100 * tot_k / tot_R:.1f}")
        mac("mnNullFlags", sum(c[4] < 0.01 for c in cells))
        mac("mnNullLoPct", f"{100 * min(c[2] / c[3] for c in cells):.1f}")
        mac("mnNullHiPct", f"{100 * max(c[2] / c[3] for c in cells):.1f}")
        say(f"P = Q nulls (N0_*), Laplace ramp at alpha = 0, every stored size: {len(cells)} cells, pooled rejection "
            f"rate {100 * tot_k / tot_R:.1f}% ({tot_k}/{tot_R}), largest {wc[2]}/{wc[3]} ({wc[0]} n={wc[1]}), flagged "
            f"{sum(c[4] < 0.01 for c in cells)}.")
    # analytic validity under the new threshold
    an_bd = an_bd_R = an_null = an_null_R = 0
    an_bd_cells = 0
    for k in KEYS:
        cellset = set()
        for (n, rep), r in analytic(k).items():
            if r.get("mode") != "alpha0":
                an_bd += r["dec"][8]
                an_bd_R += 1
                cellset.add(n)
        an_bd_cells += len(cellset)
    for k in NULLS:
        for (n, rep), r in analytic(k).items():
            an_null += r["dec0"] if r.get("mode") == "alpha0" else r["dec"][0]
            an_null_R += 1
    mac("mnAnBdRejections", an_bd)
    mac("mnAnBdCells", an_bd_cells)
    mac("mnAnNullRejections", an_null)
    say(f"Analytic test, new threshold: {an_bd} rejections at alpha = alpha* in {an_bd_R} replicates "
        f"({an_bd_cells} design x size cells); {an_null} rejections of P = Q in {an_null_R} null replicates.")
    say()

    # ------------------------------------------------------------------ real data
    say("## Real data (Table 2)")
    say()
    vb_path = MAIN / "analytic_rd" / "tables" / "r12_valid_bandwidth.csv"
    vb = list(csv.DictReader(open(vb_path))) if vb_path.exists() else []
    if RHC_COV == "full":     # the RHC rows from the 45-covariate run, relabelled rhc
        vb = [r for r in vb if r["dataset"] != "rhc"] + [
            dict(r, dataset="rhc") for r in csv.DictReader(open(MAIN / "analytic_rd_full" / "tables" /
                                                                 "r12_valid_bandwidth.csv"))
            if r["dataset"] == "rhc_full"]
    vb_old = list(csv.DictReader(open(ROOT / "results" / "v2" / "tables" / "r12_valid_bandwidth.csv")))
    route = {"rhc": "pilot, strict (pilot standardization)"}
    full_nm = {}          # sizes from the data: the pooled-median route of exp_valid_bandwidth.py uses every observation
    for r in vb:
        if r["route"] == "pooled median (paper)":
            full_nm[r["dataset"]] = (int(r["n"]), int(r["m"]))
    c0 = 1.0 + math.sqrt(math.log(1 / DELTA))
    lines = ["% Generated by scripts/aos_main_tables.py -- do not edit. Analytic: scripts/exp_valid_bandwidth.py "
             "--threshold aos (tau = min{tau^P, tau^A}), pilot route; split-sample: Laplace kernel, genuine quantile "
             "cut points, eight splits (results/aos/main/raw/realdata_laplace)."
             + (" RHC: the 45 covariates (rhc_full; --rhc-covariates full)." if RHC_COV == "full" else ""),
             r"\begin{tabular}{lrrcccccccc}", r"\toprule",
             r" & & & \multicolumn{5}{c}{analytic (" + A_AN + r"), pilot bandwidth} & & \multicolumn{2}{c}{split-sample} \\",
             r"\cmidrule(lr){4-8}\cmidrule(lr){10-11}",
             r"contrast & $n$ & $m$ & $\widehat D_0$ & $\tau(0)$ & $n_{\mathrm{fire}}$ & $\alpha_{\mathrm{last}}$ & "
             r"bisection & & " + A_SPLIT + r" (min) & " + A_SPLIT + r" (avg.\ e-value) \\", r"\midrule"]
    if RHC_COV == "full":
        say("RHC row: the 45 covariates (the 33 numeric and 0/1 ones plus sex, dnr1 and the ten admission-diagnosis "
            "indicators; scripts/aos_main_tables.py --rhc-covariates full, the default). Its Gaussian (stored) column "
            "and the stored tau_new column are the 33-covariate runs (none exists on the 45); "
            "--rhc-covariates paper gives the 33-covariate row.")
        say()
    say("Split-sample columns of Table 2: alpha-hat (min), the minimum over the eight splits (level delta), and e-value "
        "(avg), the level certified by the average over the eight splits of the final wealth K_N (an e-value, valid at "
        "delta), read upward to the first failure (records of results/aos/main/raw/realdata_laplace and, for levels a "
        "chain did not reach, results/aos/main/raw/ref_realdata_ext, scripts/aos_ref_realdata.py evalues). The median "
        "(level 2 delta) is no longer in Table 2; it is kept here and in REFEREE_RUNS.md.")
    say()
    say("| contrast | D0 | tau(0) | n_fire | alpha_last | bisection | (stored tau_new: alpha_last / bisection / scale) | Laplace order min / median | e-value (avg K_N), first failure (level, average) | per split | Laplace linear min / median | Gaussian (stored) min / median | no bet at stop |")
    say("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    gauss_rd = {}
    for name in CONTRASTS:
        recs = {}
        for line in (ROOT / "results" / "aos" / "realdata" / "wealth" / f"{name}.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                recs.setdefault(r["seed"], {})[round(r["alpha"], 6)] = r["certified"]
        lows = []
        for s in SEEDS:
            lv = 0.0
            for a in sorted(recs[s]):
                if recs[s][a]:
                    lv = a
                else:
                    break
            lows.append(lv)
        gauss_rd[name] = lows
    all_above = True
    RD = {}
    for name in CONTRASTS:
        tag = TAGS[name]
        lows = {"order": [], "linear": []}
        nobet = 0
        for s in SEEDS:
            ch = {round(r["alpha"], 6): r for r in jl(RAW / "realdata_laplace" / f"{rd_chain(name)}_{s}.jsonl")}
            for c in ("order", "linear"):
                lv, stop = 0.0, None
                for a in sorted(ch):
                    if ch[a][c]["certified"]:
                        lv = a
                    else:
                        stop = ch[a]
                        break
                lows[c].append(lv)
                if c == "order" and stop is not None and all(x == "skip" for x in stop["order"]["source"]):
                    nobet += 1
        mn, md_ = min(lows["order"]), float(np.median(lows["order"]))
        mac(f"mnRd{tag}Min", f"{mn:.2f}")
        mac(f"mnRd{tag}Med", f"{md_:.2f}")
        mac(f"mnRd{tag}LinMin", f"{min(lows['linear']):.2f}")
        mac(f"mnRd{tag}LinMed", f"{float(np.median(lows['linear'])):.2f}")
        mac(f"mnRd{tag}GaussMin", f"{min(gauss_rd[name]):.2f}")
        mac(f"mnRd{tag}GaussMed", f"{float(np.median(gauss_rd[name])):.2f}")
        mac(f"mnRd{tag}NoBet", nobet)
        if name == "rhc":
            # the Gaussian kernel on the 45 covariates (cold read B2, 2026-11-24): AOS_REALDATA_KERNEL=gaussian
            # scripts/aos_main_realdata.py run --contrasts rhc_full; genuine quantile cuts ("order"), as the Laplace runs
            gl = []
            for s in SEEDS:
                f = RAW / "realdata_gaussian" / f"rhc_full_{s}.jsonl"
                if not f.exists():
                    gl = None
                    break
                ch = {round(r["alpha"], 6): r for r in jl(f)}
                lv = 0.0
                for a in sorted(ch):
                    if ch[a]["order"]["certified"]:
                        lv = a
                    else:
                        break
                gl.append(lv)
            if gl:
                mac("mnRdRhcFullGaussMin", f"{min(gl):.2f}")
                mac("mnRdRhcFullGaussMed", f"{float(np.median(gl)):.2f}")
        mac(f"mnRd{tag}Splits", " ".join(f"{x:.2f}" for x in lows["order"]))
        ev, ev_fail = ev_final_level(rd_chain(name), name)
        mac(f"mnRd{tag}EvAvg", f"{ev:.2f}")
        if name == "rhc":                     # both covariate sets, whatever --rhc-covariates is
            mac("mnRdRhcPaperEvAvg", f"{ev_final_level('rhc', name)[0]:.2f}")
            mac("mnRdRhcFullEvAvg", f"{ev_final_level('rhc_full', name)[0]:.2f}")
        if name == "nhanes_bmi":
            mac("mnRdBmiCertZeroN", sum(x > 0 or _cert0(name, s) for x, s in zip(lows["order"], SEEDS)))
        # analytic, new threshold
        rt = route.get(name, "pilot (tune-then-test)")
        pv = [r for r in vb if r["dataset"] == name and r["route"] == rt]
        po = [r for r in vb_old if r["dataset"] == name and r["route"] == rt and r.get("threshold") == "plugin"
              and not (name == "rhc" and RHC_COV == "full")]      # stored tau_new: 33 covariates only
        n_, m_ = full_nm[name]
        assert len(pv) == 1, (name, rt, len(pv))
        if pv:
            pv = pv[0]
            fired = pv["H00_rejected"] == "True"
            last = f"{float(pv['alpha_last']):.2f}" if fired else "--"
            fine = f"{float(pv['certified_fine']):.4f}".rstrip("0") if fired else "--"   # as jmlr/v3/tools/derive.py
            if fine.endswith("."):
                fine += "0"
            if fired and float(pv["certified_fine"]) == 0.0:
                fine = "0"   # no positive level after bisection (was printed "0.0")
            fire_n = math.ceil(2 * (c0 / float(pv["D0"])) ** 2)
            mac(f"mnPv{tag}Dzero", f"{float(pv['D0']):.3f}")
            mac(f"mnPv{tag}Tauzero", f"{float(pv['tau0']):.3f}")
            mac(f"mnPv{tag}Fire", T.tex_n(fire_n))
            mac(f"mnPv{tag}Last", last)
            mac(f"mnPv{tag}Fine", fine)
            mac(f"mnPv{tag}Scale", pv["bandwidth"].split(" x ")[0])
            lines.append(" & ".join([LABEL_RD[name], T.tex_n(n_), T.tex_n(m_), f"{float(pv['D0']):.3f}",
                                     f"{float(pv['tau0']):.3f}", T.tex_n(fire_n), last, fine, "",
                                     f"{mn:.2f}", f"{ev:.2f}"]) + r" \\")
            if fired and float(pv["alpha_last"]) >= mn:
                all_above = False
            old = (f"{float(po[0]['alpha_last']):.2f} / {float(po[0]['certified_fine']):.4f} / "
                   f"{po[0]['bandwidth'].split(' x ')[0]}") if po and po[0]["H00_rejected"] == "True" else "--"
            say(f"| {name} | {float(pv['D0']):.3f} | {float(pv['tau0']):.3f} | {fire_n} | {last} | {fine} | {old} | "
                f"{mn:.2f} / {md_:.2f} | {ev:.2f}, {ev_fail[0]:.2f}, {ev_fail[1]:.3g} | "
                f"{' '.join(f'{x:.2f}' for x in lows['order'])} | "
                f"{min(lows['linear']):.2f} / {float(np.median(lows['linear'])):.2f} | "
                f"{min(gauss_rd[name]):.2f} / {float(np.median(gauss_rd[name])):.2f} | {nobet} |")
            RD[name] = (mn, md_, last, fine, old, min(gauss_rd[name]), float(np.median(gauss_rd[name])), ev)
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "aos_main_realdata.tex").write_text("\n".join(lines) + "\n")
    print("wrote", TAB / "aos_main_realdata.tex")
    mac("mnRdMinAboveAnalytic", "yes" if all_above else "no")
    say()

    # ------------------------------------------------------------------ S6 Figure 2(b)
    chk, levels, sizes, r_ana, r_cert, factors = panel_b()
    say("## S6, Figure 2(b): RHC-shaped design by sample size, analytic with tau = min{tau^P, tau^A}")
    say()
    say("Stored decisions = d_hat > tau_new in " + ", ".join(f"{k}: {a}/{t}" for k, (a, t) in chk.items())
        + " (grid levels x rows); alpha-hat recomputed from the stored paths (results/aos/main/fig2_data_aos.json, "
          "fig2_small_aos.json).")
    for i, a in enumerate(levels):
        say(f"- alpha* = {a}: " + " ".join(f"n={int(n)} {x:.2f}/{c:.2f}" for n, x, c in zip(sizes, r_ana[i], r_cert[i]))
            + "  (analytic / split-sample mean fraction)")
    say("- sample-size factors at n = 8,000 and 16,000 (split-sample size reaching the analytic fraction): "
        + ", ".join("--" if f is None else f"{f:.1f}" for f in factors))
    say()

    # ------------------------------------------------------------------ compute
    secs = {}
    for kind in ("synth_laplace", "synth_gaussian", "alpha0_laplace", "alpha0_gaussian", "realdata_laplace"):
        tot = 0.0
        for f in (RAW / kind).glob("*.jsonl"):
            if RHC_COV == "paper" and f.name.startswith("rhc_full_"):
                continue        # the 45-covariate RHC run (referee runs) is not part of the 33-covariate outputs
            for r in jl(f):
                tot += r.get("seconds", 0.0)
        secs[kind] = tot / 3600
    say("## Compute (CPU-hours, sum of per-record seconds, single-threaded workers; real data 2 BLAS threads)")
    say()
    for k, v in secs.items():
        say(f"- {k}: {v:.2f}")
    say()

    # ------------------------------------------------------------------ files and macros
    lines = ["% Generated by scripts/aos_main_tables.py -- do not edit."] + [
        f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in MACROS]
    names = [k for k, _ in MACROS]
    assert len(names) == len(set(names)), [k for k in names if names.count(k) > 1]
    (TAB / "aos_main_macros.tex").write_text("\n".join(lines) + "\n")
    print("wrote", TAB / "aos_main_macros.tex")
    md = dict(MACROS)
    # ------------------------------------------------------------------ headline numbers (inserted after the intro)
    H = ["## Headline numbers", ""]
    H.append(f"- Table 1 (power 0.8 against H_(0,0), {md['mnRepsPerSize']} replicates per size, n = {md['mnNmin']} to "
             f"{md['mnNcap'].replace('{,}', ',')}): the Laplace ramp (alpha-hat) needs the fewest observations among "
             f"the four tests that certify a level in {md['mnNeightyRampBest']} of {md['mnNdesigns']} designs "
             f"({md['mnNeightyRampBestAlone']} alone); over all six tests the permutation MMD is fastest in "
             f"{md['mnNeightyPermBest']}, MMDAgg in {md['mnNeightyAggBest']}, the ramp in {md['mnNeightyRampAllBest']}; "
             f"the ramp reaches power 0.8 nowhere in {md['mnNeightyRampNotReached']} designs, the analytic test in "
             f"{md['mnNeightyAnalyticNotReached']}.")
    H.append(f"- Ratios in Table 1: ramp / permutation MMD {md['mnNeightyRampOverPermLo']} to "
             f"{md['mnNeightyRampOverPermHi']} (median {md['mnNeightyRampOverPermMed']}, {md['mnNeightyRampOverPermN']} "
             f"designs with both interpolated); analytic / ramp {md['mnAnaOverRampLo']} to {md['mnAnaOverRampHi']} "
             f"(median {md['mnAnaOverRampMed']}, {md['mnAnaOverRampN']} designs); analytic n80 / size at which tau(0) "
             f"meets the population MMD {md['mnTauFireRatioLo']} to {md['mnTauFireRatioHi']} ({md['mnTauFireRatioN']} "
             f"designs). Laplace vs Gaussian ramp in Table 1: fewer observations in {md['mnNeightyLapBelowGauss']}, "
             f"more in {md['mnNeightyLapAboveGauss']}, equal in {md['mnNeightyLapEqGauss']} designs.")
    H.append(f"- Recovery at n = m = 800 ({md['mnRecReps']} replicates): median over designs of the mean certified "
             f"level / alpha*: Laplace ramp {md['mnRecRampMedian']} (max {md['mnRecRampMax']}, RHC-shaped "
             f"{md['mnRecRhcRamp']}), Gaussian ramp {md['mnRecGaussMedian']}, analytic {md['mnRecAnalyticMedian']} "
             f"(no positive level on average in {md['mnRecAnalyticZero']} designs, above the ramp in "
             f"{md['mnRecAnalyticAboveRamp']}); Laplace vs Gaussian at two paired s.e. {md['mnLapGaussWins']} wins / "
             f"{md['mnLapGaussTies']} ties / {md['mnLapGaussLosses']} losses (largest loss {md['mnLapGaussLossMax']}, "
             f"largest gain {md['mnLapGaussGainMax']}, oscillating nu = 4 {md['mnRecNfourGauss']} -> "
             f"{md['mnRecNfourLap']}); the ramp above the better default classifier on its batches in "
             f"{md['mnRampAboveBestMatched']}, tuned logistic above the ramp in {md['mnLogitTunedAboveRamp']} (ramp "
             f"win/tie/loss {md['mnWtlRecLogitTuned']}).")
    H.append(f"- Held-out replicates alone (50-199, extension 200050-200199; {md['mnHoReps']} per cell): Table 1 ramp "
             f"best of four in {md['mnHoNeightyRampBest']} ({md['mnHoNeightyRampBestAlone']} alone), permutation MMD "
             f"fastest in {md['mnHoNeightyPermBest']}, ramp / permutation {md['mnHoNeightyRampOverPermLo']} to "
             f"{md['mnHoNeightyRampOverPermHi']} (median {md['mnHoNeightyRampOverPermMed']}), analytic / ramp "
             f"{md['mnHoAnaOverRampLo']} to {md['mnHoAnaOverRampHi']} (median {md['mnHoAnaOverRampMed']}), Laplace "
             f"fewer / more observations than Gaussian in {md['mnHoNeightyLapBelowGauss']} / "
             f"{md['mnHoNeightyLapAboveGauss']}; recovery medians Laplace {md['mnHoRecRampMedian']}, Gaussian "
             f"{md['mnHoRecGaussMedian']}, analytic {md['mnHoRecAnalyticMedian']}; Laplace vs Gaussian "
             f"{md['mnHoLapGaussWins']}/{md['mnHoLapGaussTies']}/{md['mnHoLapGaussLosses']} (largest loss "
             f"{md['mnHoLapGaussLossMax']}, largest gain {md['mnHoLapGaussGainMax']}).")
    H.append(f"- Validity: boundary alpha = alpha* at n = 800, Laplace ramp, {md['mnBdTotal']} rejections in "
             f"{md['mnBdReps']} replicates, largest {md['mnBdMax']} ({md['mnBdMaxDesign']}, p = {md['mnBdMaxP']}), "
             f"flags {md['mnBdFlags']}; share with alpha-hat > alpha* at most {md['mnAloGtMax']}; P = Q nulls "
             f"{md['mnNullCells']} cells, pooled {md['mnNullPct']}% (cells {md['mnNullLoPct']}% to "
             f"{md['mnNullHiPct']}%), largest {md['mnNullMax']}, flags {md['mnNullFlags']}; analytic: "
             f"{md['mnAnBdRejections']} boundary rejections ({md['mnAnBdCells']} cells), {md['mnAnNullRejections']} "
             f"rejections of P = Q.")
    H.append("- Real data, split-sample bound alpha-hat (Laplace, genuine quantile cuts; min / median of eight "
             "splits) and analytic (new tau; alpha_last / bisection): " + "; ".join(
                 f"{name} {v[0]:.2f} / {v[1]:.2f} (analytic {v[2]} / {v[3]})" for name, v in RD.items()) + "; "
             "averaged final-wealth e-value (Table 2): " + ", ".join(f"{name} {v[7]:.2f}" for name, v in RD.items())
             + ".")
    H.append("")
    # ------------------------------------------------------------------ observations (computed)
    O = ["## Observations", ""]
    below1 = [k for k in KEYS if interpolated(tab, k, "ramp") and interpolated(tab, k, "perm_pilot")
              and n80val(tab, k, "ramp") < n80val(tab, k, "perm_pilot")]
    O.append("- Designs where the Laplace ramp needs fewer observations than the permutation MMD (both interpolated): "
             + (", ".join(f"{k} ({tab[(k, 'ramp')][0]} vs {tab[(k, 'perm_pilot')][0]})".replace("{,}", ",")
                          for k in below1) if below1 else "none") + ".")
    dif = sorted(res[k][0] for k in KEYS)
    O.append(f"- Laplace - Gaussian recovery per design: median paired difference {statistics.median(dif):+.3f}, "
             f"range {dif[0]:+.3f} to {dif[-1]:+.3f}; losses at two paired s.e. in "
             + ", ".join(f"{k} ({res[k][0]:+.3f})" for k in KEYS if res[k][2] == "loss")
             + "; wins in " + ", ".join(f"{k} ({res[k][0]:+.3f})" for k in KEYS if res[k][2] == "win") + ".")
    O.append("- Genuine quantile vs linear (numpy.quantile) cuts on the same Laplace fits, recovery: losses at two "
             "paired s.e. in " + (", ".join(f"{k} ({res2[k][0]:+.4f})" for k in KEYS if res2[k][2] == "loss")
                                  or "none") + "; wins in "
             + (", ".join(f"{k} ({res2[k][0]:+.4f})" for k in KEYS if res2[k][2] == "win") or "none") + ".")
    mixed = [k for k in KEYS if (n80val(tab, k, "ramp") < n80val(tab, k, "ramp_gauss") and res[k][2] == "loss")
             or (n80val(tab, k, "ramp") > n80val(tab, k, "ramp_gauss") and res[k][2] == "win")]
    O.append("- Designs where the kernel comparison points one way in Table 1 and the other in recovery: "
             + (", ".join(f"{k} (n80 {tab[(k, 'ramp')][0]} vs {tab[(k, 'ramp_gauss')][0]}, recovery "
                          f"{res[k][0]:+.3f})".replace("{,}", ",") for k in mixed) or "none") + ".")
    old_rec = {}
    for k in KEYS:
        st, n = stored(k), main_n(k)
        old_rec[k] = float(np.mean([frac(st[(n, rep)]["jlo"]["an_new"], st[(n, rep)]["levels"],
                                         st[(n, rep)]["alpha_star"]) for rep in per_rep[k]["analytic"]]))
    drop = [k for k in KEYS if abs(old_rec[k] - R[k]["analytic"][0]) > 1e-12]
    O.append("- Analytic recovery, stored tau_new (with tau^P') -> tau = min{tau^P, tau^A}: "
             + (", ".join(f"{k} {old_rec[k]:.3f} -> {R[k]['analytic'][0]:.3f}" for k in drop) or "unchanged")
             + f"; Table 1 analytic column changed in {md['mnNeightyAnalyticChanged']} designs.")
    O.append("- Analytic on real data, stored tau_new (alpha_last / bisection / scale) -> new tau: "
             + "; ".join(f"{name} {v[4]} -> {v[2]} / {v[3]}" for name, v in RD.items()) + ".")
    O.append("- Split-sample bound on real data, Laplace (genuine quantile cuts) vs Gaussian (stored), min / median: "
             + "; ".join(f"{name} {v[0]:.2f} / {v[1]:.2f} vs {v[5]:.2f} / {v[6]:.2f}" for name, v in RD.items())
             + ".")
    O.append("")
    k0 = REPORT.index("Protocol and running log: results/aos/main/NOTES.md. Raw records: results/aos/main/raw.") + 2
    REPORT[k0:k0] = H + O
    # ------------------------------------------------------------------ files and macros
    say("## Files")
    say()
    say("Tables and macros (scripts/aos_main_tables.py):")
    say()
    for f in ("aos_main_n80.tex", "aos_main_n80_heldout.tex", "aos_main_n80_gauss.tex", "aos_main_recovery_gauss.tex",
              "aos_main_realdata.tex", "aos_main_macros.tex"):
        say(f"- results/aos/tables/{f}")
    s6 = MAIN / "s6" / "files.json"
    if s6.exists():
        F = json.loads(s6.read_text())
        say()
        say("S6 tables (scripts/aos_main_s6.py; analytic under tau = min{tau^P, tau^A} = stored an_newcf; headers in "
            f"the AoS notation; validity: {F['validity_rows'][0]} rows -> {F['validity_rows'][1]} after merging "
            "identical rows; split_tests merges aos_baselines_eb.tex and aos_extra_tests.tex, all columns from the "
            "stored replicates of the study):")
        say()
        for f in F["files"]:
            say(f"- {f}")
    say()
    say("Figure data and figures:")
    say()
    say("- results/aos/main/fig_recovery_main.json, fig_recovery_gauss.json (recovery figure data), "
        "results/aos/main/fig2_data_aos.json, fig2_small_aos.json (S6 panel (b), analytic under the new tau)")
    for f in ("results/aos/main/figs/fig_recovery_main.pdf", "results/aos/main/figs/fig_recovery_gauss_s6.pdf",
              "aos/figs_aos/fig_recovery_main.pdf"):
        p_ = ROOT / f
        say(f"- {f}" + ("" if p_.exists() else " (not drawn yet: bash scripts/aos_main_figs.sh)"))
    say("  (scripts/aos_main_figs.sh: scripts/v3_fig_recovery.py with TMMD_FIGA_MAIN, TMMD_FIG_PANELS=a, "
        "TMMD_FIG_NOTATION=aos2; the main-text recovery figure is copied to aos/figs_aos/)")
    say()
    say(f"## Macros (results/aos/tables/aos_main_macros.tex, {len(MACROS)} macros)")
    say()
    say("```")
    for k, v in MACROS:
        say(f"\\{k} = {v}")
    say("```")
    (MAIN / "REPORT.md").write_text("\n".join(REPORT) + "\n")
    print("wrote", MAIN / "REPORT.md")
    names = [k for k, _ in MACROS + EXTRA]
    assert len(names) == len(set(names)), [k for k in names if names.count(k) > 1]
    (TAB / "aos_main_extra_macros.tex").write_text("\n".join(
        ["% Generated by scripts/aos_main_tables.py (extra_summaries) -- do not edit."]
        + [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in EXTRA]) + "\n")
    print("wrote", TAB / "aos_main_extra_macros.tex")


def panel_b():
    """S6, Figure 2(b): the RHC-shaped run by sample size with the analytic estimator under tau = min{tau^P, tau^A}.
    The stored rows hold the whole estimated path d_hat on the grid, and their decisions are exactly d_hat > tau_new
    (checked below), so alpha-hat = the first grid level with d_hat <= tau is recomputed with the new tau (Gaussian
    constants on the exact Gram, n < 2000; bounded constants for the cos/sin features, n >= 2000), no solving."""
    out = {}
    for src, dst in ((ROOT / "results" / "v3" / "figures" / "fig2_data_merged.json", MAIN / "fig2_data_aos.json"),
                     (ROOT / "results" / "v2" / "figures" / "fig2_small.json", MAIN / "fig2_small_aos.json")):
        D = json.loads(src.read_text())
        g = D["grid"]
        agree = tot = 0
        for r in D["rows"]:
            kc = TH.KernelConstants.bounded(1.0) if r.get("rff") else TH.GAUSSIAN
            n = r["n"]
            rej_old = [r["d_hat"][j] > TH.tau_new(n, n, a, DELTA, kc) for j, a in enumerate(g)]
            agree += sum(x == y for x, y in zip(rej_old, r["reject"]))
            tot += len(g)
            rej = [r["d_hat"][j] > TH.tau_aos(n, n, a, DELTA, kc) for j, a in enumerate(g)]
            first = next((j for j, x in enumerate(rej) if not x), len(g) - 1)
            r["alpha_hat_tau_new"] = r["alpha_hat"]
            r["alpha_hat"] = float(g[first])
            r["reject"] = rej
            r["tau_0"] = TH.tau_aos(n, n, 0.0, DELTA, kc)
            r["fires"] = bool(rej[0])
        D["threshold"] = "aos: min(tau^P closed form, tau^A) (scripts/aos_main_tables.py::panel_b)"
        dst.write_text(json.dumps(D))
        out[src.name] = (agree, tot)
    import v3_fig_recovery as VR
    VR.FIG2_SMALL = MAIN / "fig2_small_aos.json"          # load_regime reads both replaced files
    levels, sizes, r_ana, r_cert, _ = VR.load_regime(MAIN / "fig2_data_aos.json")
    assert np.all(r_ana <= r_cert + 1e-12), "analytic above the split-sample bound in a cell of the RHC-shaped run"
    factors = []
    for N in VR.GAP_SIZES:
        j = int(np.where(sizes == N)[0][0])
        for i in range(len(levels)):
            try:
                n_split = VR.size_reached(sizes, r_cert[i], r_ana[i, j])
            except AssertionError:          # reached at the smallest simulated size: only a lower bound
                n_split = None
            factors.append(None if n_split is None else N / n_split)
    big = int(sizes[-1])
    ana_big = [100 * r_ana[i, -1] for i in range(len(levels))]
    mac("mnRgNatLargest", T.tex_n(big))
    mac("mnRgAnaRecoverLo", f"{min(ana_big):.0f}")
    mac("mnRgAnaRecoverHi", f"{max(ana_big):.0f}")
    fs = [f for f in factors if f is not None]
    if fs:
        mac("mnRecGapLo", VR.sig2(min(fs)))
        mac("mnRecGapHi", VR.sig2(max(fs)))
    return out, levels, sizes, r_ana, r_cert, factors


def _cert0(name, seed):
    ch = {round(r["alpha"], 6): r for r in jl(RAW / "realdata_laplace" / f"{name}_{seed}.jsonl")}
    return bool(ch.get(0.0, {}).get("order", {}).get("certified", False))


def ev_final_level(stem, contrast):
    """Table 2, column e-value (avg): the largest grid level up to which the average over the eight splits of the
    final wealth K_N (cut rule "order") stays >= 1/delta, read upward from alpha = 0 (0.00 if none; Table 2's
    convention). Records: the chains of results/aos/main/raw/realdata_laplace/<stem>_<seed>.jsonl and, for the
    levels a chain stopped before, results/aos/main/raw/ref_realdata_ext/<stem>_<seed>.jsonl
    (scripts/aos_ref_realdata.py evalues). Returns (level, (first failing level, average there))."""
    grid = [round(0.05 * k, 6) for k in range(int(round((0.95 if contrast == "nhanes_child" else 0.75) / 0.05)) + 1)]
    ch = {}
    for s in SEEDS:
        ch[s] = {round(r["alpha"], 6): r for r in jl(RAW / "realdata_laplace" / f"{stem}_{s}.jsonl")}
        for r in jl(RAW / "ref_realdata_ext" / f"{stem}_{s}.jsonl"):
            k = round(r["alpha"], 6)
            assert k not in ch[s] or ch[s][k]["order"] == r["order"], (stem, s, k)
            ch[s].setdefault(k, r)
    lv, fail = 0.0, None
    for a in grid:
        miss = [s for s in SEEDS if a not in ch[s]]
        assert not miss, f"{stem}: level {a} missing for splits {miss}; run scripts/aos_ref_realdata.py evalues"
        avg = float(np.mean([math.exp(ch[s][a]["order"]["log_final"]) for s in SEEDS]))
        if avg >= 1.0 / DELTA:
            lv = a
        else:
            fail = (a, avg)
            break
    assert fail is not None, (stem, "the averaged e-value certifies the whole grid")
    return lv, fail


def rd_chain(name):
    """File stem of a contrast's Laplace chains: RHC on the 45 covariates (rhc_full) unless --rhc-covariates paper."""
    return "rhc_full" if (name == "rhc" and RHC_COV == "full") else name


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--rhc-covariates", choices=["full", "paper"], default="full",
                     help="RHC covariate set of Table 2 (default full = 45 covariates; paper = the 33 of the "
                          "2026-11-22 outputs)")
    RHC_COV = _ap.parse_args().rhc_covariates
    main()

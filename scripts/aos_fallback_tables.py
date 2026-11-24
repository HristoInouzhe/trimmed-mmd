"""Exploratory (AoS): tables, macros and REPORT.md for the cut-point rules OWN / FALLBACK / PLAIN and the kernel arm
(Gaussian vs Laplace). No Monte Carlo work here; reads results/aos/fallback/raw (written by
scripts/aos_fallback_realdata.py and scripts/aos_fallback_synth.py) and, for the reproduction checks, the stored
results/aos/realdata/wealth/*.jsonl and explore/harder_sims/out/*.jsonl (read-only). Writes

    results/aos/fallback/summary.md, realdata_summary.csv, synth_summary.csv, REPORT.md
    results/aos/tables/aos_fallback_realdata.tex, aos_fallback_recovery.tex, aos_fallback_macros.tex
    results/aos/tables/aos_kernel_realdata.tex, aos_kernel_recovery.tex, aos_kernel_macros.tex   (if the Laplace
    runs exist)

    python -B scripts/aos_fallback_tables.py

Conventions as scripts/aos_baselines_tables.py: two decimals rounded half up on the decimal value, exact one-sided
binomial p-value against delta = 0.05 (flag < 0.01), win / tie / loss at two paired standard errors.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
from scipy.stats import binom  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("AOS_FALLBACK_RAW", str(ROOT / "results" / "aos" / "fallback" / "raw")))
OUTD = ROOT / "results" / "aos" / "fallback"
TAB = ROOT / "results" / "aos" / "tables"
STORED_RD = ROOT / "results" / "aos" / "realdata" / "wealth"
STORED_SY = ROOT / "explore" / "harder_sims" / "out"
DELTA = 0.05
L20 = math.log(1.0 / DELTA)
RULES = ("own", "fallback", "plain")
RNAME = {"own": "OWN", "fallback": "FALLBACK", "plain": "PLAIN"}
RMAC = {"own": "Own", "fallback": "Fallback", "plain": "Plain"}
KMAC = {"gaussian": "Gauss", "laplace": "Laplace"}
CONTRASTS = [("rhc", "RHC arms"), ("nhanes_child", "NHANES ages 6-11 vs 60-79"),
             ("nhanes_age", "NHANES ages 20-39 vs 60-79"), ("nhanes_bmi", "NHANES BMI <25 vs >=30")]
CMAC = {"rhc": "Rhc", "nhanes_child": "Child", "nhanes_age": "Age", "nhanes_bmi": "Bmi"}
SEEDS = [1000 + i for i in range(8)]
GRID_TOP = {"nhanes_child": 0.95}
DESIGNS = [  # (key, label, alpha* as printed): order of Table 1
    ("D9_rhc", r"RHC-shaped, $d=33$", "0.2"),
    ("D1_d2_a10", r"overlap, $d=2$", "0.1"), ("D1_d2_a20", r"overlap, $d=2$", "0.2"),
    ("D1_d10_a10", r"overlap, $d=10$", "0.1"), ("D1_d10_a20", r"overlap, $d=10$", "0.2"),
    ("D2_d20", r"dense shift, $d=20$", "0.2"), ("D2_d50", r"dense shift, $d=50$", "0.2"),
    ("D2_d100", r"dense shift, $d=100$", "0.2"), ("D3a_loc", r"sparse shift, $d=50$", "0.2"),
    ("D3b_corr", r"correlation flip, $d=50$", "0.2"), ("D4_small", r"small contamination, $d=2$", "0.05"),
    ("D5b_mixed", r"mixed types, $m=5n$", "0.2"), ("D6_t3", r"$t_3$ tails, $d=5$", "0.2"),
    ("D7_wide", "curves, wide bump", "0.2"), ("D7_narrow", "curves, narrow bump", "0.2"),
    ("D8_N1", r"oscillating, $\nu=1$", "0.5"), ("D8_N4", r"oscillating, $\nu=4$", "0.5"),
    ("D8_N16", r"oscillating, $\nu=16$", "0.5"), ("D8_N64", r"oscillating, $\nu=64$", "0.5"),
]
SUBSET = ["D2_d20", "D2_d50", "D3a_loc", "D6_t3", "D7_wide"]
SUBSET_N = 200
REPORT_J = (0, 4, 6, 8)


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------

def r2(x):
    return str(Decimal(repr(float(x))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def r3(x):
    return str(Decimal(repr(float(x))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def lv(x):
    """A real-data bound: '--' when not even TV = 0 is rejected."""
    return "--" if x < 0 else r2(x)


STOPLOGW = {}
LOSSES = []


def mac(name, val):
    assert name.isalpha(), name
    return f"\\newcommand{{\\{name}}}{{{val}}}"


def binom_p(k, R, p=DELTA):
    return float(binom.sf(k - 1, R, p))


def wtl(d):
    d = np.asarray(d, float)
    mu = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(d.size)) if d.size > 1 else 0.0
    if mu > 0 and mu > 2 * se:
        return mu, se, "win"
    if mu < 0 and -mu > 2 * se:
        return mu, se, "loss"
    return mu, se, "tie"


def read_jsonl(path):
    out = []
    if path.exists():
        for line in path.read_bytes().split(b"\n"):
            if not line.strip() or b"\x00" in line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def md_table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


# ---------------------------------------------------------------------------
# real data
# ---------------------------------------------------------------------------

def rd_chains(kernel):
    d = RAW / f"realdata_{kernel}"
    out = {}
    for name, _ in CONTRASTS:
        for s in SEEDS:
            recs = read_jsonl(d / f"{name}_{s}.jsonl")
            if recs:
                out[(name, s)] = sorted({round(r["alpha"], 6): r for r in recs}.values(), key=lambda r: r["alpha"])
    return out


def rd_split(recs, rule, name):
    """Down-closed bound of one split under a rule, and the record at its stopping level (None if the grid top was
    certified or the chain is incomplete)."""
    bound, stop = -1.0, None
    for r in recs:
        if r[rule]["certified"]:
            bound = r["alpha"]
        else:
            stop = r
            break
    complete = stop is not None or abs(recs[-1]["alpha"] - GRID_TOP.get(name, 0.75)) < 1e-9
    return bound, stop, complete


def rd_summary(kernel):
    ch = rd_chains(kernel)
    if not ch:
        return None
    S = {}
    for name, _ in CONTRASTS:
        for rule in RULES:
            b, stops, comp = [], [], []
            for s in SEEDS:
                if (name, s) not in ch:
                    comp.append(False)
                    continue
                bound, stop, complete = rd_split(ch[(name, s)], rule, name)
                b.append(bound)
                stops.append(stop)
                comp.append(complete)
            if len(b) < len(SEEDS) or not all(comp):
                S[(name, rule)] = None
                continue
            nobet = sum(1 for st in stops if st is not None and st[rule]["pairs_bet"] == 0)
            onebat = sum(1 for st in stops if st is not None and st[rule]["source"].count("skip") == 1)
            # crossing of the own cuts (what the paper reports) at the stopping level, both batches
            own_both = sum(1 for st in stops if st is not None and all(x["own_crossed"] for x in st["batches"]))
            plain_both = sum(1 for st in stops if st is not None and all(x["plain_crossed"] for x in st["batches"]))
            srcs = {}
            for st in stops:
                if st is not None:
                    k = "/".join(st[rule]["source"])
                    srcs[k] = srcs.get(k, 0) + 1
            lmax = [st[rule]["log_max"] for st in stops if st is not None and st[rule]["pairs_bet"] > 0]
            lfin = [st[rule]["log_final"] for st in stops if st is not None and st[rule]["pairs_bet"] > 0]
            pairs = [st[rule]["pairs_bet"] / st["horizon"] for st in stops if st is not None]
            top = sum(1 for st in stops if st is None)
            # wealth at the last certified level
            last = []
            for s in SEEDS:
                recs = ch[(name, s)]
                cert = [r for r in recs if r[rule]["certified"]]
                ok = []
                for r in recs:
                    if not r[rule]["certified"]:
                        break
                    ok.append(r)
                if ok:
                    last.append(ok[-1][rule]["log_final"])
            # extra levels gained by fallback over own (per split)
            S[(name, rule)] = {
                "bounds": b, "min": float(min(b)), "median": float(np.median(b)),
                "nobet": nobet, "one_batch_skipped": onebat, "own_crossed_both": own_both,
                "plain_crossed_both": plain_both, "sources": srcs, "grid_top": top,
                "stop_alpha": [None if st is None else st["alpha"] for st in stops],
                "stop_logmax_med": float(np.median(lmax)) if lmax else float("nan"),
                "stop_logmax_max": float(max(lmax)) if lmax else float("nan"),
                "stop_logfinal_med": float(np.median(lfin)) if lfin else float("nan"),
                "stop_n_bet": len(lmax), "stop_pairs_frac_med": float(np.median(pairs)) if pairs else float("nan"),
                "last_logfinal_min": float(min(last)) if last else float("nan"),
                "last_logfinal_max": float(max(last)) if last else float("nan"),
                "levels": sum(len(ch[(name, s)]) for s in SEEDS),
                "seconds": sum(r["seconds"] for s in SEEDS for r in ch[(name, s)]),
            }
    return S


def rd_repro():
    """OWN (Gaussian) against the stored per-(split, level) wealth records."""
    ch = rd_chains("gaussian")
    tot = agree = 0
    dmax = 0.0
    for name, _ in CONTRASTS:
        st = {}
        for r in read_jsonl(STORED_RD / f"{name}.jsonl"):
            st[(r["seed"], round(r["alpha"], 6))] = r
        for s in SEEDS:
            for r in ch.get((name, s), []):
                x = st.get((s, round(r["alpha"], 6)))
                if x is None:
                    continue
                tot += 1
                skipped = [b + 1 for b, src in enumerate(r["own"]["source"]) if src == "skip"]
                agree += int(x["certified"] == r["own"]["certified"] and x["skipped"] == skipped)
                if x["pairs_bet"] > 0:
                    dmax = max(dmax, abs(x["log_final"] - r["own"]["log_final"]))
    return agree, tot, dmax


# ---------------------------------------------------------------------------
# synthetic
# ---------------------------------------------------------------------------

def sy_records(kernel, key, n=None, zero=False):
    d = RAW / f"synth_{kernel}"
    if n is None:
        n = 400 if key == "D5b_mixed" else 800
    f = d / (f"{key}_n{n}_zero.jsonl" if zero else f"{key}_n{n}.jsonl")
    recs = {}
    for r in read_jsonl(f):
        recs.setdefault(r["rep"], r)
    return recs


def frac(j, levels, ast):
    return 0.0 if j < 0 or ast == 0 else levels[j] / ast


def sy_summary(kernel, reps=None):
    """Per design and rule: recovery, power at j = 0, boundary rejections at j = 8, crossing counts."""
    S = {}
    for key, _, _ in DESIGNS:
        recs = sy_records(kernel, key)
        if not recs:
            continue
        ids = sorted(recs) if reps is None else [r for r in sorted(recs) if r in reps]
        R = len(ids)
        row = {"R": R, "reps": ids}
        for rule in RULES:
            fr, d0, d8, nobet, onebat = [], [], [], 0, 0
            for i in ids:
                r = recs[i]
                x = r["rules"][rule]
                fr.append(frac(x["jlo"], r["levels"], r["alpha_star"]))
                d0.append(x["dec"][0])
                d8.append(x["dec"][8])
                js = x["jlo"] + 1
                if js <= 10:
                    src = x["src"][js]
                    if all(s in ("skip", "none") for s in src):
                        nobet += 1
                    elif sum(s in ("skip", "none") for s in src) == 1:
                        onebat += 1
            row[rule] = {"frac": fr, "rec": float(np.mean(fr)),
                         "rec_se": float(np.std(fr, ddof=1) / math.sqrt(R)) if R > 1 else float("nan"),
                         "pow0": float(np.mean(d0)), "k0": int(sum(d0)), "k8": int(sum(d8)),
                         "p8": binom_p(int(sum(d8)), R), "nobet": nobet, "onebat": onebat}
        # own-cut crossing at OWN's stopping level, by batch; fallback / plain usage
        cross_any = 0
        for i in ids:
            r = recs[i]
            js = r["rules"]["own"]["jlo"] + 1
            if js <= 10 and any(b.get("own_crossed", False) for b in r["batches"][js]):
                cross_any += 1
        row["own_cross_any_at_stop"] = cross_any
        used = sum(1 for i in ids for j in range(len(recs[i]["levels"]))
                   if "plain" in (recs[i]["rules"]["fallback"]["src"][j] or []))
        row["fallback_plain_cells"] = used
        row["cells"] = sum(len(recs[i]["levels"]) for i in ids)
        # reproduction of the stored seq_bet (Gaussian)
        row["repro_jlo"] = sum(1 for i in ids if recs[i]["repro"].get("seq_jlo"))
        row["repro_dec"] = sum(1 for i in ids if recs[i]["repro"].get("seq_dec"))
        row["repro_sigma"] = sum(1 for i in ids if recs[i]["repro"].get("sigma"))
        row["stored_rec"] = float(np.mean([frac(recs[i]["stored"]["jlo"], recs[i]["levels"], recs[i]["alpha_star"])
                                           for i in ids])) if all("stored" in recs[i] for i in ids) else float("nan")
        row["stored_pow0"] = float(np.mean([recs[i]["stored"]["dec"][0] for i in ids])) \
            if all("stored" in recs[i] for i in ids) else float("nan")
        row["stored_an_rec"] = float(np.mean([frac(recs[i]["stored"]["an_new_jlo"], recs[i]["levels"],
                                                   recs[i]["alpha_star"]) for i in ids]))
        row["stored_an_pow0"] = float(np.mean([recs[i]["stored"]["an_new_dec"][0] for i in ids]))
        if all("analytic" in recs[i] for i in ids):
            row["an_rec"] = float(np.mean([frac(recs[i]["analytic"]["jlo"], recs[i]["levels"], recs[i]["alpha_star"])
                                           for i in ids]))
            row["an_pow0"] = float(np.mean([recs[i]["analytic"]["dec"][0] for i in ids]))
            row["an_k8"] = int(sum(recs[i]["analytic"]["dec"][3] for i in ids))      # REPORT_J index 3 = j = 8
            row["an_undecided"] = int(sum(recs[i]["analytic"]["undecided"] for i in ids))
        row["seconds"] = float(sum(recs[i]["seconds"] for i in ids))
        S[key] = row
    return S


def sy_subset(kernel):
    out = {}
    for key in SUBSET:
        recs = sy_records(kernel, key, n=SUBSET_N, zero=True)
        if not recs:
            continue
        ids = sorted(recs)
        out[key] = {"R": len(ids), **{rule: float(np.mean([recs[i]["rules"][rule]["dec"][0] for i in ids]))
                                      for rule in RULES},
                    "stored": float(np.mean([recs[i]["stored"]["dec"][0] for i in ids])),
                    "repro0": sum(1 for i in ids if recs[i]["repro"].get("seq_dec0")),
                    "seconds": float(sum(recs[i]["seconds"] for i in ids))}
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    TAB.mkdir(parents=True, exist_ok=True)
    md, macros_f, macros_k = [], [], []
    stamp = time.strftime("%Y-%m-%d %H:%M")
    hdr = "% Generated by scripts/aos_fallback_tables.py -- do not edit."
    macros_f.append(hdr)
    macros_k.append(hdr)

    # ---------------- real data, cut rules (Gaussian) ----------------
    RG = rd_summary("gaussian")
    RL = rd_summary("laplace")
    ag, tot, dmax = rd_repro()
    md += [f"# Cut-point rules for the split-sample lower bound: summary (generated {stamp})", "",
           "Generated by scripts/aos_fallback_tables.py from results/aos/fallback/raw. Exploratory.", "",
           "## Real data, Gaussian kernel (the paper's protocol)", "",
           f"Reproduction: OWN takes the stored decision and skip pattern at {ag}/{tot} stored (split, level) cells "
           f"(results/aos/realdata/wealth); max |log K_N difference| {dmax:.2g}.", ""]
    macros_f += [mac("fbRdReproAgree", ag), mac("fbRdReproTotal", tot)]
    rows_csv = []
    if RG:
        rows = []
        for name, label in CONTRASTS:
            for rule in RULES:
                s = RG.get((name, rule))
                if s is None:
                    rows.append([label, RNAME[rule]] + ["(incomplete)"] * 9)
                    continue
                rows.append([label, RNAME[rule], lv(s["min"]), lv(s["median"]),
                             " ".join(lv(x) for x in s["bounds"]), s["nobet"], s["one_batch_skipped"],
                             s["own_crossed_both"], s["grid_top"],
                             f"{s['stop_n_bet']}: {s['stop_logmax_med']:.2f} / {s['stop_logmax_max']:.2f}"
                             if s["stop_n_bet"] else "--",
                             ", ".join(f"{k} x{v}" for k, v in sorted(s["sources"].items()))])
                rows_csv.append({"kernel": "gaussian", "contrast": name, "rule": rule, "min": s["min"],
                                 "median": s["median"], "bounds": " ".join(lv(x) for x in s["bounds"]),
                                 "nobet_at_stop": s["nobet"], "one_batch_skipped_at_stop": s["one_batch_skipped"],
                                 "own_crossed_both_at_stop": s["own_crossed_both"], "grid_top": s["grid_top"],
                                 "stop_logmax_median": s["stop_logmax_med"], "stop_logmax_max": s["stop_logmax_max"],
                                 "stop_logfinal_median": s["stop_logfinal_med"],
                                 "last_logfinal_min": s["last_logfinal_min"],
                                 "last_logfinal_max": s["last_logfinal_max"], "levels": s["levels"],
                                 "seconds": s["seconds"]})
                cm, rm = CMAC[name], RMAC[rule]
                macros_f += [mac(f"fbRd{cm}{rm}Min", lv(s["min"])), mac(f"fbRd{cm}{rm}Med", lv(s["median"])),
                             mac(f"fbRd{cm}{rm}NoBet", s["nobet"])]
                if s["stop_n_bet"]:
                    # largest log max-wealth at the stopping level over the splits that bet there (AoS text, 2026-11-21)
                    macros_f.append(mac(f"fbRd{cm}{rm}LogWMax", f"{s['stop_logmax_max']:.2f}"))
                    STOPLOGW.setdefault(rm, []).append((name, s["stop_logmax_max"], s["stop_n_bet"]))
        md += md_table(["contrast", "rule", "min", "median", "per split", "splits stopped with no bet",
                        "stopped with one batch skipped", "own cuts crossed in both batches at stop",
                        "grid top certified", "splits betting at stop: median / max log max K (log 20 = 3.00)",
                        "batch sources at stop"], rows) + [""]
        md += ["'stopped with no bet': at the first non-rejected level both betting batches were skipped under the "
               "rule (cuts crossed), so lambda = 0 and the wealth stayed 1. For FALLBACK a batch is skipped only "
               "when both the own-term and the plain cuts cross.", ""]
        # LaTeX
        tl = [hdr, r"\begin{tabular}{lcccccc}", r"\toprule",
              r" & \multicolumn{3}{c}{minimum over splits} & \multicolumn{3}{c}{median over splits} \\",
              r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
              r"contrast & own & fallback & plain & own & fallback & plain \\", r"\midrule"]
        for name, label in CONTRASTS:
            ss = [RG.get((name, r)) for r in RULES]
            if any(x is None for x in ss):
                continue
            tl.append(" & ".join([label.replace(">=", r"$\ge$").replace("<", "$<$")]
                                 + [f"${lv(x['min'])}$" for x in ss] + [f"${lv(x['median'])}$" for x in ss])
                      + r" \\")
        tl += [r"\midrule", r"\multicolumn{7}{l}{splits whose bound stopped with no bet (cuts crossed in both "
               r"batches), of 8} \\"]
        for name, label in CONTRASTS:
            ss = [RG.get((name, r)) for r in RULES]
            if any(x is None for x in ss):
                continue
            tl.append(" & ".join([label.replace(">=", r"$\ge$").replace("<", "$<$")]
                                 + [f"${x['nobet']}$" for x in ss] + ["", "", ""]) + r" \\")
        tl += [r"\bottomrule", r"\end{tabular}"]
        (TAB / "aos_fallback_realdata.tex").write_text("\n".join(tl) + "\n")

    # ---------------- synthetic, cut rules (Gaussian) ----------------
    SG = sy_summary("gaussian")
    if SG:
        R = min(SG[k]["R"] for k in SG)
        md += [f"## Synthetic designs of Figure 2(a), Gaussian kernel ({len(SG)} designs, {R}-"
               f"{max(SG[k]['R'] for k in SG)} replicates each)", ""]
        tot_rep = sum(SG[k]["R"] for k in SG)
        md += [f"Reproduction of the stored seq_bet by OWN (full warm-start chain): jlo {sum(SG[k]['repro_jlo'] for k in SG)}"
               f"/{tot_rep}, decisions at j = 0, 4, 6, 8 {sum(SG[k]['repro_dec'] for k in SG)}/{tot_rep}; fold "
               f"bandwidths recomputed equal to the stored ones in {sum(SG[k]['repro_sigma'] for k in SG)}/{tot_rep} "
               f"(the stored values are used).", ""]
        rows, trow = [], []
        meds = {}
        for rule in RULES:
            meds[rule] = float(np.median([SG[k][rule]["rec"] for k in SG]))
        meds["stored"] = float(np.median([SG[k]["stored_rec"] for k in SG]))
        wins = {r: {"win": 0, "tie": 0, "loss": 0} for r in ("fallback", "plain")}
        for key, label, ast in DESIGNS:
            if key not in SG:
                continue
            s = SG[key]
            dF = wtl(np.array(s["fallback"]["frac"]) - np.array(s["own"]["frac"]))
            dP = wtl(np.array(s["plain"]["frac"]) - np.array(s["own"]["frac"]))
            wins["fallback"][dF[2]] += 1
            wins["plain"][dP[2]] += 1
            rows.append([key, ast, s["R"], r3(s["stored_rec"]), r3(s["own"]["rec"]), r3(s["fallback"]["rec"]),
                         r3(s["plain"]["rec"]), f"{dF[0]:+.3f} ({dF[1]:.3f}) {dF[2]}",
                         f"{dP[0]:+.3f} ({dP[1]:.3f}) {dP[2]}",
                         f"{r2(s['own']['pow0'])} / {r2(s['fallback']['pow0'])} / {r2(s['plain']['pow0'])}",
                         f"{s['own']['k8']} / {s['fallback']['k8']} / {s['plain']['k8']}",
                         f"{s['own']['nobet']} / {s['fallback']['nobet']} / {s['plain']['nobet']}",
                         s["own_cross_any_at_stop"], f"{s['fallback_plain_cells']}/{s['cells']}"])
            trow.append((label, ast, s))
        md += md_table(["design", "alpha*", "R", "stored seq_bet rec.", "OWN rec.", "FALLBACK rec.", "PLAIN rec.",
                        "FALLBACK - OWN (paired s.e.)", "PLAIN - OWN (paired s.e.)",
                        "power at alpha = 0 (O / F / P)", "rejections at alpha = alpha* (O / F / P)",
                        "stopped with no bet (O / F / P)", "own cuts crossed in a batch at OWN's stop",
                        "FALLBACK cells using a plain cut"], rows) + [""]
        up = {r: sum(int(a > b) for k in SG for a, b in zip(SG[k][r]["frac"], SG[k]["own"]["frac"]))
              for r in ("fallback", "plain")}
        dn = {r: sum(int(a < b) for k in SG for a, b in zip(SG[k][r]["frac"], SG[k]["own"]["frac"]))
              for r in ("fallback", "plain")}
        nb = {r: sum(SG[k][r]["nobet"] for k in SG) for r in RULES}
        md += [f"Per replicate: FALLBACK's bound is above OWN's in {up['fallback']} and below it in {dn['fallback']} "
               f"of {tot_rep} replicates (equal in the rest); PLAIN above in {up['plain']}, below in {dn['plain']}. "
               f"Replicates whose bound stopped with no bet (cuts crossed in both batches at the first non-rejected "
               f"level): OWN {nb['own']}, FALLBACK {nb['fallback']}, PLAIN {nb['plain']} of {tot_rep}.", ""]
        macros_f += [mac("fbSyFallbackUp", up["fallback"]), mac("fbSyFallbackDown", dn["fallback"]),
                     mac("fbSyPlainUp", up["plain"]), mac("fbSyPlainDown", dn["plain"]),
                     mac("fbSyOwnNoBet", nb["own"]), mac("fbSyFallbackNoBet", nb["fallback"]),
                     mac("fbSyTotalReps", tot_rep)]
        md += [f"Median over designs of the mean recovered fraction: stored seq_bet {r3(meds['stored'])}, OWN "
               f"{r3(meds['own'])}, FALLBACK {r3(meds['fallback'])}, PLAIN {r3(meds['plain'])}.",
               f"FALLBACK vs OWN, win/tie/loss at two paired s.e.: {wins['fallback']['win']}/{wins['fallback']['tie']}/"
               f"{wins['fallback']['loss']}; PLAIN vs OWN: {wins['plain']['win']}/{wins['plain']['tie']}/"
               f"{wins['plain']['loss']}.", ""]
        # validity
        md += ["Boundary validity (alpha = alpha*, a true null; exact one-sided binomial p against 0.05; flag if "
               "< 0.01):", ""]
        vrows = []
        for rule in RULES:
            k8 = [(SG[k][rule]["k8"], SG[k]["R"], k) for k in SG]
            mx = max(k8)
            flags = [k for a, b, k in k8 if binom_p(a, b) < 0.01]
            vrows.append([RNAME[rule], f"{mx[0]}/{mx[1]} ({mx[2]})", f"{binom_p(mx[0], mx[1]):.3f}", len(flags),
                          ", ".join(flags) or "--", sum(a for a, _, _ in k8), sum(b for _, b, _ in k8)])
            macros_f += [mac(f"fbSy{RMAC[rule]}BoundMax", mx[0]), mac(f"fbSy{RMAC[rule]}BoundFlags", len(flags))]
        md += md_table(["rule", "max rejections (design)", "its binomial p", "flags", "flagged designs",
                        "total rejections", "total replicates"], vrows) + [""]
        for rule in RULES:
            macros_f.append(mac(f"fbSyRecMed{RMAC[rule]}", r2(meds[rule])))
        macros_f += [mac("fbSyRecMedStored", r2(meds["stored"])), mac("fbSyReps", R),
                     mac("fbSyFallbackWins", wins["fallback"]["win"]), mac("fbSyFallbackTies", wins["fallback"]["tie"]),
                     mac("fbSyFallbackLosses", wins["fallback"]["loss"]), mac("fbSyPlainWins", wins["plain"]["win"]),
                     mac("fbSyPlainTies", wins["plain"]["tie"]), mac("fbSyPlainLosses", wins["plain"]["loss"])]
        # LaTeX recovery table
        tl = [hdr, r"\begin{tabular}{lccccccccc}", r"\toprule",
              r" &  & \multicolumn{3}{c}{certified level / $\alpha^\star$} & \multicolumn{3}{c}{power at $\alpha=0$}"
              r" & \multicolumn{2}{c}{rejections at $\alpha^\star$} \\",
              r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-10}",
              r"design & $\alpha^\star$ & own & fallback & plain & own & fallback & plain & fallback & plain \\",
              r"\midrule"]
        for label, ast, s in trow:
            rec = [r2(s[r]["rec"]) for r in RULES]
            b = max(rec, key=Decimal)
            tl.append(" & ".join([label, f"${ast}$"] + [rf"$\mathbf{{{x}}}$" if x == b else f"${x}$" for x in rec]
                                 + [f"${r2(s[r]['pow0'])}$" for r in RULES]
                                 + [f"${s[r]['k8']}/{s['R']}$" for r in ("fallback", "plain")]) + r" \\")
        tl += [r"\midrule", " & ".join(["median over designs", ""] + [f"${r2(meds[r])}$" for r in RULES]
                                       + ["", "", "", "", ""]) + r" \\", r"\bottomrule", r"\end{tabular}"]
        (TAB / "aos_fallback_recovery.tex").write_text("\n".join(tl) + "\n")
        with open(OUTD / "synth_summary.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["kernel", "design", "R", "rule", "rec", "rec_se", "pow0", "k8", "p8", "nobet_at_stop",
                        "one_batch_at_stop"])
            for key in SG:
                for rule in RULES:
                    x = SG[key][rule]
                    w.writerow(["gaussian", key, SG[key]["R"], rule, x["rec"], x["rec_se"], x["pow0"], x["k8"],
                                x["p8"], x["nobet"], x["onebat"]])
        # subset at n = 200
        sub = sy_subset("gaussian")
        if sub:
            md += [f"Power against H_(0,0) at n = m = {SUBSET_N} (levels j = 0 only; stored seq_bet in brackets; "
                   "OWN reproduces the stored decision in the count shown):", ""]
            md += md_table(["design", "R", "OWN", "FALLBACK", "PLAIN", "stored", "OWN = stored"],
                           [[k, v["R"], r2(v["own"]), r2(v["fallback"]), r2(v["plain"]), r2(v["stored"]),
                             f"{v['repro0']}/{v['R']}"] for k, v in sub.items()]) + [""]
        secs = sum(SG[k]["seconds"] for k in SG)
        md += [f"Compute: {secs / 3600:.1f} CPU-hours for the Gaussian synthetic replicates "
               f"(single-threaded, mean {secs / tot_rep:.1f} s per replicate, all 11 levels).", ""]
    for K in ("gaussian", "laplace"):
        ch = rd_chains(K)
        if ch:
            cells = sum(len(v) for v in ch.values())
            secs = sum(r["seconds"] for v in ch.values() for r in v)
            md += [f"Compute (real data, {K}): {cells} (split, level) cells, each fitted once for all three rules; "
                   f"{secs / 3600:.2f} worker-hours (2 BLAS threads per worker).", ""]

    # ---------------- kernel arm ----------------
    SL = sy_summary("laplace")
    if RL or SL:
        md += ["## Kernel arm: Gaussian vs Laplace (OWN and FALLBACK; PLAIN-Laplace stored as well)", ""]
    if RL and RG:
        rows = []
        tl = [hdr, r"\begin{tabular}{lcccccccc}", r"\toprule",
              r" & \multicolumn{4}{c}{minimum over splits} & \multicolumn{4}{c}{median over splits} \\",
              r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
              r" & \multicolumn{2}{c}{Gaussian} & \multicolumn{2}{c}{Laplace} & \multicolumn{2}{c}{Gaussian} & "
              r"\multicolumn{2}{c}{Laplace} \\",
              r"contrast & own & fallback & own & fallback & own & fallback & own & fallback \\", r"\midrule"]
        for name, label in CONTRASTS:
            cells = []
            for K, RR in (("gaussian", RG), ("laplace", RL)):
                for rule in RULES:
                    s = RR.get((name, rule))
                    cells.append(s)
            if any(c is None for c in cells):
                rows.append([label, "(incomplete)"])
                continue
            g = {r: cells[i] for i, r in enumerate(RULES)}
            l_ = {r: cells[3 + i] for i, r in enumerate(RULES)}
            rows.append([label] + [f"{lv(g[r]['min'])} / {lv(g[r]['median'])}" for r in RULES]
                        + [f"{lv(l_[r]['min'])} / {lv(l_[r]['median'])}" for r in RULES]
                        + [f"{g[r]['nobet']}" for r in RULES] + [f"{l_[r]['nobet']}" for r in RULES]
                        + [" ".join(lv(x) for x in l_["own"]["bounds"]), " ".join(lv(x) for x in l_["fallback"]["bounds"])])
            tl.append(" & ".join([label.replace(">=", r"$\ge$").replace("<", "$<$")]
                                 + [f"${lv(x['min'])}$" for x in (g["own"], g["fallback"], l_["own"], l_["fallback"])]
                                 + [f"${lv(x['median'])}$" for x in (g["own"], g["fallback"], l_["own"],
                                                                      l_["fallback"])]) + r" \\")
            for r in ("own", "fallback"):
                macros_k += [mac(f"kRd{CMAC[name]}Laplace{RMAC[r]}Min", lv(l_[r]["min"])),
                             mac(f"kRd{CMAC[name]}Laplace{RMAC[r]}Med", lv(l_[r]["median"])),
                             mac(f"kRd{CMAC[name]}Laplace{RMAC[r]}NoBet", l_[r]["nobet"])]
        tl += [r"\bottomrule", r"\end{tabular}"]
        (TAB / "aos_kernel_realdata.tex").write_text("\n".join(tl) + "\n")
        md += ["Real data (min / median over the 8 splits):", ""]
        md += md_table(["contrast", "Gauss OWN", "Gauss FALLBACK", "Gauss PLAIN", "Laplace OWN", "Laplace FALLBACK",
                        "Laplace PLAIN", "no bet at stop G-O", "G-F", "G-P", "L-O", "L-F", "L-P",
                        "Laplace OWN per split", "Laplace FALLBACK per split"], rows) + [""]
        for name, label in CONTRASTS:
            s = RL.get((name, "own"))
            if s:
                md.append(f"- {label}, Laplace OWN at the stopping level: {s['nobet']} of 8 splits with no bet, "
                          f"{s['stop_n_bet']} betting (median / max log max K {s['stop_logmax_med']:.2f} / "
                          f"{s['stop_logmax_max']:.2f}); sources {s['sources']}.")
        md.append("")
    if SL and SG:
        common = [k for k, _, _ in DESIGNS if k in SL and k in SG]
        rows, tl_rows = [], []
        meds = {}
        for K, SS in (("gaussian", SG), ("laplace", SL)):
            for rule in RULES:
                meds[(K, rule)] = float(np.median([SS[k][rule]["rec"] for k in common]))
        wins = {"win": 0, "tie": 0, "loss": 0}
        winsF = {"win": 0, "tie": 0, "loss": 0}
        for key, label, ast in DESIGNS:
            if key not in common:
                continue
            g, l_ = SG[key], SL[key]
            ids = [i for i in l_["reps"] if i in g["reps"]]
            gi = {i: n for n, i in enumerate(g["reps"])}
            li = {i: n for n, i in enumerate(l_["reps"])}
            d = wtl([l_["own"]["frac"][li[i]] - g["own"]["frac"][gi[i]] for i in ids])
            dF = wtl([l_["fallback"]["frac"][li[i]] - g["fallback"]["frac"][gi[i]] for i in ids])
            wins[d[2]] += 1
            winsF[dF[2]] += 1
            if key == "D8_N4":   # oscillating density, nu = 4: the largest Laplace gain (AoS text, 2026-11-21)
                macros_k += [mac("kSyNfourGaussRec", r2(g["own"]["rec"])), mac("kSyNfourLaplaceRec", r2(l_["own"]["rec"]))]
            if d[2] == "loss":
                LOSSES.append(-d[0])
            rows.append([key, ast, len(ids), r3(g["own"]["rec"]), r3(g["fallback"]["rec"]), r3(l_["own"]["rec"]),
                         r3(l_["fallback"]["rec"]), r3(l_["plain"]["rec"]), f"{d[0]:+.3f} ({d[1]:.3f}) {d[2]}",
                         f"{dF[0]:+.3f} ({dF[1]:.3f}) {dF[2]}",
                         f"{r2(g['own']['pow0'])} / {r2(l_['own']['pow0'])} / {r2(l_['fallback']['pow0'])}",
                         f"{g['own']['k8']} / {g['fallback']['k8']} / {l_['own']['k8']} / {l_['fallback']['k8']}",
                         f"{g['own']['nobet']} / {g['fallback']['nobet']} / {l_['own']['nobet']} / "
                         f"{l_['fallback']['nobet']}",
                         f"{r3(g['stored_an_rec'])} / {r3(l_['an_rec'])}" if "an_rec" in l_ else "--",
                         f"{r2(g['stored_an_pow0'])} / {r2(l_['an_pow0'])}" if "an_pow0" in l_ else "--"])
            tl_rows.append((label, ast, g, l_))
        md += [f"Synthetic, {len(common)} designs (paired on the same replicates):", ""]
        md += md_table(["design", "alpha*", "R", "Gauss OWN rec.", "Gauss FALLBACK rec.", "Laplace OWN rec.",
                        "Laplace FALLBACK rec.", "Laplace PLAIN rec.", "Laplace - Gauss, OWN (paired s.e.)",
                        "Laplace - Gauss, FALLBACK", "power at alpha = 0 (G-O / L-O / L-F)",
                        "rejections at alpha* (G-O / G-F / L-O / L-F)", "no bet at stop (G-O / G-F / L-O / L-F)",
                        "analytic rec. (Gauss stored / Laplace)", "analytic power at 0 (G / L)"], rows) + [""]
        macros_k += [mac("kSyLaplaceWins", wins["win"]), mac("kSyLaplaceTies", wins["tie"]),
                     mac("kSyLaplaceLosses", wins["loss"])]
        if LOSSES:
            macros_k.append(mac("kSyLaplaceLossMax", f"{max(LOSSES):.2f}"))
        cnt = {}
        for rule in ("own", "fallback"):
            u = d_ = tot_ = 0
            for key in common:
                gi = {i: n for n, i in enumerate(SG[key]["reps"])}
                for n_, i in enumerate(SL[key]["reps"]):
                    if i in gi:
                        a, b = SL[key][rule]["frac"][n_], SG[key][rule]["frac"][gi[i]]
                        u += int(a > b)
                        d_ += int(a < b)
                        tot_ += 1
            cnt[rule] = (u, d_, tot_)
        upL = sum(int(a > b) for k in common for a, b in zip(SL[k]["fallback"]["frac"], SL[k]["own"]["frac"]))
        dnL = sum(int(a < b) for k in common for a, b in zip(SL[k]["fallback"]["frac"], SL[k]["own"]["frac"]))
        nbL = {r: sum(SL[k][r]["nobet"] for k in common) for r in RULES}
        md += [f"Per replicate, Laplace above / below Gaussian: OWN {cnt['own'][0]} / {cnt['own'][1]}, FALLBACK "
               f"{cnt['fallback'][0]} / {cnt['fallback'][1]} of {cnt['own'][2]}. Within Laplace, FALLBACK above / below "
               f"OWN in {upL} / {dnL} replicates. Laplace replicates stopped with no bet: OWN {nbL['own']}, FALLBACK "
               f"{nbL['fallback']}, PLAIN {nbL['plain']}.", ""]
        macros_k += [mac("kSyLaplaceAboveOwn", cnt["own"][0]), mac("kSyLaplaceBelowOwn", cnt["own"][1]),
                     mac("kSyLaplaceFallbackUp", upL), mac("kSyLaplaceFallbackDown", dnL),
                     mac("kSyLaplaceOwnNoBet", nbL["own"])]
        md += [f"Median over designs of the mean recovered fraction: Gaussian OWN {r3(meds[('gaussian', 'own')])}, "
               f"FALLBACK {r3(meds[('gaussian', 'fallback')])}; Laplace OWN {r3(meds[('laplace', 'own')])}, FALLBACK "
               f"{r3(meds[('laplace', 'fallback')])}, PLAIN {r3(meds[('laplace', 'plain')])}.",
               f"Laplace vs Gaussian under OWN, win/tie/loss at two paired s.e.: {wins['win']}/{wins['tie']}/"
               f"{wins['loss']}; under FALLBACK: {winsF['win']}/{winsF['tie']}/{winsF['loss']}.", ""]
        vrows = []
        for K, SS in (("gaussian", SG), ("laplace", SL)):
            for rule in RULES:
                k8 = [(SS[k][rule]["k8"], SS[k]["R"], k) for k in common]
                mx = max(k8)
                flags = [k for a, b, k in k8 if binom_p(a, b) < 0.01]
                vrows.append([K, RNAME[rule], f"{mx[0]}/{mx[1]} ({mx[2]})", f"{binom_p(mx[0], mx[1]):.3f}",
                              len(flags), ", ".join(flags) or "--"])
                macros_k += [mac(f"kSy{KMAC[K]}{RMAC[rule]}BoundFlags", len(flags)),
                             mac(f"kSyRecMed{KMAC[K]}{RMAC[rule]}", r2(meds[(K, rule)]))]
        if all("an_k8" in SL[k] for k in common):
            k8 = [(SL[k]["an_k8"], SL[k]["R"], k) for k in common]
            mx = max(k8)
            flags = [k for a, b, k in k8 if binom_p(a, b) < 0.01]
            vrows.append(["laplace", "analytic", f"{mx[0]}/{mx[1]} ({mx[2]})", f"{binom_p(mx[0], mx[1]):.3f}",
                          len(flags), ", ".join(flags) or "--"])
            an_g = float(np.median([SG[k]["stored_an_rec"] for k in common]))
            an_l = float(np.median([SL[k]["an_rec"] for k in common]))
            nz_g = sum(1 for k in common if SG[k]["stored_an_rec"] == 0)
            nz_l = sum(1 for k in common if SL[k]["an_rec"] == 0)
            md += [f"Analytic test: median recovery Gaussian (stored an_new, same replicates) {r3(an_g)}, Laplace "
                   f"{r3(an_l)}; designs with no positive level certified on average 0: Gaussian {nz_g}, Laplace "
                   f"{nz_l}; undecided QP threshold comparisons (Laplace) "
                   f"{sum(SL[k]['an_undecided'] for k in common)}.", ""]
            macros_k += [mac("kSyAnRecMedGauss", r2(an_g)), mac("kSyAnRecMedLaplace", r2(an_l))]
        secsL = sum(SL[k]["seconds"] for k in common)
        md += [f"Compute: {secsL / 3600:.1f} CPU-hours for the Laplace synthetic replicates (single-threaded, all 11 "
               f"levels plus the analytic test; mean {secsL / sum(SL[k]['R'] for k in common):.1f} s per replicate).",
               ""]
        md += ["Boundary validity (alpha = alpha*):", ""]
        md += md_table(["kernel", "rule", "max rejections (design)", "binomial p", "flags", "flagged"], vrows) + [""]
        sub_g, sub_l = sy_subset("gaussian"), sy_subset("laplace")
        if sub_l:
            md += [f"Power against H_(0,0) at n = m = {SUBSET_N}:", ""]
            md += md_table(["design", "R", "Gauss OWN", "Gauss FALLBACK", "Laplace OWN", "Laplace FALLBACK",
                            "Laplace PLAIN"],
                           [[k, sub_l[k]["R"], r2(sub_g[k]["own"]) if k in sub_g else "--",
                             r2(sub_g[k]["fallback"]) if k in sub_g else "--", r2(sub_l[k]["own"]),
                             r2(sub_l[k]["fallback"]), r2(sub_l[k]["plain"])] for k in sub_l]) + [""]
        tl = [hdr, r"\begin{tabular}{lccccccccc}", r"\toprule",
              r" &  & \multicolumn{4}{c}{certified level / $\alpha^\star$} & \multicolumn{2}{c}{power at $\alpha=0$}"
              r" & \multicolumn{2}{c}{analytic, level / $\alpha^\star$} \\",
              r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
              r" &  & \multicolumn{2}{c}{Gaussian} & \multicolumn{2}{c}{Laplace} &  &  &  &  \\",
              r"design & $\alpha^\star$ & own & fallback & own & fallback & Gaussian & Laplace & Gaussian & Laplace \\",
              r"\midrule"]
        for label, ast, g, l_ in tl_rows:
            rec = [r2(g["own"]["rec"]), r2(g["fallback"]["rec"]), r2(l_["own"]["rec"]), r2(l_["fallback"]["rec"])]
            b = max(rec, key=Decimal)
            tl.append(" & ".join([label, f"${ast}$"] + [rf"$\mathbf{{{x}}}$" if x == b else f"${x}$" for x in rec]
                                 + [f"${r2(g['own']['pow0'])}$", f"${r2(l_['own']['pow0'])}$",
                                    f"${r2(g['stored_an_rec'])}$",
                                    f"${r2(l_['an_rec'])}$" if "an_rec" in l_ else "--"]) + r" \\")
        tl += [r"\midrule", " & ".join(["median over designs", ""]
                                       + [f"${r2(meds[(K, r)])}$" for K in ("gaussian", "laplace")
                                          for r in ("own", "fallback")] + ["", "", "", ""]) + r" \\",
               r"\bottomrule", r"\end{tabular}"]
        (TAB / "aos_kernel_recovery.tex").write_text("\n".join(tl) + "\n")
        with open(OUTD / "synth_summary.csv", "a", newline="") as fh:
            w = csv.writer(fh)
            for key in SL:
                for rule in RULES:
                    x = SL[key][rule]
                    w.writerow(["laplace", key, SL[key]["R"], rule, x["rec"], x["rec_se"], x["pow0"], x["k8"],
                                x["p8"], x["nobet"], x["onebat"]])
    if RL:
        for name, _ in CONTRASTS:
            for rule in RULES:
                s = RL.get((name, rule))
                if s:
                    rows_csv.append({"kernel": "laplace", "contrast": name, "rule": rule, "min": s["min"],
                                     "median": s["median"], "bounds": " ".join(lv(x) for x in s["bounds"]),
                                     "nobet_at_stop": s["nobet"], "one_batch_skipped_at_stop": s["one_batch_skipped"],
                                     "own_crossed_both_at_stop": s["own_crossed_both"], "grid_top": s["grid_top"],
                                     "stop_logmax_median": s["stop_logmax_med"],
                                     "stop_logmax_max": s["stop_logmax_max"],
                                     "stop_logfinal_median": s["stop_logfinal_med"],
                                     "last_logfinal_min": s["last_logfinal_min"],
                                     "last_logfinal_max": s["last_logfinal_max"], "levels": s["levels"],
                                     "seconds": s["seconds"]})
    if rows_csv:
        with open(OUTD / "realdata_summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_csv[0]))
            w.writeheader()
            w.writerows(rows_csv)
    crossed = [v for (nm, v, k) in STOPLOGW.get("Fallback", []) if CMAC[nm] in ("Rhc", "Child", "Age")]
    if crossed:
        macros_f += [mac("fbRdFallbackLogWMaxCrossed", f"{max(crossed):.2f}"),
                     mac("fbLogInvDelta", f"{math.log(1 / 0.05):.2f}")]
    (TAB / "aos_fallback_macros.tex").write_text("\n".join(macros_f) + "\n")
    if len(macros_k) > 1:
        (TAB / "aos_kernel_macros.tex").write_text("\n".join(macros_k) + "\n")
    (OUTD / "summary.md").write_text("\n".join(md) + "\n")
    rep = [f"# Cut-point rules (and kernel) for the split-sample lower bound: report (generated {stamp})", "",
           "Exploratory; nothing here is in the paper. Every number below is computed by "
           "scripts/aos_fallback_tables.py from results/aos/fallback/raw (no hand-typed numbers); the LaTeX tables "
           "and macros are results/aos/tables/aos_fallback_*.tex and aos_kernel_*.tex. Protocol (fixed before the "
           "runs) and running log: results/aos/fallback/NOTES.md.", "",
           "## Protocol", "",
           "- Rules, decided on the earlier batches only (each is an admissible fitting rule, so Theorem thm:betting "
           "and Corollary cor:lower hold for it): OWN = the paper's cuts, (1-alpha)- and alpha-quantiles of the fit "
           "values with each point's own kernel term removed, batch skipped if they cross; FALLBACK = OWN, and if "
           "the own-term cuts cross, the plain empirical quantile cuts of the same fitted witness (own term kept), "
           "batch skipped only if those cross too; PLAIN = plain cuts always. The variance guess of the fractions "
           "uses the fit values the cuts came from. Everything else is the paper's and is shared: each (split, "
           "level) is fitted once and the three rules are evaluated on the same fits (paired comparison).",
           "- Real data: RHC (batch-0 standardization) and the three NHANES contrasts, split seeds 1000-1007, "
           "batches (0, 1/4, 1/2, 1), batch-0 median bandwidth, per-level witnesses (package solver), grid "
           "0:0.05:0.75 (child 0:0.05:0.95), delta = 0.05; each split read upward until every rule has failed.",
           "- Synthetic: the 19 designs of Figure 2(a) at n = m = 800 (mixed types n = 400, m = 5n), the stored "
           "replicates regenerated exactly (scripts/aos_baselines_run.py::draw, stored fold bandwidths), the "
           "study's GramQP witness fits warm-started along the full grid, levels j alpha*/8 (j = 0..10), every level "
           "evaluated for every rule; recovery = alpha_lo / alpha* (0 if nothing certified); boundary validity = "
           "the single test at alpha = alpha* (j = 8, a true null).",
           "- Kernel arm (pre-stated in NOTES.md before it ran): Laplace exp(-|x-y|/h) with h = the Gaussian's "
           "median distance on the same points; analytic Laplace with the paper's tau_new at the pilot median.",
           ""]
    rep += md[2:]
    rep += ["## Files", "",
            "- raw: results/aos/fallback/raw/realdata_<kernel>/<contrast>_<seed>.jsonl (one record per split and "
            "level: per rule decision, wealth summary, batch sources and cuts; per batch own and plain cuts), "
            "results/aos/fallback/raw/synth_<kernel>/<design>_n<n>[_zero].jsonl (one record per replicate: per rule "
            "decisions and log max wealth at every level, batch sources; per level and batch the own and plain cut "
            "gaps)",
            "- summaries: results/aos/fallback/summary.md, realdata_summary.csv, synth_summary.csv",
            "- code: src/trimmedmmd/aos_fallback.py; scripts/aos_fallback_realdata.py, aos_fallback_synth.py, "
            "aos_fallback_verify.py, aos_fallback_tables.py", ""]
    (OUTD / "REPORT.md").write_text("\n".join(rep) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

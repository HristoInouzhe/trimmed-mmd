"""AoS revision: tables and macros for the referee baselines (no Monte Carlo work here).

Reads the new replicates (results/aos/baselines/raw/*.jsonl, written by scripts/aos_baselines_run.py) and, for the
methods already in the paper, the stored replicates of the synthetic study (explore/harder_sims/out/<design>.jsonl,
read-only), paired on the same (design, n, rep). Writes

    results/aos/baselines/summary.csv      one row per design x n x method: rejection rates at j = 0, 4, 6, 8,
                                           boundary CP limit and p-value, mean recovered fraction (+ s.e.)
    results/aos/baselines/paired.csv       per design: ramp minus each method (recovery and H_{0,0} power),
                                           mean, paired s.e., win/tie/loss at 2 paired s.e.
    results/aos/tables/aos_baselines_table1.tex     power against H_{0,0}, one row per design
    results/aos/tables/aos_baselines_half.tex       power against H_{0,alpha*/2}
    results/aos/tables/aos_baselines_recovery.tex   mean certified level / alpha* (Figure 2(a) made like-for-like)
    results/aos/tables/aos_baselines_eb.tex         fixed-n empirical Bernstein vs betting (single split, sequential)
    results/aos/tables/aos_baselines_univ.tex       1-d designs: KS lower bound vs analytic vs split-sample, by n
    results/aos/tables/aos_baselines_validity.tex   boundary type-I and P = Q nulls of every new test
    results/aos/tables/aos_baselines_macros.tex     summary macros (prefix \\ab)

    ~/.venvs/llm_book/bin/python -B scripts/aos_baselines_tables.py

Conventions as in the paper: rates rounded half up on their decimal value, one-sided 95% Clopper-Pearson upper
limits, red flag = exact one-sided binomial p-value < 0.01 against delta = 0.05.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
from scipy.stats import beta as _beta, binom  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("AOS_RAW", str(ROOT / "results" / "aos" / "baselines" / "raw")))
OUTD = ROOT / "results" / "aos" / "baselines"
TAB = ROOT / "results" / "aos" / "tables"
STORED = ROOT / "explore" / "harder_sims" / "out"
DELTA = 0.05
REPORT_J = (0, 4, 6, 8)
MIN_REPS = 1           # cells with fewer replicates are skipped (all cells have 200 in the final run)

DESIGNS = [  # (key, label, alpha* as printed) in the order of Table 1, the RHC-shaped design first
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
NULLS = [("N0_D5b_mixed", r"mixed, $m=5n$"), ("N0_D2_d100", r"dense, $d=100$"),
         ("N0_D6_t3", r"$t_3$, $d=5$"), ("N0_D7_narrow", "curves")]
UNIV = [("D8_N1", r"oscillating, $\nu=1$", "0.5"), ("D8_N4", r"oscillating, $\nu=4$", "0.5"),
        ("D8_N16", r"oscillating, $\nu=16$", "0.5"), ("D8_N64", r"oscillating, $\nu=64$", "0.5"),
        ("U1_shift", r"overlap, $d=1$ (new)", "0.2"), ("U1_t3scale", r"$t_3$ scale, $d=1$ (new)", "0.2")]
UNIV_NS = (100, 200, 400, 800)
OLD = ("an_new", "seq_bet", "ramp_bet", "ramp_b2s", "gbm_bet", "logit_bet")        # stored (paper / study)
NEW_BET = ("ramp0", "ramp0c", "svm", "svm05", "svm_hard",
           "gbm_m_hard", "gbm_m_soft", "logit_m_hard", "logit_m_soft",
           "gbm_mt_hard", "gbm_mt_soft", "logit_mt_hard", "logit_mt_soft",
           "gbm_ss_hard", "gbm_ss_soft", "logit_ss_hard", "logit_ss_soft",
           "hplb_bayes_gbm", "hplb_adapt_gbm", "hplb_bayes_logit", "hplb_adapt_logit")
H00 = ("perm_pilot", "perm_med", "mmdagg")
MATCHED = ("gbm_m_hard", "gbm_m_soft", "logit_m_hard", "logit_m_soft",
           "gbm_mt_hard", "gbm_mt_soft", "logit_mt_hard", "logit_mt_soft")
MATCHED_DEF = MATCHED[:4]
HPLB = ("hplb_bayes_gbm", "hplb_adapt_gbm", "hplb_bayes_logit", "hplb_adapt_logit")
LONG = {"an_new": "analytic test", "seq_bet": "betting, ramp (paper)", "ramp_bet": "single-split betting, ramp",
        "ramp_b2s": "single-split empirical Bernstein, ramp", "gbm_bet": "GBM + betting (paper)",
        "logit_bet": "logistic + betting (paper)", "perm_pilot": "permutation MMD, pilot bandwidth",
        "perm_med": "permutation MMD, pooled median bandwidth", "mmdagg": "MMDAgg",
        "ramp0": "alpha = 0 ramp (untrimmed)", "ramp0c": "alpha = 0 witness, level-alpha cut points",
        "svm": "nu-SVM normal (CV nu), ramp", "svm05": "nu-SVM normal (nu = 0.5), ramp",
        "svm_hard": "nu-SVM labels (CV nu)",
        "gbm_m_hard": "GBM matched, labels", "gbm_m_soft": "GBM matched, probabilities",
        "logit_m_hard": "logistic matched, labels", "logit_m_soft": "logistic matched, probabilities",
        "gbm_mt_hard": "GBM matched + tuned, labels", "gbm_mt_soft": "GBM matched + tuned, probabilities",
        "logit_mt_hard": "logistic matched + tuned, labels",
        "logit_mt_soft": "logistic matched + tuned, probabilities",
        "gbm_ss_hard": "GBM single split, labels (recomputed)", "gbm_ss_soft": "GBM single split, probabilities",
        "logit_ss_hard": "logistic single split, labels (recomputed)",
        "logit_ss_soft": "logistic single split, probabilities",
        "hplb_bayes_gbm": "HPLB bayes, GBM", "hplb_adapt_gbm": "HPLB adapt, GBM",
        "hplb_bayes_logit": "HPLB bayes, logistic", "hplb_adapt_logit": "HPLB adapt, logistic",
        "ks": "KS minus DKW radii"}
TEXNAME = {"perm_pilot": "permutation MMD", "perm_med": "permutation MMD, median bandwidth", "mmdagg": "MMDAgg",
           "ramp0": r"$\alpha=0$ ramp", "ramp0c": r"$\alpha=0$ witness, level cuts",
           "svm": r"$\nu$-SVM ramp (CV $\nu$)", "svm05": r"$\nu$-SVM ramp ($\nu=0.5$)",
           "svm_hard": r"$\nu$-SVM labels", "gbm_m_hard": "GBM matched, labels",
           "gbm_m_soft": "GBM matched, prob.", "logit_m_hard": "logistic matched, labels",
           "logit_m_soft": "logistic matched, prob.", "gbm_mt_hard": "GBM tuned, labels",
           "gbm_mt_soft": "GBM tuned, prob.", "logit_mt_hard": "logistic tuned, labels",
           "logit_mt_soft": "logistic tuned, prob.", "gbm_ss_hard": "GBM single split, labels",
           "gbm_ss_soft": "GBM single split, prob.", "logit_ss_hard": "logistic single split, labels",
           "logit_ss_soft": "logistic single split, prob.", "hplb_bayes_gbm": "HPLB bayes, GBM",
           "hplb_adapt_gbm": "HPLB adapt, GBM", "hplb_bayes_logit": "HPLB bayes, logistic",
           "hplb_adapt_logit": "HPLB adapt, logistic", "ks": "KS bound"}


MACNAME = {"an_new": "Analytic", "seq_bet": "Ramp", "ramp_bet": "SingleBet", "ramp_b2s": "Eb", "gbm_bet": "GbmPaper",
           "logit_bet": "LogitPaper", "ramp0": "RampZero", "ramp0c": "RampZeroCut", "svm": "Svm", "svm05": "SvmHalf",
           "svm_hard": "SvmLabels", "perm_pilot": "Perm", "perm_med": "PermMed", "mmdagg": "Agg", "ks": "Ks"}


def mname(mth):
    """Letters-only macro fragment for a method id."""
    if mth in MACNAME:
        return MACNAME[mth]
    words = {"m": "Matched", "mt": "Tuned", "ss": "Single", "hard": "Labels", "soft": "Prob"}
    out = "".join(words.get(p, p.capitalize()) for p in mth.split("_"))
    digits = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven",
              "8": "Eight", "9": "Nine"}
    out = "".join(digits.get(ch, ch) for ch in out)
    assert out.isalpha(), out
    return out


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

BAD = {}


def jsonl(path):
    """Valid records of a jsonl file, deduplicated by (n, rep); unparsable lines are counted in BAD and reported."""
    out, seen = [], set()
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                BAD[path.name] = BAD.get(path.name, 0) + 1
                continue
            k = (r.get("n"), r.get("rep"))
            if k not in seen:
                seen.add(k)
                out.append(r)
    return out


def stored(key):
    return {(r["n"], r["rep"]): r for r in jsonl(STORED / f"{key}.jsonl")}


def frac(j, levels, ast):
    return 0.0 if j < 0 or ast == 0 else levels[j] / ast


def cell_main(key, n, old):
    """Per-replicate rows (decisions at REPORT_J, recovered fraction) for every method in the main run."""
    new = {r["rep"]: r for r in jsonl(RAW / f"{key}_n{n}.jsonl")}
    reps = sorted(new)
    D = {}
    for rep in reps:
        a, s = new[rep], old[(n, rep)]
        lv, ast = a["levels"], a["alpha_star"]
        for mth in OLD:
            D.setdefault(mth, []).append((list(s["dec"][mth]), frac(s["jlo"][mth], lv, ast)))
        for mth in NEW_BET + (("ks",) if "ks" in a["dec_all"] else ()):
            D.setdefault(mth, []).append(([a["dec_all"][mth][k] for k in REPORT_J], frac(a["jlo"][mth], lv, ast)))
        for mth in H00:
            D.setdefault(mth, []).append(([a["dec_all"][mth][0], None, None, None], None))
    return reps, D, new


# ---------------------------------------------------------------------------
# statistics and formatting
# ---------------------------------------------------------------------------

def cp_upper(k, R, conf=0.95):
    return 1.0 if k >= R else float(_beta.ppf(conf, k + 1, R - k))


def binom_p(k, R, p=DELTA):
    return float(binom.sf(k - 1, R, p))


def r2(x):
    """Two decimals, half up on the decimal value."""
    return str(Decimal(repr(float(x))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def rate(k, R):
    return str((Decimal(k) / Decimal(R)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def mac(name, val):
    return f"\\newcommand{{\\{name}}}{{{val}}}"


def summarize(D, R):
    S = {}
    for mth, rows in D.items():
        s = {"R": R}
        for i, j in enumerate(REPORT_J):
            if rows[0][0][i] is None:
                continue
            k = sum(int(r[0][i]) for r in rows)
            s[f"k{j}"], s[f"pow{j}"] = k, k / R
        if "k8" in s:
            s.update({"typeI": s["pow8"], "typeI_cp": cp_upper(s["k8"], R), "typeI_p": binom_p(s["k8"], R)})
        if rows[0][1] is not None:
            rec = np.array([r[1] for r in rows])
            s["rec"] = float(rec.mean())
            s["rec_se"] = float(rec.std(ddof=1) / math.sqrt(R)) if R > 1 else float("nan")
        S[mth] = s
    return S


def wtl(d):
    """Paired difference vector -> (mean, se, 'win'|'tie'|'loss') at two paired standard errors."""
    d = np.asarray(d, float)
    mu = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(d.size)) if d.size > 1 else 0.0
    if mu > 0 and mu > 2 * se:
        return mu, se, "win"
    if mu < 0 and -mu > 2 * se:
        return mu, se, "loss"
    return mu, se, "tie"


# ---------------------------------------------------------------------------
# table writer
# ---------------------------------------------------------------------------

def write_table(path, note, designs, rows, cols, field, groups=None, extra_rule=None):
    """cols: list of (method, (header line 1, header line 2)); method 'ALPHA' prints alpha*. Bold: the largest
    value of the row over the method columns (after rounding)."""
    out = [f"% Generated by scripts/aos_baselines_tables.py -- do not edit. {note}",
           r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule"]
    if groups:
        out.append(" & " + " & ".join(g for g, _ in groups) + r" \\")
        pos, cm = 2, []
        for g, span in groups:
            if g:
                cm.append(rf"\cmidrule(lr){{{pos}-{pos + span - 1}}}")
            pos += span
        out.append("".join(cm))
    out.append("design & " + " & ".join(c[1][0] for c in cols) + r" \\")
    if any(c[1][1] for c in cols):
        out.append(" & " + " & ".join(c[1][1] for c in cols) + r" \\")
    out.append(r"\midrule")
    for key, label, ast in designs:
        vals = {c[0]: r2(rows[key][c[0]][field]) for c in cols if c[0] != "ALPHA"}
        best = max(vals.values(), key=Decimal)
        cells = [label]
        for c in cols:
            if c[0] == "ALPHA":
                cells.append(f"${ast}$")
            else:
                s = vals[c[0]]
                cells.append(rf"$\mathbf{{{s}}}$" if s == best and Decimal(best) > 0 else f"${s}$")
        out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(out) + "\n")
    print("wrote", path)


# ---------------------------------------------------------------------------
# sample size for power 0.8 against H_{0,0} (the paper's Table 1 rule)
# ---------------------------------------------------------------------------

def sig2(x):
    """Two significant figures, half up (jmlr/tools/make_harder_tables.py)."""
    q = Decimal(10) ** (math.floor(math.log10(x)) - 1)
    return int((Decimal(repr(x)) / q).quantize(Decimal(1), rounding=ROUND_HALF_UP) * q)


def tex_n(v):
    return f"{v:,}".replace(",", "{,}")


def n80(curve):
    """First size with power >= 0.8 and the size before it, log-linear interpolation, two significant figures
    (make_harder_tables.n80). Returns (tex, value): value = n for '<= n', None for '> n'."""
    c = sorted(curve.items())
    for i, (n, p) in enumerate(c):
        if p >= 0.8:
            if i == 0:
                return f"{{\\le}}{tex_n(n)}", n
            n0, p0 = c[i - 1]
            v = min(max(sig2(math.exp(math.log(n0) + (0.8 - p0) / (p - p0) * math.log(n / n0))), n0), n)
            return tex_n(v), v
    return f"{{>}}{tex_n(c[-1][0])}", None


def curve_stored(key, mth):
    """The paper's curve0: power of a stored method at every grid size, then the alpha = 0 extension."""
    pts = {}
    for r in jsonl(STORED / f"{key}.jsonl"):
        pts.setdefault(r["n"], []).append(r["dec"][mth][0])
    out = {n: float(np.mean(v)) for n, v in pts.items()}
    ext = {}
    for r in jsonl(STORED / f"{key}_n0ext.jsonl"):
        if r["n"] not in out:
            ext.setdefault(r["n"], []).append(r["dec0"][mth])
    out.update({n: float(np.mean(v)) for n, v in ext.items()})
    return out


def curve_new(key, mth):
    pts = {}
    for f in list(RAW.glob(f"{key}_n*_h00.jsonl")) + list(RAW.glob(f"{key}_n*_h00x.jsonl")) + [
            f for f in RAW.glob(f"{key}_n*.jsonl") if f.stem.split("_n")[-1].isdigit()]:
        for r in jsonl(f):
            pts.setdefault(r["n"], []).append(r["dec_all"][mth][0])
    return {n: float(np.mean(v)) for n, v in pts.items()}, {n: len(v) for n, v in pts.items()}


def paper_n80():
    """Cells of the paper's Table 1 (aos/tables/harder_n80_v2.tex, read-only), for the reproduction check."""
    f = ROOT / "aos" / "tables" / "harder_n80_v2.tex"
    rows = []
    for line in f.read_text().splitlines():
        if line.count("&") == 5 and "\\\\" in line and not line.startswith("design") and not line.startswith(" &"):
            cells = [c.strip().rstrip("\\").strip() for c in line.split("&")]
            rows.append([c.replace("$", "").replace("\\mathbf{", "").replace("}", "").replace("{", "")
                         .replace("\\le", "<=").replace(">", ">").strip() for c in cells[2:]])
    return rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    TAB.mkdir(parents=True, exist_ok=True)
    rows, per, nsize, missing = {}, {}, {}, []
    NUS = []                     # nu chosen by CV on the half-1 fit block, all designs and replicates
    repro = {}
    allkeys = [d[0] for d in DESIGNS] + [k for k, _ in NULLS]
    for key in allkeys:
        old = stored(key)
        n = 800 if any(nn == 800 for nn, _ in old) else max(nn for nn, _ in old)
        reps, D, new = cell_main(key, n, old)
        if len(reps) < MIN_REPS:
            missing.append(key)
            continue
        nsize[key] = (n, new[reps[0]]["m"], len(reps))
        rows[key] = summarize(D, len(reps))
        per[key] = D
        if key in dict((d[0], 1) for d in DESIGNS):
            NUS.extend(r["diag"]["svm_nu"][1] for r in new.values())
        for r in new.values():
            for f, v in r["repro"].items():
                repro.setdefault(f, [0, 0])
                repro[f][0] += int(v)
                repro[f][1] += 1
    if missing:
        print("missing designs (not in raw/ yet):", missing)
    if BAD:
        print("WARNING unparsable lines skipped:", BAD)
    designs = [d for d in DESIGNS if d[0] in rows]
    keys = [d[0] for d in designs]
    nulls = [k for k, _ in NULLS if k in rows]
    nd = len(designs)

    # ---------------- summary.csv ----------------
    fields = ["design", "n", "m", "R", "method", "label", "pow0", "pow4", "pow6", "typeI", "k8", "typeI_cp", "typeI_p",
              "rec", "rec_se"]
    with open(OUTD / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for key in rows:
            n, m, R = nsize[key]
            for mth, s in rows[key].items():
                w.writerow({"design": key, "n": n, "m": m, "R": R, "method": mth, "label": LONG[mth],
                            **{f: s.get(f, "") for f in fields[6:]}})

    # ---------------- paired comparisons with the ramp ----------------
    W = {}
    with open(OUTD / "paired.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["design", "method", "quantity", "R", "ramp_minus_method", "paired_se", "result"])
        for key in keys:
            D = per[key]
            ramp_rec = np.array([r[1] for r in D["seq_bet"]])
            ramp_p0 = np.array([r[0][0] for r in D["seq_bet"]], float)
            for mth in D:
                if mth == "seq_bet":
                    continue
                p0 = np.array([r[0][0] for r in D[mth]], float)
                res = {"pow0": wtl(ramp_p0 - p0)}
                if D[mth][0][1] is not None:
                    res["rec"] = wtl(ramp_rec - np.array([r[1] for r in D[mth]]))
                    res["pow4"] = wtl(np.array([r[0][1] for r in D["seq_bet"]], float)
                                      - np.array([r[0][1] for r in D[mth]], float))
                for q, (mu, se, out) in res.items():
                    W[(key, mth, q)] = out
                    w.writerow([key, mth, q, p0.size, f"{mu:.4f}", f"{se:.4f}", out])

    def col(mth, field):
        return np.array([rows[k][mth][field] for k in keys])

    def count_wtl(mth, q):
        o = [W[(k, mth, q)] for k in keys]
        return f"{o.count('win')}/{o.count('tie')}/{o.count('loss')}"

    # ---------------- LaTeX tables ----------------
    A = ("ALPHA", (r"$\alpha^\star$", ""))
    kern = [("an_new", ("analytic", "")), ("perm_pilot", ("perm.", "MMD")), ("mmdagg", ("MMD-", "Agg")),
            ("seq_bet", ("ramp", "")), ("svm", (r"$\nu$-SVM", "ramp"))]
    clf = []
    for c in ("gbm", "logit"):
        clf += [(f"{c}_bet", ("paper", "")), (f"{c}_m_hard", ("labels", "")), (f"{c}_m_soft", ("prob.", "")),
                (f"{c}_mt_hard", ("tuned", "labels")), (f"hplb_adapt_{c}", ("HPLB", ""))]
    groups = [("", 1), (r"\multicolumn{5}{c}{kernel}", 5), (r"\multicolumn{5}{c}{GBM}", 5),
              (r"\multicolumn{5}{c}{logistic}", 5)]
    write_table(TAB / "aos_baselines_table1.tex",
                "Power against H_{0,0}, n = m = 800 (mixed types n = 400, m = 5n).", designs, rows,
                [A] + kern + clf, "pow0", groups)
    kern_r = [("an_new", ("analytic", "")), ("seq_bet", ("ramp", "")), ("ramp0", (r"$\alpha{=}0$", "ramp")),
              ("ramp0c", (r"$\alpha{=}0$", "witness")), ("svm", (r"$\nu$-SVM", "ramp"))]
    groups_r = groups
    write_table(TAB / "aos_baselines_recovery.tex",
                "Mean certified level / alpha*, n = m = 800 (mixed types n = 400, m = 5n).", designs, rows,
                [A] + kern_r + clf, "rec", groups_r)
    write_table(TAB / "aos_baselines_half.tex",
                "Power against H_{0,alpha*/2}, n = m = 800 (mixed types n = 400, m = 5n).", designs, rows,
                [A] + kern_r + clf, "pow4", groups_r)
    eb_cols = [A, ("ramp_b2s", ("EB", "")), ("ramp_bet", ("betting", "")), ("seq_bet", ("betting,", "seq.")),
               ("ramp_b2s", ("EB", "")), ("ramp_bet", ("betting", "")), ("seq_bet", ("betting,", "seq."))]
    # the EB table has two blocks of the same methods (power, recovery): written by hand below
    out = ["% Generated by scripts/aos_baselines_tables.py -- do not edit. Fixed-n empirical Bernstein (single split) "
           "vs betting (single split; sequential = the paper's ramp), n = m = 800 (mixed n = 400, m = 5n).",
           r"\begin{tabular}{lccccccc}", r"\toprule",
           r" & & \multicolumn{3}{c}{power against $H_{0,0}$} & \multicolumn{3}{c}{certified level$/\alpha^\star$} \\",
           r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
           r"design & $\alpha^\star$ & EB & betting & betting, seq. & EB & betting & betting, seq. \\", r"\midrule"]
    for key, label, ast in designs:
        p = [r2(rows[key][mm]["pow0"]) for mm in ("ramp_b2s", "ramp_bet", "seq_bet")]
        q = [r2(rows[key][mm]["rec"]) for mm in ("ramp_b2s", "ramp_bet", "seq_bet")]
        bp, bq = max(p, key=Decimal), max(q, key=Decimal)
        cells = [label, f"${ast}$"] + [rf"$\mathbf{{{x}}}$" if x == bp and Decimal(bp) > 0 else f"${x}$" for x in p] \
            + [rf"$\mathbf{{{x}}}$" if x == bq and Decimal(bq) > 0 else f"${x}$" for x in q]
        out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "aos_baselines_eb.tex").write_text("\n".join(out) + "\n")
    print("wrote", TAB / "aos_baselines_eb.tex")
    del eb_cols

    # ---------------- validity table ----------------
    vt = ["% Generated by scripts/aos_baselines_tables.py -- do not edit. Validity of the new tests.",
          r"\begin{tabular}{lcccc" + "c" * len(nulls) + "}", r"\toprule",
          r" & \multicolumn{4}{c}{boundary $H_{0,\alpha^\star}$, " + str(nd) + r" designs} & \multicolumn{"
          + str(len(nulls)) + r"}{c}{$P=Q$: rejection rate of $H_{0,0}$} \\",
          r"\cmidrule(lr){2-5}\cmidrule(lr){6-" + str(5 + len(nulls)) + "}",
          "test & max rate & max $k$ & CP upper & flags & " + " & ".join(dict(NULLS)[k] for k in nulls) + r" \\",
          r"\midrule"]
    for mth in list(H00) + list(NEW_BET):
        if mth in H00:
            b = ["--", "--", "--", "--"]
        else:
            k8 = [rows[k][mth]["k8"] for k in keys]
            R = [rows[k][mth]["R"] for k in keys]
            i = int(np.argmax(k8))
            flags = sum(binom_p(a, r) < 0.01 for a, r in zip(k8, R))
            b = [f"${rate(k8[i], R[i])}$", f"${k8[i]}/{R[i]}$", f"${r2(cp_upper(k8[i], R[i]))}$", f"${flags}$"]
        nv = [f"${rate(rows[k][mth]['k0'], rows[k][mth]['R'])}$" for k in nulls]
        vt.append(TEXNAME[mth] + " & " + " & ".join(b + nv) + r" \\")
    vt += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "aos_baselines_validity.tex").write_text("\n".join(vt) + "\n")
    print("wrote", TAB / "aos_baselines_validity.tex")

    # ---------------- univariate (KS) ----------------
    uni = {}          # (key, n) -> {method: (mean recovered fraction, power at 0, R)}
    for key, _, _ in UNIV:
        for n in UNIV_NS:
            if key.startswith("U1"):
                recs = jsonl(RAW / f"{key}_n{n}_univ.jsonl")
                if not recs:
                    continue
                res = {}
                for mth in ("ks", "an_new", "seq_bet", "ramp_bet", "ramp_b2s"):
                    fr = [frac(r["jlo"][mth], r["levels"], r["alpha_star"]) for r in recs]
                    p0 = [r["dec"][mth][0] for r in recs]
                    k8 = sum(r["dec"][mth][3] for r in recs)
                    res[mth] = (float(np.mean(fr)), float(np.mean(p0)), len(recs), k8)
            else:
                old = stored(key)
                if n == 800:
                    if key not in per:
                        continue
                    ksr = jsonl(RAW / f"{key}_n{n}.jsonl")
                else:
                    ksr = jsonl(RAW / f"{key}_n{n}_ks.jsonl")
                if not ksr:
                    continue
                res = {"ks": (float(np.mean([frac(r["jlo"]["ks"], r["levels"], r["alpha_star"]) for r in ksr])),
                              float(np.mean([r["dec_all"]["ks"][0] for r in ksr])), len(ksr),
                              sum(r["dec_all"]["ks"][8] for r in ksr))}
                reps = [r["rep"] for r in ksr]
                for mth in ("an_new", "seq_bet", "ramp_bet", "ramp_b2s"):
                    ss = [old[(n, rp)] for rp in reps]
                    fr = [frac(s["jlo"][mth], s["levels"], s["alpha_star"]) for s in ss]
                    res[mth] = (float(np.mean(fr)), float(np.mean([s["dec"][mth][0] for s in ss])), len(ss),
                                sum(s["dec"][mth][3] for s in ss))
            uni[(key, n)] = res
    with open(OUTD / "univariate.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["design", "n", "method", "R", "rec", "pow0", "k8"])
        for (key, n), res in uni.items():
            for mth, (rc, p0, R, k8) in res.items():
                w.writerow([key, n, mth, R, f"{rc:.4f}", f"{p0:.4f}", k8])
    um = ("ks", "an_new", "seq_bet", "ramp_bet", "ramp_b2s")
    ut = ["% Generated by scripts/aos_baselines_tables.py -- do not edit. 1-d designs: mean certified level / alpha* "
          "by n = m.", r"\begin{tabular}{lcc" + "c" * len(um) + "}", r"\toprule",
          r"design & $\alpha^\star$ & $n$ & KS bound & analytic & betting, seq. & betting & EB \\", r"\midrule"]
    for key, label, ast in UNIV:
        first = True
        for n in UNIV_NS:
            if (key, n) not in uni:
                continue
            vals = [r2(uni[(key, n)][mm][0]) for mm in um]
            best = max(vals, key=Decimal)
            ut.append(" & ".join([label if first else "", f"${ast}$" if first else "", f"${n}$"]
                                 + [rf"$\mathbf{{{v}}}$" if v == best and Decimal(best) > 0 else f"${v}$"
                                    for v in vals]) + r" \\")
            first = False
        if not first and key != UNIV[-1][0]:
            ut.append(r"\addlinespace")
    ut += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "aos_baselines_univ.tex").write_text("\n".join(ut) + "\n")
    print("wrote", TAB / "aos_baselines_univ.tex")

    # ---------------- sample size for power 0.8 against H_{0,0} ----------------
    N80M = [("an_new", "s"), ("seq_bet", "s"), ("gbm_bet", "s"), ("logit_bet", "s"), ("perm_pilot", "n"),
            ("mmdagg", "n")]
    n80tab, n80sizes = {}, {}
    for key in keys:
        for mth, src in N80M:
            if src == "s":
                n80tab[(key, mth)] = n80(curve_stored(key, mth))
            else:
                cv, cnt = curve_new(key, mth)
                n80tab[(key, mth)] = n80(cv)
                n80sizes[(key, mth)] = cnt

    def n80val(key, mth):
        v = n80tab[(key, mth)][1]
        return math.inf if v is None else v

    out = ["% Generated by scripts/aos_baselines_tables.py -- do not edit. Sample size n = m (mixed: n, m = 5n) for "
           "power 0.8 against H_{0,0}: the paper's rule (first size with power >= 0.8, log-linear interpolation, two "
           "significant digits). Analytic, ramp, GBM, logistic: stored study incl. its alpha = 0 extension (as Table "
           "1); permutation MMD and MMDAgg: this run (sizes 100-800, and 1600 where run).",
           r"\begin{tabular}{lccccccc}", r"\toprule",
           r"design & $\alpha^\star$ & analytic & betting, & GBM & logistic & perm. & MMDAgg \\",
           r" & & & ramp & + betting & + betting & MMD & \\", r"\midrule"]
    for key, label, ast in designs:
        vals = [n80val(key, m_) for m_, _ in N80M]
        # Bold: the smallest among the four tests that certify a level of d_TV; italic: the smaller of the two
        # tests of P = Q (permutation MMD, MMDAgg), which certify no level (author, 2026-11-21).
        best_c, best_d = min(vals[:4]), min(vals[4:])
        cells = [label, f"${ast}$"]
        for i, ((m_, _), v) in enumerate(zip(N80M, vals)):
            t = n80tab[(key, m_)][0]
            if i < 4 and v == best_c and best_c < math.inf:
                cells.append(rf"$\mathbf{{{t}}}$")
            elif i >= 4 and v == best_d and best_d < math.inf:
                cells.append(rf"$\mathit{{{t}}}$")
            else:
                cells.append(f"${t}$")
        out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "aos_baselines_n80.tex").write_text("\n".join(out) + "\n")
    print("wrote", TAB / "aos_baselines_n80.tex")
    # reproduction of the paper's Table 1 from the stored replicates
    paper = paper_n80()
    rep_ok = rep_all = 0
    if len(paper) == len(DESIGNS):
        for (key, _, _), row in zip(DESIGNS, paper):
            if key not in keys:
                continue
            for m_, cell in zip(("an_new", "seq_bet", "gbm_bet", "logit_bet"), row):
                mine = n80tab[(key, m_)][0].replace("{", "").replace("}", "").replace("\\le", "<=")
                rep_all += 1
                rep_ok += int(mine == cell)

    # ---------------- macros ----------------
    L = ["% Generated by scripts/aos_baselines_tables.py -- do not edit."]
    L.append(mac("abNeightyReproduced", f"{rep_ok}/{rep_all}"))
    if NUS:
        L += [mac("abSvmNuMed", r2(np.median(NUS))), mac("abSvmNuLoQ", r2(np.quantile(NUS, 0.1))),
              mac("abSvmNuHiQ", r2(np.quantile(NUS, 0.9)))]
    allv = {k: min(n80val(k, m_) for m_, _ in N80M) for k in keys}
    for m_, name in (("perm_pilot", "Perm"), ("mmdagg", "Agg"), ("seq_bet", "Ramp")):
        L.append(mac(f"abNeighty{name}Best", sum(n80val(k, m_) == allv[k] for k in keys)))
    ratios = [n80val(k, "seq_bet") / n80val(k, "perm_pilot") for k in keys
              if n80val(k, "seq_bet") < math.inf and n80val(k, "perm_pilot") < math.inf
              and n80tab[(k, "perm_pilot")][0].find("le") < 0]
    if ratios:
        L += [mac("abNeightyRampOverPermMed", r2(np.median(ratios))), mac("abNeightyRampOverPermN", len(ratios)),
              mac("abNeightyRampOverPermLo", r2(min(ratios))), mac("abNeightyRampOverPermHi", r2(max(ratios)))]
    L += [mac("abNeightyPermLeHundred", sum(n80tab[(k, "perm_pilot")][0].startswith("{\\le}") for k in keys)),
          mac("abNeightyPermNotReached", sum(n80tab[(k, "perm_pilot")][1] is None for k in keys)),
          mac("abNeightyAggNotReached", sum(n80tab[(k, "mmdagg")][1] is None for k in keys))]
    Rs = sorted({rows[k]["seq_bet"]["R"] for k in keys})
    L += [mac("abNdesigns", nd), mac("abReps", Rs[0] if len(Rs) == 1 else f"{Rs[0]}--{Rs[-1]}"),
          mac("abNnulls", len(nulls)), mac("abPermB", 999), mac("abAggL", 10), mac("abAggB", 500),
          mac("abHplbS", 2000)]
    ramp = col("seq_bet", "rec")

    def best_of(ms, field="rec"):
        return np.array([max(rows[k][m][field] for m in ms) for k in keys])

    paper_clf = best_of(("gbm_bet", "logit_bet"))
    m_def, m_all, hp = best_of(MATCHED_DEF), best_of(MATCHED), best_of(HPLB)
    clf_all = best_of(MATCHED + HPLB + ("gbm_bet", "logit_bet", "gbm_ss_soft", "logit_ss_soft"))
    L += [mac("abRampAbovePaperClf", int(np.sum(ramp > paper_clf))),
          mac("abRampAboveBestMatched", int(np.sum(ramp > m_def))),
          mac("abRampAboveBestMatchedTuned", int(np.sum(ramp > m_all))),
          mac("abBestMatchedTunedAboveRamp", int(np.sum(m_all > ramp))),
          mac("abRampAboveBestHplb", int(np.sum(ramp > hp))),
          mac("abRampAboveEveryClf", int(np.sum(ramp > clf_all))),
          mac("abEveryClfAboveRamp", int(np.sum(clf_all > ramp))),
          mac("abRampAboveSvm", int(np.sum(ramp > col("svm", "rec")))),
          mac("abSvmAboveRamp", int(np.sum(col("svm", "rec") > ramp))),
          mac("abRampAboveRampZero", int(np.sum(ramp > col("ramp0", "rec")))),
          mac("abRampZeroAboveRamp", int(np.sum(col("ramp0", "rec") > ramp))),
          mac("abRampAboveRampZeroCut", int(np.sum(ramp > col("ramp0c", "rec")))),
          mac("abRampZeroCutAboveRamp", int(np.sum(col("ramp0c", "rec") > ramp)))]
    for mth in ("svm", "svm_hard", "ramp0", "ramp0c") + MATCHED + HPLB + ("gbm_bet", "logit_bet"):
        name = mname(mth)
        L.append(mac(f"abWtlRec{name}", count_wtl(mth, "rec")))
        L.append(mac(f"abWtlPow{name}", count_wtl(mth, "pow0")))
    for mth in ("perm_pilot", "mmdagg", "an_new"):
        name = mname(mth)
        L.append(mac(f"abWtlPow{name}", count_wtl(mth, "pow0")))
    for mth in ("svm", "ramp0", "ramp0c"):
        name = mname(mth)
        L.append(mac(f"abWtlHalf{name}", count_wtl(mth, "pow4")))
    for mth in OLD + NEW_BET:
        name = mname(mth)
        v = col(mth, "rec")
        L += [mac(f"abRecMed{name}", r2(np.median(v))), mac(f"abRecMax{name}", r2(v.max()))]
    lab = {d[0]: d[1] + (f" ($\\alpha^\\star={d[2]}$)" if d[2] not in ("0.2",) else "") for d in designs}
    above = [k for k, a, b in zip(keys, m_all, ramp) if a > b]
    L += [mac("abDesignsBestMatchedTunedAboveRamp", "; ".join(lab[k] for k in above) if above else "none")]
    for nm, ref in (("abDesignsRampAboveBestMatched", m_def), ("abDesignsRampAboveBestMatchedTuned", m_all)):
        below = [k for k, a, b in zip(keys, ref, ramp) if b > a]
        L.append(mac(nm, "; ".join(lab[k] for k in below) if below else "none"))
    for mth in MATCHED + HPLB + ("svm", "ramp0c"):
        L.append(mac(f"abNAboveRamp{mname(mth)}", int(np.sum(col(mth, "rec") > ramp))))
    L += [mac("abRecMedBestMatched", r2(np.median(m_def))), mac("abRecMedBestMatchedTuned", r2(np.median(m_all))),
          mac("abRecMedBestHplb", r2(np.median(hp))), mac("abRecMedBestPaperClf", r2(np.median(paper_clf)))]
    # power against H_{0,0}
    for mth in ("perm_pilot", "perm_med", "mmdagg", "seq_bet", "svm", "an_new"):
        name = mname(mth)
        v = col(mth, "pow0")
        L += [mac(f"abPow{name}Lo", r2(v.min())), mac(f"abPow{name}Hi", r2(v.max())),
              mac(f"abPow{name}Med", r2(np.median(v)))]
    rp = col("seq_bet", "pow0")
    for mth, name in (("perm_pilot", "Perm"), ("mmdagg", "Agg"), ("an_new", "Analytic")):
        v = col(mth, "pow0")
        L += [mac(f"ab{name}AboveRampPow", int(np.sum(v > rp + 1e-12))),
              mac(f"abRampAbove{name}Pow", int(np.sum(rp > v + 1e-12)))]
    L.append(mac("abAggAbovePermPow", int(np.sum(col("mmdagg", "pow0") > col("perm_pilot", "pow0") + 1e-12))))
    L.append(mac("abPermAboveAggPow", int(np.sum(col("perm_pilot", "pow0") > col("mmdagg", "pow0") + 1e-12))))
    bm0 = best_of(MATCHED, "pow0")
    L += [mac("abRampAboveBestMatchedPow", int(np.sum(rp > bm0 + 1e-12))),
          mac("abBestMatchedAboveRampPow", int(np.sum(bm0 > rp + 1e-12)))]
    # H_{0, alpha*/2}
    r4 = col("seq_bet", "pow4")
    for mth, name in (("ramp0", "RampZero"), ("ramp0c", "RampZeroCut"), ("svm", "Svm")):
        v = col(mth, "pow4")
        L += [mac(f"abHalfRampAbove{name}", int(np.sum(r4 > v + 1e-12))),
              mac(f"abHalf{name}AboveRamp", int(np.sum(v > r4 + 1e-12)))]
    L += [mac("abHalfPowRampMed", r2(np.median(r4))), mac("abHalfPowRampZeroMed", r2(np.median(col("ramp0", "pow4")))),
          mac("abHalfPowSvmMed", r2(np.median(col("svm", "pow4"))))]
    # fixed-n empirical Bernstein vs betting
    eb, sb = col("ramp_b2s", "rec"), col("ramp_bet", "rec")
    L += [mac("abEbRecMed", r2(np.median(eb))), mac("abSingleBetRecMed", r2(np.median(sb))),
          mac("abSingleBetAboveEb", int(np.sum(sb > eb))), mac("abEbAboveSingleBet", int(np.sum(eb > sb))),
          mac("abSeqAboveEb", int(np.sum(ramp > eb))),
          mac("abEbPowMed", r2(np.median(col("ramp_b2s", "pow0")))),
          mac("abSingleBetPowMed", r2(np.median(col("ramp_bet", "pow0"))))]
    # validity of the new tests
    newk = [(rows[k][m]["k8"], rows[k][m]["R"], m, k) for k in keys for m in NEW_BET]
    i = int(np.argmax([a[0] for a in newk]))
    L += [mac("abBoundaryCells", len(newk)),
          mac("abBoundaryFlags", sum(binom_p(a, r) < 0.01 for a, r, _, _ in newk)),
          mac("abBoundaryMax", rate(newk[i][0], newk[i][1])), mac("abBoundaryMaxK", newk[i][0]),
          mac("abBoundaryMaxMethod", TEXNAME[newk[i][2]]), mac("abBoundaryMaxDesign", dict((d[0], d[1]) for d in designs)[newk[i][3]]),
          mac("abBoundaryMaxCP", r2(cp_upper(newk[i][0], newk[i][1])))]
    if nulls:
        nk = [(rows[k][m]["k0"], rows[k][m]["R"], m, k) for k in nulls for m in list(H00) + list(NEW_BET)]
        j = int(np.argmax([a[0] for a in nk]))
        L += [mac("abNullCells", len(nk)), mac("abNullMax", rate(nk[j][0], nk[j][1])), mac("abNullMaxK", nk[j][0]),
              mac("abNullMaxMethod", TEXNAME[nk[j][2]]), mac("abNullMaxDesign", dict(NULLS)[nk[j][3]]),
              mac("abNullMaxP", f"{binom_p(nk[j][0], nk[j][1]):.3f}"),
              mac("abNullFlags", sum(binom_p(a, r) < 0.01 for a, r, _, _ in nk))]
        for mth, name in (("perm_pilot", "Perm"), ("mmdagg", "Agg")):
            v = [rows[k][mth]["k0"] for k in nulls]
            L += [mac(f"abNull{name}Max", rate(max(v), rows[nulls[0]][mth]["R"]))]
    # univariate
    if uni:
        cells = list(uni)
        ks_above = sum(uni[c]["ks"][0] > max(uni[c]["seq_bet"][0], uni[c]["an_new"][0]) for c in cells)
        L += [mac("abUnivCells", len(cells)), mac("abUnivKsAboveKernel", ks_above),
              mac("abUnivKernelAboveKs", sum(max(uni[c]["seq_bet"][0], uni[c]["an_new"][0]) > uni[c]["ks"][0]
                                              for c in cells))]
        for key, name in (("U1_shift", "Shift"), ("U1_t3scale", "Scale"), ("D8_N1", "OscOne"),
                          ("D8_N4", "OscFour")):
            if (key, 800) in uni:
                L += [mac(f"abUniv{name}Ks", r2(uni[(key, 800)]["ks"][0])),
                      mac(f"abUniv{name}Ramp", r2(uni[(key, 800)]["seq_bet"][0])),
                      mac(f"abUniv{name}Analytic", r2(uni[(key, 800)]["an_new"][0]))]
        ksk = [(uni[c][m][3], uni[c][m][2]) for c in cells for m in ("ks",)]
        L.append(mac("abUnivKsBoundaryMaxK", max(a for a, _ in ksk)))
    for f, (a, b) in sorted(repro.items()):
        rn = {"sigma": "Sigma", "sigma_close": "SigmaClose", "seq0": "SeqZero", "hard_gbm": "HardGbm",
              "diff_gbm": "DiffGbm", "hard_logit": "HardLogit", "diff_logit": "DiffLogit"}[f]
        L.append(mac("abRepro" + rn, f"{a}/{b}"))
    pr = OUTD / "probes.json"
    if pr.exists():
        P = json.loads(pr.read_text())
        L += [mac("abProbePermR", P["perm"]["R"]), mac("abProbePermRate", r2(P["perm"]["rate"])),
              mac("abProbePermKsP", r2(P["perm"]["ks_p"])), mac("abProbeHplbR", P["hplb"]["R"]),
              mac("abProbeHplbNull", f"{P['hplb']['null_rate']:.3f}"),
              mac("abProbeHplbBoundary", f"{P['hplb']['boundary_rate']:.3f}"), mac("abProbeAggR", P["mmdagg"]["R"]),
              mac("abProbeAggInter", f"{P['mmdagg']['rate_interSample']:.3f}"),
              mac("abProbeAggPooled", f"{P['mmdagg']['rate_pooled']:.3f}"),
              mac("abProbeAggSe", f"{P['mmdagg']['se']:.3f}")]
    (TAB / "aos_baselines_macros.tex").write_text("\n".join(L) + "\n")
    print("wrote", TAB / "aos_baselines_macros.tex")
    for ln in L[1:]:
        print(" ", ln)


if __name__ == "__main__":
    main()

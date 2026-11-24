"""JMLR v3, Section 6 (revision plan v1.1, S1): how much of alpha* each lower bound recovers.

Drawing and macros only, from the raw replicates of the synthetic study (explore/harder_sims/out/<design>.jsonl,
200 replicates per design and size). For every design, at n = m = 800 (mixed types: n = 400, m = 5n, its
largest planned size), the figure shows the mean over replicates of (largest certified level) / alpha*, for the analytic
test (an_new), the betting test with the ramp (seq_bet, the estimator alpha-hat_lo) and the same betting test on the
labels of gradient boosting (gbm_bet) and logistic regression (logit_bet). Levels are the study's grid
alpha_j = j alpha*/8, so a fraction is a multiple of 1/8; a replicate certifying no level counts as 0.

Panel (b) (2026-11-19) reads the RHC-shaped run of Appendix E.2 (results/v2/figures/fig2_data.json, sizes 500 to
16,000, and fig2_small.json, sizes 100 to 400; alpha* in {0.15, 0.2, 0.3, 0.4}, grid of mesh 0.025) and draws, against
the sample size, the mean over replicates of alpha-hat / alpha* (analytic) and of the certified level / alpha*
(split-sample bound): a line for the mean over the four levels and a band for their range. From n = 2,000 that run uses
random Fourier features (Appendix E.2). It also writes the sample-size factor quoted in the introduction: at the two
largest sizes, the size at which the split-sample bound's mean curve first reaches the analytic estimator's mean
fraction (log-linear interpolation between simulated sizes), divided into the analytic estimator's size. The script
asserts that the analytic estimator never certifies more on average than the split-sample bound, in any design and
size of the synthetic study or of the RHC-shaped run.

    python3 -I scripts/v3_fig_recovery.py

Writes jmlr/v3/figs_v2/fig_recovery_v3.pdf (copied to figs_print by jmlr/v3/tools/sync.sh) and
jmlr/v3/tables/recovery_macros.tex.

TMMD_FIG2_DATA (AoS, 2026-11-20; default: results/v2/figures/fig2_data.json) replaces the fig2_data.json read by
panel (b), e.g. results/v3/figures/fig2_data_merged.json (v2 rows below n = 2,000, cos/sin random features from
there, written by scripts/aos_rerun_macros.py); fig2_small.json is still read from results/v2/figures. It needs
TMMD_FIG_ONLY=1 (the macros of that data are written by scripts/aos_rerun_macros.py, not here).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from _plate import BLUE, GREEN, GRID, INK, INK2, MUTED, ORANGE  # noqa: E402

OUT = ROOT / "explore" / "harder_sims" / "out"
REG = ROOT / "results" / "v2" / "figures"          # the RHC-shaped run of Appendix E.2
FIG2_DATA = Path(os.environ.get("TMMD_FIG2_DATA") or REG / "fig2_data.json")   # panel (b), n >= 500
# AoS restructure (2026-11-21; default unchanged): TMMD_FIG2_SMALL replaces fig2_small.json (sizes 100-400), e.g. the
# copy with the analytic estimator under tau = min{tau^P, tau^A} written by scripts/aos_main_tables.py.
FIG2_SMALL = Path(os.environ.get("TMMD_FIG2_SMALL") or REG / "fig2_small.json")
GAP_SIZES = (8000, 16000)                          # sizes at which the sample-size factor is read
FIG = Path(os.environ.get("TMMD_FIG_OUT", ROOT / "jmlr" / "v3" / "figs_v2" / "fig_recovery_v3.pdf"))
# Print-size options for other text widths (AoS version, 2026-11-20); the defaults give the JMLR figure unchanged.
FIG_WIDTH = float(os.environ.get("TMMD_FIG_WIDTH", "6.3"))   # inches
FIG_MINPT = float(os.environ.get("TMMD_FIG_MINPT", "0"))     # smallest type size in pt (0: as designed)
FIG_ONLY = os.environ.get("TMMD_FIG_ONLY") == "1"            # draw the figure, do not rewrite the macros
FIG_LEFT = float(os.environ.get("TMMD_FIG_LEFT", "0.30"))    # left margin (fraction), room for the row labels
FIG_RATIO = float(os.environ.get("TMMD_FIG_RATIO", "1.45"))  # width of panel (a) relative to panel (b)


def fs(x):
    return max(x, FIG_MINPT)
MAC = ROOT / "jmlr" / "v3" / "tables" / "recovery_macros.tex"

DESIGNS = [  # (key, label) in the order of Table 1, the RHC-shaped design first
    ("D9_rhc", "RHC-shaped, $d=33$"),
    ("D1_d2_a10", r"overlap, $d=2$, $\alpha^\star=0.1$"), ("D1_d2_a20", r"overlap, $d=2$"),
    ("D1_d10_a10", r"overlap, $d=10$, $\alpha^\star=0.1$"), ("D1_d10_a20", r"overlap, $d=10$"),
    ("D2_d20", r"dense shift, $d=20$"), ("D2_d50", r"dense shift, $d=50$"), ("D2_d100", r"dense shift, $d=100$"),
    ("D3a_loc", r"sparse shift, $d=50$"), ("D3b_corr", r"correlation flip, $d=50$"),
    ("D4_small", r"small contamination, $\alpha^\star=0.05$"), ("D5b_mixed", r"mixed types, $m=5n$"),
    ("D6_t3", r"$t_3$ tails, $d=5$"), ("D7_wide", "curves, wide bump"), ("D7_narrow", "curves, narrow bump"),
    ("D8_N1", r"oscillating, $\nu=1$, $\alpha^\star=0.5$"), ("D8_N4", r"oscillating, $\nu=4$"),
    ("D8_N16", r"oscillating, $\nu=16$"), ("D8_N64", r"oscillating, $\nu=64$"),
]
# AoS restructure (2026-11-21; default "jmlr" unchanged): TMMD_FIG_NOTATION=aos2 names the split-sample bound
# alpha-hat and the analytic estimator alpha-hat_an in the legend (the AoS paper's notation).
NOTATION = os.environ.get("TMMD_FIG_NOTATION", "jmlr")
RAMP_LABEL = r"betting, ramp ($\widehat\alpha$)" if NOTATION == "aos2" else r"betting, ramp ($\widehat\alpha_{\mathrm{lo}}$)"
AN_LABEL = r"analytic ($\widehat\alpha_{\mathrm{an}}$)" if NOTATION == "aos2" else "analytic"
METHODS = [  # key, label, colour, marker, filled
    ("an_new", AN_LABEL, INK, "o", False),
    ("seq_bet", RAMP_LABEL, BLUE, "o", True),
    ("gbm_bet", "GBM + betting", ORANGE, "s", True),
    ("logit_bet", "logistic + betting", GREEN, "^", True),
]


# AoS revision (2026-11-21; default unchanged): TMMD_FIGA_MATCHED=<summary.csv of scripts/aos_baselines_tables.py>
# replaces, in panel (a), the paper's classifiers (trained once on half of the data, betting on the other half) by the
# same classifiers on the ramp's batch schedule (scikit-learn defaults and, for logistic regression, tuned by CV),
# betting their labels, read from the mean recoveries of that summary at the design's planned size.
FIGA_MATCHED = os.environ.get("TMMD_FIGA_MATCHED")
METHODS_MATCHED = [
    ("an_new", AN_LABEL, INK, "o", False),
    ("seq_bet", RAMP_LABEL, BLUE, "o", True),
    ("gbm_m_hard", "GBM + betting", ORANGE, "s", True),
    ("logit_m_hard", "logistic + betting", GREEN, "^", True),
    ("logit_mt_hard", "tuned logistic + betting", GREEN, "^", False),
]


# AoS restructure (2026-11-21; default unchanged): TMMD_FIGA_MAIN=<json of scripts/aos_main_tables.py> draws panel (a)
# from that file (the ramp with the kernel named in its "ramp_label", the analytic test with tau = min{tau^P, tau^A},
# and the classifiers on the batches of the ramp), and TMMD_FIG_PANELS=a draws panel (a) alone. Both need
# TMMD_FIG_ONLY=1 (their macros are written by scripts/aos_main_tables.py).
FIGA_MAIN = os.environ.get("TMMD_FIGA_MAIN")
FIG_PANELS = os.environ.get("TMMD_FIG_PANELS", "ab")


def main_recovery():
    D = json.loads(Path(FIGA_MAIN).read_text())
    lab = D.get("ramp_label", "")
    methods = [
        ("analytic", AN_LABEL, INK, "o", False),
        ("ramp", RAMP_LABEL + (f", {lab} kernel" if lab else ""), BLUE, "o", True),
        ("gbm_m_hard", "GBM + betting", ORANGE, "s", True),
        ("logit_m_hard", "logistic + betting", GREEN, "^", True),
        ("logit_mt_hard", "tuned logistic + betting", GREEN, "^", False),
    ]
    rows = {r["design"]: (r["n"], {k: tuple(v) for k, v in r["rec"].items()}) for r in D["rows"]}
    return methods, rows


def matched_recovery(key, nmax):
    import csv
    out = {}
    for row in csv.DictReader(open(FIGA_MATCHED)):
        if row["design"] == key and int(row["n"]) == nmax and row["rec"] != "":
            out[row["method"]] = (float(row["rec"]), int(row["R"]))
    return out


def load(key):
    f = OUT / f"{key}.jsonl"
    if not f.exists():
        return None
    recs = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    return recs


def recovery(recs):
    sizes = {r["n"] for r in recs}
    nmax = 800 if 800 in sizes else max(sizes)       # the common planned size (tier-2 n = 1600 runs are not used)
    rs = [r for r in recs if r["n"] == nmax]
    out = {}
    for mkey, *_ in METHODS:
        fr = []
        for r in rs:
            j = r["jlo"][mkey]
            fr.append(0.0 if j < 0 else r["levels"][j] / r["alpha_star"])
        out[mkey] = (float(np.mean(fr)), len(fr))
    return nmax, out


def dominance(recs):
    """Cells (n) of a synthetic design where the analytic test certifies more on average than the ramp."""
    bad = []
    for n in sorted({r["n"] for r in recs}):
        rs = [r for r in recs if r["n"] == n]
        frac = {k: np.mean([0.0 if r["jlo"][k] < 0 else r["levels"][r["jlo"][k]] / r["alpha_star"] for r in rs])
                for k in ("an_new", "seq_bet")}
        if frac["an_new"] > frac["seq_bet"]:
            bad.append(n)
    return bad


def load_regime(fig2_data=None):
    """Mean fractions of alpha* by level and size in the RHC-shaped run (fig2_data.json + fig2_small.json)."""
    rows, cells = [], {}
    fig2_data = FIG2_DATA if fig2_data is None else Path(fig2_data)
    for name, path in (("fig2_data.json", fig2_data), ("fig2_small.json", FIG2_SMALL)):
        D = json.loads(path.read_text())
        for r in D["rows"]:
            if r["alpha_star"] <= 0:
                continue
            key = (r["alpha_star"], r["n"])
            cells.setdefault(key, set()).add(name)
            rows.append(r)
        grid = np.array(D["grid"])
    clash = [k for k, v in cells.items() if len(v) > 1]
    assert not clash, f"cells in both files: {clash[:4]}"
    levels = sorted({r["alpha_star"] for r in rows})
    sizes = sorted({r["n"] for r in rows})
    ana = np.array([[np.mean([r["alpha_hat"] for r in rows if r["alpha_star"] == a and r["n"] == n]) / a
                     for n in sizes] for a in levels])
    cert = np.array([[np.mean([r["certified"] for r in rows if r["alpha_star"] == a and r["n"] == n]) / a
                      for n in sizes] for a in levels])
    ceiling = np.array([grid[grid < a - 1e-12].max() / a for a in levels])
    return levels, np.array(sizes), ana, cert, ceiling


def size_reached(sizes, frac, f):
    """Smallest size at which the curve ``frac`` reaches ``f``, log-linear between simulated sizes."""
    for i, v in enumerate(frac):
        if v >= f:
            assert i > 0, "reached at the smallest simulated size: the factor would be only a lower bound"
            x0, x1, y0, y1 = np.log(sizes[i - 1]), np.log(sizes[i]), frac[i - 1], frac[i]
            return float(np.exp(x0 + (f - y0) / (y1 - y0) * (x1 - x0)))
    return None


def sig2(x):
    return f"{float(f'{x:.2g}'):,.0f}".replace(",", "{,}")


def main():
    if FIGA_MATCHED and not FIG_ONLY:
        raise SystemExit("TMMD_FIGA_MATCHED needs TMMD_FIG_ONLY=1 (its macros: scripts/aos_baselines_tables.py)")
    if os.environ.get("TMMD_FIG2_DATA") and not FIG_ONLY:
        raise SystemExit("TMMD_FIG2_DATA needs TMMD_FIG_ONLY=1 (its macros: scripts/aos_rerun_macros.py)")
    if (FIGA_MAIN or FIG_PANELS != "ab") and not FIG_ONLY:
        raise SystemExit("TMMD_FIGA_MAIN / TMMD_FIG_PANELS need TMMD_FIG_ONLY=1 (macros: scripts/aos_main_tables.py)")
    main_methods, main_rows = main_recovery() if FIGA_MAIN else (None, None)
    rows = []
    for key, label in DESIGNS:
        recs = load(key)
        if recs is None:
            print(f"skip {key}: no replicates yet")
            continue
        bad = dominance(recs)
        assert not bad, f"{key}: analytic above the ramp at n = {bad}"
        nmax, rec = recovery(recs)
        if FIGA_MAIN:
            nmax, rec = main_rows[key]
            rows.append((key, label, nmax, rec))
            continue
        if FIGA_MATCHED:
            mr = matched_recovery(key, nmax)
            for mkey, *_ in METHODS_MATCHED[2:]:
                rec[mkey] = mr[mkey]
            assert abs(mr["seq_bet"][0] - rec["seq_bet"][0]) < 1e-9, (key, "ramp recovery differs from the summary")
        rows.append((key, label, nmax, rec))

    levels, sizes, r_ana, r_cert, ceiling = load_regime()
    assert np.all(r_ana <= r_cert + 1e-12), "RHC-shaped run: analytic above the split-sample bound in some cell"
    factors = []
    for N in GAP_SIZES:
        j = int(np.where(sizes == N)[0][0])
        for i in range(len(levels)):
            n_split = size_reached(sizes, r_cert[i], r_ana[i, j])
            assert n_split is not None
            factors.append(N / n_split)

    plt.rcParams.update({"font.family": "serif", "font.size": fs(8), "axes.linewidth": 0.6,
                         "pdf.fonttype": 42, "text.usetex": False, "mathtext.fontset": "cm"})
    h = float(os.environ.get("TMMD_FIG_ROWH", "0.24")) * len(rows) + 0.9   # AoS (2026-11-21): row height option
    fig = plt.figure(figsize=(FIG_WIDTH, h))
    if FIG_PANELS == "a":
        ax = fig.add_subplot(1, 1, 1)
        bx = None
    else:
        gs = fig.add_gridspec(1, 2, width_ratios=[FIG_RATIO, 1.0], wspace=0.11)
        ax = fig.add_subplot(gs[0, 0])
        bx = fig.add_subplot(gs[0, 1])
    y = np.arange(len(rows))[::-1]
    for yi in y:
        ax.axhline(yi, color=GRID, lw=0.5, zorder=0)
    for mkey, mlabel, col, mk, filled in (main_methods if FIGA_MAIN else METHODS_MATCHED if FIGA_MATCHED else METHODS):
        xs = [r[3][mkey][0] for r in rows]
        ax.scatter(xs, y, s=30, marker=mk, facecolors=col if filled else "none", edgecolors=col if filled else INK,
                   linewidths=0.9 if not filled else 0.6, label=mlabel, zorder=4 if not filled else 3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r[1]}" + (r"  ($n=400$)" if r[2] != 800 else "") for r in rows], color=INK)
    ax.set_xlim(-0.02, 1.0)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel(r"mean certified level / $\alpha^\star$", color=INK)
    ax.tick_params(colors=INK2, length=2.5, width=0.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    if FIG_PANELS != "a":
        ax.set_title(r"(a) all designs, $n=m=800$", fontsize=fs(7.6), loc="left", color=INK)
    if bx is None:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
                   fontsize=fs(7.5), handletextpad=0.3, columnspacing=1.4)
        fig.subplots_adjust(left=FIG_LEFT, right=0.98, bottom=0.48 / h, top=1 - 0.78 / h)
        FIG.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG)
        print("wrote", FIG)
        return

    # (b) the RHC-shaped run against the sample size: mean over the four levels (line), their range (band)
    bx.axhspan(ceiling.min(), ceiling.max(), color=GRID, alpha=0.55, lw=0, zorder=0)
    bx.text(sizes[0] * 1.05, ceiling.max() + 0.012, "largest grid level below $\\alpha^\\star$", fontsize=fs(6.2),
            color=MUTED, va="bottom")
    bx.axvline(800, color=MUTED, ls=(0, (1, 2)), lw=0.7, zorder=1)
    bx.text(800 * 1.08, 0.015, r"$n=800$", fontsize=fs(6.2), color=MUTED, va="bottom")
    for arr, col, filled, z in ((r_cert, BLUE, True, 3), (r_ana, INK, False, 4)):
        bx.fill_between(sizes, arr.min(axis=0), arr.max(axis=0), color=col, alpha=0.13, lw=0, zorder=z - 1)
        bx.plot(sizes, arr.mean(axis=0), color=col, lw=1.1, zorder=z, marker="o", ms=4.2,
                mfc=col if filled else "white", mec=col, mew=0.9)
    bx.set_xscale("log")
    bx.set_xlim(sizes[0] / 1.25, sizes[-1] * 1.25)
    bx.set_xticks([100, 1000, 10000])
    bx.set_xticklabels(["100", "1,000", "10,000"])
    bx.minorticks_off()
    bx.set_ylim(-0.02, 1.0)
    bx.yaxis.tick_right()
    bx.yaxis.set_label_position("right")
    bx.set_ylabel(r"mean certified level / $\alpha^\star$", color=INK)
    bx.set_xlabel(r"sample size per arm, $n=m$", color=INK)
    bx.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    bx.tick_params(colors=INK2, length=2.5, width=0.5)
    for s in ("top", "left"):
        bx.spines[s].set_visible(False)
    for s in ("bottom", "right"):
        bx.spines[s].set_color(MUTED)
    bx.set_title("(b) RHC-shaped, by sample size", fontsize=fs(7.6), loc="left", color=INK)

    handles, labels = ax.get_legend_handles_labels()
    five = bool(FIGA_MATCHED or FIGA_MAIN)          # five legend entries: three columns, two rows
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3 if five else 4, frameon=False,
               fontsize=fs(7.5), handletextpad=0.3, columnspacing=1.4)
    fig.subplots_adjust(left=FIG_LEFT, right=0.915, bottom=0.48 / h, top=1 - (0.78 if five else 0.56) / h)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG)
    print("wrote", FIG)
    if FIGA_MAIN:          # rows keyed by the methods of scripts/aos_main_tables.py, which writes their macros
        return

    # macros
    def mac(name, val):
        return f"\\newcommand{{\\{name}}}{{{val}}}"
    lines = ["% Generated by scripts/v3_fig_recovery.py -- do not edit."]
    have = {r[0]: r for r in rows}
    if "D9_rhc" in have:
        r = have["D9_rhc"][3]
        lines += [mac("recRhcAnalytic", f"{r['an_new'][0]:.2f}"), mac("recRhcRamp", f"{r['seq_bet'][0]:.2f}"),
                  mac("recRhcGbm", f"{r['gbm_bet'][0]:.2f}"), mac("recRhcLogit", f"{r['logit_bet'][0]:.2f}")]
    ramp = [r[3]["seq_bet"][0] for r in rows]
    ana = [r[3]["an_new"][0] for r in rows]
    best_clf = [max(r[3]["gbm_bet"][0], r[3]["logit_bet"][0]) for r in rows]
    lines += [mac("recNdesigns", str(len(rows))),
              mac("recRampMedian", f"{np.median(ramp):.2f}"),
              mac("recRampMax", f"{max(ramp):.2f}"),
              mac("recAnalyticMedian", f"{np.median(ana):.2f}"),
              mac("recAnalyticZero", str(sum(a == 0 for a in ana))),
              mac("recRampAboveAnalytic", str(sum(a > b for a, b in zip(ramp, ana)))),
              mac("recRampAboveClf", str(sum(a > b for a, b in zip(ramp, best_clf))))]
    lines += [mac("recGapLo", sig2(min(factors))), mac("recGapHi", sig2(max(factors))),
              mac("recGapNlo", sig2(GAP_SIZES[0])), mac("recGapNhi", sig2(GAP_SIZES[1]))]
    if FIG_ONLY:
        return
    MAC.write_text("\n".join(lines) + "\n")
    print("RHC-shaped run, analytic / split-sample by size:")
    for i, a in enumerate(levels):
        print(f"  alpha*={a}: " + " ".join(f"{n}:{x:.2f}/{c:.2f}" for n, x, c in zip(sizes, r_ana[i], r_cert[i])))
    print("sample-size factors at", GAP_SIZES, ":", " ".join(f"{f:.1f}" for f in factors))
    for key, label, nmax, rec in rows:
        print(f"{key:12s} n={nmax:4d} " + " ".join(f"{k}={v[0]:.2f}" for k, v in rec.items()))


if __name__ == "__main__":
    main()

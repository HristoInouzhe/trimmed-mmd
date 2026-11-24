"""Figures from the CSV summaries in out/ (no recomputation)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / "figs"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

STYLE = {  # method -> (label, color, marker, linestyle)
    "boot": ("bootstrap, Thm 16 (asymptotic)", INK, "o", "-"),
    "ramp_bet": ("ramp + betting (finite-sample)", SLOT[0], "s", "-"),
    "dir0_ramp_bet": ("ramp + betting, α=0 witness", SLOT[6], "D", "-"),
    "seq3": ("sequential batches + betting", SLOT[2], "^", "-"),
    "ramp_b2s": ("ramp + 2-sample emp. Bernstein", SLOT[1], "v", "--"),
    "psi_bet": ("unclipped ψ + betting", SLOT[4], "P", "--"),
    "tlm": ("DKW band (T&LM)", SLOT[3], "X", ":"),
    "prop17": ("Prop 17 (DKW, 2κβ)", SLOT[7], "x", ":"),
    "gbm_hoef": ("GBM classifier + Hoeffding (App. E.3)", INK2, "*", "-."),
    "gbm_bet": ("GBM classifier + betting", SLOT[5], "h", "-."),
}


def _ax_style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(INK2)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def power_figure(rates, seq=None, out="power.png"):
    alphas = [0.0, 0.1, 0.15, 0.2]
    titles = ["α = 0 (power)", "α = 0.10 (power)", "α = 0.15 (power)", "α = 0.20 = TV (type-I)"]
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.9), sharey=False)
    for ax, a, t in zip(axs, alphas, titles):
        d = rates[np.isclose(rates.alpha, a)].sort_values("n")
        for m, (lab, col, mk, ls) in STYLE.items():
            if m == "seq3":
                if seq is None:
                    continue
                dd = seq[np.isclose(seq.alpha, a)].sort_values("n")
                x, y = dd.n, dd["seq3"]
            else:
                if m not in d.columns:
                    continue
                x, y = d.n, d[m]
            ax.plot(x, y, color=col, marker=mk, ls=ls, lw=2, ms=6, label=lab)
        ax.set_xscale("log", base=2)
        ax.set_xticks([100, 200, 400, 800, 1600])
        ax.set_xticklabels(["100", "200", "400", "800", "1600"])
        ax.set_title(t, fontsize=10, color=INK)
        ax.set_xlabel("n = m (total per sample)", fontsize=9, color=INK2)
        if a == 0.2:
            ax.axhline(0.05, color=INK2, lw=1, ls="--")
            ax.set_ylim(-0.005, 0.15)
        else:
            ax.set_ylim(-0.02, 1.02)
        _ax_style(ax)
    axs[0].set_ylabel("rejection rate", fontsize=9, color=INK2)
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("disjoint_d2_a20 (TV = 0.2), δ = 0.05, same splits and witnesses for all certificates",
                 fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIG / out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def alphalo_figure(means, out="alpha_lo.png"):
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for m, (lab, col, mk, ls) in STYLE.items():
        if m not in means.columns:
            continue
        ax.plot(means.index, means[m], color=col, marker=mk, ls=ls, lw=2, ms=6, label=lab)
    ax.axhline(0.2, color=INK2, lw=1, ls="--")
    ax.text(105, 0.203, "TV = 0.2", fontsize=8, color=INK2)
    ax.set_xscale("log", base=2)
    ax.set_xticks([100, 200, 400, 800])
    ax.set_xticklabels(["100", "200", "400", "800"])
    ax.set_ylim(-0.005, 0.22)
    ax.set_xlabel("n = m", fontsize=9, color=INK2)
    ax.set_ylabel("mean certified lower bound α̂_lo", fontsize=9, color=INK2)
    _ax_style(ax)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / out, dpi=150)
    plt.close(fig)


def real_figure(wide, out="real.png"):
    order = ["rhc", "nhanes_child", "nhanes_age", "nhanes_bmi", "adult", "acs"]
    labels = ["RHC", "NHANES\n6–11 vs 60–79", "NHANES\n20–39 vs 60–79", "NHANES\nBMI", "Adult", "ACS"]
    meths = ["boot", "ramp_bet", "dir0_ramp_bet", "ramp_b2s", "tlm", "prop17"]
    meths = [m for m in meths if m in wide.columns]
    wide = wide.reindex([o for o in order if o in wide.index])
    x = np.arange(len(wide))
    wbar = 0.8 / len(meths)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for k, m in enumerate(meths):
        vals = wide[m].clip(lower=0).values
        ax.bar(x + (k - (len(meths) - 1) / 2) * wbar, vals, width=wbar * 0.92, color=STYLE[m][1],
               label=STYLE[m][0])
    ax.set_xticks(x)
    ax.set_xticklabels([labels[order.index(i)] for i in wide.index], fontsize=8)
    ax.set_ylabel("certified TV lower bound\n(min over 8 splits, δ = 0.05)", fontsize=9, color=INK2)
    ax.set_title("Adult and ACS: one α = 0 witness per split for every method (matched)", fontsize=8, color=INK2)
    _ax_style(ax)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    what = sys.argv[1:] or ["power", "alphalo", "real"]
    if "power" in what:
        r = pd.read_csv(HERE / "out" / "rates_disjoint_d2_a20_all.csv")
        sq = HERE / "out" / "seq_rates_disjoint_d2_a20.csv"
        power_figure(r, pd.read_csv(sq) if sq.exists() else None)
    if "alphalo" in what:
        alphalo_figure(pd.read_csv(HERE / "out" / "alphalo_mean_disjoint_d2_a20.csv", index_col=0))
    if "real" in what:
        w = pd.read_csv(HERE / "out" / "real_min.csv", index_col=0)
        f = HERE / "out" / "real_dir0.jsonl"
        if f.exists():
            recs = [json.loads(l) for l in f.read_text().splitlines()]
            for d in ("adult", "acs"):
                rr = [r for r in recs if r["dataset"] == d]
                if rr:
                    mn = lambda k: min(r["lower"][k] for r in rr)
                    w.loc[d] = np.nan
                    w.loc[d, "boot"] = mn("boot_dir0")
                    for k in ("ramp_bet", "ramp_b2s", "tlm", "prop17"):
                        w.loc[d, k] = mn(k)
                    w.loc[d, "dir0_ramp_bet"] = mn("ramp_bet")
        real_figure(w)

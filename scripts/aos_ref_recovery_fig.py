"""AoS referee 2, M5: S6 figure, recovery of alpha* against n for each design (small multiples): the Laplace ramp
(alpha-hat), the oracle (the better of 1{p > q} and the best likelihood-ratio level set, same betting test), the oracle
1{p > q} alone, the Gaussian ramp at the main size, and the largest grid level below alpha* (0.875).

Data: results/aos/main/ref_recovery_summary.json (scripts/aos_ref_recovery_tables.py); nothing is computed here.
Print size: imsart text width 5.62 in, smallest type 8 pt (as aos/tools/figs_aos.sh). Writes
results/aos/main/figs/fig_recovery_curves.pdf and copies it to aos/figs_aos/fig_recovery_curves.pdf.

    python3 -I scripts/aos_ref_recovery_fig.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from _plate import BLUE, GRID, INK, INK2, MUTED, ORANGE  # noqa: E402

SUMMARY = ROOT / "results" / "aos" / "main" / "ref_recovery_summary.json"
OUT = ROOT / "results" / "aos" / "main" / "figs" / "fig_recovery_curves.pdf"
PAPER = ROOT / "aos" / "figs_aos" / "fig_recovery_curves.pdf"
WIDTH, MINPT = 5.62, 8.0
NCOL, NROW = 4, 5
SIZES = (100, 200, 400, 800)
TITLE = {                      # the labels of Table 1, shortened to fit a panel at 8 pt
    "D9_rhc": r"RHC-shaped, $d=33$", "D1_d2_a10": r"overlap $d\!=\!2$, $\alpha^\star\!=\!0.1$",
    "D1_d2_a20": r"overlap $d\!=\!2$, $\alpha^\star\!=\!0.2$", "D1_d10_a10": r"overlap $d\!=\!10$, $\alpha^\star\!=\!0.1$",
    "D1_d10_a20": r"overlap $d\!=\!10$, $\alpha^\star\!=\!0.2$", "D2_d20": r"dense shift, $d=20$",
    "D2_d50": r"dense shift, $d=50$", "D2_d100": r"dense shift, $d=100$", "D3a_loc": r"sparse shift, $d=50$",
    "D3b_corr": r"corr. flip, $d=50$", "D4_small": r"small contam., $d=2$", "D5b_mixed": r"mixed types, $m=5n$",
    "D6_t3": r"$t_3$ tails, $d=5$", "D7_wide": "curves, wide bump", "D7_narrow": "curves, narrow bump",
    "D8_N1": r"oscillating, $\nu=1$", "D8_N4": r"oscillating, $\nu=4$", "D8_N16": r"oscillating, $\nu=16$",
    "D8_N64": r"oscillating, $\nu=64$",
}


def main():
    S = json.loads(SUMMARY.read_text())
    cells = S["cells"]
    keys = S["designs"]
    assert len(keys) == 19 and set(keys) == set(TITLE)
    ceil = S["grid_ceiling"]
    plt.rcParams.update({"font.family": "serif", "font.size": MINPT, "axes.linewidth": 0.6, "pdf.fonttype": 42,
                         "text.usetex": False, "mathtext.fontset": "cm", "axes.titlesize": MINPT,
                         "xtick.labelsize": MINPT, "ytick.labelsize": MINPT, "legend.fontsize": MINPT})
    H = 6.1
    fig, axes = plt.subplots(NROW, NCOL, figsize=(WIDTH, H), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flat):
        if i >= len(keys):
            ax.axis("off")
            continue
        k = keys[i]
        ns = [n for n in SIZES if f"{k}|{n}" in cells]
        c = [cells[f"{k}|{n}"] for n in ns]
        ax.axhline(ceil, color=MUTED, lw=0.7, ls=(0, (1, 1.5)), zorder=1)
        ax.plot(ns, [x["psi"] for x in c], color=MUTED, lw=0.8, ls=(0, (3, 1.5)), zorder=2)
        ax.plot(ns, [x["oracle"] for x in c], color=INK, lw=0.9, marker="s", ms=3.4, mfc="white", mec=INK, mew=0.8,
                zorder=3)
        ax.plot(ns, [x["ramp"] for x in c], color=BLUE, lw=1.1, marker="o", ms=3.4, mfc=BLUE, mec=BLUE, zorder=4)
        ax.plot([ns[-1]], [c[-1]["gauss"]], ls="none", marker="D", ms=3.6, mfc="none", mec=ORANGE, mew=0.9, zorder=5)
        ax.set_title(TITLE[k], loc="left", color=INK, pad=2.5)
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(SIZES)
        ax.set_xticklabels([str(n) for n in SIZES])
        ax.set_xlim(85, 940)
        ax.set_ylim(-0.03, 1.0)
        ax.set_yticks([0, 0.25, 0.5, 0.75])
        ax.set_yticklabels(["0", "0.25", "0.5", "0.75"])
        ax.grid(axis="y", color=GRID, lw=0.45, zorder=0)
        ax.tick_params(colors=INK2, length=2.0, width=0.5, pad=1.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("bottom", "left"):
            ax.spines[s].set_color(MUTED)
    # the last row's panels that are not drawn still carry the shared x axis of their column
    for j in range(len(keys), NROW * NCOL):
        above = axes.flat[j - NCOL]
        above.xaxis.set_tick_params(labelbottom=True)
    leg_ax = axes.flat[-1]
    handles = [Line2D([], [], color=BLUE, lw=1.1, marker="o", ms=3.4, label=r"$\widehat\alpha$, Laplace ramp"),
               Line2D([], [], color=INK, lw=0.9, marker="s", ms=3.4, mfc="white", mew=0.8, label="oracle"),
               Line2D([], [], color=MUTED, lw=0.8, ls=(0, (3, 1.5)), label=r"oracle $\mathbf{1}\{p>q\}$"),
               Line2D([], [], ls="none", marker="D", ms=3.6, mfc="none", mec=ORANGE, mew=0.9,
                      label=r"$\widehat\alpha$, Gaussian"),
               Line2D([], [], color=MUTED, lw=0.7, ls=(0, (1, 1.5)), label=r"grid ceiling $7/8$")]
    leg_ax.legend(handles=handles, loc="center left", bbox_to_anchor=(-0.08, 0.5), frameon=False,
                  handlelength=1.8, handletextpad=0.5, labelspacing=0.45, borderaxespad=0.0)
    fig.supylabel(r"mean certified level / $\alpha^\star$", x=0.006, fontsize=MINPT, color=INK)
    fig.supxlabel(r"sample size per arm, $n=m$ (mixed types: $n$, $m=5n$)", y=0.008, fontsize=MINPT, color=INK)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.075, top=0.97, wspace=0.10, hspace=0.42)

    # checks: no text below 8 pt; no title running into the panel on its right
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_text() and t.get_visible():
            assert t.get_fontsize() >= MINPT - 1e-9, (t.get_text(), t.get_fontsize())
    for r in range(NROW):
        for cidx in range(NCOL - 1):
            a, b = axes[r, cidx], axes[r, cidx + 1]
            if r * NCOL + cidx + 1 >= len(keys):
                continue
            ta = a._left_title.get_window_extent(rend)
            gap = (b._left_title.get_window_extent(rend).x0 - ta.x1) / fig.dpi
            assert gap >= 0.08, (a.get_title(loc="left"), b.get_title(loc="left"), gap)   # inches between titles
    for r in range(NROW):
        ax_ = axes[r, NCOL - 1]
        if ax_.get_title(loc="left"):
            room = (fig.bbox.x1 - ax_._left_title.get_window_extent(rend).x1) / fig.dpi
            assert room >= 0.01, ("clipped", ax_.get_title(loc="left"), room)
    assert all(ax_.get_title(loc="left") for ax_ in axes.flat[:len(keys)])
    leg = leg_ax.get_legend().get_window_extent(rend)
    assert leg.x1 <= fig.bbox.x1 + 0.5, "legend clipped"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print("wrote", OUT)
    tmp = PAPER.with_name("." + PAPER.name + ".tmp")
    shutil.copyfile(OUT, tmp)
    os.replace(tmp, PAPER)
    print("copied to", PAPER)


if __name__ == "__main__":
    main()

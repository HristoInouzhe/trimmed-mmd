r"""AoS Section 5.4 figure: why the margin is pessimistic (Proposition prop:excess).

For the shift P = N(1,1), Q = N(-1,1) (alpha* = 2 Phi(1) - 1) and two kernels, the rescaled path
u(alpha) = (1-alpha)D_alpha near alpha*, the supporting line at alpha_0 = alpha* - eps with slope -(t*-u*) (the gap
between the population cut points, a subgradient of u by Proposition prop:excess(i)), drawn past its zero, and the
zero itself. The excess of the population ramp at alpha_0 is at least the distance from alpha_0 to that zero,
u(alpha_0)/(t*-u*). Panel (b) has an inset that enlarges the region between alpha_0 and alpha*.

Data: explore/split_rate_excess/out/q1_{laplace,gauss}_shift_1.0.json (rows[].disc: U = u(alpha*-eps), gap_n =
t*-u*, excess = E_P psi* - E_Q psi* - alpha), written by explore/split_rate_excess/scripts/q1_population.py; the
Laplace values agree with the certified path of results/aos/margin_check/laplace_shift_1.0.json. Dots: computed
levels; line: monotone (PCHIP) interpolation in alpha between them.

Writes aos/figs_aos/fig_excess_aos.pdf and the numbers of its caption, results/aos/tables/aos_fig_excess_macros.tex
(prefix \fx: alpha*, the two levels, the distance to the zero as a fraction of alpha* - alpha in each panel, and the
Gaussian fraction and level at the alpha* - alpha of panel (a), for a like-for-like comparison).

    ~/.venvs/llm_book/bin/python -B scripts/aos_fig_excess.py
"""
from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "explore" / "split_rate_excess" / "out"
OUT = ROOT / "aos" / "figs_aos" / "fig_excess_aos.pdf"
MACROS = ROOT / "results" / "aos" / "tables" / "aos_fig_excess_macros.tex"
INK, MUTED, LINE = "#222222", "#888888", "#d95f02"
WIDTH, HEIGHT = 5.62, 1.98


def rnd(x, q):
    return str(Decimal(repr(float(x))).quantize(Decimal(q), rounding=ROUND_HALF_UP))


def load(kernel):
    o = json.loads((SRC / f"q1_{kernel}_shift_1.0.json").read_text())
    a_star = o["alpha_star"]
    rows = [(r["eps"], r["disc"]["U"], r["disc"]["gap_n"], r["disc"]["excess"]) for r in o["rows"]
            if r.get("disc") and "U" in r["disc"]]
    rows.sort()
    return a_star, rows


def draw(ax, path, a_star, a0, u0, gap, xlim, ylim, ms, arrow=True, labels=True, text_right=False):
    """Path, dots, supporting line extended below zero with its zero marked, and the excess arrow."""
    grid, fgrid, alpha, U = path
    aT = a0 + u0 / gap
    ax.plot(grid, fgrid, color=INK, lw=1.0, zorder=2)
    ax.plot(alpha, U, "o", ms=ms, color=INK, zorder=2)
    ax.plot([a_star, xlim[1]], [0, 0], color=INK, lw=1.0, zorder=2)
    ax.axhline(0, color=MUTED, lw=0.4, zorder=1)
    x_lo = max(xlim[0], a0 - 0.6 * (aT - a0))
    x_hi = min(xlim[1], aT + 0.35 * (aT - a0))
    tl = np.linspace(x_lo, x_hi, 80)
    ax.plot(tl, u0 - gap * (tl - a0), color=LINE, lw=1.4, ls=(0, (4, 1.6)), zorder=3)
    ax.plot([a0], [u0], "o", ms=ms + 1.0, color=LINE, zorder=4)
    ax.plot([aT], [0], marker="|", ms=9, mew=1.6, color=LINE, zorder=4)
    span = ylim[1] - ylim[0]
    if arrow:
        yb = ylim[0] + 0.13 * span
        ax.annotate("", xy=(aT, yb), xytext=(a0, yb),
                    arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8, shrinkA=0, shrinkB=0))
        if text_right:
            # inside the inset frame: to the right of the arrow, where the path is flat at zero
            ax.text(aT + 0.025 * (xlim[1] - xlim[0]), yb, "excess bound", color=INK, ha="left", va="center",
                    fontsize=7)
        else:
            ax.text(0.5 * (a0 + aT), yb - 0.04 * span, "excess bound", color=INK, ha="center", va="top", fontsize=7)
    for x in (a0, a_star):
        ax.axvline(x, color=MUTED, lw=0.5, ls=":", zorder=1)
    if labels:
        for x, lab in ((a0, r"$\alpha$"), (a_star, r"$\alpha^\star$")):
            ax.text(x, ylim[1], lab, ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    return aT


def panel(ax, kernel, eps0, window, title, inset=False):
    a_star, rows = load(kernel)
    eps = np.array([r[0] for r in rows])
    U = np.array([r[1] for r in rows])
    keep = eps <= window * 1.0001
    eps, U = eps[keep], U[keep]
    alpha = a_star - eps
    order = np.argsort(alpha)
    alpha, U = alpha[order], U[order]
    # the path is zero from alpha* on; add that point for the interpolation
    alpha_i = np.append(alpha, a_star)
    U_i = np.append(U, 0.0)
    f = PchipInterpolator(alpha_i, U_i)
    grid = np.linspace(alpha_i[0], a_star, 600)
    path = (grid, f(grid), alpha, U)
    r0 = min(rows, key=lambda r: abs(r[0] - eps0))
    e0, u0, gap, exc = r0
    a0 = a_star - e0
    ymax = float(U.max())
    xlim = (a_star - window, a_star + 0.06 * window)
    ylim = (-0.3 * ymax, 1.12 * ymax)
    aT = draw(ax, path, a_star, a0, u0, gap, xlim, ylim, ms=2.4, arrow=not inset)
    ax.annotate(r"slope $-(t^\star-u^\star)$", xy=(a0, u0), xytext=(a_star - 0.97 * window, 0.04 * ymax),
                color=LINE, fontsize=7, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color=LINE, lw=0.5, shrinkA=0, shrinkB=2))
    if inset:
        # upper right, above the path, which is flat near zero there; tick labels at 7 pt as in the main panels
        axi = ax.inset_axes([0.47, 0.54, 0.52, 0.42])
        xl = (a0 - 0.25 * (aT - a0), a_star + 0.06 * (a_star - a0))
        yl = (-1.3 * u0, 1.45 * u0)
        draw(axi, path, a_star, a0, u0, gap, xl, yl, ms=2.0, arrow=True, labels=False, text_right=True)
        axi.set_xticks([0.65, 0.67])
        axi.tick_params(labelsize=7, length=1.5, pad=1.5)
        axi.set_yticks([])
        for s in axi.spines.values():
            s.set_linewidth(0.5)
            s.set_color(MUTED)
        ax.indicate_inset_zoom(axi, edgecolor=MUTED, lw=0.5)
    ax.set_xlabel(r"level $\alpha$", fontsize=8)
    ax.set_title(title, fontsize=8, loc="left", color=INK, pad=10)
    ax.tick_params(labelsize=7, length=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return a_star, a0, e0, u0 / gap, exc


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42,
                         "text.usetex": False, "mathtext.fontset": "cm"})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, HEIGHT))
    s1 = panel(a1, "laplace", 0.1, 0.32, r"(a) Laplace kernel")
    s2 = panel(a2, "gauss", 0.0341, 0.11, r"(b) Gaussian kernel", inset=True)
    # the Gaussian fraction at the level of panel (a), for a like-for-like comparison in the caption
    a_star_g, rows_g = load("gauss")
    e_g, u_g, gap_g, _ = min(rows_g, key=lambda r: abs(r[0] - s1[2]))
    assert abs(e_g - s1[2]) < 0.01, (e_g, s1[2])
    frac_g_at_lap = u_g / gap_g / e_g
    print(f"gauss at the level of (a): eps = {e_g:.4g}, alpha = {a_star_g - e_g:.4g}, fraction = {frac_g_at_lap:.3f}")
    a1.set_ylabel(r"$(1-\alpha)D_\alpha$", fontsize=8)
    fig.tight_layout(pad=0.3, w_pad=1.2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    for name, (_, _, e0, lb, exc) in (("laplace", s1), ("gauss", s2)):
        print(f"{name}: eps = {e0:.4g}, bound u/gap = {lb:.4g} ({lb / e0:.2f} eps), excess = {exc:.4g} ({exc / e0:.2f} eps)")
    print("wrote", OUT)
    lines = ["% Generated by scripts/aos_fig_excess.py -- do not edit.",
             f"\\newcommand{{\\fxAlphaStar}}{{{rnd(s1[0], '0.001')}}}",
             f"\\newcommand{{\\fxLevelLap}}{{{rnd(s1[1], '0.01')}}}",
             f"\\newcommand{{\\fxLevelGauss}}{{{rnd(s2[1], '0.01')}}}",
             f"\\newcommand{{\\fxFracLap}}{{{rnd(s1[3] / s1[2], '0.01')}}}",
             f"\\newcommand{{\\fxFracGauss}}{{{rnd(s2[3] / s2[2], '0.01')}}}",
             f"\\newcommand{{\\fxLevelGaussAtLap}}{{{rnd(a_star_g - e_g, '0.01')}}}",
             f"\\newcommand{{\\fxFracGaussAtLap}}{{{rnd(frac_g_at_lap, '0.01')}}}"]
    MACROS.parent.mkdir(parents=True, exist_ok=True)
    MACROS.write_text("\n".join(lines) + "\n")
    print("wrote", MACROS)


if __name__ == "__main__":
    main()

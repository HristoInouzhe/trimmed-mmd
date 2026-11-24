"""Figure 2: where the method works, on data whose contamination is known.

Drawing only. Every number comes from ``results/figures/fig2_data.json``,
written by ``scripts/export_fig2_data.py``; this script re-solves nothing.

    PYTHONPATH=src python scripts/fig2.py
    PYTHONPATH=src python scripts/fig2.py --threshold plugin --outdir results/v2   # revision v2

With ``--outdir D`` the inputs are read from ``D/figures`` (fig2_data.json,
fig2_null.json, fig1_data.json) and the outputs go to ``D/figures`` and
``D/tables/regime_macros.tex``. ``--threshold`` defaults to the threshold
recorded in fig2_data.json (``paper`` for v1 files); it sets the constant of
the firing rule and the computed alpha-hat boundaries. For the plug-in
threshold the firing rule keeps its closed form (tau_new(0) = 2C/sqrt(n) at
n = m, with C = :func:`trimmedmmd.threshold.closed_form_constant`), but the
alpha-hat boundaries are found numerically, since tau_new(alpha) is not
tau_new(0)/(1-alpha).

Revision v2: if ``fig2_small.json`` (sizes 100-400, written by
``export_fig2_data.py --sizes 100 200 300 400 --out fig2_small.json``) sits
next to fig2_data.json, its rows are merged (:func:`load_rows`), so the fired /
did-not-fire markers, panel 3 and the macros (\\rgNmin, \\rgReps, ...) cover
those sizes too; it must carry the same threshold, certificate, grid, delta
and separation. Without the file nothing changes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib.patches import Rectangle                      # noqa: E402
from _plate import (BLUE, GREEN, GRID, INK, INK2, MUTED, ORANGE,  # noqa: E402
                    VIOLET, RC, banner, head, ledger, legend, note,
                    panels, provenance)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
plt.rcParams.update(RC)

# One hue per contamination level, assigned in fixed order and never cycled.
LEVEL_COLOR = [BLUE, ORANGE, VIOLET, GREEN]   # validated 4-slot set
LEVEL_MARK = ["o", "s", "^", "D"]


def _by(rows, key, levels, sizes):
    """mean / min / max of ``key`` for each (level, size), ragged-safe."""
    out = {}
    for a in levels:
        m, lo, hi, xs = [], [], [], []
        for n in sizes:
            v = [r[key] for r in rows
                 if r["alpha_star"] == a and r["n"] == n and r.get(key) is not None]
            if not v:
                continue
            xs.append(n); m.append(np.mean(v))
            lo.append(np.min(v)); hi.append(np.max(v))
        out[a] = (np.array(xs), np.array(m), np.array(lo), np.array(hi))
    return out


def closed_C(delta, threshold="paper"):
    """C with threshold(n, n, 0) = 2C/sqrt(n): 4 + sqrt(2 log(4/delta)) for v1.

    Revision v2 (threshold "plugin"): eq:samplesize of the paper, n > 2 (C_0/D_0)^2
    with C_0 = 1 + sqrt(log(1/delta)) (Gaussian kernel, kappa = 1), i.e. this C is
    C_0 / sqrt(2), so that :func:`_fire_n`'s (2C/D_0)^2 is exactly 2 (C_0/D_0)^2.
    """
    if threshold == "paper":
        return 4.0 + np.sqrt(2.0 * np.log(4.0 / delta))
    from trimmedmmd.threshold import samplesize_constant
    return samplesize_constant(delta) / math.sqrt(2.0)


@lru_cache(maxsize=None)
def _n_cross(target, b, delta, exact=True):
    """Smallest n (n = m) with tau_new(n, n, b, delta) < target (Gaussian kernel).

    A fixed-point step on the n^{-1/2} law gets within a fraction of a
    percent; ``exact`` then brackets and bisects to the integer.
    """
    from trimmedmmd.threshold import tau_new
    if not target > 0:
        return float("inf")

    def f(k):
        return tau_new(int(k), int(k), float(b), float(delta))

    n = 1000
    for _ in range(8):
        new = max(2, int(math.ceil(n * (f(n) / target) ** 2)))
        if new == n:
            break
        n = new
    if not exact:
        return float(n)
    lo, hi = n, n
    while f(hi) >= target:
        hi = int(hi * 1.02) + 1
    while lo > 2 and f(lo) < target:
        lo = int(lo / 1.02)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if f(mid) < target:
            hi = mid
        else:
            lo = mid
    return float(hi)


def _fire_n(a, d0_per_a, C):
    """Sample size at which the analytic test first fires, from Section 11.3:
    tau(0) = 2C/sqrt(n) and D_0 ~ d0_per_a * alpha*, so n > (2C/D_0)^2
    (v2: = 2 (C_0/D_0)^2 of eq:samplesize, see :func:`closed_C`; the regime map
    uses the measured population D_0 = d0_per_a * alpha*)."""
    return (2.0 * C / (d0_per_a * a)) ** 2


def _sharp_n(a, f, d0_per_a, C):
    """Sample size at which alpha-hat first reaches a fraction f of alpha*.

    ``required_n_sharp`` of the package: alpha-hat >= b exactly when
    tau(0) < D(b)(1-b). Along this scenario the path decays essentially
    linearly to zero at alpha*, so D(f alpha*) ~ d0_per_a alpha* (1 - f).
    This treats the measured path as the population one, so it is a design
    guide, not a guarantee -- which is why it is drawn as a boundary and not
    as a claim.
    """
    b = f * a
    t = d0_per_a * a * (1.0 - f) * (1.0 - b)
    return (2.0 * C / np.maximum(t, 1e-12)) ** 2


def _sharp_n_v2(a, f, d0_per_a, delta, threshold):
    """``_sharp_n`` for a threshold without the tau(0)/(1-alpha) form.

    alpha-hat >= b = f alpha* once threshold(b) < D(b) ~ d0_per_a alpha* (1 - f),
    solved for n numerically. Arrays of more than a few points (the drawn
    curves) are solved on 30 log-spaced levels and interpolated in log-log;
    scalars (the macros) exactly, to the integer.
    """
    if threshold == "paper":
        return _sharp_n(a, f, d0_per_a, closed_C(delta))
    a = np.atleast_1d(np.asarray(a, dtype=float))
    if a.size <= 8:
        return np.array([_n_cross(float(d0_per_a * x * (1.0 - f)), float(f * x), float(delta))
                         for x in a])
    knots = np.geomspace(a.min(), a.max(), 30)
    nk = np.array([_n_cross(float(d0_per_a * x * (1.0 - f)), float(f * x), float(delta),
                            exact=False) for x in knots])
    return np.exp(np.interp(np.log(a), np.log(knots), np.log(nk)))


def _macros(D, rows, levels, sizes, d0_per_a, C, delta, threshold="paper", tables=None):
    """Numbers quoted in the text, generated rather than transcribed."""
    pos = [a for a in levels if a > 0]
    cov = float(np.mean([r["covers"] for r in rows]))
    big = max(sizes)
    lines = ["% Generated by scripts/fig2.py -- do not edit."]

    def add(k, v):
        lines.append(rf"\newcommand{{\{k}}}{{{v}}}")

    add("rgSep", f"{D.get('sep', 3.0):g}")
    add("rgLevels", ", ".join(f"{a:g}" for a in pos))
    add("rgNmin", f"{min(sizes):,}")
    add("rgNmax", f"{big:,}")
    add("rgReps", f"{len(rows):,}")
    add("rgDzeroPerA", f"{d0_per_a:.2f}")
    add("rgCoverMiss", f"{100 * (1 - cov):.0f}")
    add("rgDelta", f"{delta:g}")
    # LaTeX macro names may contain letters only, so a level such as 0.15
    # becomes the tag "OneFive" rather than "15".
    DIGIT = ["Zero", "One", "Two", "Three", "Four",
             "Five", "Six", "Seven", "Eight", "Nine"]
    recover = {"cert": [], "ana": []}
    for a in pos:
        tag = "".join(DIGIT[int(c)] for c in f"{int(round(100 * a)):02d}")
        add(f"rgFire{tag}", f"{_fire_n(a, d0_per_a, C):,.0f}")
        if threshold == "paper":
            add(f"rgHalf{tag}", f"{_sharp_n(np.array([a]), 0.5, d0_per_a, C)[0]:,.0f}")
            add(f"rgSharp{tag}", f"{_sharp_n(np.array([a]), 0.9, d0_per_a, C)[0]:,.0f}")
        else:
            add(f"rgHalf{tag}", f"{_sharp_n_v2(np.array([a]), 0.5, d0_per_a, delta, threshold)[0]:,.0f}")
            add(f"rgSharp{tag}", f"{_sharp_n_v2(np.array([a]), 0.9, d0_per_a, delta, threshold)[0]:,.0f}")
        rr = [r for r in rows if r["alpha_star"] == a and r["n"] == big]
        if not rr:
            rr = [r for r in rows if r["alpha_star"] == a
                  and r["n"] == max(x["n"] for x in rows if x["alpha_star"] == a)]
        add(f"rgAhat{tag}", f"{np.mean([r['alpha_hat'] for r in rr]):.3f}")
        add(f"rgCert{tag}", f"{np.mean([r['certified'] for r in rr]):.3f}")
        # revision v2, --certificate all: rgCert is the sequential certificate; the
        # single-split betting and the bootstrap columns get their own macros
        for col, name in (("certified_betting", "rgCertOneSplit"),
                          ("certified_bootstrap", "rgCertBoot")):
            vals = [r[col] for r in rr if r.get(col) is not None]
            if vals:
                add(f"{name}{tag}", f"{np.mean(vals):.3f}")
        add(f"rgNat{tag}", f"{rr[0]['n']:,}")
        rec = [r["recall"] for r in rr if r.get("recall") is not None]
        if rec:
            add(f"rgRecall{tag}", f"{np.mean(rec):.2f}")
        recover["cert"].append(100 * np.mean([r["certified"] for r in rr]) / a)
        recover["ana"].append(100 * np.mean([r["alpha_hat"] for r in rr]) / a)
    # Section 6 (06_splitting.tex): "where TV(P,Q) is known, the certificate
    # recovers Lo-Hi% of it at sample sizes where the analytic estimator
    # recovers Lo-Hi%": the share of alpha* = TV(P,Q) returned, on average,
    # at the largest simulated size of each level (the cells of \rgCert<tag>,
    # \rgAhat<tag>), smallest and largest over the levels.
    add("rgCertRecoverLo", f"{min(recover['cert']):.0f}")
    add("rgCertRecoverHi", f"{max(recover['cert']):.0f}")
    add("rgAnaRecoverLo", f"{min(recover['ana']):.0f}")
    add("rgAnaRecoverHi", f"{max(recover['ana']):.0f}")
    tables = (ROOT / "results" / "tables") if tables is None else Path(tables)
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "regime_macros.tex").write_text(
        "\n".join(lines) + "\n")
    print(f"  wrote {tables / 'regime_macros.tex'}")


def cert_label(D):
    """Legend name of the certificate stored in ``certified`` (v1: the bootstrap)."""
    cert = D.get("certificate", "bootstrap")
    if cert in ("sequential", "all"):
        return "sequential betting certificate"
    if cert in ("betting", "both"):
        return "split-sample betting certificate"
    return "split-sample certificate"


def resolve_threshold(D, threshold):
    """The threshold the stored decisions were made with; a different request is an error."""
    stored = D.get("threshold", "paper")
    if threshold is None:
        return stored
    if threshold != stored:
        raise SystemExit(f"fig2_data.json was made with threshold={stored!r}, "
                         f"not {threshold!r}; rerun export_fig2_data.py first")
    return threshold


SMALL = "fig2_small.json"


def _same_design(D, E, name):
    """``E`` (an extra run merged into fig2_data.json) must use the same design and methods."""
    problems = []
    if E.get("certificate", "bootstrap") != D.get("certificate", "bootstrap"):
        problems.append(f"certificate {E.get('certificate', 'bootstrap')!r} vs "
                        f"{D.get('certificate', 'bootstrap')!r}")
    for k in ("delta", "sep"):
        if not math.isclose(float(E.get(k, 0.0)), float(D.get(k, 0.0))):
            problems.append(f"{k} {E.get(k)} vs {D.get(k)}")
    if len(E["grid"]) != len(D["grid"]) or not np.allclose(E["grid"], D["grid"]):
        problems.append("grid differs")
    if problems:
        raise SystemExit(f"{name} cannot be merged with fig2_data.json: " + "; ".join(problems))


def load_rows(FIGD, threshold=None, require_null=False):
    """The regime-study rows: fig2_data.json, plus fig2_null.json and fig2_small.json when present.

    Returns ``(D, rows, threshold)`` with ``D`` the fig2_data.json meta.
    The null (alpha* = 0, nothing to find) is run separately, because it
    needs no contamination level and no recall; merged so panel 3 can show it
    beside the four contaminated levels (``require_null``: its absence is an
    error). ``fig2_small.json`` (revision v2) adds sizes below the v1 window.
    """
    FIGD = Path(FIGD)
    D = json.loads((FIGD / "fig2_data.json").read_text())
    threshold = resolve_threshold(D, threshold)
    rows = list(D["rows"])
    null_path = FIGD / "fig2_null.json"
    if null_path.exists():
        Dn = json.loads(null_path.read_text())
        resolve_threshold(Dn, threshold)
        rows += Dn["rows"]
    elif require_null:
        raise SystemExit(f"missing {null_path} (run export_fig2_data.py for the null cells)")
    small_path = FIGD / SMALL
    if small_path.exists():
        Ds = json.loads(small_path.read_text())
        resolve_threshold(Ds, threshold)
        _same_design(D, Ds, small_path)
        have = {(r["alpha_star"], r["n"]) for r in rows}
        clash = sorted(have & {(r["alpha_star"], r["n"]) for r in Ds["rows"]})
        if clash:
            raise SystemExit(f"{small_path} repeats cells of fig2_data.json: {clash[:4]}")
        rows += Ds["rows"]
        print(f"  merged {len(Ds['rows'])} rows of {small_path.name} "
              f"(sizes {sorted({r['n'] for r in Ds['rows']})})")
    return D, rows, threshold


def build(fig_dir=None, tables_dir=None, threshold=None):
    FIGD = FIG if fig_dir is None else Path(fig_dir)
    D, rows, threshold = load_rows(FIGD, threshold, require_null=fig_dir is not None)
    levels = sorted({r["alpha_star"] for r in rows})
    sizes = sorted({r["n"] for r in rows})
    grid = np.array(D["grid"])
    delta = D["delta"]
    C = closed_C(delta, threshold)

    def sharp(aa, f):
        return (_sharp_n(aa, f, d0_per_a, C) if threshold == "paper"
                else _sharp_n_v2(aa, f, d0_per_a, delta, threshold))
    # Signal per unit contamination, measured rather than assumed. The
    # null cells carry alpha* = 0 and are excluded: there is no signal to
    # divide by there, and they say nothing about the ratio.
    d0_per_a = float(np.mean([r["d_hat_0"] / r["alpha_star"] for r in rows
                              if r["alpha_star"] > 0]))

    fig = plt.figure(figsize=(6.3, 9.3))
    bg = fig.add_axes([0, 0, 1, 1], zorder=-10)
    bg.set_xlim(0, 1); bg.set_ylim(0, 1); bg.axis("off")
    banner(bg, "Where it works and where it does not: contamination known exactly",
           fontsize=8.4)

    axs, frs = panels(fig, nrow=3, ncol=2)
    a1, a2, a3, a4, a5, a6 = axs
    f1, f2, f3, f4, f5, f6 = frs

    # ---- 1. power curves at one level, several n ------------------------
    pos_levels = [a for a in levels if a > 0]
    a_show = pos_levels[len(pos_levels) // 2]
    # Colour follows the level, never the loop index: merging the null cells
    # changes how many series a panel draws, and a filter that repaints the
    # survivors would make the panels disagree with each other.
    CMAP = {a: LEVEL_COLOR[i % len(LEVEL_COLOR)]
            for i, a in enumerate(pos_levels)}
    MMAP = {a: LEVEL_MARK[i % len(LEVEL_MARK)]
            for i, a in enumerate(pos_levels)}
    ns_show = [n for n in sizes if n >= 1000][:4]
    for i, n in enumerate(ns_show):
        rr = [r for r in rows if r["alpha_star"] == a_show and r["n"] == n]
        if not rr:
            continue
        pw = np.mean([r["reject"] for r in rr], axis=0)
        a1.plot(grid, pw, "-", lw=1.5, color=LEVEL_COLOR[i % len(LEVEL_COLOR)],
                marker=LEVEL_MARK[i % len(LEVEL_MARK)], ms=2.2,
                label=f"$n={n:,}$")
    a1.axvline(a_show, color=INK, lw=1.1, ls="--")
    a1.axhline(delta, color=MUTED, lw=0.9, ls=":")
    a1.text(a_show, 1.02, r" $\alpha^\star$", fontsize=5.6, color=INK,
            transform=a1.get_xaxis_transform())
    a1.set_xlabel(r"level $\alpha$ of the null $\mathrm{TV}(P,Q)\leq\alpha$",
                  labelpad=1.5)
    a1.set_ylabel("rejection rate", labelpad=1.5)
    a1.set_ylim(-0.04, 1.04)
    reach = []
    for n in ns_show:
        rr = [r for r in rows if r["alpha_star"] == a_show and r["n"] == n]
        if rr:
            pw = np.mean([r["reject"] for r in rr], axis=0)
            reach.append(grid[pw >= 0.5].max() if (pw >= 0.5).any() else 0.0)
    note(bg, f1,
        rf"the null holds from $\alpha^\star={a_show}$ rightward, where the rate is $0$, inside $\delta={delta}$. But power reaches only $\alpha\approx{max(reach):.2f}$ at $n={max(ns_show):,}$, not $\alpha^\star$: closing that gap is boundaries two and three of panel 2.")
    head(bg, f1, 1, "The test acquires power",
         r"and keeps its level on the other side of $\alpha^\star$")

    # ---- 2. the whole field, simulated where we can and computed elsewhere
    aa = np.geomspace(0.02, 0.5, 200)
    nf = _fire_n(aa, d0_per_a, C)
    n50 = sharp(aa, 0.5)
    n90 = sharp(aa, 0.9)
    y_lo = min(1e2, min(sizes) / 1.6)   # v2 (fig2_small.json, n = 100): room under the lowest row
    a2.fill_between(aa, y_lo, nf, color=MUTED, alpha=0.13, lw=0)
    a2.fill_between(aa, nf, n50, color=BLUE, alpha=0.10, lw=0)
    a2.fill_between(aa, n50, n90, color=ORANGE, alpha=0.10, lw=0)
    a2.fill_between(aa, n90, 1e8, color=VIOLET, alpha=0.10, lw=0)
    a2.plot(aa, nf, "-", lw=1.4, color=BLUE, label=r"test fires, $(2C/D_0)^2$")
    a2.plot(aa, n50, "-", lw=1.3, color=ORANGE,
            label=r"$\widehat\alpha$ reaches $\frac{1}{2}\alpha^\star$")
    a2.plot(aa, n90, "-", lw=1.2, color=VIOLET,
            label=r"$\widehat\alpha$ reaches $0.9\,\alpha^\star$")
    # what this simulation actually covers
    lo_a, hi_a = min(pos_levels), max(pos_levels)
    lo_n, hi_n = min(sizes), max(sizes)
    a2.add_patch(Rectangle((lo_a, lo_n), hi_a - lo_a, hi_n - lo_n,
                           fill=False, ec=INK, lw=1.1, ls="--", zorder=5))
    if lo_n * 0.62 > y_lo:  # under the box, inside the axes
        a2.text(hi_a, lo_n * 0.62, "simulated here", fontsize=5.1, color=INK,
                ha="right", va="top", zorder=6)
    else:                   # the box starts at the bottom of the axes (v2: n from 100)
        a2.text(hi_a, hi_n * 1.25, "simulated here", fontsize=5.1, color=INK,
                ha="right", va="bottom", zorder=6)
    for a in pos_levels:
        for n in sizes:
            rr = [r for r in rows if r["alpha_star"] == a and r["n"] == n]
            if not rr:
                continue
            fired = np.mean([r["fires"] for r in rr])
            a2.plot([a], [n], marker="o" if fired > 0.5 else "x", ms=3.0,
                    mew=0.9, color=INK if fired > 0.5 else MUTED, zorder=7)
    try:
        F1 = json.loads((FIGD / "fig1_data.json").read_text())
        a2.plot([F1["cert_median"]], [min(F1["n"], F1["m"])], marker="*",
                ms=9, color="#b3261e", zorder=8, lw=0,
                label="the RHC study of Figure 1")
    except Exception:
        pass
    a2.set_xscale("log"); a2.set_yscale("log")
    a2.set_xlim(0.02, 0.5); a2.set_ylim(y_lo, 1e7)
    a2.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5])
    a2.set_xticklabels(["0.02", "0.05", "0.1", "0.2", "0.5"])
    a2.minorticks_off()
    a2.set_xlabel(r"contamination level $\alpha^\star$", labelpad=1.5)
    a2.set_ylabel("sample size per arm (log)", labelpad=1.5)
    for txt, yy, col in (
            (r"analytic route silent", 0.30, INK2),
            (r"fires, but $\widehat\alpha\ll\alpha^\star$", 0.60, BLUE),
            (r"$\widehat\alpha$ within a factor 2", 0.80, ORANGE),
            (r"$\widehat\alpha$ sharp", 0.94, VIOLET)):
        a2.text(0.023, yy, txt, transform=a2.transAxes, fontsize=5.0,
                color=col, ha="left", va="center", zorder=9)
    note(bg, f2,
        "circles fired, crosses did not. Everything above the dashed box is computed, not run: the sharp regime is out of reach of any simulation we can afford.")
    head(bg, f2, 2, "The whole field, not a chosen corner",
         r"three computed boundaries; we verify the lowest")

    # ---- 3. the first boundary, verified -------------------------------
    for a in pos_levels:
        xs, m, lo, hi = _by(rows, "fires", levels, sizes)[a]
        a3.plot(xs, m, "-", lw=1.5, color=CMAP[a],
                marker=MMAP[a], ms=2.8, label=rf"$\alpha^\star={a}$")
        a3.axvline(_fire_n(a, d0_per_a, C), color=CMAP[a], lw=1.0,
                   ls=(0, (3, 2)), alpha=0.8)
    if 0.0 in levels:
        xs, m, _, _ = _by(rows, "fires", levels, sizes)[0.0]
        a3.plot(xs, m, "-", lw=1.5, color=INK, marker="v", ms=2.8,
                label=r"$\alpha^\star=0$ (nothing to find)")
    a3.set_xscale("log"); a3.set_ylim(-0.04, 1.04)
    a3.set_xlabel("sample size per arm (log)", labelpad=1.5)
    a3.set_ylabel(r"rate of firing at $\alpha=0$", labelpad=1.5)
    note(bg, f3,
        "dashed verticals are panel 2's first boundary, computed, not fitted.")
    head(bg, f3, 3, "The first boundary holds",
         "four levels, four predicted crossings, and a null")

    # ---- 4. alpha-hat against the certificate, same axes ----------------
    ah = _by(rows, "alpha_hat", levels, sizes)
    ce = _by(rows, "certified", levels, sizes)
    for a in pos_levels:
        xs, m, lo, hi = ah[a]
        a4.plot(xs, m, "-", lw=1.5, color=CMAP[a],
                marker=MMAP[a], ms=2.6)
        xs2, m2, _, _ = ce[a]
        if xs2.size:
            a4.plot(xs2, m2, "--", lw=1.3, color=CMAP[a], alpha=0.85)
        a4.axhline(a, color=CMAP[a], lw=0.8, ls=":", alpha=0.7)
    a4.plot([], [], "-", color=INK2, lw=1.5, label=r"analytic $\widehat\alpha$")
    a4.plot([], [], "--", color=INK2, lw=1.3, label=cert_label(D))
    a4.plot([], [], ":", color=INK2, lw=0.8, label=r"the truth $\alpha^\star$")
    a4.set_xscale("log")
    a4.set_xlabel("sample size per arm (log)", labelpad=1.5)
    a4.set_ylabel(r"bound returned on $\mathrm{TV}(P,Q)$", labelpad=1.5)
    cov = float(np.mean([r["covers"] for r in rows]))
    note(bg, f4,
        rf"both stay under $\alpha^\star$: {1 - cov:.0%} of {len(rows)} replicates crossed, and $\delta={delta}$ is allowed.")
    head(bg, f4, 4, "Sharpness is a different sample size",
         r"firing is boundary one; reaching $\alpha^\star$ is boundary three")

    # ---- 5. the trimming finds the contamination ------------------------
    agg = _by(rows, "recall", levels, sizes)
    for a in pos_levels:
        xs, m, lo, hi = agg[a]
        if xs.size == 0:
            continue
        a5.plot(xs, m, "-", lw=1.5, color=CMAP[a], marker=MMAP[a],
                ms=2.8, label=rf"$\alpha^\star={a}$")
        a5.fill_between(xs, lo, hi, color=CMAP[a], alpha=0.13, lw=0)
    a5.set_xscale("log"); a5.set_ylim(0, 1.04)
    a5.set_xlabel("sample size per arm (log)", labelpad=1.5)
    a5.set_ylabel("discarded that were\nplanted (share)", labelpad=1.5)
    note(bg, f5,
        "the witness is fitted on one half and applied to the other, so this is out-of-sample. It cannot reach $1$: a contaminant inside the core is unidentifiable.")
    head(bg, f5, 5, "And the trimming discards the right points",
         r"out-of-sample, at the true level $\alpha=\alpha^\star$")

    # ---- 6. what it cost ------------------------------------------------
    for a in pos_levels:
        xs, m, lo, hi = _by(rows, "path_seconds", levels, sizes)[a]
        a6.plot(xs, m, "-", lw=1.4, color=CMAP[a], marker=MMAP[a],
                ms=2.6, label=rf"$\alpha^\star={a}$")
    xs = np.array(sizes, dtype=float)
    ref = xs / xs[0] * max(1e-9, np.mean([r["path_seconds"] for r in rows
                                          if r["n"] == sizes[0]]))
    a6.plot(xs, ref, ls="--", lw=1.0, color=MUTED, label="linear in $n$")
    a6.set_xscale("log"); a6.set_yscale("log")
    a6.set_xlabel("sample size per arm (log)", labelpad=1.5)
    a6.set_ylabel("seconds per replicate (log)", labelpad=1.5)
    n_rff = min([r["n"] for r in rows if r["rff"]], default=None)
    if n_rff:
        a6.axvline(n_rff, color=INK, lw=0.9, ls=":")
        a6.text(n_rff, 0.04, " random features\n from here",
                fontsize=5.0, color=INK2, transform=a6.get_xaxis_transform())
    note(bg, f6,
        f"{len(grid)} levels per path, warm-started. The exact kernel is quadratic in $n$; random features bring it back to linear.")
    head(bg, f6, 6, "What the whole path costs",
         "one convex program per level, on one core")

    for ax, fr, nc in ((a1, f1, 4), (a2, f2, 2), (a3, f3, 3),
                       (a4, f4, 3), (a5, f5, 4), (a6, f6, 3)):
        legend(fig, ax, fr, ncol=nc)

    # The validity sentence states what the replicates show, so it is computed.
    type1 = max([float(np.mean([r["reject"][j] for r in rows
                                if r["alpha_star"] == a and r["n"] == n]))
                 for a in levels for n in sizes
                 if any(r["alpha_star"] == a and r["n"] == n for r in rows)
                 for j in range(len(grid)) if grid[j] >= a - 1e-12] or [0.0])
    if type1 == 0.0 and cov == 1.0:
        valid_txt = (r"Validity: the rejection rate at and above $\alpha^\star$ "
                     r"stays at $0$ in every cell, and $\widehat\alpha\leq\alpha^\star$ "
                     r"in every replicate -- Theorem 12(i) only asks for $1-\delta$.")
    else:
        valid_txt = (rf"Validity: the largest rejection rate at and above $\alpha^\star$ "
                     rf"in any cell is ${type1:.2f}$, and $\widehat\alpha\leq\alpha^\star$ "
                     rf"in {cov:.0%} of the replicates; the guarantee is $1-\delta$.")
    ledger(bg, "What the simulation shows, and what it cost", [
        (BLUE, valid_txt),
        (ORANGE, r"Detection: the test fires at the sample size the closed form "
                 r"$(2C/D_0)^2$ predicts, independently at four contamination "
                 r"levels; power then rises quickly with $n$."),
        (VIOLET, r"Recovery: at the true level the trimming discards planted "
                 r"contamination out-of-sample, and the certificate is already "
                 r"near $\alpha^\star$ where the analytic route is silent."),
        (INK2, r"Honesty: panel 2 is the whole field; only its lower bands were "
               r"simulated. The RHC study of Figure 1 sits in the silent one."),
    ])
    provenance(bg, "trimmedmmd + scripts/export_fig2_data.py, scripts/fig2.py")

    _macros(D, rows, levels, sizes, d0_per_a, C, delta, threshold=threshold,
            tables=tables_dir)

    FIGD.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGD / "fig2_regimes.pdf")
    fig.savefig(FIGD / "fig2_regimes.png", dpi=240)
    plt.close(fig)
    print(f"wrote {FIGD}/fig2_regimes.{{pdf,png}}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", choices=["paper", "plugin"], default=None,
                    help="default: the threshold recorded in fig2_data.json")
    ap.add_argument("--outdir", default=None,
                    help="read <outdir>/figures, write <outdir>/figures and "
                         "<outdir>/tables/regime_macros.tex (v2: results/v2)")
    a = ap.parse_args(argv)
    if a.outdir is None:
        build(threshold=a.threshold)
        return
    root = Path(a.outdir)
    if not root.is_absolute():
        root = ROOT / root
    build(fig_dir=root / "figures", tables_dir=root / "tables", threshold=a.threshold)


if __name__ == "__main__":
    main()

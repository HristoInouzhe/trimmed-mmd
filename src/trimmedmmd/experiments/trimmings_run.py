"""Experiment E9: the structure of the recovered trimming.

Three claims of Section "Recovering the trimming" are simulable, and each
can come out wrong here:

* **bang-bang** (Proposition on structure, part ii). Below the contact level
  the optimal weights should be at the cap or at zero, with at most one
  fractional exception. A large fractional count below alpha-star would
  falsify it.
* **collapse above the contact level** (part iii). At alpha > alpha-star the
  minimizer should stop being *identified*. The direct test is to solve the
  same problem from two different random feasible starting points: below
  alpha-star the two runs must agree on which observations to discard, and
  above it they may attain the same objective value while discarding
  different ones. Agreement at every level would falsify the proposition.
  The weights smearing out is the same phenomenon seen through the
  fractional count.
* **the two geometries agree on the transition but not on the subsample.**
  Both the kernel and the partial-transport trimming should be bang-bang
  below alpha-star; they should *not* discard the same observations, since
  one thresholds a witness function and the other a reduced cost.

A fourth, softer quantity is reported and is the one an applied reader
cares about: *recall*, the fraction of the observations the trimming
discards that were in fact the planted contaminants. It cannot reach 1 --- a
contaminant drawn where the two components overlap is not identifiable from
the data --- so the number is descriptive, not a target.

Run: ``python -m trimmedmmd.experiments.trimmings_run``
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..trimmings import (HAVE_POT, optimal_trimming_mmd,
                         optimal_trimming_wasserstein)

RESULTS = Path(__file__).resolve().parents[3] / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"


def two_sided_contamination(n, m, alpha_star, rng):
    """X = (1-a)N(-1,1) + a N(2,1);  Y = (1-a)N(-1,1) + a N(-4,1).

    The shared core is N(-1,1) and TV(P,Q) = alpha_star by construction, up
    to the negligible overlap of the two contaminating components. Planted
    contaminants are the trailing indices, which is what makes recall
    measurable.
    """
    nc, mc = int(round(alpha_star * n)), int(round(alpha_star * m))
    X = np.concatenate([rng.normal(-1, 1, n - nc), rng.normal(2, 1, nc)])
    Y = np.concatenate([rng.normal(-1, 1, m - mc), rng.normal(-4, 1, mc)])
    return X[:, None], Y[:, None], np.arange(n - nc, n), np.arange(m - mc, m)


def structure_table(n=400, m=400, alpha_star=0.10, seed=20260921,
                    alphas=(0.0, 0.02, 0.04, 0.06, 0.08, 0.10,
                            0.12, 0.15, 0.20, 0.30)):
    rng = np.random.default_rng(seed)
    X, Y, cx, cy = two_sided_contamination(n, m, alpha_star, rng)
    rows = []
    for a in alphas:
        for geom in ("mmd", "wasserstein"):
            if geom == "mmd":
                # select="none": the structure claims are about what the bare
                # QP returns. The least-trimming selection is a *choice* among
                # minimizers and would confound the diagnostic it responds to.
                t = optimal_trimming_mmd(X, Y, a, rng=np.random.default_rng(0),
                                         select="none")
            else:
                if a == 0.0:          # no trimming: the plan is the plain OT
                    t = optimal_trimming_wasserstein(X, Y, 1e-12)
                else:
                    t = optimal_trimming_wasserstein(X, Y, a)
            dx, dy = t.dropped("x"), t.dropped("y")
            rows.append({
                "geometry": geom, "alpha": a, "value": t.value,
                "kept_x": int(t.kept_x.size), "kept_y": int(t.kept_y.size),
                "frac_x": t.fractional_count("x"),
                "frac_y": t.fractional_count("y"),
                "frac_share_x": t.fractional_count("x") / max(1, t.kept_x.size),
                "ess_x": t.ess("x"), "ess_y": t.ess("y"),
                "budget_used_x": t.budget_used("x"),
                "recall_x": float(np.isin(dx, cx).mean()) if dx.size else np.nan,
                "recall_y": float(np.isin(dy, cy).mean()) if dy.size else np.nan,
                "unique": t.unique,
            })
            print(f"  {geom:12s} a={a:.2f} D={t.value:.5f} "
                  f"kept={rows[-1]['kept_x']} frac={rows[-1]['frac_x']} "
                  f"recall={rows[-1]['recall_x']}")
    return rows, X, Y, cx, cy


def identifiability(X, Y, alphas, seed=11):
    """Two starts, same problem: do they discard the same observations?

    A direct test of non-uniqueness, and the one that does not depend on a
    tolerance. Each run starts from an independent random feasible point, so
    if the minimizer is identified the two must coincide; if it is not, they
    can differ while attaining the same value.
    """
    from ..kernels import gaussian_kernel, median_heuristic
    from ..simplex import project_capped_simplex
    from ..solver import solve_gram

    rng = np.random.default_rng(seed)
    sigma = median_heuristic(X, Y, rng=np.random.default_rng(0))
    Kxx = gaussian_kernel(X, X, sigma)
    Kyy = gaussian_kernel(Y, Y, sigma)
    Kxy = gaussian_kernel(X, Y, sigma)
    n, m = X.shape[0], Y.shape[0]
    out = []
    for a in alphas:
        cw, cv = 1.0 / (n * (1 - a)), 1.0 / (m * (1 - a))
        res = []
        for _ in range(2):
            # Start at a VERTEX of the capped simplex chosen by a random
            # direction, not at a near-uniform point: projecting uniform
            # noise lands every start in the middle of the polytope, where
            # they would agree for trivial reasons.
            w0 = project_capped_simplex(
                1e6 * rng.standard_normal(n), cw)
            v0 = project_capped_simplex(
                1e6 * rng.standard_normal(m), cv)
            res.append(solve_gram(Kxx, Kyy, Kxy, a, w0=w0, v0=v0,
                                  tol=1e-12, max_iter=8000,
                                  rng=np.random.default_rng(0)))
        d = [set(np.nonzero(r.w <= 1e-10 * cw)[0].tolist()) for r in res]
        union = d[0] | d[1]
        jac = len(d[0] & d[1]) / len(union) if union else 1.0
        out.append({"alpha": a,
                    "value_1": res[0].discrepancy,
                    "value_2": res[1].discrepancy,
                    "value_gap": abs(res[0].discrepancy - res[1].discrepancy),
                    "dropped_1": len(d[0]), "dropped_2": len(d[1]),
                    "jaccard_two_starts": jac})
        print(f"  identifiability a={a:.2f} D=({res[0].discrepancy:.5f},"
              f"{res[1].discrepancy:.5f}) jaccard={jac:.3f}")
    return out


def overlap(n=400, m=400, alpha_star=0.10, seed=20260921, alpha_read=0.08):
    """How much do the two geometries agree on *which* points to drop?"""
    rng = np.random.default_rng(seed)
    X, Y, _, _ = two_sided_contamination(n, m, alpha_star, rng)
    tk = optimal_trimming_mmd(X, Y, alpha_read, rng=np.random.default_rng(0),
                              select="none")
    tw = optimal_trimming_wasserstein(X, Y, alpha_read)
    dk, dw = set(tk.dropped("x").tolist()), set(tw.dropped("x").tolist())
    jac = len(dk & dw) / max(1, len(dk | dw))
    return {"alpha_read": alpha_read, "n_dropped_mmd": len(dk),
            "n_dropped_w2": len(dw), "n_common": len(dk & dw),
            "jaccard": jac}


def figure(rows, ident, X, Y, cx, cy, alpha_star=0.10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    BLUE, ORANGE, GREY = "#2b5d8a", "#d1701a", "#666666"
    grid = np.linspace(-8, 6, 600)
    rng0 = np.random.default_rng(0)

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))

    def wkde(Z, w):
        return gaussian_kde(Z.ravel(), weights=w)(grid)

    for k, (geom, solve) in enumerate([
            ("kernel (trimmed MMD)",
             lambda a: optimal_trimming_mmd(X, Y, a, rng=rng0,
                                            select="none")),
            ("transport (trimmed $W_2$)",
             lambda a: optimal_trimming_wasserstein(X, Y, a))]):
        A = ax[k]
        A.fill_between(grid, wkde(X, np.full(X.shape[0], 1 / X.shape[0])),
                       color=GREY, alpha=.18, lw=0)
        A.fill_between(grid, wkde(Y, np.full(Y.shape[0], 1 / Y.shape[0])),
                       color=GREY, alpha=.18, lw=0)
        A.plot(grid, wkde(X, np.full(X.shape[0], 1 / X.shape[0])),
               color="k", lw=1.6, label="$P_n$ (untrimmed)")
        A.plot(grid, wkde(Y, np.full(Y.shape[0], 1 / Y.shape[0])),
               color="k", lw=1.6, ls="--", label="$Q_m$ (untrimmed)")
        t = solve(alpha_star)
        A.plot(grid, wkde(X, t.w), color=BLUE, lw=1.8,
               label=rf"trimmed $P_n$, $\alpha={alpha_star}$")
        A.plot(grid, wkde(Y, t.v), color=ORANGE, lw=1.8, ls="--",
               label=rf"trimmed $Q_m$, $\alpha={alpha_star}$")
        A.set_title(geom, fontsize=10)
        A.set_xlabel("$x$")
        A.legend(fontsize=7, frameon=False)
    ax[0].set_ylabel("density")

    A = ax[2]
    for geom, col, mk in [("mmd", BLUE, "o"), ("wasserstein", ORANGE, "s")]:
        r = [x for x in rows if x["geometry"] == geom]
        A.plot([x["alpha"] for x in r], [x["frac_share_x"] for x in r],
               mk + "-", color=col, label=geom)
    A.plot([x["alpha"] for x in ident],
           [1 - x["jaccard_two_starts"] for x in ident], "^--", color=GREY,
           label="kernel, disagreement of two starts")
    A.axvline(alpha_star, color="k", ls=":", lw=1)
    A.text(alpha_star, .92, r" $\alpha^\star$", fontsize=9,
           transform=A.get_xaxis_transform())
    A.set_xlabel(r"$\alpha$")
    A.set_ylabel("fraction")
    A.set_title("bang-bang below $\\alpha^\\star$, collapse above", fontsize=10)
    A.legend(fontsize=7, frameon=False)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / "f11_trimmings.pdf")
    fig.savefig(FIGURES / "f11_trimmings.png", dpi=150)
    plt.close(fig)
    print("  wrote f11_trimmings")


def macros(rows, ov, ident, alpha_star):
    def get(geom, a):
        return next(x for x in rows
                    if x["geometry"] == geom and abs(x["alpha"] - a) < 1e-9)
    below, above = get("mmd", 0.08), get("mmd", 0.20)
    wbelow = get("wasserstein", 0.08)
    lines = [
        "% Generated by trimmedmmd.experiments.trimmings_run -- do not edit.",
        rf"\newcommand{{\trimFracBelow}}{{{below['frac_x']}}}",
        rf"\newcommand{{\trimFracAbove}}{{{above['frac_x']}}}",
        rf"\newcommand{{\trimFracShareAbove}}{{{above['frac_share_x']:.2f}}}",
        rf"\newcommand{{\trimWFracBelow}}{{{wbelow['frac_x']}}}",
        rf"\newcommand{{\trimRecall}}{{{below['recall_x']:.2f}}}",
        rf"\newcommand{{\trimWRecall}}{{{wbelow['recall_x']:.2f}}}",
        rf"\newcommand{{\trimJaccard}}{{{ov['jaccard']:.2f}}}",
        rf"\newcommand{{\trimAlphaStar}}{{{alpha_star:.2f}}}",
        rf"\newcommand{{\trimBudgetAbove}}{{{above['budget_used_x']:.2f}}}",
        rf"\newcommand{{\trimJacTwoBelow}}"
        rf"{{{min(x['jaccard_two_starts'] for x in ident if x['alpha'] <= 0.08):.2f}}}",
        rf"\newcommand{{\trimJacTwoAbove}}"
        rf"{{{min(x['jaccard_two_starts'] for x in ident if x['alpha'] >= 0.20):.2f}}}",
        rf"\newcommand{{\trimValGapAbove}}"
        rf"{{{max(x['value_gap'] for x in ident if x['alpha'] >= 0.20):.1e}}}",
    ]
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "trim_macros.tex").write_text("\n".join(lines) + "\n")
    print("  wrote trim_macros.tex")


def main():
    print(f"E9 trimming structure (POT available: {HAVE_POT})")
    alpha_star = 0.10
    rows, X, Y, cx, cy = structure_table(alpha_star=alpha_star)
    ov = overlap(alpha_star=alpha_star)
    print("  overlap:", ov)
    ident = identifiability(X, Y, [0.02, 0.06, 0.08, 0.12, 0.20, 0.30])

    TABLES.mkdir(parents=True, exist_ok=True)
    import csv
    with open(TABLES / "e9_trimming_structure.csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    (TABLES / "e9_overlap.json").write_text(
        json.dumps({"overlap": ov, "identifiability": ident}, indent=2))
    figure(rows, ident, X, Y, cx, cy, alpha_star)
    macros(rows, ov, ident, alpha_star)


if __name__ == "__main__":
    main()

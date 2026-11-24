"""Figures from the experiment tables."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS = Path(__file__).resolve().parents[3] / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"

# Revision v2: callers that only import this module and call a figure function
# (jmlr/tools/print_figs.py draws f1, f8 and f10 = Figure 4 this way) read the
# v2 tables when TRIMMEDMMD_OUTDIR is set, e.g. TRIMMEDMMD_OUTDIR=results/v2.
# Unset (the default) nothing changes.
import os as _os  # noqa: E402

if _os.environ.get("TRIMMEDMMD_OUTDIR"):
    _od = Path(_os.environ["TRIMMEDMMD_OUTDIR"])
    _od = _od if _od.is_absolute() else RESULTS.parent / _od
    TABLES, FIGURES = _od / "tables", _od / "figures"

BLUE, ORANGE, GREY = "#2b5d8a", "#d1701a", "#666666"


def _is_plugin(df) -> bool:
    """Revision v2 tables carry a ``threshold`` column; v1 tables do not."""
    return ("threshold" in getattr(df, "columns", ())
            and str(df["threshold"].iloc[0]) == "plugin")


def _tau_label(df, short=False):
    # v2 naming of the paper (jmlr/sections/04_test.tex): tau_{n,m} is the minimum of the
    # plug-in thresholds and tau^A; v1's threshold is r_n(alpha, delta/2) + r_m(alpha, delta/2).
    if _is_plugin(df):
        return "plug-in" if short else r"threshold $\tau_{n,m}$"
    return "analytic" if short else r"analytic threshold $\tau$"


def _save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png", dpi=150)
    plt.close(fig)
    print(f"  wrote {name}")


def fig_conservativeness():
    df = pd.read_csv(TABLES / "e1_calibration.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    a = df["alpha"].to_numpy()
    ax[0].plot(a, df["analytic_tau"], "o-", color=ORANGE, label=_tau_label(df))
    if "tau_paper" in df.columns:
        ax[0].plot(a, df["tau_paper"], "o:", color=ORANGE, alpha=0.5,
                   label=r"v1 threshold $r_n+r_m$")
    ax[0].plot(a, df["q_dhat"], "s-", color=BLUE,
               label=r"true $1-\delta$ quantile of $\hat D_\alpha$")
    ax[0].plot(a, df["mean_dhat"], "^--", color=GREY, label=r"mean $\hat D_\alpha$")
    ax[0].axvline(df["alpha_star"].iloc[0], color="k", ls=":", lw=1)
    ax[0].set_yscale("log")
    ax[0].set_xlabel(r"$\alpha$")
    ax[0].set_ylabel("discrepancy scale")
    ax[0].set_title(f"Threshold vs. actual null spread (n={df['n'].iloc[0]})")
    ax[0].legend(fontsize=8)

    ax[1].plot(a, df["conservativeness"], "o-", color=ORANGE, label=_tau_label(df, True))
    if "conservativeness_paper" in df.columns:
        ax[1].plot(a, df["conservativeness_paper"], "o:", color=ORANGE, alpha=0.5,
                   label="v1")
        ax[1].legend(fontsize=8)
    ax[1].axvline(df["alpha_star"].iloc[0], color="k", ls=":", lw=1)
    ax[1].set_xlabel(r"$\alpha$")
    ax[1].set_ylabel(r"$\tau\ /\ q_{1-\delta}$")
    ax[1].set_title("Factor by which the bound overshoots")
    ax[1].set_yscale("log")
    _save(fig, "f1_conservativeness")


def fig_power():
    df = pd.read_csv(TABLES / "e1_power_vs_n.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(df["n"], df["mean_dhat"], "o-", color=BLUE,
               label=r"mean $\hat D_0$ (signal)")
    ax[0].plot(df["n"], df["analytic_tau"], "s-", color=ORANGE,
               label=(r"threshold $\tau_{\mathrm{new}}$" if _is_plugin(df)
                      else r"threshold $\tau$"))
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("n"); ax[0].set_ylabel("value")
    ax[0].set_title(r"Signal vs. threshold at $\alpha=0$")
    ax[0].legend(fontsize=8)

    ax[1].plot(df["n"], df["ratio_dhat_to_tau"], "o-", color=GREY)
    ax[1].axhline(1.0, color="k", ls="--", lw=1)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("n")
    ax[1].set_ylabel(r"mean $\hat D_0\ /\ \tau$")
    ax[1].set_title("Power requires this to exceed 1")
    _save(fig, "f2_power_vs_n")


def fig_alpha_hat():
    detail = json.loads((TABLES / "e2_detail.json").read_text())
    df = pd.read_csv(TABLES / "e2_alpha_hat.csv")
    astar = df["alpha_star"].iloc[0]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bins = np.linspace(0, 0.62, 32)
    ax.hist(detail["analytic"], bins=bins, alpha=0.65, color=ORANGE,
            label=("plug-in threshold" if detail.get("threshold") == "plugin"
                   else "analytic threshold"))
    ax.hist(detail["oracle"], bins=bins, alpha=0.65, color=BLUE,
            label="oracle-calibrated threshold")
    ax.axvline(astar, color="k", ls="--", lw=1.5, label=r"true $\alpha^\star$")
    ax.set_xlabel(r"$\hat\alpha$"); ax.set_ylabel("count")
    ax.set_title(r"Threshold-crossing estimator, $n=%d$" % df["n"].iloc[0])
    ax.legend(fontsize=8)
    _save(fig, "f3_alpha_hat")


def fig_margin():
    df = pd.read_csv(TABLES / "e3_margin.csv")
    fit = json.loads((TABLES / "e3_margin_fit.json").read_text())
    floor = fit["noise_floor"]
    e = df["eps"].to_numpy()
    d = df["D_alpha"].to_numpy()
    dc = d - floor

    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    ax.loglog(e, d, "o", color=GREY, mfc="none",
              label=r"raw $\bar{\hat D}_{\alpha^\star-\epsilon}$")
    ax.loglog(e[dc > 0], dc[dc > 0], "o", color=BLUE,
              label="after subtracting the floor")
    xs = np.linspace(e.min(), e.max(), 50)
    ax.loglog(xs, fit["c0_hat"] * xs ** fit["gamma_hat"], "-", color=ORANGE,
              label=fr"fit to corrected: $\gamma={fit['gamma_hat']:.2f}$")
    ax.loglog(xs, fit["c0_hat_all_points"] * xs ** fit["gamma_hat_all_points"],
              "--", color=GREY,
              label=fr"fit to raw: $\gamma={fit['gamma_hat_all_points']:.2f}$")
    ax.axhline(floor, color="k", ls=":", lw=1)
    ax.annotate("noise floor", xy=(e.max(), floor), xytext=(-6, 4),
                textcoords="offset points", ha="right", fontsize=8)
    ax.set_xlabel(r"$\epsilon=\alpha^\star-\alpha$")
    ax.set_ylabel(r"discrepancy")
    ax.set_title(f"Margin exponent (n={fit['n']}): the floor biases it downward")
    ax.legend(fontsize=8, loc="upper left")
    _save(fig, "f4_margin")


def fig_assumption():
    detail = json.loads((TABLES / "e4_detail.json").read_text())
    df = pd.read_csv(TABLES / "e4_assumption.csv")
    astar = df["alpha_star"].iloc[0]
    grid = np.array(detail["grid"])
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(grid, detail["gaussian"], "o-", color=BLUE,
            label="Gaussian kernel (characteristic)")
    ax.plot(grid, detail["linear"], "s-", color=ORANGE,
            label="linear kernel (NOT characteristic)")
    ax.axvline(astar, color="k", ls="--", lw=1.5, label=r"true $\alpha^\star$")
    ax.set_xlabel(r"$\alpha$"); ax.set_ylabel(r"mean $\hat D_{\alpha,k}$")
    ax.set_title("Removing ass:regularity(A3): mean-matched contamination")
    ax.legend(fontsize=8)
    _save(fig, "f5_assumption_violation")


def fig_scaling():
    df = pd.read_csv(TABLES / "e5_scaling.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(df["n"], df["seconds_exact_path"], "o-", color=BLUE,
               label="exact kernel")
    ax[0].plot(df["n"], df["seconds_rff_path"], "s-", color=ORANGE,
               label=f"random features (r={df['n_features'].iloc[0]})")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("n"); ax[0].set_ylabel("seconds for full $\\alpha$-path")
    ax[0].set_title("Cost of one warm-started path")
    ax[0].legend(fontsize=8)

    ax[1].plot(df["n"], df["max_abs_diff_exact_vs_rff"], "o-", color=GREY)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("n")
    ax[1].set_ylabel(r"$\max_\alpha|\hat D-\hat D^{(r)}|$")
    ax[1].set_title("Random-feature approximation error")
    _save(fig, "f6_scaling")


def fig_calibrations():
    df = pd.read_csv(TABLES / "e6_calibrations.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    for k, regime in enumerate(["boundary (type-I)", "alpha=0 (power)"]):
        sub = df[df["regime"] == regime]
        x = np.arange(len(sub)) * 5.0
        w = 0.8
        a = ax[k]
        a.bar(x - 1.5 * w, sub["analytic_tau"], w, color=ORANGE, label=_tau_label(df, True))
        a.bar(x - 0.5 * w, sub["mmd_screen_crit"], w, color="#7a9e3f",
              label="MMD screen")
        a.bar(x + 0.5 * w, sub["oracle_crit"], w, color=BLUE, label="oracle")
        a.bar(x + 1.5 * w, sub["mean_perm_crit"], w, color="#a03050",
              label="permutation")
        a.plot(x, sub["mean_dhat"], "k*", ms=13, label=r"statistic $\hat D$")
        a.set_yscale("log")
        a.set_xticks(x); a.set_xticklabels([f"n={int(v)}" for v in sub["n"]])
        a.set_ylabel("critical value (log scale)")
        rep = _reported_certificate(sub)
        if rep is not None and "rate_split" in sub.columns:
            # revision v2: the reported split certificate's rejection rate per n
            rates = ", ".join(f"{float(r):.2f}" for r in sub["rate_split"])
            a.set_title(f"{regime}\n{CERT_LABEL.get(rep, rep)} certificate rejects: {rates}",
                        fontsize=9)
        else:
            a.set_title(regime)
        if k == 0:
            a.legend(fontsize=7.5, ncol=2)
    fig.suptitle("Four critical values on identical data: a star above a bar "
                 "means that rule rejects", fontsize=10)
    _save(fig, "f7_calibrations")


CERT_LABEL = {"sequential": "sequential betting", "betting": "single-split betting",
              "bootstrap": "bootstrap (Thm 16)"}


def _reported_certificate(df):
    """The certificate whose bracket a v2 table reports under the v1 column
    names (``reported_certificate``, else read off ``certificate``); None for v1."""
    if "reported_certificate" in getattr(df, "columns", ()):
        return str(df["reported_certificate"].iloc[0])
    if "certificate" in getattr(df, "columns", ()):
        c = str(df["certificate"].iloc[0])
        return {"all": "sequential", "both": "betting"}.get(c, c)
    return None


def fig_split():
    df = pd.read_csv(TABLES / "e7_split_bounds.csv")
    astar = df["alpha_star"].iloc[0]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
    n = df["n"].to_numpy()
    rep = _reported_certificate(df)
    ax[0].fill_between(n, df["mean_lower"], df["mean_upper"], alpha=0.25, color=BLUE,
                       label=("split-sample interval" if rep is None else
                              f"split-sample interval ({CERT_LABEL.get(rep, rep)} lower end)"))
    ax[0].plot(n, df["mean_lower"], "o-", color=BLUE)
    ax[0].plot(n, df["mean_upper"], "o-", color=BLUE)
    # revision v2: the other certificates' lower ends, same draws
    for mth, style in (("betting", "v-."), ("bootstrap", "d:")):
        col = f"mean_lower_{mth}"
        if col in df.columns:
            ax[0].plot(n, df[col], style, color=BLUE, alpha=0.6, lw=1,
                       label=f"lower end, {CERT_LABEL[mth]}")
    ax[0].axhline(astar, color="k", ls="--", lw=1.5, label=r"true $\alpha^\star$")
    if "mean_alpha_hat_analytic" in df.columns:
        # revision v2: the analytic estimator on the same draws (E7 path)
        ax[0].plot(n, df["mean_alpha_hat_analytic"], "s--", color=ORANGE,
                   label=(r"mean $\hat\alpha$, " + _tau_label(df, True)
                          + " threshold (no upper bound)"))
    else:
        # v1: the analytic alpha-hat was 0 at every size of this table
        ax[0].plot(n, np.zeros_like(n, dtype=float), "s--", color=ORANGE,
                   label=r"analytic $\hat\alpha$ (no upper bound)")
    if "mean_upper_cor" in df.columns:
        ax[0].plot(n, df["mean_upper_cor"], "^:", color=GREY,
                   label="finite-sample upper end (Cor. upperfs)")
    ax[0].set_xscale("log"); ax[0].set_xlabel("n")
    ax[0].set_ylabel(r"bound on $d_{\mathrm{TV}}(P,Q)$")
    ax[0].set_xlabel("n")
    ax[0].set_title("Two-sided bounds on the contamination level")
    ax[0].legend(fontsize=8)

    ax[1].loglog(n, df["mean_width"], "o-", color=BLUE, label="observed width")
    c = float((df["mean_width"] * np.sqrt(n)).mean())
    ns = np.logspace(np.log10(n.min()), np.log10(6000), 40)
    ax[1].loglog(ns, c / np.sqrt(ns), ":", color=GREY,
                 label=fr"$\approx{c:.2f}/\sqrt{{n}}$")
    ax[1].axhline(0.04, color=ORANGE, ls="--", lw=1,
                  label=r"width 0.04 (i.e. $[0.18,0.22]$)")
    ax[1].set_xlabel("n"); ax[1].set_ylabel("interval width")
    ax[1].set_title("Width shrinks at the parametric rate")
    ax[1].legend(fontsize=8)
    _save(fig, "f8_split_bounds")


def fig_nhanes():
    """The functional application: what the curves look like, and what the
    method concludes about how much of the population they separate."""
    import numpy as np
    from ..realdata import load_nhanes_activity, nhanes_groups

    curves, cov, _ = load_nhanes_activity()
    # The child contrast: the one where the analytic test fires, so the
    # only panel that can show a crossing rather than a gap.
    X, Y, la, lb = nhanes_groups(curves, cov, by="child")
    paths = pd.read_csv(TABLES / "r1_paths.csv")
    low = pd.read_csv(TABLES / "r2_lower_bounds.csv").set_index("dataset")
    p = paths[paths.dataset == "nhanes_child"]
    summ = pd.read_csv(TABLES / "r1_summary.csv").set_index("dataset")

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
    h = np.arange(24)
    for Z, lab, col in ((X, la, BLUE), (Y, lb, ORANGE)):
        mu = Z.mean(0)
        se = Z.std(0, ddof=1) / np.sqrt(len(Z))
        ax[0].plot(h, mu, "-", color=col, label=f"{lab} (n={len(Z)})")
        ax[0].fill_between(h, mu - 2 * se, mu + 2 * se, color=col, alpha=0.25)
    ax[0].set_xlabel("clock hour"); ax[0].set_ylabel(r"mean $\log(1+\mathrm{MIMS})$")
    ax[0].set_xticks([0, 6, 12, 18, 23])
    ax[0].set_title("NHANES daily activity profiles")
    ax[0].legend(fontsize=8)

    ax[1].plot(p.alpha, p.d_hat, "o-", color=BLUE, label=r"$\hat D_\alpha$")
    ax[1].plot(p.alpha, p.analytic_tau, "s--", color=ORANGE, label=_tau_label(p))
    lo = float(low.loc["nhanes_child", "median_lower"])
    a_hat = float(summ.loc["nhanes_child", "alpha_hat_analytic"])
    ax[1].axvline(a_hat, color="#7a9e3f", ls="-.", lw=1.5,
                  label=fr"analytic $\hat\alpha={a_hat:.2f}$")
    ax[1].axvline(lo, color="k", ls=":", lw=1.5,
                  label=fr"certified $d_{{\mathrm{{TV}}}}\geq{lo:.2f}$")
    # Log scale: tau grows without bound in alpha while D-hat decays to the
    # noise floor, so on a linear axis the only visible feature is tau.
    ax[1].set_yscale("log")
    ax[1].set_xlabel(r"$\alpha$"); ax[1].set_ylabel("discrepancy (log scale)")
    ax[1].set_title("Both routes, on the contrast where both work")
    ax[1].legend(fontsize=8)
    _save(fig, "f9_nhanes_activity")


def fig_rhc():
    """The causal-overlap application: the propensity picture everyone draws,
    and the certified answer next to the rules of thumb."""
    import json as _json

    import numpy as np
    from ..realdata import load_rhc
    from .realdata_run import _logistic_fit

    X, Y, _, _ = load_rhc()
    Z = np.vstack([X, Y])
    y = np.r_[np.ones(len(X)), np.zeros(len(Y))]
    _, e = _logistic_fit(Z, y)
    comp = _json.loads((TABLES / "r4_rhc_propensity.json").read_text())
    paths = pd.read_csv(TABLES / "r1_paths.csv")
    low = pd.read_csv(TABLES / "r2_lower_bounds.csv").set_index("dataset")
    summ = pd.read_csv(TABLES / "r1_summary.csv").set_index("dataset")
    p = paths[paths.dataset == "rhc"]

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
    bins = np.linspace(0, 1, 41)
    ax[0].hist(e[y == 1], bins=bins, alpha=0.6, color=BLUE,
               label=f"catheterized (n={int((y == 1).sum())})")
    ax[0].hist(e[y == 0], bins=bins, alpha=0.6, color=ORANGE,
               label=f"not catheterized (n={int((y == 0).sum())})")
    a = comp["crump_alpha_cutoff"]
    for c in (a, 1 - a):
        ax[0].axvline(c, color="k", ls="--", lw=1)
    ax[0].set_xlabel("estimated propensity score")
    ax[0].set_ylabel("count")
    ax[0].set_title("RHC: where the overlap problem is usually read off")
    ax[0].legend(fontsize=8)

    ax[1].plot(p.alpha, p.d_hat, "o-", color=BLUE, label=r"$\hat D_\alpha$")
    ax[1].plot(p.alpha, p.analytic_tau, "s--", color=ORANGE, label=_tau_label(p))
    lo = float(low.loc["rhc", "median_lower"])
    a_hat = float(summ.loc["rhc", "alpha_hat_analytic"])
    ax[1].axvline(a_hat, color="#7a9e3f", ls="-.", lw=1.5,
                  label=fr"analytic $\hat\alpha={a_hat:.2f}$")
    ax[1].axvline(lo, color="k", ls=":", lw=1.5,
                  label=fr"certified $d_{{\mathrm{{TV}}}}\geq{lo:.2f}$")
    ax[1].axvline(comp["frac_discarded_crump"], color="#7a9e3f", ls="-.", lw=1.5,
                  label=f"Crump et al. discard {comp['frac_discarded_crump']:.2f}")
    ax[1].axvline(comp["frac_outside_0.1_0.9"], color="#a03050", ls="-.", lw=1.5,
                  label=f"$e\\notin[0.1,0.9]$: {comp['frac_outside_0.1_0.9']:.2f}")
    ax[1].set_yscale("log")
    ax[1].set_xlabel(r"$\alpha$"); ax[1].set_ylabel("discrepancy (log scale)")
    ax[1].set_title("Certificate vs. the standard trimming rules")
    ax[1].legend(fontsize=7.5)
    _save(fig, "f10_rhc_overlap")


def all_figures(outdir=None):
    """Every figure from the experiment tables.

    ``outdir`` (revision v2): read ``<outdir>/tables`` and write
    ``<outdir>/figures`` instead of ``results/``. A figure whose tables are
    missing there is skipped with a message (as in v1); nothing falls back
    to the v1 tables.
    """
    global TABLES, FIGURES
    saved = (TABLES, FIGURES)
    if outdir is not None:
        from .v2 import out_root
        root = out_root(outdir)
        TABLES, FIGURES = root / "tables", root / "figures"
    try:
        print("figures:" if outdir is None else f"figures ({TABLES} -> {FIGURES}):")
        for fn in (fig_conservativeness, fig_power, fig_alpha_hat, fig_margin,
                   fig_assumption, fig_scaling, fig_calibrations, fig_split,
                   fig_nhanes, fig_rhc):
            try:
                fn()
            except FileNotFoundError as e:
                print(f"  skipped {fn.__name__}: {e}")
    finally:
        TABLES, FIGURES = saved


if __name__ == "__main__":
    all_figures()

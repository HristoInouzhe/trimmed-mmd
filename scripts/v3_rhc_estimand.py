"""RHC case study, JMLR v3 (revision plan v1.1, issue C1 / decision D3): what each weighting estimates.

The kernel trimming at the split-sample lower bound targets the common core of the two arms, whose
covariate law has density proportional to min(p, q). In propensity terms, with e(x) the propensity score
and pi the treated fraction, min(p, q)(x) is proportional to f(x) * min{e(x)/pi, (1-e(x))/(1-pi)}: an
overlap-population target in the balancing-weights family of Li, Morgan & Zaslavsky (2018), equal to the
matching-weights target (ATM) of Li & Greene (2013) when pi = 1/2. The balancing weights of Table 3
(energy and kernel balancing, each arm to the pooled sample) target the pooled population (ATE).

This script puts all of them side by side, adds the two propensity-based overlap targets

    overlap weights (ATO):  treated (1 - e),            controls e
    matching weights (ATM): treated min(e, 1-e) / e,    controls min(e, 1-e) / (1 - e)

and reports, for every weighting, the 30-day mortality risk difference (dth30, catheterized minus others)
in the weighted cohort, with a 95% percentile interval from a stratified bootstrap of patients in which the
weights are held fixed (renormalized within each resampled arm). The interval therefore ignores the
uncertainty of the weights themselves; it is illustrative, as the paper says.

Everything else is exactly as scripts/exp_rhc_balancing.py --alpha certified --threshold plugin
--outdir results/v2 (same covariates, standardization, bandwidth, trimming solver and seed, same SMD and
overlap definitions); the trimming row is checked against results/v2/tables/r13_rhc_balancing.csv.

    PYTHONPATH=src python3 -I scripts/v3_rhc_estimand.py --threads 3

Writes results/v3/tables/rhc_estimand.{csv,json} and, for the JMLR v3 build,
jmlr/v3/tables/estimand_v3.tex and jmlr/v3/tables/estimand_macros.tex.
"""

from __future__ import annotations

import os
import sys


def _threads_from_argv(default="3"):
    for i, a in enumerate(sys.argv):
        if a == "--threads" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


_T = _threads_from_argv()
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _T

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import exp_rhc_balancing as EB  # noqa: E402
from trimmedmmd.experiments.realdata_run import DELTA, _logistic_fit  # noqa: E402
from trimmedmmd.kernels import gaussian_kernel, median_heuristic  # noqa: E402
from trimmedmmd.realdata import RHC_BINARY, RHC_CONTINUOUS, data_root, load_rhc  # noqa: E402
from trimmedmmd.threshold import threshold_function  # noqa: E402
from trimmedmmd.trimmings import optimal_trimming_mmd  # noqa: E402

V2 = ROOT / "results" / "v2" / "tables"
OUT = ROOT / "results" / "v3" / "tables"
JT = ROOT / "jmlr" / "v3" / "tables"
SEED = 0
B = 2000
BOOT_SEED = 20261117


def rhc_outcome():
    """dth30 (1 = died within 30 days), aligned with load_rhc's rows: same complete-case mask and arm split."""
    df = pd.read_csv(data_root() / "rhc" / "rhc.csv", low_memory=False)
    Z = df[RHC_CONTINUOUS + RHC_BINARY].to_numpy(dtype=float)
    treated = df["swang1"].to_numpy() == "RHC"
    keep = np.isfinite(Z).all(axis=1)
    y = (df["dth30"].to_numpy() == "Yes").astype(float)[keep]
    t = treated[keep]
    return y[t], y[~t]


def risk_difference(yx, yy, w, v):
    return float((w / w.sum()) @ yx - (v / v.sum()) @ yy)


def boot_ci(yx, yy, w, v, rng):
    n, m = len(yx), len(yy)
    out = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        j = rng.integers(0, m, m)
        out[b] = risk_difference(yx[i], yy[j], w[i], v[j])
    lo, hi = np.quantile(out, [0.025, 0.975])
    return float(lo), float(hi)


def fmt(x, nd=2):
    return f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    lo = pd.read_csv(V2 / "r2_lower_bounds.csv").set_index("dataset")
    alpha = float(lo.loc["rhc", "median_lower"])

    X, Y, names, meta = load_rhc()
    yx, yy = rhc_outcome()
    assert len(yx) == len(X) and len(yy) == len(Y)
    n, m = len(X), len(Y)
    Z = np.vstack([X, Y])
    _, prop = _logistic_fit(Z, np.r_[np.ones(n), np.zeros(m)])
    ex, ey = prop[:n], prop[n:]
    pi = n / (n + m)

    rng = np.random.default_rng(SEED)
    sigma = float(median_heuristic(X, Y, rng=rng))
    thr = float(threshold_function("plugin", n, m, DELTA)(alpha))
    Kfull = gaussian_kernel(Z, Z, sigma)
    Dfull = cdist(Z, Z)

    weights, timing = {}, {}
    weights["unweighted"] = (np.ones(n), np.ones(m), "observed arms (no reweighting)")
    t0 = time.time()
    tr = optimal_trimming_mmd(X, Y, alpha, sigma=sigma, rng=np.random.default_rng(SEED), tau=thr)
    timing["trimming"] = time.time() - t0
    weights["trimming"] = (tr.w, tr.v, "common core (overlap population)")

    a_all = Dfull.mean(axis=1)
    c = float(Dfull.mean())
    t0 = time.time()
    w_e, _ = EB.energy_per_arm(Dfull[:n, :n], a_all[:n], c)
    v_e, _ = EB.energy_per_arm(Dfull[n:, n:], a_all[n:], c)
    timing["energy"] = time.time() - t0
    weights["energy"] = (w_e, v_e, "pooled population (ATE)")

    b_all = Kfull.mean(axis=1)
    cK = float(Kfull.mean())
    t0 = time.time()
    w_k, _ = EB.kernel_per_arm(Kfull[:n, :n], b_all[:n], cK)
    v_k, _ = EB.kernel_per_arm(Kfull[n:, n:], b_all[n:], cK)
    timing["kernel"] = time.time() - t0
    weights["kernel"] = (w_k, v_k, "pooled population (ATE)")

    weights["ato"] = (1.0 - ex, ey, "overlap population, tilting e(1-e) (ATO)")
    mx, my = np.minimum(ex, 1 - ex), np.minimum(ey, 1 - ey)
    weights["atm"] = (mx / ex, my / (1 - ey), "overlap population, tilting min(e,1-e) (ATM)")

    labels = {"unweighted": "all patients", "trimming": f"trimming, alpha = {alpha:g}",
              "energy": "energy balancing", "kernel": "kernel balancing",
              "ato": "overlap weights", "atm": "matching weights"}
    rows = []
    brng = np.random.default_rng(BOOT_SEED)
    for key, (w, v, target) in weights.items():
        r, _ = EB.describe(labels[key], X, Y, names, w, v, prop, Kfull, Dfull)
        rd = risk_difference(yx, yy, w, v)
        ci = boot_ci(yx, yy, np.asarray(w, float), np.asarray(v, float), brng)
        r.update({"key": key, "target": target, "rd_dth30": rd, "rd_lo": ci[0], "rd_hi": ci[1]})
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "rhc_estimand.csv", index=False)

    # reproduction check of the trimming row against the published v2 table
    r13 = pd.read_csv(V2 / "r13_rhc_balancing.csv")
    pub = r13[r13.method.str.startswith("kernel trimming")].iloc[0]
    mine = out[out.key == "trimming"].iloc[0]
    check = {k: [float(mine[k]), float(pub[k])] for k in
             ("max_abs_smd", "mean_abs_smd", "ess_frac_x", "ess_frac_y", "prop_overlap")}
    rel = max(abs(x - y) / max(abs(y), 1e-300) for x, y in check.values())
    check["max_relative_difference"] = rel
    check["reproduces_v2"] = bool(rel < 1e-3)

    # common-core tilting written through the propensity score (for the text): min(e/pi, (1-e)/(1-pi))
    meta_out = {"script": "scripts/v3_rhc_estimand.py",
                "command": f"PYTHONPATH=src python3 -I scripts/v3_rhc_estimand.py --threads {a.threads}",
                "alpha": alpha, "tau": thr, "sigma": sigma, "pi_treated": pi, "bootstrap_B": B,
                "bootstrap_seed": BOOT_SEED, "bootstrap": "stratified by arm, weights held fixed",
                "outcome": "dth30 == 'Yes'", "n_treated": n, "n_control": m,
                "crude_mortality_treated": float(yx.mean()), "crude_mortality_control": float(yy.mean()),
                "trimming_check": check, "timing_seconds": {k: round(v, 1) for k, v in timing.items()},
                "seconds_total": round(time.time() - t_start, 1)}
    (OUT / "rhc_estimand.json").write_text(json.dumps(meta_out, indent=1))

    # LaTeX: table and macros
    order = ["unweighted", "trimming", "atm", "ato", "energy", "kernel"]
    tlabel = {"unweighted": "all patients", "trimming": r"trimming, $\alpha=\cohAlpha$",
              "atm": "matching weights", "ato": "overlap weights", "energy": "energy balancing",
              "kernel": "kernel balancing"}
    ttarget = {"unweighted": "--", "trimming": "common core", "atm": "overlap (ATM)",
               "ato": "overlap (ATO)", "energy": "pooled (ATE)", "kernel": "pooled (ATE)"}
    o = out.set_index("key")
    lines = ["% Generated by scripts/v3_rhc_estimand.py -- do not edit.",
             r"\begin{tabular}{llcccccc}", r"\toprule",
             r"RHC & target & \multicolumn{2}{c}{effective fraction} & max $|$SMD$|$ & mean $|$SMD$|$ & overlap & risk difference \\",
             r" & population & catheterized & others & & & & (95\% interval) \\", r"\midrule"]
    for k in order:
        r = o.loc[k]
        lines.append(f"{tlabel[k]} & {ttarget[k]} & {fmt(r.ess_frac_x)} & {fmt(r.ess_frac_y)} & "
                     f"{fmt(r.max_abs_smd, 3)} & {fmt(r.mean_abs_smd, 3)} & {fmt(r.prop_overlap)} & "
                     f"{fmt(r.rd_dth30, 3)} ({fmt(r.rd_lo, 3)}, {fmt(r.rd_hi, 3)}) \\\\")
        if k == "trimming":
            lines.append(r"\addlinespace")
        if k == "ato":
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (JT / "estimand_v3.tex").write_text("\n".join(lines) + "\n")

    def mac(name, val):
        return f"\\newcommand{{\\{name}}}{{{val}}}"
    cap = {"unweighted": "Unw", "trimming": "Trim", "atm": "Atm", "ato": "Ato", "energy": "Energy",
           "kernel": "Kernel"}
    ml = ["% Generated by scripts/v3_rhc_estimand.py -- do not edit."]
    for k in order:
        r = o.loc[k]
        ml += [mac(f"est{cap[k]}RD", fmt(r.rd_dth30, 3)), mac(f"est{cap[k]}RDlo", fmt(r.rd_lo, 3)),
               mac(f"est{cap[k]}RDhi", fmt(r.rd_hi, 3)), mac(f"est{cap[k]}SMD", fmt(r.max_abs_smd, 3)),
               mac(f"est{cap[k]}EssX", fmt(r.ess_frac_x)), mac(f"est{cap[k]}EssY", fmt(r.ess_frac_y))]
    ml += [mac("estPiTreated", fmt(pi)), mac("estBoot", f"{B:,}".replace(",", "{,}")),
           mac("estCrudeX", fmt(yx.mean(), 3)), mac("estCrudeY", fmt(yy.mean(), 3))]
    (JT / "estimand_macros.tex").write_text("\n".join(ml) + "\n")

    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out[["key", "target", "ess_frac_x", "ess_frac_y", "max_abs_smd", "mean_abs_smd",
                   "prop_overlap", "rd_dth30", "rd_lo", "rd_hi"]].to_string(index=False))
    print(json.dumps(check, indent=1))
    print(f"done in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()

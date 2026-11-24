"""Balancing-weights baselines for the RHC trimmed cohort (Section 10.4, A2).

The paper trims RHC at its certified level alpha = 0.25 with the Gaussian
kernel at the pooled median bandwidth and reports covariate balance of the
retained cohort (results/tables/r10_trimmed_cohorts.csv, r11_rhc_balance.csv,
cohort_macros.tex, all written by trimmedmmd.experiments.realdata_run). This
script puts balancing weights next to it, with the SAME covariates (the 33 of
realdata.load_rhc, standardized by pooled moments), the SAME standardized
mean difference (realdata_run.standardized_differences: weighted means over
the square root of the average of the two weighted variances) and the SAME
propensity overlap coefficient (realdata_run._overlap on the logistic
propensity of realdata_run._logistic_fit).

Methods (each arm's weights lie on its probability simplex, no caps):

  energy           energy balancing weights (Huling & Mak 2024, J. Causal
                   Inference): for each arm separately, the weights minimizing
                   the energy distance between the weighted arm and the pooled
                   sample of both arms, a convex QP on the simplex;
  energy_improved  the variant of the same paper that adds the energy distance
                   between the two weighted arms to the two arm-to-pooled terms
                   (the default of WeightIt's method = "energy" for the ATE);
                   solved jointly;
  kernel           kernel (MMD) balancing: for each arm, weights minimizing
                   the Gaussian-kernel MMD between the weighted arm and the
                   pooled sample, same kernel and bandwidth as the trimming;
  trimming         the paper's kernel trimming at alpha = 0.25, recomputed with
                   exactly the call of realdata_run.trimmed_cohort, as the
                   reference (it must reproduce r10_trimmed_cohorts.csv).

The estimands differ. Balancing weights reweight both arms towards the pooled
population (an ATE-type target) and discard no unit by design; the trimming
discards at most a fraction alpha of each arm and changes the target to the
common core of the two arms.

All QPs are solved by FISTA with Euclidean projection onto the simplex
(trimmedmmd.simplex.project_capped_simplex with cap 1), stopped on the
Frank-Wolfe duality gap, which bounds the suboptimality of the objective.

    PYTHONPATH=src python3 -I scripts/exp_rhc_balancing.py --threads 4

Revision v2 (defaults unchanged): ``--alpha`` sets the trimming level (v1:
0.25; v2: the RHC certified level of the reported certificate (sequential
betting; the single-split betting gives 0.30),
or ``--alpha certified`` to read the median of <outdir>/tables/
r2_lower_bounds.csv), ``--threshold plugin`` uses tau_new as the
least-trimming cutoff (as realdata_run.trimmed_cohort does in v2), and
``--outdir D`` writes r13_* to D/tables and checks the trimming row against
the v2 r10 in D/tables.

    PYTHONPATH=src python3 -I scripts/exp_rhc_balancing.py --threads 4 \
        --alpha certified --threshold plugin --outdir results/v2
"""

from __future__ import annotations

import os
import sys


def _threads_from_argv(default="4"):
    for i, a in enumerate(sys.argv):
        if a == "--threads" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


_T = _threads_from_argv()
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _T

import argparse                                                # noqa: E402
import json                                                    # noqa: E402
import platform                                                # noqa: E402
import time                                                    # noqa: E402
from pathlib import Path                                       # noqa: E402

import numpy as np                                             # noqa: E402
import pandas as pd                                            # noqa: E402
from scipy.spatial.distance import cdist                       # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trimmedmmd.experiments.realdata_run import (DELTA, _logistic_fit,  # noqa: E402
                                                 _overlap,
                                                 standardized_differences)
from trimmedmmd.kernels import gaussian_kernel, median_heuristic  # noqa: E402
from trimmedmmd.realdata import load_rhc                       # noqa: E402
from trimmedmmd.simplex import project_capped_simplex          # noqa: E402
from trimmedmmd.threshold import tau                           # noqa: E402
from trimmedmmd.threshold import threshold_function            # noqa: E402
from trimmedmmd.trimmings import optimal_trimming_mmd          # noqa: E402

TABLES = ROOT / "results" / "tables"
ALPHA = 0.25          # the certified level used for the RHC cohort
SEED = 0              # as realdata_run.trimmed_cohort
GAP_TOL = 1e-9        # absolute Frank-Wolfe gap at which a QP is stopped
MAX_ITER = 100_000


# ---------------------------------------------------------------------------
# a block-simplex FISTA for convex quadratics
# ---------------------------------------------------------------------------

def fista_simplex(grad, value, sizes, hess_matvec, max_iter=MAX_ITER,
                  gap_tol=GAP_TOL, seed=0):
    """min f(x) over a product of probability simplices of the given sizes.

    ``grad(x)`` and ``value(x)`` act on the concatenated vector;
    ``hess_matvec`` is the Hessian, used only to bound the curvature on the
    directions that keep every block sum fixed (the energy-distance Hessian
    is indefinite on all of R^n, but positive semidefinite there).
    """
    cuts = np.cumsum([0] + list(sizes))
    blocks = [slice(cuts[i], cuts[i + 1]) for i in range(len(sizes))]

    def center(u):
        u = u.copy()
        for b in blocks:
            u[b] -= u[b].mean()
        return u

    rng = np.random.default_rng(seed)
    u = center(rng.standard_normal(cuts[-1]))
    u /= np.linalg.norm(u)
    lam = 1.0
    for _ in range(200):
        z = center(hess_matvec(u))
        nz = np.linalg.norm(z)
        if nz <= 0:
            break
        u, lam = z / nz, nz
    L = 1.05 * lam     # small safety margin on the power-iteration estimate

    def project(y):
        out = np.empty_like(y)
        for b in blocks:
            out[b] = project_capped_simplex(y[b], 1.0)
        return out

    def fw_gap(x, g):
        return float(sum(g[b] @ x[b] - g[b].min() for b in blocks))

    x = np.concatenate([np.full(s, 1.0 / s) for s in sizes])
    y, t, prev, it, gap = x.copy(), 1.0, None, 0, np.inf
    for it in range(1, max_iter + 1):
        x_new = project(y - grad(y) / L)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x, t = x_new, t_new
        if it % 25 == 0:
            f = value(x)
            if prev is not None and f > prev:
                y, t = x.copy(), 1.0          # restart, as solver._fista
            prev = f
            gap = fw_gap(x, grad(x))
            if gap <= gap_tol:
                break
    return x, {"n_iter": it, "fw_gap": float(gap), "lipschitz": float(L),
               "objective": float(value(x))}


# ---------------------------------------------------------------------------
# the balancing problems
# ---------------------------------------------------------------------------

def energy_per_arm(DAA, a, c):
    """E(w) = 2 a'w - w'DAA w - c: energy distance, weighted arm vs pooled."""
    n = len(a)
    return fista_simplex(lambda w: 2.0 * a - 2.0 * (DAA @ w),
                         lambda w: float(2.0 * a @ w - w @ (DAA @ w) - c),
                         [n], lambda u: -2.0 * (DAA @ u))


def energy_improved(D11, D00, D10, a1, a0, c):
    n1, n0 = len(a1), len(a0)

    def split(x):
        return x[:n1], x[n1:]

    def value(x):
        w1, w0 = split(x)
        e1 = 2 * a1 @ w1 - w1 @ (D11 @ w1) - c
        e0 = 2 * a0 @ w0 - w0 @ (D00 @ w0) - c
        e10 = 2 * w1 @ (D10 @ w0) - w1 @ (D11 @ w1) - w0 @ (D00 @ w0)
        return float(e1 + e0 + e10)

    def grad(x):
        w1, w0 = split(x)
        return np.concatenate([2 * a1 - 4 * (D11 @ w1) + 2 * (D10 @ w0),
                               2 * a0 - 4 * (D00 @ w0) + 2 * (D10.T @ w1)])

    def hess(u):
        u1, u0 = split(u)
        return np.concatenate([-4 * (D11 @ u1) + 2 * (D10 @ u0),
                               -4 * (D00 @ u0) + 2 * (D10.T @ u1)])

    return fista_simplex(grad, value, [n1, n0], hess)


def kernel_per_arm(KAA, b, c):
    """M(w) = w'KAA w - 2 b'w + c: squared MMD, weighted arm vs pooled."""
    n = len(b)
    return fista_simplex(lambda w: 2.0 * (KAA @ w) - 2.0 * b,
                         lambda w: float(w @ (KAA @ w) - 2.0 * b @ w + c),
                         [n], lambda u: 2.0 * (KAA @ u))


# ---------------------------------------------------------------------------
# diagnostics, identical definitions to realdata_run
# ---------------------------------------------------------------------------

def kish(w):
    return float(w.sum() ** 2 / (w ** 2).sum())


def describe(method, X, Y, names, w, v, prop, Kfull, Dfull, extra=None):
    n, m = len(X), len(Y)
    w = np.asarray(w, float)
    v = np.asarray(v, float)
    smd = standardized_differences(X, Y, w / w.sum(), v / v.sum())
    smd0 = standardized_differences(X, Y)
    # the same differences on a fixed scale (the unweighted pooled SD)
    mx, my = (w / w.sum()) @ X, (v / v.sum()) @ Y
    sd0 = np.sqrt((X.var(0) + Y.var(0)) / 2.0)
    smd_fixed = (mx - my) / sd0
    worst_before = int(np.argmax(np.abs(smd0)))
    worst_after = int(np.argmax(np.abs(smd)))
    ex, ey = prop[:n], prop[n:]
    # between-arm discrepancies of the weighted arms, same yardsticks for all
    u = np.r_[w / w.sum(), -(v / v.sum())]
    mmd2 = float(u @ (Kfull @ u))
    energy = float(-(u @ (Dfull @ u)))
    row = {
        "method": method,
        "max_abs_smd": float(np.abs(smd).max()),
        "mean_abs_smd": float(np.abs(smd).mean()),
        "worst_covariate_after": names[worst_after],
        "worst_covariate_before": names[worst_before],
        "smd_worst_before_cov_after": float(smd[worst_before]),
        "n_abs_smd_above_0.1": int(np.sum(np.abs(smd) > 0.1)),
        "max_abs_smd_fixed_sd": float(np.abs(smd_fixed).max()),
        "mean_abs_smd_fixed_sd": float(np.abs(smd_fixed).mean()),
        "ess_x": kish(w), "ess_y": kish(v),
        "ess_frac_x": kish(w) / n, "ess_frac_y": kish(v) / m,
        "support_x": int(np.sum(w > 1e-10 * w.max())),
        "support_y": int(np.sum(v > 1e-10 * v.max())),
        "max_weight_x_times_n": float(w.max() / w.sum() * n),
        "max_weight_y_times_m": float(v.max() / v.sum() * m),
        "prop_overlap": _overlap(ex, ey, w, v),
        "mmd_between_arms": float(np.sqrt(max(mmd2, 0.0))),
        "energy_between_arms": energy,
        "n": n, "m": m,
    }
    row.update(extra or {})
    return row, smd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--alpha", default=None,
                    help="trimming level (default 0.25 as in v1), or 'certified' for the "
                         "median of <tables>/r2_lower_bounds.csv (rhc row)")
    ap.add_argument("--threshold", choices=["paper", "plugin"], default="paper")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    global TABLES, ALPHA
    if a.outdir is not None:
        od = Path(a.outdir)
        TABLES = (od if od.is_absolute() else ROOT / od) / "tables"
        TABLES.mkdir(parents=True, exist_ok=True)
    if a.alpha == "certified":
        lo = pd.read_csv(TABLES / "r2_lower_bounds.csv").set_index("dataset")
        ALPHA = float(lo.loc["rhc", "median_lower"])
    elif a.alpha is not None:
        ALPHA = float(a.alpha)
    t_start = time.time()

    X, Y, names, meta = load_rhc()
    n, m = len(X), len(Y)
    Z = np.vstack([X, Y])
    N = len(Z)
    y = np.r_[np.ones(n), np.zeros(m)]
    _, prop = _logistic_fit(Z, y)

    # kernel and bandwidth exactly as realdata_run.trimmed_cohort
    rng = np.random.default_rng(SEED)
    sigma = float(median_heuristic(X, Y, rng=rng))
    thr = (float(tau(n, m, ALPHA, DELTA)) if a.threshold == "paper"
           else float(threshold_function(a.threshold, n, m, DELTA)(ALPHA)))
    Kfull = gaussian_kernel(Z, Z, sigma)
    Dfull = cdist(Z, Z)

    rows, smds, timing, solver = [], {}, {}, {}

    # -- reference: the unweighted arms ------------------------------------
    r, s = describe("unweighted", X, Y, names, np.ones(n), np.ones(m), prop,
                    Kfull, Dfull)
    rows.append(r)
    smds["unweighted"] = s

    # -- the paper's kernel trimming at alpha = ALPHA (v1: 0.25) ----------
    t0 = time.time()
    tr = optimal_trimming_mmd(X, Y, ALPHA, sigma=sigma,
                              rng=np.random.default_rng(SEED), tau=thr)
    timing["trimming"] = time.time() - t0
    r, s = describe(f"kernel trimming (paper), alpha = {ALPHA:g}", X, Y, names,
                    tr.w, tr.v, prop, Kfull, Dfull,
                    {"kept_x": int(tr.kept_x.size), "kept_y": int(tr.kept_y.size),
                     "estimand": f"common core (discards up to {100 * ALPHA:g}% "
                                 "of each arm)"})
    rows.append(r)
    smds["trimming"] = s

    # -- energy balancing, per arm to the pooled sample --------------------
    a_all = Dfull.mean(axis=1)                 # a_i = mean_j ||Z_i - Z_j||
    c = float(Dfull.mean())                    # (1/N^2) sum_jl ||Z_j - Z_l||
    D11, D00, D10 = Dfull[:n, :n], Dfull[n:, n:], Dfull[:n, n:]
    t0 = time.time()
    w_e, info_w = energy_per_arm(D11, a_all[:n], c)
    v_e, info_v = energy_per_arm(D00, a_all[n:], c)
    timing["energy"] = time.time() - t0
    solver["energy"] = {"treated": info_w, "control": info_v}
    r, s = describe("energy balancing (per arm to pooled)", X, Y, names,
                    w_e, v_e, prop, Kfull, Dfull,
                    {"estimand": "pooled population (ATE); no unit discarded "
                                 "by design",
                     "energy_to_pooled_x": info_w["objective"],
                     "energy_to_pooled_y": info_v["objective"]})
    rows.append(r)
    smds["energy"] = s

    # -- improved energy balancing (adds the between-arm term) -------------
    t0 = time.time()
    x_ei, info_ei = energy_improved(D11, D00, D10, a_all[:n], a_all[n:], c)
    timing["energy_improved"] = time.time() - t0
    solver["energy_improved"] = info_ei
    r, s = describe("energy balancing, improved (adds arm-to-arm term)", X, Y,
                    names, x_ei[:n], x_ei[n:], prop, Kfull, Dfull,
                    {"estimand": "pooled population (ATE); no unit discarded "
                                 "by design"})
    rows.append(r)
    smds["energy_improved"] = s

    # -- kernel (MMD) balancing, same kernel and bandwidth -----------------
    b_all = Kfull.mean(axis=1)
    cK = float(Kfull.mean())
    t0 = time.time()
    w_k, info_kw = kernel_per_arm(Kfull[:n, :n], b_all[:n], cK)
    v_k, info_kv = kernel_per_arm(Kfull[n:, n:], b_all[n:], cK)
    timing["kernel"] = time.time() - t0
    solver["kernel"] = {"treated": info_kw, "control": info_kv}
    r, s = describe("kernel (MMD) balancing (per arm to pooled)", X, Y, names,
                    w_k, v_k, prop, Kfull, Dfull,
                    {"estimand": "pooled population (ATE); no unit discarded "
                                 "by design",
                     "mmd_to_pooled_x": float(np.sqrt(max(info_kw["objective"], 0))),
                     "mmd_to_pooled_y": float(np.sqrt(max(info_kv["objective"], 0)))})
    rows.append(r)
    smds["kernel"] = s

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "r13_rhc_balancing.csv", index=False)
    per_cov = pd.DataFrame({"covariate": names, **smds})
    per_cov.to_csv(TABLES / "r13_rhc_balancing_smd.csv", index=False)

    # consistency with the published cohort (r10, r11)
    r10 = pd.read_csv(TABLES / "r10_trimmed_cohorts.csv")
    r10 = r10[(r10.dataset == "rhc") & (r10.geometry == "mmd")].iloc[0]
    if abs(float(r10.alpha) - ALPHA) > 1e-9:
        print(f"WARNING: r10's RHC level {float(r10.alpha):g} != ALPHA {ALPHA:g}; "
              "the reproduction check compares different cohorts")
    trow = out.iloc[1]
    check = {
        "max_abs_smd": [float(trow.max_abs_smd), float(r10.max_abs_smd_after)],
        "mean_abs_smd": [float(trow.mean_abs_smd), float(r10.mean_abs_smd_after)],
        "ess_x": [float(trow.ess_x), float(r10.ess_x)],
        "ess_y": [float(trow.ess_y), float(r10.ess_y)],
        "prop_overlap": [float(trow.prop_overlap), float(r10.prop_overlap_after)],
        "prop_overlap_before": [float(out.iloc[0].prop_overlap),
                                float(r10.prop_overlap_before)],
        "kept": [[int(trow.kept_x), int(trow.kept_y)],
                 [int(r10.kept_x), int(r10.kept_y)]],
    }
    # The solver runs many BLAS reductions, so a different thread count moves
    # the last digits; agreement is judged relatively.
    rel = max(abs(v[0] - v[1]) / max(abs(v[1]), 1e-300)
              for k, v in check.items() if k != "kept")
    check["max_relative_difference"] = float(rel)
    check["reproduces_r10"] = bool(rel < 1e-5
                                   and check["kept"][0] == check["kept"][1])

    meta_out = {
        "script": "scripts/exp_rhc_balancing.py",
        "command": f"PYTHONPATH=src python3 -I scripts/exp_rhc_balancing.py "
                   f"--threads {a.threads}"
                   + (f" --alpha {a.alpha}" if a.alpha is not None else "")
                   + (f" --threshold {a.threshold}" if a.threshold != "paper" else "")
                   + (f" --outdir {a.outdir}" if a.outdir is not None else ""),
        "data": meta, "n_treated": n, "n_control": m, "d": int(X.shape[1]),
        "alpha_trimming": ALPHA, "delta": DELTA, "tau": thr, "sigma": sigma,
        "seed": SEED,
        "smd_definition": "realdata_run.standardized_differences: weighted "
                          "mean difference / sqrt((weighted var_x + weighted "
                          "var_y)/2); *_fixed_sd columns use the unweighted "
                          "pooled SD instead",
        "overlap_definition": "realdata_run._overlap, 40 bins on [0,1], "
                              "logistic propensity of realdata_run._logistic_fit",
        "energy_balancing": "Huling & Mak (2024), J. Causal Inference 12: "
                            "per arm, argmin over the simplex of "
                            "2 sum_i w_i mean_j|X_i - Z_j| - sum_ik w_i w_k "
                            "|X_i - X_k| - mean_jl |Z_j - Z_l| (Z pooled); "
                            "'improved' adds the energy distance between the "
                            "two weighted arms and is solved jointly",
        "kernel_balancing": "per arm, argmin over the simplex of the squared "
                            "MMD to the pooled sample, Gaussian kernel at the "
                            "trimming's bandwidth",
        "solver": "FISTA with projection onto the simplex, stopped when the "
                  f"Frank-Wolfe gap <= {GAP_TOL} or after {MAX_ITER} iterations",
        "solver_info": solver,
        "trimming_reproduces_r10": check,
        "timing_seconds": {k: round(v, 1) for k, v in timing.items()},
        "seconds_total": round(time.time() - t_start, 1),
        "numpy": np.__version__, "python": platform.python_version(),
        "threads": a.threads,
    }
    if a.threshold != "paper" or a.alpha is not None or a.outdir is not None:
        meta_out["v2"] = {"threshold": a.threshold, "alpha": ALPHA, "outdir": a.outdir}
    (TABLES / "r13_rhc_balancing.json").write_text(json.dumps(meta_out, indent=1))
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out[["method", "max_abs_smd", "mean_abs_smd",
                   "worst_covariate_after", "smd_worst_before_cov_after",
                   "ess_frac_x", "ess_frac_y", "support_x", "support_y",
                   "prop_overlap", "mmd_between_arms"]].to_string(index=False))
    print(json.dumps(check, indent=1))
    print(json.dumps(solver, indent=1))
    print(f"done in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()

"""AoS revision: the RHC trimmed cohort as a certified minimizer of the empirical program (2.10).

The published cohort (aos/tables/cohort_macros.tex, Section sec:appcohort) is trimmings.optimal_trimming_mmd with its
default select="least": an 8,000-iteration FISTA solve that stopped at its limit, followed by a ridge continuation
toward uniform weights. Here (2.10) is solved at the same level alpha = rhcLoMin with no penalty of any kind, by the
solver of scripts/aos_realdata_qp.py (FISTA warm start, SMO to a KKT violation of 1e-14, exact polish on the face),
and certified by the Frank-Wolfe duality gap.

Runs (``--run``):

    old        the published cohort, recomputed exactly as scripts/v3_rhc_estimand.py does (pooled standardization,
               pooled median bandwidth with default_rng(0), plugin tau, select="least"); its unpenalized gap is reported
    pooled     the converged unpenalized minimizer with the published protocol (pooled moments, pooled median
               bandwidth): isolates the solver
    batch0     the converged unpenalized minimizer with the covariates standardized by the moments of batch 0 of
               split seed ``--seed`` (default 1000, the first of the eight) and the median bandwidth of that batch,
               exactly the map and bandwidth of splitting.sequential_order(standardize_batch0=True): the primary run

Each run writes results/aos/realdata/cohort/<label>.npz (weights, witness values) and <label>.json (solver record,
counts, cut, balance). Rerunning skips finished labels unless --force.

    PYTHONPATH=src python3 -I scripts/aos_realdata_cohort.py --run old pooled batch0 --threads 4
    PYTHONPATH=src python3 -I scripts/aos_realdata_cohort.py --run batch0 --seed 1001 ... (sensitivity)

Kernel (``--kernel``, added 2026-11-22 for the restructured paper; the default ``gaussian`` is the code above, unchanged):

    gaussian   exp(-|x - y|^2 / (2 sigma^2)), trimmedmmd.kernels.gaussian_kernel (the stored cohort)
    laplace    exp(-|x - y| / sigma), trimmedmmd.aos_fallback.kernel_matrix(kernel="laplace"), the kernel of the
               split-sample bound (scripts/aos_main_realdata.py), with the SAME sigma as the Gaussian run of that label
               (pooled: median heuristic on all patients; batch0: median heuristic on batch 0, = order.sigma of the
               bound). Both kernels have k(x, x) = 1, so the certificate of certify_sets holds as written. The level is
               checked against the Laplace bound (min over the eight splits of results/aos/main/raw/realdata_laplace,
               cut rule "order"). Output: results/aos/main/cohort/<label>.{json,npz}; the "old" run is Gaussian only.

    PYTHONPATH=src python3 -B scripts/aos_realdata_cohort.py --kernel laplace --run pooled batch0 \
        --seed 1000 1001 1002 1003 1004 1005 1006 1007 --threads 4

Covariates (``--covariates``, AoS referee runs 2026-11-22; default ``paper``, the code above unchanged): ``full`` uses
the 45 covariates of realdata.load_rhc(covariates="full") (the 33 plus sex, dnr1 and the ten admission-diagnosis
indicators, 0/1) everywhere -- standardization, bandwidth, Gram matrices, the propensity model of the overlap
statistic and the balance (SMD) -- by setting TRIMMEDMMD_RHC_COVARIATES=full for the process; Laplace kernel only. The
level is the minimum over the eight splits of the 45-covariate Laplace bound (results/aos/main/raw/realdata_laplace/
rhc_full_<seed>.jsonl, scripts/aos_main_realdata.py --contrasts rhc_full); output results/aos/main/cohort_full/.

    PYTHONPATH=src python3 -B scripts/aos_realdata_cohort.py --kernel laplace --covariates full --run pooled batch0 \
        --seed 1000 1001 1002 1003 1004 1005 1006 1007 --threads 4 --notes results/aos/main/ref_realdata/cohort_full.log

Gaussian trimming at the Laplace level (2026-11-24, author: "Laplace for the bound, Gaussian for the trimming"):
``--kernel gaussian --covariates full`` solves (2.10) with the Gaussian kernel at the level of the 45-covariate Laplace
bound (as above) and writes results/aos/main/cohort_full_gauss/.

    PYTHONPATH=src python3 -B scripts/aos_realdata_cohort.py --kernel gaussian --covariates full --run pooled batch0 \
        --seed 1000 1001 1002 1003 1004 1005 1006 1007 --threads 4 --notes results/aos/main/ref_realdata/cohort_full_gauss.log

``--out`` / ``--notes`` redirect the outputs (e.g. to check that the Gaussian default reproduces the stored runs without
touching them). JSON and NPZ files are written to a temporary name and renamed (the pCloud mount corrupted appended
files on 2026-11-21); the notes line is appended by rewrite + rename for the same reason.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path


def _arg(name, default):
    for i, a in enumerate(sys.argv):
        if a == name and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


_T = _arg("--threads", "4")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = _T

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from aos_realdata_qp import certificate, solve_exact  # noqa: E402
from trimmedmmd.experiments.realdata_run import (DELTA, _logistic_fit, _overlap,  # noqa: E402
                                                 standardized_differences)
from trimmedmmd.kernels import gaussian_kernel, median_heuristic  # noqa: E402
from trimmedmmd.realdata import load_rhc  # noqa: E402
from trimmedmmd.splitting import SEQ_EDGES, sequential_order  # noqa: E402
from trimmedmmd.threshold import threshold_function  # noqa: E402
from trimmedmmd.trimmings import optimal_trimming_mmd  # noqa: E402

OUT = ROOT / "results" / "aos" / "realdata" / "cohort"
NOTES = ROOT / "results" / "aos" / "realdata" / "NOTES.md"
OUT_KERNEL = {"gaussian": OUT, "laplace": ROOT / "results" / "aos" / "main" / "cohort"}
NOTES_KERNEL = {"gaussian": NOTES, "laplace": ROOT / "results" / "aos" / "main" / "NOTES.md"}
LAPLACE_RAW = ROOT / "results" / "aos" / "main" / "raw" / "realdata_laplace"
B_BOOT, BOOT_SEED = 2000, 20261117        # as scripts/v3_rhc_estimand.py


def paper_alpha():
    """alpha = \\rhcLoMin of aos/tables/app_macros.tex (read only), checked against the batch-0 rerun of the bound."""
    txt = (ROOT / "aos" / "tables" / "app_macros.tex").read_text()
    a = float(re.search(r"\\newcommand\{\\rhcLoMin\}\{([0-9.]+)\}", txt).group(1))
    b0 = json.loads((ROOT / "results" / "v3" / "rhc_lower_batch0.json").read_text())
    assert abs(b0["min_lower"] - a) < 1e-12, (a, b0["min_lower"])
    return a


def laplace_alpha(contrast="rhc"):
    """The certified RHC level of the Laplace bound: min over the eight splits of the down-closed level (cut rule
    "order") of results/aos/main/raw/realdata_laplace/<contrast>_<seed>.jsonl; for "rhc" checked equal to
    paper_alpha() (for "rhc_full", the 45-covariate run, there is no earlier value to check against)."""
    lows = {}
    for seed in range(1000, 1008):
        recs = sorted((json.loads(s) for s in (LAPLACE_RAW / f"{contrast}_{seed}.jsonl").read_text().splitlines()
                       if s.strip()), key=lambda r: r["alpha"])
        lo = -1.0
        for r in recs:
            assert r["kernel"] == "laplace"
            if not r["order"]["certified"]:
                break
            lo = r["alpha"]
        lows[seed] = lo
    a = min(lows.values())
    if contrast == "rhc":
        a0 = paper_alpha()
        assert abs(a - a0) < 1e-12, (a, a0, lows)
    return a, lows


def gram(A, B, sigma, kernel, same=False):
    """k(A_i, B_j). Gaussian: the package's gaussian_kernel, exactly the stored runs. Laplace: the kernel of the
    split-sample bound (aos_fallback.kernel_matrix; same=True puts exact zeros on the diagonal distance)."""
    if kernel == "gaussian":
        return gaussian_kernel(A, B, sigma)
    from trimmedmmd.aos_fallback import kernel_matrix
    return kernel_matrix(A, B, sigma, kernel="laplace", same=same)


def _replace_write(path, write):
    """write(tmp_path) then rename onto path (same directory)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp" + path.suffix)
    write(tmp)
    os.replace(tmp, path)


def kish(w):
    return float(w.sum() ** 2 / (w ** 2).sum())


def rhc_outcome_and_propensity():
    from v3_rhc_estimand import rhc_outcome
    Xp, Yp, names, _ = load_rhc()                       # pooled standardization, as the published propensity fit
    Z = np.vstack([Xp, Yp])
    _, prop = _logistic_fit(Z, np.r_[np.ones(len(Xp)), np.zeros(len(Yp))])
    yx, yy = rhc_outcome()
    return names, prop, yx, yy


def describe(w, v, Xraw, Yraw, prop, yx, yy, alpha, tol_rel=1e-9):
    """Counts, balance and the mortality difference of a weighting (SMD on the raw covariates: SMDs are invariant
    under the coordinatewise affine maps of either standardization)."""
    from v3_rhc_estimand import boot_ci, risk_difference
    n, m = len(w), len(v)
    cx, cy = 1.0 / (n * (1.0 - alpha)), 1.0 / (m * (1.0 - alpha))
    out = {}
    for arm, u, c in (("x", w, cx), ("y", v, cy)):
        out[f"cap_{arm}"] = int(np.sum(u >= c * (1 - tol_rel)))
        out[f"zero_{arm}"] = int(np.sum(u <= c * tol_rel))
        out[f"frac_{arm}"] = int(np.sum((u > c * tol_rel) & (u < c * (1 - tol_rel))))
        out[f"kept_{arm}"] = int(np.sum(u > c * tol_rel))
        out[f"kept_exact_{arm}"] = int(np.sum(u > 0))
        out[f"ess_frac_{arm}"] = kish(u) / len(u)
        out[f"frac_mass_{arm}"] = float(u[(u > c * tol_rel) & (u < c * (1 - tol_rel))].sum())
    smd = standardized_differences(Xraw, Yraw, w / w.sum(), v / v.sum())
    out["smd"] = smd.tolist()
    out["max_abs_smd"] = float(np.abs(smd).max())
    out["mean_abs_smd"] = float(np.abs(smd).mean())
    out["n_abs_smd_above_0.1"] = int(np.sum(np.abs(smd) > 0.1))
    out["prop_overlap"] = _overlap(prop[:n], prop[n:], w, v)
    out["rd_dth30"] = risk_difference(yx, yy, w, v)
    lo, hi = boot_ci(yx, yy, w, v, np.random.default_rng(BOOT_SEED))
    out["rd_lo"], out["rd_hi"] = lo, hi
    return out


def structure(w, v, dx, dy, alpha, D, tol_rel=1e-9):
    """The level-set reading of Proposition prop:bangbang(ii) at (w, v): the cut, and whether the cap / zero / fractional
    sets sit below / above / at it."""
    n, m = len(w), len(v)
    cx, cy = 1.0 / (n * (1.0 - alpha)), 1.0 / (m * (1.0 - alpha))
    na, ma = int(np.floor(n * (1 - alpha) + 1e-12)), int(np.floor(m * (1 - alpha) + 1e-12))
    res = {"n_alpha": na, "m_alpha": ma, "n_alpha_plus_1": na + 1, "m_alpha_plus_1": ma + 1}
    for arm, u, c, d, k, lower in (("x", w, cx, dx, na, True), ("y", v, cy, dy, ma, False)):
        capm = u >= c * (1 - tol_rel)
        zero = u <= c * tol_rel
        frac = ~capm & ~zero
        ds = np.sort(d) if lower else np.sort(d)[::-1]
        t_order = float(ds[k]) if k < len(d) else float("inf")      # (k+1)-th smallest (x) / largest (y)
        rng_d = float(d.max() - d.min())
        sd_d = float(d.std())
        r = {"cut_order_stat": t_order, "witness_range": rng_d, "witness_sd": sd_d}
        if frac.any():
            r["cut_frac_mean"] = float(d[frac].mean())
            r["frac_witness_spread"] = float(d[frac].max() - d[frac].min())
        t = r.get("cut_frac_mean", t_order)
        r["cut"] = t
        r["cut_normalized"] = t / D if D > 0 else float("nan")
        # sign conditions: x keeps the SMALL witness values, y the LARGE ones
        if lower:
            r["max_cap_minus_cut"] = float((d[capm] - t).max()) if capm.any() else float("nan")
            r["min_zero_minus_cut"] = float((d[zero] - t).min()) if zero.any() else float("nan")
        else:
            r["min_cap_minus_cut"] = float((d[capm] - t).min()) if capm.any() else float("nan")
            r["max_zero_minus_cut"] = float((d[zero] - t).max()) if zero.any() else float("nan")
        for e in (1e-12, 1e-10, 1e-8, 1e-6):
            r[f"n_within_{e:g}_sd_of_cut"] = int(np.sum(np.abs(d - t) <= e * sd_d))
        res[arm] = r
    return res


def certify_sets(w, v, dx, dy, alpha, gap, lam_min, face=None, tol_rel=1e-9):
    """Turn the duality gap into statements about the exact (unique) minimizer z*.

    F(z) - F* >= (z - z*)' H (z - z*) >= lam_min ||z - z*||^2 by first-order optimality at z*, with lam_min the
    smallest eigenvalue of H (= that of the Gram matrix G, H = S G S). Hence ||z - z*||_2 <= sqrt(gap / lam_min), and,
    since k(x, x) = 1, |d(x) - d*(x)| <= ||d - d*||_H <= sqrt(gap) at every point x. If the cap units' witness values lie
    below the cut, the zero units' above it, both by more than 2 sqrt(gap) plus the spread of the fractional units, then
    at z* the same units are at the cap and at zero (the mass constraint pins the cut of z* inside the window of the
    fractional units), and if every fractional weight is farther than sqrt(gap / lam_min) from both bounds, every one
    of them is fractional at z* as well: the counts are then exact for the minimizer itself."""
    n, m = len(w), len(v)
    cx, cy = 1.0 / (n * (1.0 - alpha)), 1.0 / (m * (1.0 - alpha))
    g = max(gap, 0.0)
    l2 = float(np.sqrt(g / lam_min)) if lam_min > 0 else float("inf")
    sup = float(np.sqrt(g))
    out = {"lam_min_gram": float(lam_min), "l2_bound_to_minimizer": l2, "witness_sup_bound": sup}
    ok = True
    for arm, u, c, d, low in (("x", w, cx, dx, True), ("y", v, cy, dy, False)):
        capm, zero = u >= c * (1 - tol_rel), u <= c * tol_rel
        frac = ~capm & ~zero
        t = float(d[frac].mean()) if frac.any() else float("nan")
        spread = float(d[frac].max() - d[frac].min()) if frac.any() else 0.0
        sgn = 1.0 if low else -1.0          # x keeps small witness values, y large ones
        cap_margin = float(np.min(sgn * (t - d[capm]))) if capm.any() else float("inf")
        zero_margin = float(np.min(sgn * (d[zero] - t))) if zero.any() else float("inf")
        need = 2.0 * sup + spread
        dist = float(np.min(np.minimum(u[frac], c - u[frac]))) if frac.any() else float("inf")
        r = {"cap_margin": cap_margin, "zero_margin": zero_margin, "window_needed": need,
             "min_frac_distance_to_bounds": dist, "min_frac_distance_over_cap": dist / c,
             "sets_certified": bool(cap_margin > need and zero_margin > need),
             "fractional_certified": bool(dist > l2)}
        ok = ok and r["sets_certified"] and r["fractional_certified"]
        out[arm] = r
    # Second route to the fractional count: once the cap and zero sets are certified, z* minimizes F over the face
    # {z_cap = cap, z_zero = 0, sums = 1} intersected with the box; if the minimizer over the affine face (the exact
    # polish solve, solver_info.polish) lies strictly inside the box, it is z* (strict convexity), so every window
    # unit is fractional at z*. The linear solve is accurate to about cond * 1e-16 relative.
    sets = all(out[a]["sets_certified"] for a in ("x", "y"))
    if face is not None and face.get("feasible") and face.get("min_dist_over_cap") is not None:
        margin = float(face["min_dist_over_cap"])
        out["face_min_dist_over_cap"] = margin
        out["face_route_certified"] = bool(sets and margin > 1e3 * face.get("cond", 1e4) * 1e-16)
    else:
        out["face_route_certified"] = False
    out["counts_exact_for_minimizer"] = bool(ok or (sets and out["face_route_certified"]))
    return out


def build(run, seed):
    Xraw, Yraw, names, _ = load_rhc(standardize=False)
    n, m = len(Xraw), len(Yraw)
    if run in ("old", "pooled"):
        Xs, Ys, _, _ = load_rhc()
        sigma = float(median_heuristic(Xs, Ys, rng=np.random.default_rng(0)))
        meta = {"standardization": "pooled moments of all patients (realdata.load_rhc)",
                "bandwidth": "median heuristic on all patients, 2,000-point subsample, default_rng(0)"}
    else:
        split_seed = int(np.random.default_rng(seed).integers(0, 2**31 - 1))
        rng = np.random.default_rng(split_seed)
        ix, iy = rng.permutation(n), rng.permutation(m)
        b1x, b1y = int(round(SEQ_EDGES[1] * n)), int(round(SEQ_EDGES[1] * m))
        B0 = np.vstack([Xraw[ix][:b1x], Yraw[iy][:b1y]])
        mu, sd = B0.mean(axis=0), B0.std(axis=0)
        sd[sd == 0] = 1.0
        order = sequential_order(Xraw, Yraw, np.random.default_rng(split_seed), edges=SEQ_EDGES,
                                 standardize_batch0=True)
        assert np.allclose(order.X, (Xraw[ix] - mu) / sd) and np.allclose(order.Y, (Yraw[iy] - mu) / sd)
        sigma = float(order.sigma)
        Xs, Ys = (Xraw - mu) / sd, (Yraw - mu) / sd
        meta = {"standardization": f"moments of batch 0 of split seed {seed} (split_seed {split_seed}), "
                                   f"{b1x} + {b1y} patients", "bandwidth": "median heuristic on that batch 0",
                "split_seed": split_seed, "batch0_sizes": [b1x, b1y]}
    return Xraw, Yraw, names, Xs, Ys, sigma, meta


def note(line, notes=None):
    notes = Path(notes or NOTES)
    old = notes.read_text() if notes.exists() else ""
    _replace_write(notes, lambda p: Path(p).write_text(old + f"- {time.strftime('%F %H:%M')}  {line}\n"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", default=["old", "pooled", "batch0"])
    ap.add_argument("--seed", type=int, nargs="+", default=[1000])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--eps", type=float, default=1e-14)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--kernel", choices=["gaussian", "laplace"], default="gaussian")
    ap.add_argument("--out", default=None, help="output directory (default: by kernel, see the docstring)")
    ap.add_argument("--notes", default=None, help="notes file (default: by kernel)")
    ap.add_argument("--covariates", choices=["paper", "full"], default="paper",
                    help="RHC covariate set (realdata.load_rhc); see the docstring")
    a = ap.parse_args()
    if a.covariates == "full":
        from trimmedmmd.realdata import RHC_COVARIATES_ENV
        os.environ[RHC_COVARIATES_ENV] = "full"       # every load_rhc() of this process (build, propensity, SMD)
    full_dir = "cohort_full" if a.kernel == "laplace" else "cohort_full_gauss"
    out = Path(a.out) if a.out else (ROOT / "results" / "aos" / "main" / full_dir if a.covariates == "full"
                                     else OUT_KERNEL[a.kernel])
    notes = Path(a.notes) if a.notes else NOTES_KERNEL[a.kernel]
    if a.kernel != "gaussian" and "old" in a.run:
        raise SystemExit("--run old is the published Gaussian solve; it has no Laplace version")
    out.mkdir(parents=True, exist_ok=True)
    if a.kernel == "gaussian" and a.covariates != "full":
        alpha = paper_alpha()
    else:
        alpha, split_lows = laplace_alpha("rhc_full" if a.covariates == "full" else "rhc")
        print(f"alpha = {alpha}: min over splits of the Laplace bound {split_lows}", flush=True)
    names, prop, yx, yy = rhc_outcome_and_propensity()
    for run in a.run:
        for seed in (a.seed if run == "batch0" else [None]):
            label = run if run != "batch0" else f"batch0_seed{seed}"
            if (out / f"{label}.json").exists() and not a.force:
                print(f"{label}: done, skipping")
                continue
            t0 = time.time()
            Xraw, Yraw, names, Xs, Ys, sigma, meta = build(run, seed)
            n, m = len(Xs), len(Ys)
            tau = float(threshold_function("plugin", n, m, DELTA)(alpha))
            Kxx = gram(Xs, Xs, sigma, a.kernel, same=True)
            Kyy = gram(Ys, Ys, sigma, a.kernel, same=True)
            Kxy = gram(Xs, Ys, sigma, a.kernel)
            if a.kernel == "laplace":
                assert np.all(np.diag(Kxx) == 1.0) and np.all(np.diag(Kyy) == 1.0)
                meta["bandwidth"] += "; Laplace kernel exp(-|x - y| / sigma) at that sigma"
            caps = np.array([1.0 / (n * (1.0 - alpha)), 1.0 / (m * (1.0 - alpha))])
            print(f"{label}: kernel={a.kernel} n={n} m={m} alpha={alpha} sigma={sigma:.6f} tau={tau:.6f}", flush=True)
            if run == "old":
                tr = optimal_trimming_mmd(Xs, Ys, alpha, sigma=sigma, rng=np.random.default_rng(0), tau=tau)
                w, v = np.asarray(tr.w, float), np.asarray(tr.v, float)
                N = n + m
                H = np.empty((N, N))
                H[:n, :n], H[n:, n:], H[:n, n:], H[n:, :n] = Kxx, Kyy, -Kxy, -Kxy.T
                z = np.concatenate([w, v])
                F, gap, gaps, Hz = certificate(H, z, n, caps)
                Fl, gapl, _, _ = certificate(H, z, n, caps, dtype=np.longdouble)
                del H
                info = {"solver": "trimmings.optimal_trimming_mmd(select='least'): FISTA max_iter=8000 tol=1e-12, then "
                                  "4-step ridge continuation toward uniform weights",
                        "converged_flag": bool(tr.meta.get("converged")), "n_iter": int(tr.meta.get("n_iter")),
                        "unique_flag": bool(tr.unique), "value_reported": float(tr.value),
                        "F": F, "gap": gap, "gaps": gaps, "F_longdouble": Fl, "gap_longdouble": gapl,
                        "D_hat": float(np.sqrt(F)), "D_hat_lower": float(np.sqrt(max(F - max(gap, gapl), 0.0)))}
                dx, dy = Hz[:n], -Hz[n:]
            else:
                w, v, info = solve_exact(Kxx, Kyy, Kxy, alpha, eps=a.eps,
                                         log=lambda s: print(s, flush=True))
                dx, dy = info.pop("_witness_x"), info.pop("_witness_y")
                info["solver"] = ("FISTA warm start (solver.solve_gram, ridge 0, 2000 it), SMO with second-order "
                                  "working-set selection to KKT violation eps, exact polish on the face; no penalty")
            if run != "old":
                G = np.empty((n + m, n + m))
                G[:n, :n], G[n:, n:], G[:n, n:], G[n:, :n] = Kxx, Kyy, Kxy, Kxy.T
                lam = np.linalg.eigvalsh(G)
                info["gram_eig_min"], info["gram_eig_max"] = float(lam[0]), float(lam[-1])
                del G, lam
            del Kxx, Kyy, Kxy
            rec = {"label": label, "run": run, "seed": seed, "alpha": alpha, "sigma": sigma, "tau_plugin": tau,
                   "n": n, "m": m, "kernel": a.kernel, **meta, "solver_info": info}
            if a.covariates != "paper":
                rec["covariates"], rec["names"] = a.covariates, list(names)
            rec["structure"] = structure(w, v, dx, dy, alpha, info["D_hat"])
            if run != "old":
                rec["certify"] = certify_sets(w, v, dx, dy, alpha, max(info["gap"], info["gap_longdouble"]),
                                              info["gram_eig_min"], face=info.get("polish"))
            rec["describe"] = describe(w, v, Xraw, Yraw, prop, yx, yy, alpha)
            rec["seconds"] = round(time.time() - t0, 1)
            _replace_write(out / f"{label}.npz", lambda p: np.savez_compressed(p, w=w, v=v, dx=dx, dy=dy))
            js = json.dumps(rec, indent=1, default=float)
            _replace_write(out / f"{label}.json", lambda p: Path(p).write_text(js))
            assert (out / f"{label}.json").read_text() == js
            d = rec["describe"]
            msg = (f"cohort {label} ({a.kernel}): D={info['D_hat']:.6g} gap={info['gap']:.3g} (longdouble {info['gap_longdouble']:.3g}); "
                   f"x cap/zero/frac/kept {d['cap_x']}/{d['zero_x']}/{d['frac_x']}/{d['kept_x']}, "
                   f"y {d['cap_y']}/{d['zero_y']}/{d['frac_y']}/{d['kept_y']}; max|SMD| {d['max_abs_smd']:.3f} "
                   f"mean {d['mean_abs_smd']:.3f}; {rec['seconds']} s")
            print(msg, flush=True)
            note(msg, notes)


if __name__ == "__main__":
    main()

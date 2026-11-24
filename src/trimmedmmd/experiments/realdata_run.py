"""Real-data applications.

Two datasets, chosen because they stress different claims:

* **RHC** -- 33 baseline covariates, 2184 catheterized vs 3551 not. The
  question ("how much of each arm has to be discarded before the two are
  comparable?") is one the causal-inference literature answers with a rule
  of thumb on the propensity score. We answer it with a certificate, and
  report the rule of thumb alongside so the two can be compared.
* **NHANES** -- 24-hour activity profiles, ~5900 participants. Functional
  observations, where the guarantee is supposed to be dimension-free.

Like the synthetic study, each analysis can come out badly: the lower bound
can certify nothing, the split-to-split spread can swamp the estimate, and
the margin-based upper bound can land below the certified lower bound --
which would be a visible contradiction rather than a quiet one. All three
are reported when they happen.

Everything long-running is checkpointed per replicate, so an interrupted
run resumes instead of restarting.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..checkpoint import (ReplicateStore, cell_key, default_root,
                          map_replicates)
from ..discrepancy import plain_mmd, trimmed_mmd_path
from ..estimate import alpha_hat_from_path
from ..kernels import RFF, median_heuristic
from ..realdata import load_nhanes_activity, load_rhc, nhanes_groups
from ..resources import detect
from ..solver import solve_features
from ..splitting import lower_bound_tv, split_upper_bound, upper_bound_tv
from ..threshold import mmd_radius, tau
from ..threshold import closed_form_constant, sample_size_for, tau_A, threshold_function

RESULTS = Path(__file__).resolve().parents[3] / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
# Run state, not results: kept off a synced mount, where checkpoint
# files have been seen to vanish mid-run (see checkpoint.default_root).
CHECKPOINTS = default_root(RESULTS / "checkpoints")

# Bumped when the NHANES cycles were pooled (G -> G+H): the dataset name in
# the checkpoint key did not change, but the data behind it did, so every
# stored replicate had to be invalidated.
CODE_VERSION = "2026-09-18-realdata-b"

ALPHA_GRID = np.round(np.arange(0.0, 0.751, 0.05), 3)
DELTA = 0.05

# Revision v2 (plan items V8-V9). Without the new flags nothing below changes.
# With them: the threshold is selected by name ("paper" = v1 tau, "plugin" =
# tau_new with the Gaussian coupled constants, exact Gram matrices throughout
# this module); the certificate is the finite-sample betting certificate
# (Theorem 1) with per-level witnesses, plus the bootstrap of Theorem 16 on
# the same splits and witnesses when certificate="both"; the child contrast is
# certified on the extended grid 0:0.05:0.95 directly (v1 merged it by hand
# from r6_extended_grid.json, commit 2f0e4c6).
# certificate="sequential" (the paper's main split result after author
# decision Q7: Theorem thm:betting with batches e = (0, 1/4, 1/2, 1), one
# wealth, horizon fixed in advance) reports the sequential certificate in the
# v1 columns; certificate="all" adds the single-split betting (*_betting), the
# Theorem 16 bootstrap on the single split (*_bootstrap, reproduces v1), the
# closed-form Bernstein variant (*_bernstein) and the single-split alpha = 0
# witness (*_alpha0). Split seeds 1000 + i as in v1 for every method.
V2_GRID_MAX = {"nhanes_child": 0.95}
V2_CODE_VERSION = "2026-11-08-realdata-v2-seq"   # bumped: method sets changed (sequential)

# certificate -> (methods of lower_bound_betting, the reported one first; alpha0 extra?)
V2_METHODS = {
    "betting": (("betting", "bernstein"), True),
    "both": (("betting", "bernstein", "bootstrap"), True),
    "sequential": (("sequential",), False),
    "all": (("sequential", "betting", "bernstein", "bootstrap"), True),
}


def _grid_for(name: str, v2: bool) -> np.ndarray:
    if v2 and name in V2_GRID_MAX:
        return np.round(np.arange(0.0, V2_GRID_MAX[name] + 1e-9, 0.05), 3)
    return ALPHA_GRID


def _thr(threshold: str, n: int, m: int, delta: float = DELTA):
    """alpha -> the selected threshold (exact Gaussian Gram: coupled constants)."""
    if threshold == "paper":
        return lambda a: float(tau(n, m, a, delta))
    return threshold_function(threshold, n, m, delta)


def configure_outdir(outdir) -> None:
    """Send tables, figures and the manifest to ``outdir`` (v1 layout).

    Checkpoints of a v2 run live under ``outdir/checkpoints`` (unless
    TRIMMEDMMD_CHECKPOINTS is set), so no v1 run state is touched or reused.
    """
    global RESULTS, TABLES, FIGURES, CHECKPOINTS
    root = Path(outdir)
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[3] / root
    RESULTS = root
    TABLES = root / "tables"
    FIGURES = root / "figures"
    CHECKPOINTS = default_root(root / "checkpoints")
    TABLES.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# workers (top level so they pickle)
# --------------------------------------------------------------------------

_CACHE: dict = {}


def _dataset(name: str):
    """Load (and memoize) one of the named two-sample problems."""
    if name in _CACHE:
        return _CACHE[name]
    if name == "rhc":
        X, Y, _, _ = load_rhc()
    elif name.startswith("nhanes_"):
        curves, cov, _ = load_nhanes_activity()
        X, Y, _, _ = nhanes_groups(curves, cov, by=name.split("_", 1)[1])
    else:
        raise ValueError(name)
    _CACHE[name] = (np.asarray(X, float), np.asarray(Y, float))
    return _CACHE[name]


def w_lower(args):
    """One split-sample lower bound on TV, at one random split."""
    name, seed = args
    X, Y = _dataset(name)
    rng = np.random.default_rng(seed)
    lo, _ = lower_bound_tv(X, Y, ALPHA_GRID, delta=DELTA, rng=rng)
    return {"lower": float(lo), "seed": int(seed)}


def w_lower_v2(args):
    """One split of the v2 lower bound, all methods on the same split seed
    (``default_rng(seed)`` as ``w_lower``, so the bootstrap reproduces v1).

    ``certificate`` selects the method set (:data:`V2_METHODS`): the reported
    certificate is the first method -- the sequential betting certificate
    (Theorem thm:betting, batches (0, 1/4, 1/2, 1)) for "sequential"/"all",
    the single-split betting certificate for "betting"/"both" -- with
    per-level witnesses; the others are extra columns: single-split betting,
    the closed-form empirical-Bernstein variant (Corollary 1), the Theorem 16
    bootstrap, and the single-split betting certificate with ONE alpha = 0
    witness (level-specific cut points).

    Bounds are computed with ``none_value=-1`` (nothing certified, not even
    TV > 0); the r2 table maps that to 0 as v1 does and counts it separately.
    """
    from ..splitting import lower_bound_betting
    name, certificate, grid, seed = args
    X, Y = _dataset(name)
    methods, with_alpha0 = V2_METHODS[certificate]
    n_boot = 1000 if "bootstrap" in methods else 0
    b, rows = lower_bound_betting(X, Y, np.asarray(grid), delta=DELTA,
                                  rng=np.random.default_rng(seed), methods=methods,
                                  n_boot=n_boot, none_value=-1.0)
    keep = ("alpha", "sequential", "seq_log_final_wealth", "seq_horizon", "seq_pairs_bet",
            "seq_skipped", "betting", "bernstein", "bootstrap", "s", "sp", "delta_hat",
            "log_max_wealth", "lcb", "T_hat", "norm")
    main = methods[0]
    out = {"seed": int(seed), "certificate": main, "lower": float(b[main]),
           "levels": [{k: r[k] for k in keep if k in r} for r in rows]}
    for mth in methods[1:]:
        out[f"lower_{mth}"] = float(b[mth])
    if with_alpha0:
        b0, rows0 = lower_bound_betting(X, Y, np.asarray(grid), delta=DELTA,
                                        rng=np.random.default_rng(seed), witness="alpha0",
                                        methods=("betting",), none_value=-1.0)
        out["lower_alpha0"] = float(b0["betting"])
        out["levels_alpha0"] = [{"alpha": r["alpha"], "betting": r["betting"],
                                 "delta_hat": r.get("delta_hat")} for r in rows0]
    return out


def w_upper(args):
    """One split-sample upper bound on D_alpha, at one random split."""
    name, alpha, seed = args
    X, Y = _dataset(name)
    rng = np.random.default_rng(seed)
    _, d_up = split_upper_bound(X, Y, alpha, delta=DELTA, rng=rng)
    return {"d_upper": float(d_up), "alpha": float(alpha), "seed": int(seed)}


def _map(fn, jobs, workers, experiment, params, seed_index=-1, code_version=None):
    """Checkpointed parallel map; seeds are the last element of each job."""
    params = dict(params, _code_version=code_version or CODE_VERSION)
    key = cell_key(experiment, params)
    store = ReplicateStore(CHECKPOINTS)
    store.write_meta(key, params)
    seeds = [int(j[seed_index]) for j in jobs]
    have = len(store.load(key))
    if have:
        print(f"    resuming {experiment}: {have}/{len(jobs)} already stored")
    return map_replicates(fn, jobs, seeds, workers, store=store, key=key)


# --------------------------------------------------------------------------
# the path and the point estimate (one deterministic fit per dataset)
# --------------------------------------------------------------------------

def discrepancy_path(name: str, seed: int = 0, threshold: str = "paper", reuse=None):
    """The alpha-path, the analytic threshold, and alpha-hat.

    The path solve does not depend on the threshold (no early stop): only the
    ``analytic_tau``/``rejects``/``alpha_hat_analytic`` columns do. With
    ``threshold != "paper"`` those hold the selected threshold and
    ``tau_paper``/``rejects_paper``/``alpha_hat_paper`` keep v1's. ``reuse`` =
    ``(paths_df, summary_df)`` of an earlier run (e.g. v1's r1_*.csv) skips
    the solve and re-applies the threshold to its D-hat path.
    """
    X, Y = _dataset(name)
    n, m = len(X), len(Y)
    if reuse is not None:
        pr, sr = reuse
        g = pr[pr.dataset == name].sort_values("alpha")
        if not np.allclose(g.alpha.to_numpy(), ALPHA_GRID):
            raise ValueError(f"reused path for {name} is not on ALPHA_GRID")
        d = g.d_hat.to_numpy(float)
        conv = [bool(c) for c in g.converged]
        srow = sr.set_index("dataset").loc[name]
        if (int(srow.n), int(srow.m)) != (n, m):
            raise ValueError(f"reused summary for {name} has other sample sizes")
        sigma, secs, mmd = float(srow.sigma), float(srow.seconds), float(srow.plain_mmd)
    else:
        rng = np.random.default_rng(seed)
        sigma = float(median_heuristic(X, Y, rng=rng))

        t0 = time.time()
        mmd = float(plain_mmd(X, Y, sigma=sigma, rng=rng))
        _, d, res = trimmed_mmd_path(X, Y, ALPHA_GRID, sigma=sigma, rng=rng)
        secs = time.time() - t0
        conv = [bool(r.converged) for r in res]

    fn = _thr(threshold, n, m)
    thr = np.array([fn(a) for a in ALPHA_GRID])
    rows = pd.DataFrame({
        "dataset": name, "alpha": ALPHA_GRID, "d_hat": d,
        "analytic_tau": thr, "rejects": d > thr, "converged": conv,
    })
    summary = {
        "dataset": name, "n": n, "m": m, "d": int(X.shape[1]),
        "sigma": sigma, "seconds": secs,
        "plain_mmd": mmd,
        "mmd_radius": float(mmd_radius(n, m, DELTA)),
        "mmd_rejects_equality": bool(mmd > mmd_radius(n, m, DELTA)),
        "alpha_hat_analytic": float(alpha_hat_from_path(ALPHA_GRID, d, n, m, DELTA,
                                                        threshold_fn=fn)),
        "all_converged": bool(all(conv)),
    }
    if threshold != "paper":
        tp = np.array([tau(n, m, a, DELTA) for a in ALPHA_GRID])
        rows["tau_paper"] = tp
        rows["rejects_paper"] = d > tp
        rows["threshold"] = threshold
        summary["alpha_hat_paper"] = float(alpha_hat_from_path(ALPHA_GRID, d, n, m, DELTA))
        rej = ALPHA_GRID[d > thr]
        summary["alpha_last"] = float(rej.max()) if rej.size else np.nan
        summary["threshold"] = threshold
        summary["reused_path"] = reuse is not None
    return rows, summary


def convergence_audit(name: str, alphas, max_iter=40000, seed: int = 0):
    """Re-solve the flagged levels with a much larger iteration budget.

    The stopping rule is *relative* in the objective, which is a squared
    discrepancy. As alpha approaches the level where the regions touch, the
    objective goes to zero and a fixed relative tolerance becomes an ever
    stricter absolute one, so hitting ``max_iter`` there is expected. What
    matters is not whether the flag fired but whether the reported number
    moves when the budget is raised -- so we measure that instead of
    asserting it.
    """
    X, Y = _dataset(name)
    rng = np.random.default_rng(seed)
    sigma = float(median_heuristic(X, Y, rng=rng))
    alphas = np.asarray(sorted(alphas), dtype=float)

    _, d_ref, res_ref = trimmed_mmd_path(
        X, Y, alphas, sigma=sigma, rng=np.random.default_rng(seed))
    _, d_big, res_big = trimmed_mmd_path(
        X, Y, alphas, sigma=sigma, rng=np.random.default_rng(seed),
        max_iter=max_iter)

    return pd.DataFrame({
        "dataset": name, "alpha": alphas,
        "d_hat_default": d_ref, "d_hat_long": d_big,
        "abs_change": np.abs(d_big - d_ref),
        "converged_default": [bool(r.converged) for r in res_ref],
        "converged_long": [bool(r.converged) for r in res_big],
        "iters_default": [int(r.n_iter) for r in res_ref],
        "iters_long": [int(r.n_iter) for r in res_big],
    })


# --------------------------------------------------------------------------
# experiments
# --------------------------------------------------------------------------

def extended_grid_check(name: str, upper: float = 0.95, step: float = 0.05,
                        reps: int = 8, workers: int = 1):
    """Re-certify one contrast on a longer grid, to uncensor the bound.

    When every split certifies the last level on the standard grid, the
    reported bound is a property of the grid rather than of the data. This
    pushes the grid out and reports where the certificate actually stops,
    plus whether it is still censored at the new maximum.
    """
    X, Y = _dataset(name)
    grid = np.round(np.arange(0.0, upper + 1e-9, step), 3)
    out = []
    for s in range(reps):
        lo, _ = lower_bound_tv(X, Y, grid, delta=DELTA,
                               rng=np.random.default_rng(1000 + s))
        out.append(float(lo))
    lo = np.array(out)
    gmax = float(grid.max())
    return {
        "dataset": name, "grid_max": gmax, "splits": reps,
        "median_lower": float(np.median(lo)), "min_lower": float(lo.min()),
        "max_lower": float(lo.max()),
        "frac_at_grid_max": float((lo >= gmax).mean()),
        "values": lo.tolist(),
    }


def single_cycle_check(by: str = "child", threshold: str = "paper"):
    """Does the headline contrast fire on one NHANES cycle alone?

    The paper claims pooling the two cycles is what carries this contrast
    over the analytic threshold. That is a claim about a counterfactual
    dataset, so it gets measured rather than asserted: this evaluates the
    same contrast on each cycle separately and on the pool.
    """
    rows = []
    for cycles in (("G",), ("H",), ("G", "H")):
        curves, cov, _ = load_nhanes_activity(cycles=cycles)
        X, Y, la, lb = nhanes_groups(curves, cov, by=by)
        n, m = len(X), len(Y)
        rng = np.random.default_rng(0)
        sigma = float(median_heuristic(X, Y, rng=rng))
        d0 = float(plain_mmd(X, Y, sigma=sigma, rng=rng))
        t0 = float(_thr(threshold, n, m)(0.0))
        row = {"cycles": "+".join(cycles), "contrast": f"{la} vs {lb}",
               "n": n, "m": m, "d_hat0": d0, "tau0": t0,
               "fires": bool(d0 > t0), "required_n": required_n(d0, threshold=threshold)}
        if threshold != "paper":
            row.update(threshold=threshold, tau0_paper=float(tau(n, m, 0.0, DELTA)),
                       required_n_paper=required_n(d0))
        rows.append(row)
    return pd.DataFrame(rows)


def r_paths(names, threshold="paper", reuse=None):
    print("paths and point estimates:")
    rows, summaries = [], []
    for nm in names:
        r, s = discrepancy_path(nm, threshold=threshold, reuse=reuse)
        rows.append(r)
        summaries.append(s)
        print(f"  {nm}: n={s['n']} m={s['m']} d={s['d']} "
              f"MMD={s['plain_mmd']:.4f} (radius {s['mmd_radius']:.4f}) "
              f"alpha_hat={s['alpha_hat_analytic']:.2f} [{s['seconds']:.0f}s]")
        if not s["all_converged"]:
            print("    WARNING: some path solves did not converge")
    return pd.concat(rows, ignore_index=True), pd.DataFrame(summaries)


def r_lower_bounds_v2(names, workers, reps, certificate="betting"):
    """r2 with the finite-sample certificates of revision v2.

    ``min/median/max_lower`` etc. hold the reported certificate: the
    sequential betting certificate (Theorem thm:betting, batches
    (0, 1/4, 1/2, 1)) for ``certificate`` in {"sequential", "all"}, the
    single-split betting certificate for {"betting", "both"}; per-level
    witnesses in both cases. Extra columns (see :data:`V2_METHODS`):
    ``*_betting`` (single split, with "all"), ``*_bernstein`` (Corollary 1),
    ``*_alpha0`` (single-split betting, one alpha = 0 witness) and
    ``*_bootstrap`` (Theorem 16 on the single split; it reproduces v1's r2
    split by split). The child contrast uses the extended grid. ``lowers*``
    list the per-split bounds (the v2 analogue of r6's ``values``);
    ``frac_certifying_tv0`` is the fraction of splits that certify TV > 0
    (level 0 rejected), which v1's ``frac_certifying_any`` (bound > 0) does
    not distinguish. Split seeds 1000 + i, as v1.
    """
    if certificate not in V2_METHODS:
        raise ValueError(certificate)
    methods, with_alpha0 = V2_METHODS[certificate]
    main = methods[0]
    print(f"certified lower bounds, {certificate}: reported {main} ({reps} splits each):")
    out = []
    extra = list(methods[1:]) + (["alpha0"] if with_alpha0 else [])
    for nm in names:
        grid = _grid_for(nm, True)
        jobs = [(nm, certificate, tuple(float(a) for a in grid), 1000 + i) for i in range(reps)]
        params = {"dataset": nm, "reps": reps, "grid": grid.tolist(), "delta": DELTA,
                  "certificate": certificate, "witness": "per_level+alpha0",
                  "c": 0.5, "w0": 20.0, "n_boot": 1000 if "bootstrap" in methods else 0}
        if main == "sequential":
            params.update(methods=list(methods), edges=[0.0, 0.25, 0.5, 1.0],
                          horizon="fixed", witness="per_level"
                          + ("+alpha0" if with_alpha0 else ""))
        vals = _map(w_lower_v2, jobs, workers, "r_lower_v2", params,
                    code_version=V2_CODE_VERSION)
        grid_max = float(grid.max())

        def summ(key, suffix=""):
            raw = np.array([v[key] for v in vals], float)
            lo = np.maximum(raw, 0.0)
            return {
                f"mean_lower{suffix}": float(lo.mean()),
                f"median_lower{suffix}": float(np.median(lo)),
                f"min_lower{suffix}": float(lo.min()), f"max_lower{suffix}": float(lo.max()),
                f"sd_lower{suffix}": float(lo.std(ddof=1)) if reps > 1 else np.nan,
                f"frac_certifying_any{suffix}": float((lo > 0).mean()),
                f"frac_certifying_tv0{suffix}": float((raw >= 0).mean()),
                f"frac_at_grid_max{suffix}": float((lo >= grid_max).mean()),
                f"lowers{suffix}": ";".join(f"{x:g}" for x in raw),
            }
        row = {"dataset": nm, "splits": reps}
        row.update(summ("lower"))
        row.update(grid_max=grid_max, certificate=main)
        for e in extra:
            row.update(summ(f"lower_{e}", f"_{e}"))
        out.append(row)
        if row["frac_at_grid_max"] > 0:
            print(f"    NOTE: {row['frac_at_grid_max']:.0%} of splits certified the grid "
                  f"maximum {grid_max:g}; this bound is censored, extend the grid")
        print(f"  {nm}: {main} median {row['median_lower']:.2f}, range "
              f"[{row['min_lower']:.2f}, {row['max_lower']:.2f}]"
              + "".join(f"; {e} [{row[f'min_lower_{e}']:.2f}, {row[f'max_lower_{e}']:.2f}]"
                        for e in extra))
    return pd.DataFrame(out)


def r_lower_bounds(names, workers, reps):
    """Certified lower bounds, across independent random splits.

    Repeating over splits is the point: a single split gives one valid
    certificate, but the spread across splits is what tells the reader how
    much of the reported number is the data and how much is the split.
    """
    print(f"certified lower bounds ({reps} splits each):")
    out = []
    for nm in names:
        jobs = [(nm, 1000 + i) for i in range(reps)]
        vals = _map(w_lower, jobs, workers, "r_lower",
                    {"dataset": nm, "reps": reps,
                     "grid": ALPHA_GRID.tolist(), "delta": DELTA})
        lo = np.array([v["lower"] for v in vals])
        # A bound that lands on the last grid point is censored, not measured:
        # the procedure certified everything it was offered and would likely
        # have gone further. Reporting the grid maximum as an estimate would
        # understate the contamination level, so flag it.
        grid_max = float(ALPHA_GRID.max())
        censored = float((lo >= grid_max).mean())
        out.append({
            "dataset": nm, "splits": reps,
            "mean_lower": float(lo.mean()), "median_lower": float(np.median(lo)),
            "min_lower": float(lo.min()), "max_lower": float(lo.max()),
            "sd_lower": float(lo.std(ddof=1)) if reps > 1 else np.nan,
            "frac_certifying_any": float((lo > 0).mean()),
            "frac_at_grid_max": censored, "grid_max": grid_max,
        })
        if censored > 0:
            print(f"    NOTE: {censored:.0%} of splits certified the grid "
                  f"maximum {grid_max:g}; this bound is censored, extend the grid")
        print(f"  {nm}: median {np.median(lo):.2f}, range "
              f"[{lo.min():.2f}, {lo.max():.2f}], "
              f"{(lo > 0).mean():.0%} of splits certify something")
    return pd.DataFrame(out)


def r_upper_sensitivity(names, lowers, workers, reps,
                        gammas=(0.8, 1.0, 1.2), c0s=(0.8, 1.2, 1.6), paths=None,
                        summary=None):
    """The other side, as a sensitivity analysis rather than a guarantee.

    Section~\\ref{sec:upper} proves no nontrivial upper bound exists without
    an assumption, and the synthetic sweep showed margin constants do not
    transfer between families. So on real data, where the constants are not
    knowable, the only honest object is a grid: here is the bound you get
    *if* you are willing to assume this much margin.
    """
    print(f"upper-bound sensitivity ({reps} splits each):")
    out = []
    for nm in names:
        a_up = float(lowers.set_index("dataset").loc[nm, "median_lower"])
        jobs = [(nm, a_up, 2000 + i) for i in range(reps)]
        vals = _map(w_upper, jobs, workers, "r_upper",
                    {"dataset": nm, "alpha": a_up, "reps": reps, "delta": DELTA},
                    seed_index=2)
        d_up = float(np.median([v["d_upper"] for v in vals]))
        cor = None
        if paths is not None and summary is not None:
            # Corollary cor:upperfs (v2): with probability >= 1 - delta,
            # TV <= inf_alpha { alpha + omega^<-(D-hat_alpha + tau_A(alpha, delta)) },
            # omega(t) = c0 t^gamma; the infimum over the path grid is valid
            # (the event holds for every alpha simultaneously). No split.
            g_ = paths[paths.dataset == nm].sort_values("alpha")
            s_ = summary.set_index("dataset").loc[nm]
            cor = (g_.alpha.to_numpy(float), g_.d_hat.to_numpy(float), int(s_.n), int(s_.m))
        for g in gammas:
            for c in c0s:
                row = {
                    "dataset": nm, "alpha_for_upper": a_up, "d_alpha_upper": d_up,
                    "gamma": g, "c0": c,
                    "tv_upper": float(min(upper_bound_tv(d_up, a_up, 0.0, c, g), 1.0)),
                }
                if cor is not None:
                    al, dh, n_, m_ = cor
                    cand = [a + ((d + tau_A(n_, m_, a, DELTA)) / c) ** (1.0 / g)
                            for a, d in zip(al, dh)]
                    row["tv_upper_corollary"] = float(min(min(cand), 1.0))
                out.append(row)
        print(f"  {nm}: D_alpha upper {d_up:.4f} at alpha={a_up:.2f}; "
              f"TV upper ranges "
              f"{min(o['tv_upper'] for o in out if o['dataset'] == nm):.2f}"
              f"-{max(o['tv_upper'] for o in out if o['dataset'] == nm):.2f} "
              f"over the constant grid")
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# the comparator: how the overlap literature currently answers this
# --------------------------------------------------------------------------

def _logistic_fit(Z, y, l2=1e-2):
    """Ridge-penalised logistic regression by Newton's method.

    Hand-rolled because scikit-learn is not a dependency here; validated in
    the tests against a scipy.optimize solution of the same objective, since
    an unvalidated hand-rolled optimiser is exactly how the synthetic study
    lost a day to a silently wrong FISTA.
    """
    Z = np.c_[np.ones(len(Z)), Z]
    beta = np.zeros(Z.shape[1])
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-Z @ beta))
        W = np.clip(p * (1 - p), 1e-8, None)
        g = Z.T @ (y - p) - l2 * beta
        H = (Z * W[:, None]).T @ Z + l2 * np.eye(Z.shape[1])
        step = np.linalg.solve(H, g)
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta, 1.0 / (1.0 + np.exp(-Z @ beta))


def rhc_propensity_comparison():
    """What the standard overlap diagnostics say about the same data.

    Two comparators: the textbook ``0.1 <= e(x) <= 0.9`` rule, and Crump,
    Hotz, Imbens and Mitnik's (2009) variance-optimal cutoff, which solves
    for the threshold at which trimming stops paying. Both answer a related
    but different question -- they discard units to make a *weighted*
    estimator well behaved, and their answer depends on a correctly
    specified propensity model. The point of the comparison is not that one
    number is right, but that these numbers are currently produced without
    any accompanying statement of uncertainty.
    """
    X, Y, names, meta = load_rhc()
    Z = np.vstack([X, Y])
    y = np.r_[np.ones(len(X)), np.zeros(len(Y))]
    _, e = _logistic_fit(Z, y)

    rule = float(np.mean((e < 0.1) | (e > 0.9)))

    # Crump et al.: discard x with e(x) outside [a, 1-a], where a solves
    # 1/(a(1-a)) = 2 E[1/(e(1-e)) ; e in [a,1-a]] / P(e in [a,1-a]).
    w = 1.0 / np.clip(e * (1 - e), 1e-12, None)
    grid = np.linspace(0.001, 0.499, 999)
    a_star, gap = np.nan, np.inf
    for a in grid:
        keep = (e >= a) & (e <= 1 - a)
        if keep.sum() < 10:
            continue
        lhs = 1.0 / (a * (1 - a))
        rhs = 2.0 * w[keep].mean()
        if abs(lhs - rhs) < gap:
            gap, a_star = abs(lhs - rhs), a
    crump = float(np.mean((e < a_star) | (e > 1 - a_star)))

    return {
        "propensity_min": float(e.min()), "propensity_max": float(e.max()),
        "frac_outside_0.1_0.9": rule,
        "crump_alpha_cutoff": float(a_star),
        "frac_discarded_crump": crump,
        "n": int(len(Z)), "d": int(Z.shape[1]),
    }


# --------------------------------------------------------------------------
# R10: from a certified level to an actual trimmed cohort
# --------------------------------------------------------------------------

def _weighted_moments(Z, w):
    mu = w @ Z
    var = w @ (Z - mu) ** 2
    return mu, np.maximum(var, 0.0)


def standardized_differences(X, Y, w=None, v=None):
    """Per-covariate standardized mean difference, optionally weighted.

    The standard balance diagnostic in the matching literature: the
    difference in means divided by the pooled within-group spread. Reported
    here weighted by the trimming, which is what makes "before" and "after"
    comparable on the same scale.
    """
    w = np.full(len(X), 1.0 / len(X)) if w is None else w
    v = np.full(len(Y), 1.0 / len(Y)) if v is None else v
    mx, vx = _weighted_moments(X, w)
    my, vy = _weighted_moments(Y, v)
    pooled = np.sqrt(np.maximum((vx + vy) / 2.0, 1e-12))
    return (mx - my) / pooled


def trimmed_cohort(name, alpha, seed=0, geometries=("mmd", "wasserstein"),
                   ot_max=1400, propensity=None, threshold="paper"):
    """Recover the trimming at a certified level and describe the cohort.

    ``alpha`` is meant to be the split-sample certified lower bound on
    ``alpha-star``, i.e. the level the data supports. Note what that means
    for interpretation: the certificate says *at least* this much of each
    arm has no counterpart, so the trimming at that level removes the mass
    the evidence demands, not the mass that would make the two arms match.

    ``ot_max`` caps the sample size sent to the transport solver, which is
    a dense LP in ``n*m`` variables; above the cap a stratified subsample is
    used and the row is flagged.
    """
    from ..threshold import tau as tau_fn
    from ..trimmings import (optimal_trimming_mmd,
                             optimal_trimming_wasserstein)

    X, Y = _dataset(name)
    n, m = len(X), len(Y)
    rng = np.random.default_rng(seed)
    sigma = float(median_heuristic(X, Y, rng=rng))
    thr = (float(tau_fn(n, m, alpha, DELTA)) if threshold == "paper"
           else float(_thr(threshold, n, m)(alpha)))

    smd0 = standardized_differences(X, Y)
    out = []
    for geom in geometries:
        if geom == "mmd":
            t = optimal_trimming_mmd(X, Y, alpha, sigma=sigma,
                                     rng=np.random.default_rng(seed),
                                     tau=thr)
            ix, iy = np.arange(n), np.arange(m)
            sub = False
        else:
            # Dense LP: subsample if the cost matrix would be too large.
            sub = n > ot_max or m > ot_max
            ix = (rng.choice(n, min(n, ot_max), replace=False) if sub
                  else np.arange(n))
            iy = (rng.choice(m, min(m, ot_max), replace=False) if sub
                  else np.arange(m))
            ix, iy = np.sort(ix), np.sort(iy)
            t = optimal_trimming_wasserstein(X[ix], Y[iy], alpha)
        smd = standardized_differences(X[ix], Y[iy], t.w, t.v)
        # "Before" must be computed on the SAME rows as "after", or the
        # transport row (which subsamples) would be compared against a
        # full-sample baseline and look worse than it is.
        smd_pre = standardized_differences(X[ix], Y[iy])
        row = {
            "dataset": name, "geometry": geom, "alpha": alpha,
            "n": len(ix), "m": len(iy), "subsampled": sub,
            "d_hat": t.value, "tau": thr if geom == "mmd" else np.nan,
            "identified": t.unique,
            "kept_x": int(t.kept_x.size), "kept_y": int(t.kept_y.size),
            "ess_x": t.ess("x"), "ess_y": t.ess("y"),
            "frac_x": t.fractional_count("x"),
            "max_abs_smd_before": float(np.abs(smd_pre).max()),
            "max_abs_smd_after": float(np.abs(smd).max()),
            "mean_abs_smd_before": float(np.abs(smd_pre).mean()),
            "mean_abs_smd_after": float(np.abs(smd).mean()),
            "max_abs_smd_fullsample": float(np.abs(smd0).max()),
        }
        if propensity is not None and not sub:
            ex, ey = propensity[:n][ix], propensity[n:][iy]
            row["prop_overlap_before"] = _overlap(ex, ey)
            row["prop_overlap_after"] = _overlap(ex, ey, t.w, t.v)
        out.append(row)
        print(f"  {name:14s} {geom:12s} a={alpha:.2f} kept="
              f"{row['kept_x']}/{len(ix)},{row['kept_y']}/{len(iy)} "
              f"|SMD|max {row['max_abs_smd_before']:.3f} -> "
              f"{row['max_abs_smd_after']:.3f} identified={t.unique}")
    return out, smd0


def _overlap(ex, ey, w=None, v=None, bins=40):
    """Overlapping coefficient of two (possibly weighted) histograms.

    1 means the two propensity distributions coincide, 0 that they are
    disjoint. A blunt instrument, but the one the overlap literature reads.
    """
    lo, hi = 0.0, 1.0
    edges = np.linspace(lo, hi, bins + 1)
    hx, _ = np.histogram(ex, bins=edges, weights=w, density=False)
    hy, _ = np.histogram(ey, bins=edges, weights=v, density=False)
    hx = hx / max(hx.sum(), 1e-300)
    hy = hy / max(hy.sum(), 1e-300)
    return float(np.minimum(hx, hy).sum())


def cohort_macros(cohorts, balance):
    """LaTeX macros for the trimmed-cohort numbers quoted in Section 11."""
    r = cohorts[(cohorts.dataset == "rhc") & (cohorts.geometry == "mmd")].iloc[0]
    w = cohorts[(cohorts.dataset == "rhc") &
                (cohorts.geometry == "wasserstein")]
    nh = cohorts[(cohorts.dataset == "nhanes_child") &
                 (cohorts.geometry == "mmd")]
    lines = [
        "% Generated by trimmedmmd.experiments.realdata_run -- do not edit.",
        rf"\newcommand{{\cohAlpha}}{{{r.alpha:.2f}}}",
        rf"\newcommand{{\cohKeptX}}{{{int(r.kept_x):,}}}",
        rf"\newcommand{{\cohKeptY}}{{{int(r.kept_y):,}}}",
        rf"\newcommand{{\cohNX}}{{{int(r.n):,}}}",
        rf"\newcommand{{\cohNY}}{{{int(r.m):,}}}",
        rf"\newcommand{{\cohSmdBefore}}{{{r.max_abs_smd_before:.2f}}}",
        rf"\newcommand{{\cohSmdAfter}}{{{r.max_abs_smd_after:.2f}}}",
        rf"\newcommand{{\cohSmdMeanBefore}}{{{r.mean_abs_smd_before:.2f}}}",
        rf"\newcommand{{\cohSmdMeanAfter}}{{{r.mean_abs_smd_after:.2f}}}",
        rf"\newcommand{{\cohFracX}}{{{int(r.frac_x)}}}",
        rf"\newcommand{{\cohEssX}}{{{r.ess_x:.0f}}}",
        rf"\newcommand{{\cohIdentified}}{{{'yes' if r.identified else 'no'}}}",
        rf"\newcommand{{\cohDhat}}{{{r.d_hat:.3f}}}",
        rf"\newcommand{{\cohTau}}{{{r.tau:.3f}}}",
    ]
    if not w.empty:
        ww = w.iloc[0]
        lines += [
            rf"\newcommand{{\cohWSmdBefore}}{{{ww.max_abs_smd_before:.2f}}}",
            rf"\newcommand{{\cohWSmdAfter}}{{{ww.max_abs_smd_after:.2f}}}",
            rf"\newcommand{{\cohWN}}{{{int(ww.n):,}}}",
            rf"\newcommand{{\cohWFracX}}{{{int(ww.frac_x)}}}",
        ]
    if not nh.empty:
        nn = nh.iloc[0]
        lines += [
            rf"\newcommand{{\cohNhAlpha}}{{{nn.alpha:.2f}}}",
            rf"\newcommand{{\cohNhKeptX}}{{{int(nn.kept_x):,}}}",
            rf"\newcommand{{\cohNhKeptY}}{{{int(nn.kept_y):,}}}",
        ]
    if "prop_overlap_before" in cohorts.columns and \
            not pd.isna(r.get("prop_overlap_before", np.nan)):
        lines += [
            rf"\newcommand{{\cohPropBefore}}{{{r.prop_overlap_before:.2f}}}",
            rf"\newcommand{{\cohPropAfter}}{{{r.prop_overlap_after:.2f}}}",
        ]
    worst = balance.iloc[0]
    lines += [
        rf"\newcommand{{\cohWorstCov}}{{\texttt{{{worst.covariate.replace('_', '-')}}}}}",
        rf"\newcommand{{\cohWorstBefore}}{{{worst.smd_before:.2f}}}",
        rf"\newcommand{{\cohWorstAfter}}{{{worst.smd_after:.2f}}}",
    ]
    (TABLES / "cohort_macros.tex").write_text("\n".join(lines) + "\n")
    print("  wrote cohort_macros.tex")


def r_trimmed_cohorts(names, lowers, threshold="paper"):
    """Table r10: the trimmed cohort at each dataset's certified level."""
    from ..realdata import load_rhc

    prop = None
    try:
        Xr, Yr, _, _ = load_rhc()
        Z = np.vstack([Xr, Yr])
        y = np.r_[np.ones(len(Xr)), np.zeros(len(Yr))]
        _, prop = _logistic_fit(Z, y)
    except Exception as exc:          # RHC absent: the rest still runs
        print(f"  (propensity comparison skipped: {exc})")

    rows, smds = [], {}
    for name in names:
        sel = lowers[lowers["dataset"] == name]
        if sel.empty:
            continue
        a = float(sel["median_lower"].iloc[0])
        if a <= 0:
            print(f"  {name}: certified level is 0, nothing to trim")
            continue
        r, smd0 = trimmed_cohort(name, a,
                                 propensity=prop if name == "rhc" else None,
                                 threshold=threshold)
        rows.extend(r)
        smds[name] = smd0
    return pd.DataFrame(rows), smds


def rhc_balance_table(top=8, seed=0, alpha=0.25, threshold="paper"):
    """Which covariates the trimming rebalances, and which it cannot.

    ``alpha`` is the RHC certified level (v1: 0.25, hard-coded; v2 passes the
    r2 median of the selected certificate) and ``threshold`` the cutoff of the
    least-trimming continuation, as in :func:`trimmed_cohort`.
    """
    from ..realdata import load_rhc
    from ..threshold import tau as tau_fn
    from ..trimmings import optimal_trimming_mmd

    X, Y, names, _ = load_rhc()
    n, m = len(X), len(Y)
    a = float(alpha)                # the certified level for RHC (v1: 0.25)
    rng = np.random.default_rng(seed)
    sigma = float(median_heuristic(X, Y, rng=rng))
    thr = (float(tau_fn(n, m, a, DELTA)) if threshold == "paper"
           else float(_thr(threshold, n, m)(a)))
    t = optimal_trimming_mmd(X, Y, a, sigma=sigma,
                             rng=np.random.default_rng(seed),
                             tau=thr)
    before = standardized_differences(X, Y)
    after = standardized_differences(X, Y, t.w, t.v)
    order = np.argsort(-np.abs(before))[:top]
    return pd.DataFrame({
        "covariate": [names[i] for i in order],
        "smd_before": before[order],
        "smd_after": after[order],
    }), t


# --------------------------------------------------------------------------
# LaTeX tables, generated rather than transcribed
# --------------------------------------------------------------------------

def required_n(d_hat0: float, delta: float = DELTA, kappa: float = 1.0,
               threshold: str = "paper") -> int:
    """Balanced sample size per group at which the test would fire.

    At ``alpha=0`` the threshold is
    ``tau = kappa[4 + sqrt(2 log(4/delta))] (1/sqrt(n) + 1/sqrt(m))``, so with
    ``n=m`` the test rejects once ``n > (2 C / D_0)^2`` for
    ``C = kappa[4 + sqrt(2 log(4/delta))]``. Treating the observed
    ``D_0`` as the population value, this is the sample size the contrast
    would need -- which makes the threshold's cost something a reader can
    evaluate in advance rather than discover afterwards.
    """
    if threshold != "paper":
        # v2, eq:samplesize as printed: the test rejects once n > 2 (C_0 / D_0)^2,
        # C_0 = kappa [1 + sqrt(log(1/delta))] (Gaussian kernel; tau_{n,n}(0, delta)
        # = C_0 sqrt(2/n) for the closed form, the binding member at alpha = 0),
        # with the observed D-hat_0 for the data contrasts; rounded up as v1 did.
        from ..threshold import fire_sample_size
        if d_hat0 <= 0:
            return -1
        return int(np.ceil(fire_sample_size(d_hat0, delta, kappa)))
    c = kappa * (4.0 + np.sqrt(2.0 * np.log(4.0 / delta)))
    if d_hat0 <= 0:
        return -1
    return int(np.ceil((2.0 * c / d_hat0) ** 2))


def required_n_sharp(d_hat_at_a: float, a: float, delta: float = DELTA,
                     kappa: float = 1.0, threshold: str = "paper") -> int:
    """Balanced sample size at which the estimator would return at least ``a``.

    Firing is one thing; returning a number close to the truth is another.
    Because ``tau(alpha) = tau(0)/(1-alpha)`` while the path decreases, the
    crossing level satisfies ``alpha-hat >= a`` exactly when
    ``tau(0) < D(a)(1-a)``, so with ``n=m``

        n > ( 2C / [ D(a) (1-a) ] )^2 .

    Unlike the margin-based route this needs no constants: ``D(a)`` is read
    off the estimated path. It treats the observed path as the population
    one, so it is a design guide rather than a guarantee -- but it is the
    honest way to say how far the analytic estimator is from being sharp,
    and by how much data that distance would close.
    """
    if threshold != "paper":
        # v2, eq:sharpsize as printed: n_a = min{n : tau_{n,n}(a, delta) < D_a},
        # solved numerically with tau_new (Gaussian constants), D_a off the path.
        from ..threshold import KernelConstants, sharp_sample_size
        if d_hat_at_a <= 0:
            return -1
        kc = None if kappa == 1.0 else KernelConstants.gaussian(kappa)
        return int(sharp_sample_size(d_hat_at_a, a, delta, kernel_consts=kc))
    c = kappa * (4.0 + np.sqrt(2.0 * np.log(4.0 / delta)))
    t = d_hat_at_a * (1.0 - a)
    if t <= 0:
        return -1
    return int(np.ceil((2.0 * c / t) ** 2))


def sharpness_table(paths, lowers, fractions=(0.5, 0.8), threshold="paper"):
    """What each contrast would need for a non-trivial ``alpha-hat``."""
    low = lowers.set_index("dataset")
    rows = []
    for nm, g in paths.groupby("dataset", sort=False):
        cert = float(low.loc[nm, "median_lower"])
        g = g.sort_values("alpha")
        for f in fractions:
            a = round(cert * f / 0.05) * 0.05
            if a <= 0:
                continue
            r = g.iloc[(g.alpha - a).abs().argmin()]
            rows.append({"dataset": nm, "certified": cert, "fraction": f,
                         "target_alpha": float(r.alpha),
                         "d_hat_at_target": float(r.d_hat),
                         "required_n": required_n_sharp(float(r.d_hat),
                                                        float(r.alpha),
                                                        threshold=threshold)})
    return pd.DataFrame(rows)


LABELS = {
    "rhc": "RHC, treated vs.\\ control",
    "nhanes_child": "NHANES, age 6--11 vs.\\ 60--79",
    "nhanes_age": "NHANES, age 20--39 vs.\\ 60--79",
    "nhanes_bmi": "NHANES, BMI $<25$ vs.\\ $\\geq30$",
}


def write_latex_tables(paths, summary, lowers, ups, comp, threshold="paper"):
    """Emit the application tables as \\input-able LaTeX.

    The paper reads these files rather than carrying typed copies of the
    numbers. Retyping a table is how a corrected re-run silently fails to
    reach the manuscript, and this project has already paid that cost once.
    """
    out = []
    out.append("% Generated by trimmedmmd.experiments.realdata_run -- do not edit.")

    s = summary.set_index("dataset")
    out.append(r"\begin{tabular}{lrrrrrcr}")
    out.append(r"\toprule")
    out.append(r"contrast & $n$ & $m$ & $d$ & $\widehat D_0$ & $\tau(0)$ "
               r"& fires & needs $n=m>$ \\")
    out.append(r"\midrule")
    for nm in s.index:
        r = s.loc[nm]
        tau0 = float(paths[(paths.dataset == nm) & (paths.alpha == 0.0)]
                     .analytic_tau.iloc[0])
        fires = bool(paths[(paths.dataset == nm) & (paths.alpha == 0.0)]
                     .rejects.iloc[0])
        need = required_n(r.plain_mmd, threshold=threshold)
        out.append(f"{LABELS[nm]} & {int(r.n)} & {int(r.m)} & {int(r.d)} & "
                   f"{r.plain_mmd:.3f} & {tau0:.3f} & "
                   f"{'yes' if fires else 'no'} & {need:,} \\\\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    (TABLES / "app_paths.tex").write_text("\n".join(out) + "\n")

    lo = lowers.set_index("dataset")
    out = ["% Generated by trimmedmmd.experiments.realdata_run -- do not edit.",
           r"\begin{tabular}{lcccc}", r"\toprule",
           r"contrast & median & min & max & splits certifying $>0$ \\",
           r"\midrule"]
    for nm in lo.index:
        r = lo.loc[nm]
        out.append(f"{LABELS[nm]} & {r.median_lower:.2f} & {r.min_lower:.2f} & "
                   f"{r.max_lower:.2f} & {r.frac_certifying_any:.0%}".replace("%", r"\%")
                   + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "app_cert.tex").write_text("\n".join(out) + "\n")

    gammas = sorted(ups.gamma.unique())
    c0s = sorted(ups.c0.unique())
    out = ["% Generated by trimmedmmd.experiments.realdata_run -- do not edit.",
           r"\begin{tabular}{ll" + "c" * len(c0s) + "}", r"\toprule",
           r"contrast & $\gamma$ & "
           + " & ".join(fr"$c_0={c:g}$" for c in c0s) + r" \\", r"\midrule"]
    for nm in ups.dataset.unique():
        for i, g in enumerate(gammas):
            cells = []
            for c in c0s:
                v = ups[(ups.dataset == nm) & (ups.gamma == g) & (ups.c0 == c)]
                cells.append(f"{float(v.tv_upper.iloc[0]):.2f}")
            head = LABELS[nm] if i == 0 else ""
            out.append(f"{head} & {g:g} & " + " & ".join(cells) + r" \\")
        out.append(r"\addlinespace")
    out += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "app_upper.tex").write_text("\n".join(out) + "\n")

    # Scalars the prose quotes, as \newcommand macros, for the same reason.
    m = ["% Generated by trimmedmmd.experiments.realdata_run -- do not edit."]
    for nm in s.index:
        key = nm.replace("_", "")
        r = s.loc[nm]
        tau0 = float(paths[(paths.dataset == nm) & (paths.alpha == 0.0)]
                     .analytic_tau.iloc[0])
        m.append(fr"\newcommand{{\{key}Dzero}}{{{r.plain_mmd:.3f}}}")
        m.append(fr"\newcommand{{\{key}Tauzero}}{{{tau0:.3f}}}")
        m.append(fr"\newcommand{{\{key}Gap}}{{{tau0 / r.plain_mmd:.1f}}}")
        m.append(fr"\newcommand{{\{key}Screen}}{{{r.mmd_radius:.3f}}}")
        m.append(fr"\newcommand{{\{key}Ahat}}{{{r.alpha_hat_analytic:.2f}}}")
        # The last level the analytic test rejects. thm:consistency(i) controls the
        # infimum of the crossing set; the grid estimator returns the smallest
        # ACCEPTED level, up to one mesh above it. The certified statement is
        # therefore alpha* > (last rejected level), which is what the paper
        # quotes. Zero when nothing is rejected: the test certifies nothing.
        rej = paths[(paths.dataset == nm) & (paths.rejects)]
        last = float(rej.alpha.max()) if len(rej) else 0.0
        m.append(fr"\newcommand{{\{key}Last}}{{{last:.2f}}}")
        m.append(fr"\newcommand{{\{key}Need}}{{{required_n(r.plain_mmd, threshold=threshold):,}}}")
        m.append(fr"\newcommand{{\{key}N}}{{{int(r.n):,}}}")
        m.append(fr"\newcommand{{\{key}M}}{{{int(r.m):,}}}")
        m.append(fr"\newcommand{{\{key}Lo}}{{{lo.loc[nm, 'median_lower']:.2f}}}")
        m.append(fr"\newcommand{{\{key}LoMin}}{{{lo.loc[nm, 'min_lower']:.2f}}}")
        m.append(fr"\newcommand{{\{key}LoMax}}{{{lo.loc[nm, 'max_lower']:.2f}}}")
        # How many of the splits certified anything, as a count rather than a
        # percentage: the prose reads better as "four of the eight".
        cert_n = int(round(lo.loc[nm, "frac_certifying_any"] * lo.loc[nm, "splits"]))
        m.append(fr"\newcommand{{\{key}CertN}}{{{cert_n}}}")
    if threshold != "paper":
        # Revision v2: eq:sharpsize numbers quoted in Section 11 ("reaching half the
        # certified level would need about ... per group on the child contrast and ...
        # per arm on RHC") and Appendix F (half: child, adult age, RHC; four fifths:
        # range over the contrasts). Same rule as r8 (sharpness_table): the target is
        # the grid level nearest fraction x the reported certificate's median, D_a
        # read off the path, n_a = min{n : tau_{n,n}(a, delta) < D_a}.
        sh = sharpness_table(paths, lowers, fractions=(0.5, 0.8), threshold=threshold)
        tags = {0.5: "SharpHalf", 0.8: "SharpFourFifths"}
        for _, row in sh.iterrows():
            key = row.dataset.replace("_", "")
            tag = tags[float(row.fraction)]
            m.append(fr"\newcommand{{\{key}{tag}}}{{{int(row.required_n):,}}}")
            m.append(fr"\newcommand{{\{key}{tag}Alpha}}{{{float(row.target_alpha):.2f}}}")
        ff = sh[(sh.fraction == 0.8) & (sh.required_n > 0)]
        if len(ff):
            m.append(fr"\newcommand{{\sharpFourFifthsLo}}{{{int(ff.required_n.min()):,}}}")
            m.append(fr"\newcommand{{\sharpFourFifthsHi}}{{{int(ff.required_n.max()):,}}}")
        sh.to_csv(TABLES / "r8_sharpness.csv", index=False)
    m.append(fr"\newcommand{{\rhcCrumpCut}}{{{comp['crump_alpha_cutoff']:.3f}}}")
    m.append(fr"\newcommand{{\rhcCrumpDrop}}{{{comp['frac_discarded_crump']:.2f}}}")
    m.append(fr"\newcommand{{\rhcRuleDrop}}{{{comp['frac_outside_0.1_0.9']:.2f}}}")
    (TABLES / "app_macros.tex").write_text("\n".join(m) + "\n")
    print("  wrote app_paths.tex, app_cert.tex, app_upper.tex, app_macros.tex")


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def run_all(reps=None, workers=None, datasets=None, threshold="paper",
            certificate="bootstrap", outdir=None, only=None, reuse_paths=None):
    """The real-data pass. Defaults reproduce v1 exactly.

    Revision v2: ``threshold="plugin"`` (tau_new for the analytic test, the
    estimator, "needs n" and the trimming cutoff), ``certificate`` in
    {"sequential", "all"} (the paper's main certificate, sequential betting;
    "all" adds single-split betting, Bernstein, alpha = 0 witness and
    bootstrap columns) or {"betting", "both"} (single-split betting reported)
    -- r2 via :func:`r_lower_bounds_v2`; the certified level that r3, r10 and
    r11 use is the reported certificate's median -- ``outdir``
    (tables/manifest/checkpoints under it), ``only`` (comma-separated
    datasets). Any v2 flag also writes r11 and ``cohort_macros.tex`` (v1 made
    them by hand / in realdata_extras) and adds the corollary upper end to r3.
    ``reuse_paths`` = a directory whose ``tables/r1_paths.csv`` and
    ``r1_summary.csv`` hold threshold-independent D-hat paths to reuse.
    """
    v2 = (threshold != "paper" or certificate != "bootstrap" or outdir is not None)
    if outdir is not None:
        configure_outdir(outdir)
    TABLES.mkdir(parents=True, exist_ok=True)
    env = detect()
    workers = env.max_workers if workers is None else workers
    reps = (8 if env.profile in ("smoke", "minimal", "modest") else 16) \
        if reps is None else reps
    if only is not None and datasets is None:
        datasets = [d.strip() for d in str(only).split(",") if d.strip()]
    names = datasets or ["rhc", "nhanes_child", "nhanes_age", "nhanes_bmi"]
    print(f"[real data] profile={env.profile} workers={workers} splits={reps}"
          + (f" threshold={threshold} certificate={certificate} -> {RESULTS}" if v2 else "")
          + "\n")

    reuse = None
    if reuse_paths is not None:
        rp = Path(reuse_paths)
        rp = rp if rp.name == "tables" else rp / "tables"
        reuse = (pd.read_csv(rp / "r1_paths.csv"), pd.read_csv(rp / "r1_summary.csv"))
        print(f"  reusing D-hat paths from {rp} (threshold re-applied)")

    t0 = time.time()
    paths, summary = r_paths(names, threshold=threshold, reuse=reuse)
    paths.to_csv(TABLES / "r1_paths.csv", index=False)
    summary.to_csv(TABLES / "r1_summary.csv", index=False)

    if certificate == "bootstrap":
        lowers = r_lower_bounds(names, workers, reps)
    else:
        lowers = r_lower_bounds_v2(names, workers, reps, certificate=certificate)
    lowers.to_csv(TABLES / "r2_lower_bounds.csv", index=False)

    if v2:
        ups = r_upper_sensitivity(names, lowers, workers, reps, paths=paths,
                                  summary=summary)
    else:
        ups = r_upper_sensitivity(names, lowers, workers, reps)
    ups.to_csv(TABLES / "r3_upper_sensitivity.csv", index=False)

    cohorts, _ = r_trimmed_cohorts(names, lowers, threshold=threshold)
    cohorts.to_csv(TABLES / "r10_trimmed_cohorts.csv", index=False)

    comp = rhc_propensity_comparison()
    (TABLES / "r4_rhc_propensity.json").write_text(json.dumps(comp, indent=2))
    print(f"\npropensity comparators: {comp['frac_outside_0.1_0.9']:.1%} outside "
          f"[0.1,0.9]; Crump cutoff {comp['crump_alpha_cutoff']:.3f} discards "
          f"{comp['frac_discarded_crump']:.1%}")

    write_latex_tables(paths, summary, lowers, ups, comp, threshold=threshold)

    if v2 and "rhc" in names and not cohorts.empty:
        a_rhc = float(lowers.set_index("dataset").loc["rhc", "median_lower"])
        print(f"R11 RHC balance table at the certified level {a_rhc:.2f} ...")
        bal, _ = rhc_balance_table(top=8, seed=0, alpha=a_rhc, threshold=threshold)
        bal.to_csv(TABLES / "r11_rhc_balance.csv", index=False)
        cohort_macros(cohorts, bal)

    manifest = {
        "code_version": CODE_VERSION,
        "profile": env.profile, "cpus": env.cpu_count,
        "ram_gb": round(env.ram_gb, 2) if env.ram_gb else None,
        "workers": workers, "splits": reps,
        "grid": ALPHA_GRID.tolist(), "delta": DELTA,
        "minutes": round((time.time() - t0) / 60, 1),
    }
    if v2:
        manifest.update(v2_code_version=V2_CODE_VERSION, threshold=threshold,
                        certificate=certificate, datasets=names,
                        certificate_grids={nm: _grid_for(nm, True).tolist() for nm in names},
                        reused_paths=None if reuse_paths is None else str(reuse_paths),
                        checkpoints=str(CHECKPOINTS))
    (RESULTS / "realdata_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\ndone in {manifest['minutes']} min")


def main(argv=None):
    """``python -m trimmedmmd.experiments.realdata_run`` (no flags: v1 run_all())."""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--datasets", default=None, help="comma-separated")
    ap.add_argument("--threshold", choices=["paper", "plugin"], default="paper")
    ap.add_argument("--certificate",
                    choices=["bootstrap", "betting", "both", "sequential", "all"],
                    default="bootstrap",
                    help="v2: 'all' reports the sequential certificate with single-split "
                         "betting, Bernstein, alpha=0-witness and bootstrap columns")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--reuse-paths", default=None,
                    help="directory (e.g. results) whose tables/r1_paths.csv and "
                         "r1_summary.csv hold D-hat paths to reuse")
    a = ap.parse_args(argv)
    run_all(reps=a.reps, workers=a.workers, only=a.datasets, threshold=a.threshold,
            certificate=a.certificate, outdir=a.outdir, reuse_paths=a.reuse_paths)


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------
# Large-sample study: does alpha-hat actually converge on real data?
# --------------------------------------------------------------------------

def covertype_subsample(n, seed=0, classes=(1, 2)):
    """A balanced subsample of the two cover types."""
    from ..realdata import load_covertype
    X, Y, _, _ = load_covertype(classes=classes)
    rng = np.random.default_rng(seed)
    return (X[rng.choice(len(X), n, replace=False)],
            Y[rng.choice(len(Y), n, replace=False)])


def alpha_hat_bisect(X, Y, sigma, n_features=2048, delta=DELTA, seed=0,
                     max_iter=2000, tol=1e-3):
    """Locate the crossing level by bisection rather than on a grid.

    A grid costs one quadratic program per level; bisection costs about
    log2(1/tol) of them, which is what makes the large-sample end of this
    study affordable at all.
    """
    from ..estimate import bisect_alpha_hat
    n, m = len(X), len(Y)
    rff = RFF(dim=X.shape[1], n_features=n_features, sigma=sigma,
              rng=np.random.default_rng(seed))
    Zx, Zy = rff.transform(X), rff.transform(Y)

    def d_hat_at(a):
        r = solve_features(Zx, Zy, a, rng=np.random.default_rng(seed),
                           max_iter=max_iter)
        return r.discrepancy

    return float(bisect_alpha_hat(d_hat_at, lo=0.0, hi=0.95, n=n, m=m,
                                  delta=delta, tol=tol))


def r9_large_sample(sizes=(2000, 4000, 8000, 16000, 32000),
                    cert_n=10000, splits=8, n_features=2048, seed=0):
    """alpha-hat against n on Covertype, with a certificate for comparison.

    The certified lower bound is a statement about the population and does
    not depend on the sample size used to obtain it, so one run at a
    tractable ``cert_n`` gives the reference line that alpha-hat should be
    climbing towards.
    """
    rows = []
    Xc, Yc = covertype_subsample(cert_n, seed=seed)
    rng = np.random.default_rng(seed)
    sigma_c = float(median_heuristic(Xc, Yc, rng=rng))
    grid = np.round(np.arange(0.0, 0.951, 0.05), 3)
    cert = [lower_bound_tv(Xc, Yc, grid, delta=DELTA, sigma=sigma_c,
                           rng=np.random.default_rng(1000 + s))[0]
            for s in range(splits)]
    cert = np.array(cert)
    print(f"certificate at n=m={cert_n}: median {np.median(cert):.2f}, "
          f"range [{cert.min():.2f}, {cert.max():.2f}]")

    for n in sizes:
        X, Y = covertype_subsample(n, seed=seed)
        rng = np.random.default_rng(seed)
        sigma = float(median_heuristic(X, Y, rng=rng))
        d0 = float(plain_mmd(X, Y, sigma=sigma, rng=rng))
        t0 = float(tau(n, n, 0.0, DELTA))
        t0_ = time.time()
        ah = alpha_hat_bisect(X, Y, sigma, n_features=n_features, seed=seed)
        rows.append({"n": n, "m": n, "sigma": sigma, "d_hat0": d0, "tau0": t0,
                     "ratio": d0 / t0, "alpha_hat": ah,
                     "cert_median": float(np.median(cert)),
                     "seconds": time.time() - t0_})
        print(f"  n=m={n:6d}: D0={d0:.4f} tau0={t0:.4f} r={d0/t0:5.2f} "
              f"alpha_hat={ah:.3f}  [{time.time()-t0_:.0f}s]")
    return pd.DataFrame(rows), cert

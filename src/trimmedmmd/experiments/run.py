"""Synthetic experiments.

Design principle: each experiment must be able to come out badly. The null
calibration can reveal type-I violation; the estimator study can reveal that
alpha-hat is biased in the wrong direction or that thm:consistency(i) fails; the
margin study can return gamma < 1, contradicting lem:continuity; and the
assumption-violation study is run precisely to see the method break when its
hypotheses are removed.

All sizes come from :mod:`trimmedmmd.resources` at run time and are recorded
in the manifest alongside the results.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .. import dgp
from ..checkpoint import (ReplicateStore, _pin_blas_threads, cell_key,
                          default_root, map_replicates)
from ..discrepancy import plain_mmd, trimmed_mmd_path
from ..estimate import alpha_hat_from_path
from ..kernels import gaussian_kernel, linear_kernel, median_heuristic
from ..resources import detect
from ..solver import solve_gram, solve_path
from ..splitting import trimmed_mean
from ..threshold import (leading_term_share, mmd_radius,
                         reject_mmd_screen, tau)

RESULTS = Path(__file__).resolve().parents[3] / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
# Run state, not results: kept off a synced mount, where checkpoint
# files have been seen to vanish mid-run (see checkpoint.default_root).
CHECKPOINTS = default_root(RESULTS / "checkpoints")


# --------------------------------------------------------------------------
# workers (top level so they pickle)
# --------------------------------------------------------------------------

def _build(spec):
    """Build a scenario from a picklable spec.

    Specs are plain tuples so they survive the process boundary: worker
    processes do not share the parent's module state, which is why the
    scenario cannot be passed by registering it in a global catalogue.
    """
    if isinstance(spec, str):
        return dgp.make(spec)
    kind = spec[0]
    if kind == "disjoint":
        return dgp.disjoint_contamination(dim=spec[1], a=spec[2])
    if kind == "gaussian":
        return dgp.gaussian_contamination(dim=spec[1], a=spec[2], mu_norm=spec[3])
    if kind == "identical":
        return dgp.identical(dim=spec[1])
    if kind == "mean_matched":
        return dgp.mean_matched_disjoint(dim=spec[1], a=spec[2])
    if kind == "functional":
        return dgp.functional_contamination()
    raise ValueError(f"unknown scenario spec: {spec!r}")


def _draw(spec, n, m, seed):
    sc = _build(spec)
    rng = np.random.default_rng(seed)
    return sc, sc.sample(n, rng, "P"), sc.sample(m, rng, "Q"), rng


def w_dhat_at(args):
    """Discrepancy at one alpha, one replicate."""
    spec, n, m, alpha, seed = args
    _, X, Y, rng = _draw(spec, n, m, seed)
    sigma = median_heuristic(X, Y, rng=rng)
    Kxx = gaussian_kernel(X, X, sigma)
    Kyy = gaussian_kernel(Y, Y, sigma)
    Kxy = gaussian_kernel(X, Y, sigma)
    return solve_gram(Kxx, Kyy, Kxy, alpha, rng=rng, max_iter=4000,
                      tol=1e-11).discrepancy


def w_path(args):
    """Whole alpha-path for one replicate, with warm starts."""
    spec, n, m, alphas, seed, kernel = args
    _, X, Y, rng = _draw(spec, n, m, seed)
    sigma = median_heuristic(X, Y, rng=rng)
    if kernel == "gaussian":
        Kxx = gaussian_kernel(X, X, sigma)
        Kyy = gaussian_kernel(Y, Y, sigma)
        Kxy = gaussian_kernel(X, Y, sigma)
    else:  # deliberately non-characteristic
        s = float(np.sqrt(np.mean(np.sum(np.vstack([X, Y]) ** 2, axis=1))))
        Kxx = linear_kernel(X, X, s)
        Kyy = linear_kernel(Y, Y, s)
        Kxy = linear_kernel(X, Y, s)

    def solve_one(a, w0, v0):
        return solve_gram(Kxx, Kyy, Kxy, a, w0=w0, v0=v0, rng=rng,
                          max_iter=4000, tol=1e-11)

    res = solve_path(solve_one, np.asarray(alphas))
    if any(not r.converged for r in res):
        import warnings
        warnings.warn(f"trimmed-MMD path did not converge at "
                      f"{sum(not r.converged for r in res)} of {len(res)} levels",
                      RuntimeWarning)
    return np.array([r.discrepancy for r in res])


def _thr(threshold, n, m, alpha, delta):
    """The test threshold of the run: v1's tau ("paper", default) or the v2
    plug-in threshold tau_new ("plugin"; exact Gaussian Gram, so the coupled
    Gaussian constants apply). See :mod:`trimmedmmd.experiments.v2`."""
    if threshold == "paper":
        return float(tau(n, m, alpha, delta))
    from .v2 import threshold_for
    return float(threshold_for(threshold, n, m, delta, backend="gram")(alpha))


def _v2_extra(threshold, n, m, alpha, delta, stat=None, prefix=""):
    """Columns kept next to the selected threshold when it is not v1's: the
    threshold name and the v1 value (and the v1 rejection rate of ``stat``)."""
    if threshold == "paper":
        return {}
    t = float(tau(n, m, alpha, delta))
    out = {"threshold": threshold, f"{prefix}tau_paper": t}
    if stat is not None:
        out[f"{prefix}rate_paper"] = float(np.mean(np.asarray(stat) > t))
    return out


# Bump when any computation changes: checkpoint keys hash the parameters, so
# without this a re-run after a bug fix would silently reuse results produced
# by the old, wrong code.
CODE_VERSION = "2026-09-18-b"

_STORE = ReplicateStore(CHECKPOINTS)


def _map(fn, jobs, workers, experiment=None, params=None, seed_index=-1):
    """Run jobs, checkpointing per replicate when told which cell this is.

    ``seed_index`` says where the seed sits in each job tuple; seeds are what
    identify a replicate across runs, and :func:`map_replicates` rejects
    duplicates, so pointing this at the wrong field fails loudly rather than
    silently collapsing every replicate onto one key. The default ``-1``
    suits the 5-tuple jobs of :func:`w_dhat_at`; the 6-tuple jobs of
    :func:`w_path` and :func:`w_calibrations` pass ``seed_index=4``.
    Without ``experiment``/``params`` this degrades to a plain parallel map.
    """
    if experiment is None or params is None:
        if workers <= 1:
            return [fn(j) for j in jobs]
        with ProcessPoolExecutor(max_workers=workers,
                                 initializer=_pin_blas_threads) as ex:
            return list(ex.map(fn, jobs, chunksize=1))

    params = dict(params, _code_version=CODE_VERSION)
    key = cell_key(experiment, params)
    _STORE.write_meta(key, params)
    seeds = [j[seed_index] for j in jobs]
    already = len(_STORE.load(key))
    if already:
        print(f"    [resume] {experiment}: {already}/{len(jobs)} replicates "
              f"already on disk", flush=True)
    return map_replicates(fn, jobs, seeds, workers, store=_STORE, key=key)


# --------------------------------------------------------------------------
# E1: null calibration and how conservative the analytic threshold is
# --------------------------------------------------------------------------

def e1_calibration(env, delta=0.05, n=300, scenario="disjoint_d2_a20",
                   threshold="paper"):
    """Type-I error of the analytic test, and the price of the guarantee.

    At alpha = alpha* the null holds with equality, so this is the hardest
    point of the null. We record the rejection rate, and separately the
    Monte Carlo (1-delta) quantile of D-hat, whose ratio to the analytic
    threshold is the factor by which the bound overshoots.

    ``threshold="plugin"`` (revision v2): ``analytic_tau``, ``reject_rate``
    and ``conservativeness`` use tau_new; ``tau_paper``, ``rate_paper`` and
    ``conservativeness_paper`` keep v1's. ``leading_share`` is a property of
    v1's formula (the 4/sqrt(n) share of tau_{n,m}) and is kept as is. The
    D-hat replicates do not depend on the threshold, so the checkpoint key
    is unchanged and v1's replicates are reused.
    """
    sc = dgp.make(scenario)
    reps = env.n_mc_null
    alphas = [0.0, sc.alpha_star * 0.5, sc.alpha_star,
              min(sc.alpha_star + 0.1, 0.9), min(sc.alpha_star + 0.2, 0.9)]
    rows = []
    for a in alphas:
        jobs = [(scenario, n, n, a, 10_000 + i) for i in range(reps)]
        d = np.array(_map(w_dhat_at, jobs, env.max_workers,
                          experiment="e1_calibration",
                          params=dict(scenario=scenario, n=n, alpha=float(a),
                                      reps=reps, delta=delta)))
        thr = _thr(threshold, n, n, a, delta)
        q = float(np.quantile(d, 1.0 - delta))
        row = dict(
            scenario=scenario, n=n, alpha=a, alpha_star=sc.alpha_star,
            in_null=bool(a >= sc.alpha_star - 1e-12),
            reps=reps, delta=delta,
            mean_dhat=float(d.mean()), q_dhat=q,
            analytic_tau=float(thr),
            reject_rate=float(np.mean(d > thr)),
            conservativeness=float(thr / q) if q > 0 else np.nan,
            leading_share=leading_term_share(n, n, delta),
        )
        extra = _v2_extra(threshold, n, n, a, delta, stat=d)
        if extra:
            extra["conservativeness_paper"] = (float(extra["tau_paper"] / q)
                                               if q > 0 else np.nan)
        row.update(extra)
        rows.append(row)
    return rows


def e1_power_vs_n(env, delta=0.05, scenario="disjoint_d2_a20",
                  alpha_test=0.0, ns=None, threshold="paper"):
    """At what sample size does the analytic test acquire any power?

    Tested at alpha = 0 against alpha* > 0, i.e. the easiest alternative
    available in this scenario.
    """
    sc = dgp.make(scenario)
    ns = ns or sorted({100, 200, 400, 800, 1600, 3200,
                       min(6400, env.max_n_gram)})
    reps = max(30, env.n_reps // 4)
    rows = []
    for n in ns:
        jobs = [(scenario, n, n, alpha_test, 20_000 + i) for i in range(reps)]
        d = np.array(_map(w_dhat_at, jobs, env.max_workers,
                          experiment="e1_power",
                          params=dict(scenario=scenario, n=int(n),
                                      alpha=float(alpha_test), reps=reps,
                                      delta=delta)))
        thr = _thr(threshold, n, n, alpha_test, delta)
        row = dict(
            scenario=scenario, n=int(n), alpha=alpha_test,
            alpha_star=sc.alpha_star, reps=reps,
            mean_dhat=float(d.mean()), analytic_tau=float(thr),
            power=float(np.mean(d > thr)),
            ratio_dhat_to_tau=float(d.mean() / thr),
        )
        extra = _v2_extra(threshold, n, n, alpha_test, delta, stat=d)
        if extra:
            extra["power_paper"] = extra.pop("rate_paper")
        row.update(extra)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# E2: the estimator, analytic vs oracle-calibrated threshold
# --------------------------------------------------------------------------

def e2_alpha_hat(env, delta=0.05, n=300, scenario="disjoint_d2_a20",
                 n_grid=13, threshold="paper"):
    """Behaviour of alpha-hat, and a direct check of thm:consistency(i).

    Two thresholds are compared on identical data: the analytic tau, and an
    oracle calibration that replaces tau(alpha) by the true (1-delta)
    quantile of D-hat under a null with contamination exactly alpha. The
    oracle is not implementable -- it needs the truth -- but it isolates how
    much of the estimator's behaviour is the threshold's doing.
    """
    sc = dgp.make(scenario)
    grid = np.linspace(0.0, 0.6, n_grid)

    # Oracle quantiles: for each alpha on the grid, simulate a DGP whose true
    # TV equals that alpha and take the (1-delta) quantile of D-hat.
    oracle_reps = max(40, env.n_mc_null // 4)
    oracle_q = {}
    for a in grid:
        if a <= 1e-12:
            spec = ("identical", sc.dim)
        else:
            spec = ("disjoint", sc.dim, float(a))
        jobs = [(spec, n, n, float(a), 30_000 + i) for i in range(oracle_reps)]
        d = np.array(_map(w_dhat_at, jobs, env.max_workers,
                          experiment="e2_oracle",
                          params=dict(spec=list(spec), n=n, alpha=float(a),
                                      reps=oracle_reps, delta=delta)))
        oracle_q[float(a)] = float(np.quantile(d, 1.0 - delta))

    # Replicates: full path once per replicate, then both estimators.
    reps = env.n_reps
    jobs = [(scenario, n, n, grid.tolist(), 40_000 + i, "gaussian")
            for i in range(reps)]
    paths = [np.asarray(p) for p in _map(
        w_path, jobs, env.max_workers, experiment="e2_paths",
        params=dict(scenario=scenario, n=n, reps=reps, grid=grid.tolist()),
        seed_index=4)]

    ana, orc, ana_paper = [], [], []
    for d in paths:
        ana.append(alpha_hat_from_path(
            grid, d, n, n, delta,
            threshold_fn=lambda a: _thr(threshold, n, n, a, delta)))
        orc.append(alpha_hat_from_path(
            grid, d, n, n, delta,
            threshold_fn=lambda a: oracle_q[float(min(grid, key=lambda g: abs(g - a)))]))
        if threshold != "paper":
            ana_paper.append(alpha_hat_from_path(grid, d, n, n, delta))
    ana, orc = np.array(ana), np.array(orc)

    def summarize(x, label):
        return dict(
            scenario=scenario, n=n, reps=reps, delta=delta,
            alpha_star=sc.alpha_star, threshold=label,
            mean=float(x.mean()), bias=float(x.mean() - sc.alpha_star),
            rmse=float(np.sqrt(np.mean((x - sc.alpha_star) ** 2))),
            # thm:consistency(i) predicts this is at least 1 - delta.
            frac_at_most_alpha_star=float(np.mean(x <= sc.alpha_star + 1e-12)),
        )

    if threshold == "paper":
        return ([summarize(ana, "analytic"), summarize(orc, "oracle")],
                {"grid": grid.tolist(), "oracle_q": oracle_q,
                 "analytic": ana.tolist(), "oracle": orc.tolist()})
    # revision v2: "analytic" is the selected threshold; v1's kept alongside
    ana_paper = np.array(ana_paper)
    return ([summarize(ana, "analytic"), summarize(orc, "oracle"),
             summarize(ana_paper, "analytic (paper tau)")],
            {"grid": grid.tolist(), "oracle_q": oracle_q, "threshold": threshold,
             "analytic": ana.tolist(), "oracle": orc.tolist(),
             "analytic_paper": ana_paper.tolist()})


# --------------------------------------------------------------------------
# E3: the margin exponent
# --------------------------------------------------------------------------

def e3_margin(env, scenario="disjoint_d2_a20", n=None, reps=8):
    """Estimate the margin exponent gamma of ass:margin.

    A caution that turns out to matter: D-hat estimates D with an upward
    bias of order n^{-1/2}, because D-hat is a nonnegative distance and the
    truth it estimates goes to zero at the boundary. Fitting a slope through
    points where the signal is comparable to that noise floor therefore
    flattens the fit and can produce gamma < 1, which lem:continuity forbids -- an
    artifact of the measurement, not evidence against the lemma. We measure
    the floor explicitly as the mean of D-hat at alpha = alpha* (where the
    population value is exactly zero) and fit only where the signal clears
    it, reporting both fits so the difference is visible.
    """
    sc = dgp.make(scenario)
    n = n or min(env.max_n_gram, 2000)
    eps = np.array([0.02, 0.04, 0.06, 0.09, 0.12, 0.16, 0.20])
    eps = eps[eps < sc.alpha_star]
    grid = np.sort(sc.alpha_star - eps)

    jobs = [(scenario, n, n, grid.tolist(), 50_000 + i, "gaussian")
            for i in range(reps)]
    paths = np.array(_map(w_path, jobs, env.max_workers,
                          experiment="e3_margin",
                          params=dict(scenario=scenario, n=n, reps=reps,
                                      grid=grid.tolist()), seed_index=4))
    dbar = paths.mean(axis=0)

    # Noise floor: the population discrepancy is exactly zero at alpha*.
    floor_jobs = [(scenario, n, n, float(sc.alpha_star), 55_000 + i)
                  for i in range(reps)]
    floor = float(np.mean(_map(w_dhat_at, floor_jobs, env.max_workers,
                               experiment="e3_floor",
                               params=dict(scenario=scenario, n=n, reps=reps))))

    e_sorted = sc.alpha_star - grid
    rows = [dict(scenario=scenario, n=n, reps=reps, eps=float(e),
                 D_alpha=float(v), alpha=float(a), floor=floor,
                 clears_floor=bool(v > 4.0 * floor))
            for e, v, a in zip(e_sorted, dbar, grid)]
    return rows, margin_fits(e_sorted, dbar, floor, scenario, n, reps)


def margin_fits(e_sorted, dbar, floor, scenario, n, reps, floor_mult=4.0):
    """Three fits of D ~ c0 eps^gamma, in increasing order of honesty.

    The raw fit is biased downward because E[D-hat] carries a roughly
    constant positive offset -- D-hat is a nonnegative distance estimating a
    quantity that vanishes at the boundary. Restricting to points that clear
    the floor helps; subtracting the offset helps more, and is the fit to
    quote, because the offset is measurable (it is D-hat at alpha*, where
    the population value is exactly zero).
    """
    e_sorted = np.asarray(e_sorted, float)
    dbar = np.asarray(dbar, float)

    def _fit(e_, d_):
        if len(e_) < 3:
            return None, None, int(len(e_))
        sl, ic = np.polyfit(np.log(e_), np.log(d_), 1)
        return float(sl), float(np.exp(ic)), int(len(e_))

    clear = dbar > floor_mult * floor
    g_all, c_all, k_all = _fit(e_sorted, dbar)
    g_cl, c_cl, k_cl = _fit(e_sorted[clear], dbar[clear])
    corr = dbar - floor
    pos = corr > 0
    g_bc, c_bc, k_bc = _fit(e_sorted[pos], corr[pos])

    local = []
    for k in range(len(e_sorted) - 1):
        if corr[k] > 0 and corr[k + 1] > 0:
            local.append(dict(
                eps_from=float(e_sorted[k]), eps_to=float(e_sorted[k + 1]),
                gamma_local=float((np.log(corr[k + 1]) - np.log(corr[k]))
                                  / (np.log(e_sorted[k + 1]) - np.log(e_sorted[k])))))

    return dict(
        scenario=scenario, n=n, reps=reps, noise_floor=float(floor),
        floor_multiple_required=floor_mult,
        gamma_hat_all_points=g_all, c0_hat_all_points=c_all, n_points_all=k_all,
        gamma_hat_floor_restricted=g_cl, c0_hat_floor_restricted=c_cl,
        n_points_floor_restricted=k_cl,
        gamma_hat=g_bc, c0_hat=c_bc, n_points_used=k_bc,
        local_slopes=local,
        lemma11_requires_gamma_at_least_1=True,
        consistent_with_lemma11=(None if g_bc is None else bool(g_bc >= 0.9)),
    )


# --------------------------------------------------------------------------
# E4: what happens when characteristicness is removed
# --------------------------------------------------------------------------

def e4_assumption_violation(env, delta=0.05, n=400,
                            scenario="mean_matched_d1_a25", n_grid=13):
    """Remove ass:regularity(A3) and watch the estimator break.

    The scenario has TV(P,Q) = alpha* > 0 but identical means, so a linear
    (non-characteristic) kernel sees no difference at any trimming level and
    alpha-hat collapses to 0. The Gaussian kernel, which is characteristic,
    should not collapse. This is the experiment that would expose the
    characteristicness hypothesis as decorative if it were.
    """
    sc = dgp.make(scenario)
    grid = np.linspace(0.0, 0.6, n_grid)
    reps = max(30, env.n_reps // 4)
    out = []
    detail = {}
    for kernel in ("gaussian", "linear"):
        jobs = [(scenario, n, n, grid.tolist(), 60_000 + i, kernel)
                for i in range(reps)]
        paths = np.array(_map(w_path, jobs, env.max_workers,
                              experiment="e4_assumption",
                              params=dict(scenario=scenario, n=n, reps=reps,
                                          kernel=kernel, grid=grid.tolist()),
                              seed_index=4))
        mean_path = paths.mean(axis=0)
        # Where the path effectively dies: first grid point at or below 5% of
        # its value at alpha = 0. For a characteristic kernel this should sit
        # near alpha*; for a non-characteristic one it can sit far below.
        thresh = 0.05 * mean_path[0]
        below = np.nonzero(mean_path <= thresh)[0]
        collapse = float(grid[below[0]]) if below.size else float(grid[-1])
        out.append(dict(
            scenario=scenario, kernel=kernel, characteristic=(kernel == "gaussian"),
            n=n, reps=reps, alpha_star=sc.alpha_star,
            mean_dhat_at_0=float(mean_path[0]),
            mean_dhat_at_alpha_star=float(
                mean_path[int(np.argmin(np.abs(grid - sc.alpha_star)))]),
            collapse_alpha=collapse,
            collapse_relative_to_alpha_star=float(collapse / sc.alpha_star),
        ))
        detail[kernel] = paths.mean(axis=0).tolist()
    detail["grid"] = grid.tolist()
    return out, detail


def w_calibrations(args):
    """One replicate: the statistic, plus every competing critical value.

    Permutation calibration pools the two samples and re-splits them B
    times. Under P = Q that is exact; under the composite null TV <= alpha
    with P != Q the samples are not exchangeable, so the paper treats it as
    a heuristic. Measuring its realized level is the point of this
    experiment -- the caution is either vindicated or it is not.
    """
    spec, n, m, alpha, seed, n_perm = args
    _, X, Y, rng = _draw(spec, n, m, seed)
    sigma = median_heuristic(X, Y, rng=rng)
    Kxx = gaussian_kernel(X, X, sigma)
    Kyy = gaussian_kernel(Y, Y, sigma)
    Kxy = gaussian_kernel(X, Y, sigma)
    d_hat = solve_gram(Kxx, Kyy, Kxy, alpha, rng=rng, max_iter=4000,
                       tol=1e-12).discrepancy
    mmd_hat = plain_mmd(X, Y, sigma=sigma)

    # Permutation null: pool, re-split, recompute, using one pooled Gram.
    Z = np.vstack([X, Y])
    Kzz = gaussian_kernel(Z, Z, sigma)
    perm_stats = np.empty(n_perm)
    for b in range(n_perm):
        idx = rng.permutation(n + m)
        i1, i2 = idx[:n], idx[n:]
        perm_stats[b] = solve_gram(
            Kzz[np.ix_(i1, i1)], Kzz[np.ix_(i2, i2)], Kzz[np.ix_(i1, i2)],
            alpha, rng=rng, max_iter=4000, tol=1e-12).discrepancy
    return d_hat, mmd_hat, perm_stats


def w_e6_split(args):
    """Revision v2: the split certificates of H_{0,alpha} on the e6 draws.

    Same data as :func:`w_calibrations` (seed, ``run._draw``); the split,
    fold-one bandwidth and witness follow ``splitting.split_test``'s stream
    from the rng left after sampling, which is also the stream of
    ``explore/split_adaptive/synth.py`` at its first level, so the decisions
    can be checked replicate by replicate against that study. Returns the
    single-split betting decision (Theorem 1 with e = (0, 1/2, 1)), if
    ``n_boot > 0`` the Theorem 16 bootstrap decision on the same witness and,
    if ``sequential``, the sequential certificate (the paper's main one,
    e = (0, 1/4, 1/2, 1)) drawn from a COPY of the same post-sampling stream
    (:func:`trimmedmmd.splitting.sequential_order`: same row permutations as
    the single split, bandwidth on batch 0), so it leaves the single-split
    stream untouched.
    """
    import copy
    spec, n, m, alpha, seed, n_boot, delta = args[:7]
    sequential = bool(args[7]) if len(args) > 7 else False
    _, X, Y, rng = _draw(spec, n, m, seed)
    from ..splitting import sequential_certificate, split_certificate
    rng_seq = copy.deepcopy(rng)
    r = split_certificate(X, Y, alpha, delta=delta, rng=rng, n_boot=n_boot)
    out = dict(betting=bool(r.certified),
               bootstrap=(None if r.boot_certified is None else bool(r.boot_certified)),
               delta_hat=float(r.ramp.delta_hat),
               log_max_wealth=float(r.ramp.log_max_wealth))
    if sequential:
        rs = sequential_certificate(X, Y, alpha, delta=delta, rng=rng_seq)
        out.update(sequential=bool(rs.certified), seq_log_final_wealth=float(rs.log_final_wealth),
                   seq_skipped=list(rs.skipped_batches))
    return out


def e6_alternative_calibrations(env, delta=0.05, scenario="disjoint_d2_a20",
                                ns=None, n_perm=None, reps=None,
                                threshold="paper", certificate="bootstrap"):
    """How the less conservative routes already in the paper behave.

    Four critical values on identical data: the analytic threshold; the
    ordinary-MMD screen of equation (17); permutation calibration; and the
    oracle quantile, which is not implementable but bounds what any
    calibration could achieve. Evaluated both at alpha = alpha* (where the
    null holds with equality, so rejection is a type-I error) and at
    alpha = 0 (where it is power).

    Revision v2: ``threshold="plugin"`` puts tau_new in ``analytic_tau`` /
    ``rate_analytic`` (v1's in ``tau_paper`` / ``rate_paper``); the D-hat
    checkpoints are threshold-independent and reused. ``certificate`` (see
    :func:`trimmedmmd.experiments.v2.certificate_methods`) adds the split
    certificates on the same draws, checkpointed as experiment "e6_split":
    ``rate_split_sequential`` (the paper's main certificate, sequential
    betting with e = (0, 1/4, 1/2, 1); "sequential" and "all"),
    ``rate_split_betting`` (single-split betting; always computed, it is the
    cheap part of the job) and ``rate_split_bootstrap`` (Theorem 16, 1000
    resamples, same witness; "both" and "all"). ``rate_split`` repeats the
    reported certificate's rate.
    """
    sc = dgp.make(scenario)
    ns = ns or [200, 400]
    n_perm = n_perm or (30 if env.profile == "smoke" else 100)
    reps = reps or (20 if env.profile == "smoke" else min(400, env.n_reps))
    rows = []
    for n in ns:
        for alpha, label in ((sc.alpha_star, "boundary (type-I)"),
                             (0.0, "alpha=0 (power)")):
            jobs = [(scenario, n, n, float(alpha), 70_000 + i, n_perm)
                    for i in range(reps)]
            res = _map(w_calibrations, jobs, env.max_workers,
                       experiment="e6_calibrations",
                       params=dict(scenario=scenario, n=int(n),
                                   alpha=float(alpha), reps=reps,
                                   n_perm=n_perm, delta=delta),
                       seed_index=4)
            d = np.array([r[0] for r in res])
            mm = np.array([r[1] for r in res])
            perm_crit = np.array([np.quantile(np.asarray(r[2]), 1.0 - delta)
                                  for r in res])
            # Oracle critical value: the 1-delta quantile of the statistic
            # under a null with TV exactly alpha. At the boundary that is the
            # scenario itself; at alpha = 0 the null is P = Q, and taking the
            # quantile of the observed draws instead would be circular, since
            # those come from the alternative.
            null_spec = (("identical", sc.dim) if alpha <= 1e-12
                         else ("disjoint", sc.dim, float(alpha)))
            null_jobs = [(null_spec, n, n, float(alpha), 75_000 + i)
                         for i in range(reps)]
            null_d = np.array(_map(
                w_dhat_at, null_jobs, env.max_workers,
                experiment="e6_oraclenull",
                params=dict(spec=list(null_spec), n=int(n), alpha=float(alpha),
                            reps=reps, delta=delta)))
            oracle_crit = float(np.quantile(null_d, 1.0 - delta))
            thr = _thr(threshold, n, n, alpha, delta)
            row = dict(
                scenario=scenario, n=int(n), alpha=float(alpha), regime=label,
                reps=reps, n_perm=n_perm, delta=delta,
                alpha_star=sc.alpha_star,
                mean_dhat=float(d.mean()),
                analytic_tau=float(thr),
                mean_perm_crit=float(perm_crit.mean()),
                oracle_crit=oracle_crit,
                mmd_screen_crit=float(2 * alpha + mmd_radius(n, n, delta)),
                mean_mmd=float(mm.mean()),
                rate_analytic=float(np.mean(d > thr)),
                rate_permutation=float(np.mean(d > perm_crit)),
                rate_mmd_screen=float(np.mean(
                    [reject_mmd_screen(v, n, n, alpha, delta) for v in mm])),
                rate_oracle=float(np.mean(d > oracle_crit)),
            )
            row.update(_v2_extra(threshold, n, n, alpha, delta, stat=d))
            if certificate != "bootstrap":
                from .v2 import certificate_methods
                methods = certificate_methods(certificate)
                n_boot = 1000 if "bootstrap" in methods else 0
                seq = "sequential" in methods
                sjobs = [(scenario, n, n, float(alpha), 70_000 + i, n_boot, delta, seq)
                         for i in range(reps)]
                sparams = dict(scenario=scenario, n=int(n), alpha=float(alpha), reps=reps,
                               delta=delta, n_boot=n_boot)
                if seq:
                    from ..splitting import SEQ_EDGES
                    sparams.update(sequential=True, seq_edges=list(SEQ_EDGES),
                                   seq_horizon="fixed")
                sres = _map(w_e6_split, sjobs, env.max_workers,
                            experiment="e6_split", params=sparams, seed_index=4)
                row["certificate"] = certificate
                row["rate_split_betting"] = float(np.mean([r["betting"] for r in sres]))
                if n_boot:
                    row["rate_split_bootstrap"] = float(
                        np.mean([r["bootstrap"] for r in sres]))
                if seq:
                    row["rate_split_sequential"] = float(
                        np.mean([r["sequential"] for r in sres]))
                row["rate_split"] = row[f"rate_split_{methods[0]}"]
            rows.append(row)
    return rows


def w_split(args):
    """One replicate of the split-sample interval."""
    spec, n, m, seed, grid, c0, gamma, delta = args
    sc, X, Y, rng = _draw(spec, n, m, seed)
    from ..splitting import lower_bound_tv, split_upper_bound, upper_bound_tv
    lo, details = lower_bound_tv(X, Y, grid, delta=delta / 2, rng=rng,
                                 max_iter=3000, tol=1e-10)
    a_up = lo
    _, d_up = split_upper_bound(X, Y, a_up, delta=delta / 2, rng=rng,
                                max_iter=3000, tol=1e-10)
    hi = min(upper_bound_tv(d_up, a_up, 0.0, c0, gamma), 1.0)
    return dict(lo=float(lo), hi=float(hi), d_up=float(d_up))


def _split_bracket_v2(X, Y, rng, grid, delta, c0, gamma, certificate, d0=None):
    """Revision v2 bracket on one replicate: the certificate's lower end
    (down-closed over ``grid``, per-level witness, level delta/2) and the
    split-heuristic upper end at a_up = that lower end, exactly as v1 builds
    them. The reported certificate (``lo``, ``hi``, ``d_up``) is the first of
    :func:`trimmedmmd.experiments.v2.certificate_methods`: the sequential
    betting certificate for "sequential" and "all" (the paper's main one,
    e = (0, 1/4, 1/2, 1)), the single-split betting certificate for "betting"
    and "both"; the others come back with a ``_<method>`` suffix ("all":
    ``*_betting`` and ``*_bootstrap``; "both": ``*_bootstrap``).

    ``rng`` is the replicate's stream after the draw (and after the d0 solve
    in E8): the lower bound takes one integer from it (the split seed, as
    ``lower_bound_tv`` does; the sequential certificate and the single split
    then share the row permutations of ``default_rng(split_seed)``), and each
    upper end is computed from a copy of the stream left after that, which is
    the state v1's ``split_upper_bound`` call saw. The bootstrap bound
    (Theorem 16, 1000 resamples, same splits and witnesses) and its bracket
    reproduce v1's ``lo``, ``hi`` and ``d_up`` exactly (same random streams).
    """
    import copy
    from ..splitting import lower_bound_betting, split_upper_bound, upper_bound_tv
    from .v2 import certificate_methods
    methods = certificate_methods(certificate)
    has_boot = "bootstrap" in methods
    bounds, _ = lower_bound_betting(X, Y, grid, delta=delta / 2, rng=rng,
                                    methods=methods, n_boot=1000 if has_boot else 0,
                                    max_iter=3000, tol=1e-10)
    state = copy.deepcopy(rng)
    out = {}
    for k, mth in enumerate(methods):
        suffix = "" if k == 0 else f"_{mth}"
        lo = float(bounds[mth])
        _, d_up = split_upper_bound(X, Y, lo, delta=delta / 2, rng=copy.deepcopy(state),
                                    max_iter=3000, tol=1e-10)
        hi = min(upper_bound_tv(d_up, lo, 0.0, c0, gamma), 1.0)
        out.update({f"lo{suffix}": lo, f"hi{suffix}": float(hi), f"d_up{suffix}": float(d_up)})
    if d0 is not None:
        out["d0"] = float(d0)
    return out


def w_split_v2(args):
    """Revision v2 replicate of the split-sample interval (see
    :func:`_split_bracket_v2`); same draw as :func:`w_split`."""
    spec, n, m, seed, grid, c0, gamma, delta, certificate = args
    sc, X, Y, rng = _draw(spec, n, m, seed)
    return _split_bracket_v2(X, Y, rng, grid, delta, c0, gamma, certificate)


# Grid of the finite-sample upper end of Corollary cor:upperfs in E7: the
# infimum over alpha is taken over this grid (any alpha gives a valid upper
# bound on the event of Theorem A, so a coarser grid only loosens it).
E7_PATH_GRID = [round(0.05 * k, 2) for k in range(19)]          # 0, 0.05, ..., 0.9


def w_path_upper(args):
    """Revision v2, E7: full-sample path for Corollary cor:upperfs and the
    analytic alpha-hat, on the same draw as :func:`w_path` (seed, median
    bandwidth, warm-started FISTA with max_iter 4000, tol 1e-11).

    The path is solved level by level and stopped as soon as (i) the
    remaining grid cannot lower the corollary's infimum -- D-hat >= 0, so
    every later level costs at least alpha + (tau_A(alpha, delta/2)/c0)^(1/gamma),
    which is increasing in alpha -- and (ii) both thresholds (v1's tau and
    tau_new) have crossed. The infimum is therefore exactly the one over the
    whole grid, and both alpha-hats are the grid crossings (alphas[-1] if
    none). Unsolved levels are NaN. Threshold-independent output.
    """
    spec, n, m, alphas, seed, delta, c0, gamma = args
    from ..threshold import tau_A, tau_new
    _, X, Y, rng = _draw(spec, n, m, seed)
    sigma = median_heuristic(X, Y, rng=rng)
    Kxx = gaussian_kernel(X, X, sigma)
    Kyy = gaussian_kernel(Y, Y, sigma)
    Kxy = gaussian_kernel(X, Y, sigma)
    d = [float("nan")] * len(alphas)
    best = 1.0
    ahat = {"paper": None, "plugin": None}
    w0 = v0 = None
    for i, a in enumerate(alphas):
        floor = a + (tau_A(n, m, a, delta / 2) / c0) ** (1.0 / gamma)
        if floor >= best and all(v is not None for v in ahat.values()):
            break
        r = solve_gram(Kxx, Kyy, Kxy, a, w0=w0, v0=v0, rng=rng,
                       max_iter=4000, tol=1e-11)
        w0, v0 = r.w, r.v
        d[i] = float(r.discrepancy)
        best = min(best, a + (max(d[i] + tau_A(n, m, a, delta / 2), 0.0) / c0) ** (1.0 / gamma))
        if ahat["paper"] is None and d[i] <= tau(n, m, a, delta):
            ahat["paper"] = float(a)
        if ahat["plugin"] is None and d[i] <= tau_new(n, m, a, delta):
            ahat["plugin"] = float(a)
    for k in ahat:
        if ahat[k] is None:
            ahat[k] = float(alphas[-1])
    return dict(d=d, hi_cor=float(best), ahat_paper=ahat["paper"],
                ahat_plugin=ahat["plugin"], sigma=float(sigma))

# Replicates of E7. The function's own default drifts with the tier; the
# published table (results/tables/e7_split_bounds.csv, \splitReps) has 200
# replicates while run_all pins 100. run_all(e7_reps=...) overrides it.
E7_REPS = 100


def upper_end_corollary(alphas, d_hats, n, m, delta_upper, c0, gamma, kernel_consts=None):
    """Corollary cor:upperfs: TV <= inf_alpha { alpha + omega^{<-}(D-hat_alpha +
    tau_A(n, m, alpha, delta_upper)) } with omega(t) = c0 t^gamma, capped at 1.
    ``d_hats`` may be FISTA iterate values (upper bounds on the empirical
    distance), which only loosens the bound."""
    from ..threshold import tau_A
    best = 1.0
    for a, d in zip(alphas, d_hats):
        if a >= 1.0:
            continue
        val = a + (max(float(d) + tau_A(n, m, a, delta_upper, kernel_consts), 0.0) / c0) ** (1.0 / gamma)
        best = min(best, val)
    return float(best)


def _cert_params(params, certificate):
    """Checkpoint parameters of a v2 certificate cell: the certificate name and,
    when the sequential certificate is among the methods, its batch fractions
    and the fixed-horizon rule (so no earlier cache of another method set or of
    the pre-fix sequential code is ever reused)."""
    from .v2 import certificate_methods
    params = dict(params, certificate=certificate)
    if "sequential" in certificate_methods(certificate):
        from ..splitting import SEQ_EDGES
        params.update(seq_edges=list(SEQ_EDGES), seq_horizon="fixed")
    return params


def _extra_brackets(res, certificate, astar):
    """``*_<method>`` summary columns for the non-reported certificates."""
    from .v2 import certificate_methods
    out = {"reported_certificate": certificate_methods(certificate)[0]}
    for mth in certificate_methods(certificate)[1:]:
        lob = np.array([r[f"lo_{mth}"] for r in res])
        hib = np.array([r[f"hi_{mth}"] for r in res])
        out.update({
            f"mean_lower_{mth}": float(lob.mean()), f"mean_upper_{mth}": float(hib.mean()),
            f"mean_width_{mth}": float((hib - lob).mean()),
            f"lower_coverage_{mth}": float(np.mean(lob <= astar + 1e-12)),
            f"upper_coverage_{mth}": float(np.mean(hib >= astar - 1e-12)),
            f"interval_coverage_{mth}": float(np.mean((lob <= astar + 1e-12)
                                                      & (hib >= astar - 1e-12)))})
    return out


def e7_split_bounds(env, scenario="disjoint_d2_a20", delta=0.05,
                    ns=None, reps=None, c0=1.49, gamma=1.03,
                    threshold="paper", certificate="bootstrap"):
    """Two-sided bounds on TV from sample splitting.

    The point of the exercise: resolving alpha* exactly is the hard,
    nonregular problem, but an interval around it is a regular one. The
    lower end is a certificate and needs no margin condition; the upper end
    inverts the margin condition measured in E3, so it is only as good as
    those constants.

    Revision v2 (``certificate`` in {"sequential", "all", "betting",
    "both"}): the lower end is the reported finite-sample certificate
    (``mean_lower``, ...; the sequential betting certificate for "sequential"
    and "all", the single-split one for "betting" and "both"), the
    split-heuristic upper end is computed at that lower end (``mean_upper``,
    ``mean_width``, coverages), and the other methods' brackets are kept in
    ``*_betting`` / ``*_bootstrap`` columns (the bootstrap one identical to
    v1's). In
    any v2 mode (also ``threshold="plugin"`` alone) the full-sample path on
    :data:`E7_PATH_GRID` is computed once per replicate (experiment
    "e7_path", threshold-independent) for the upper end of Corollary
    cor:upperfs at delta/2 (``*_cor`` columns, interval with the
    certificate's lower end at delta/2) and the analytic alpha-hat with the
    selected threshold (``mean_alpha_hat_analytic``).
    """
    sc = dgp.make(scenario)
    ns = ns or [200, 400, 800]
    reps = reps or (20 if env.profile == "smoke" else min(800, env.n_reps))
    grid = [0.05, 0.08, 0.11, 0.14, 0.17, 0.20, 0.23, 0.26]
    v2 = threshold != "paper" or certificate != "bootstrap"
    rows = []
    for n in ns:
        if certificate == "bootstrap":
            jobs = [(scenario, n, n, 80_000 + i, grid, c0, gamma, delta)
                    for i in range(reps)]
            res = _map(w_split, jobs, env.max_workers,
                       experiment="e7_split",
                       params=dict(scenario=scenario, n=int(n), reps=reps,
                                   grid=grid, delta=delta, c0=c0, gamma=gamma),
                       seed_index=3)
        else:
            jobs = [(scenario, n, n, 80_000 + i, grid, c0, gamma, delta, certificate)
                    for i in range(reps)]
            res = _map(w_split_v2, jobs, env.max_workers,
                       experiment="e7_split",
                       params=_cert_params(dict(scenario=scenario, n=int(n), reps=reps,
                                                grid=grid, delta=delta, c0=c0, gamma=gamma),
                                           certificate),
                       seed_index=3)
        lo = np.array([r["lo"] for r in res])
        hi = np.array([r["hi"] for r in res])
        rows.append(dict(
            scenario=scenario, n=int(n), reps=reps, delta=delta,
            alpha_star=sc.alpha_star,
            mean_lower=float(lo.mean()), mean_upper=float(hi.mean()),
            mean_width=float((hi - lo).mean()),
            # the lower end is a one-sided certificate: it must not exceed TV
            lower_coverage=float(np.mean(lo <= sc.alpha_star + 1e-12)),
            upper_coverage=float(np.mean(hi >= sc.alpha_star - 1e-12)),
            interval_coverage=float(np.mean((lo <= sc.alpha_star + 1e-12)
                                            & (hi >= sc.alpha_star - 1e-12))),
            frac_lower_at_least_half=float(np.mean(lo >= 0.5 * sc.alpha_star)),
        ))
        if not v2:
            continue
        row = rows[-1]
        row["certificate"] = certificate
        row["threshold"] = threshold
        astar = sc.alpha_star
        if certificate != "bootstrap":
            row.update(_extra_brackets(res, certificate, astar))
        pjobs = [(scenario, n, n, E7_PATH_GRID, 80_000 + i, delta, c0, gamma)
                 for i in range(reps)]
        pres = _map(w_path_upper, pjobs, env.max_workers, experiment="e7_path",
                    params=dict(scenario=scenario, n=int(n), reps=reps, grid=E7_PATH_GRID,
                                delta=delta, c0=c0, gamma=gamma), seed_index=4)
        hic = np.array([r["hi_cor"] for r in pres])
        key = "ahat_plugin" if threshold == "plugin" else "ahat_paper"
        ahat = np.array([r[key] for r in pres])
        d0 = np.array([r["d"][0] for r in pres])
        row.update(
            path_grid_max=float(E7_PATH_GRID[-1]),
            mean_upper_cor=float(hic.mean()), mean_width_cor=float((hic - lo).mean()),
            upper_coverage_cor=float(np.mean(hic >= astar - 1e-12)),
            interval_coverage_cor=float(np.mean((lo <= astar + 1e-12) & (hic >= astar - 1e-12))),
            mean_alpha_hat_analytic=float(ahat.mean()),
            frac_alpha_hat_at_most_alpha_star=float(np.mean(ahat <= astar + 1e-12)),
            rate_analytic_rejects_0=float(np.mean(d0 > _thr(threshold, n, n, 0.0, delta))),
            mean_alpha_hat_paper=float(np.mean([r["ahat_paper"] for r in pres])))
    return rows


def w_dimsweep(args):
    """Signal and split interval for one replicate on one sample space."""
    spec, n, m, seed, grid, delta, c0, gamma = args
    sc, X, Y, rng = _draw(spec, n, m, seed)
    from ..splitting import lower_bound_tv, split_upper_bound, upper_bound_tv
    from ..discrepancy import trimmed_mmd_gram
    d0 = trimmed_mmd_gram(X, Y, 0.0, rng=rng, max_iter=3000,
                          tol=1e-10).discrepancy
    lo, _ = lower_bound_tv(X, Y, grid, delta=delta / 2, rng=rng,
                           max_iter=3000, tol=1e-10)
    _, d_up = split_upper_bound(X, Y, lo, delta=delta / 2, rng=rng,
                                max_iter=3000, tol=1e-10)
    hi = min(upper_bound_tv(d_up, lo, 0.0, c0, gamma), 1.0)
    return dict(d0=float(d0), lo=float(lo), hi=float(hi))


def w_dimsweep_v2(args):
    """Revision v2 replicate of :func:`w_dimsweep`: same draw and d0 solve,
    then the betting bracket of :func:`_split_bracket_v2`."""
    spec, n, m, seed, grid, delta, c0, gamma, certificate = args
    sc, X, Y, rng = _draw(spec, n, m, seed)
    from ..discrepancy import trimmed_mmd_gram
    d0 = trimmed_mmd_gram(X, Y, 0.0, rng=rng, max_iter=3000,
                          tol=1e-10).discrepancy
    return _split_bracket_v2(X, Y, rng, grid, delta, c0, gamma, certificate, d0=d0)


def e8_sample_spaces(env, n=400, reps=None, delta=0.05,
                     c0=1.49, gamma=1.03, threshold="paper", certificate="bootstrap"):
    """Does any of this depend on the dimension? It should not.

    The concentration radius of thm:concentration contains no dimension: ass:regularity
    asks only for a bounded characteristic kernel on some sample space. This
    sweep holds alpha* fixed and varies the space, from R^2 up to a function
    space, in two regimes -- contamination separated in every coordinate
    (which gets easier as d grows) and contamination confined to a single
    coordinate (which gets harder, since the remaining d-1 coordinates are
    pure noise). The second is the honest test.

    Revision v2: ``threshold="plugin"`` puts tau_new(n, n, 0) in
    ``analytic_tau_at_0`` (v1's in ``tau_paper_at_0``) and adds
    ``rate_analytic_at_0`` (fraction of replicates with d0 > threshold);
    ``certificate`` in {"sequential", "all", "betting", "both"} uses the
    reported certificate's lower end (sequential betting for "sequential" and
    "all") and the split-heuristic upper end at it, with the other methods'
    brackets in ``*_betting`` / ``*_bootstrap`` columns (the bootstrap one =
    v1's), as in E7; new checkpoint keys ("e8_spaces" + certificate).
    """
    reps = reps or (10 if env.profile == "smoke" else min(400, env.n_reps))
    grid = [0.05, 0.08, 0.11, 0.14, 0.17, 0.20, 0.23, 0.26]
    dims = [2, 5, 10, 20, 50]
    specs = ([("disjoint", d, 0.20) for d in dims]
             + [("gaussian", d, 0.30, 1.0) for d in dims]
             + [("functional",)])
    rows = []
    for spec in specs:
        sc = _build(spec)
        if certificate == "bootstrap":
            jobs = [(spec, n, n, 90_000 + i, grid, delta, c0, gamma)
                    for i in range(reps)]
            res = _map(w_dimsweep, jobs, env.max_workers,
                       experiment="e8_spaces",
                       params=dict(spec=list(spec), n=n, reps=reps, grid=grid,
                                   delta=delta, c0=c0, gamma=gamma), seed_index=3)
        else:
            jobs = [(spec, n, n, 90_000 + i, grid, delta, c0, gamma, certificate)
                    for i in range(reps)]
            res = _map(w_dimsweep_v2, jobs, env.max_workers,
                       experiment="e8_spaces",
                       params=_cert_params(dict(spec=list(spec), n=n, reps=reps, grid=grid,
                                                delta=delta, c0=c0, gamma=gamma),
                                           certificate), seed_index=3)
        lo = np.array([r["lo"] for r in res])
        hi = np.array([r["hi"] for r in res])
        d0 = np.array([r["d0"] for r in res])
        rows.append(dict(
            family=spec[0], dim=sc.dim, name=sc.name, n=n, reps=reps,
            alpha_star=sc.alpha_star,
            signal_d0=float(d0.mean()),
            analytic_tau_at_0=_thr(threshold, n, n, 0.0, delta),
            mean_lower=float(lo.mean()), mean_upper=float(hi.mean()),
            mean_width=float((hi - lo).mean()),
            lower_coverage=float(np.mean(lo <= sc.alpha_star + 1e-12)),
            upper_coverage=float(np.mean(hi >= sc.alpha_star - 1e-12)),
            interval_coverage=float(np.mean((lo <= sc.alpha_star + 1e-12)
                                            & (hi >= sc.alpha_star - 1e-12))),
        ))
        row = rows[-1]
        if threshold != "paper":
            row.update(threshold=threshold, tau_paper_at_0=float(tau(n, n, 0.0, delta)),
                       rate_analytic_at_0=float(np.mean(d0 > row["analytic_tau_at_0"])),
                       rate_paper_at_0=float(np.mean(d0 > tau(n, n, 0.0, delta))))
        if certificate != "bootstrap":
            row["certificate"] = certificate
            row.update(_extra_brackets(res, certificate, sc.alpha_star))
    return rows


# --------------------------------------------------------------------------
# E5: computational scaling
# --------------------------------------------------------------------------

def e5_scaling(env, scenario="disjoint_d2_a20", delta=0.05):
    """Runtime and solver effort, exact kernel vs random features."""
    from ..discrepancy import plain_mmd, trimmed_mmd_path as path_fn
    sc = dgp.make(scenario)
    rng = np.random.default_rng(99)
    grid = np.linspace(0.0, 0.4, 9)
    rows = []
    ns = sorted({200, 400, 800, min(1600, env.max_n_gram)})
    for n in ns:
        X, Y = sc.sample(n, rng, "P"), sc.sample(n, rng, "Q")
        t0 = time.perf_counter()
        _, d_exact, res = path_fn(X, Y, grid, rng=rng, max_iter=4000, tol=1e-11)
        t_exact = time.perf_counter() - t0
        iters = int(sum(r.n_iter for r in res))

        t0 = time.perf_counter()
        _, d_rff, res_r = path_fn(X, Y, grid, rng=rng, use_rff=True,
                                  n_features=env.n_features,
                                  max_iter=4000, tol=1e-11)
        t_rff = time.perf_counter() - t0
        rows.append(dict(
            scenario=scenario, n=int(n), n_features=env.n_features,
            seconds_exact_path=float(t_exact), seconds_rff_path=float(t_rff),
            total_fista_iters_exact=iters,
            max_abs_diff_exact_vs_rff=float(np.max(np.abs(d_exact - d_rff))),
            mean_dhat_exact=float(d_exact.mean()),
        ))
    return rows


# --------------------------------------------------------------------------
# E9: the boundary-law conjecture (conj:boundary), r=1 tractable special case
#
# The conjecture is stated for a fixed feature dimension r with a single
# contact point and known tangent cones. r=1 (one linear feature, i.e. work
# directly with the identity map on R) is the simplest instance satisfying
# its hypotheses (B1)-(B3): trimmed regions are intervals, a single contact
# point is just two intervals touching at one endpoint, and the tangent
# cones there are half-lines, so K = cl(T_Q - T_P) = [0, infinity) and the
# conjectured limit collapses to a half-normal.
#
# Construction: P ~ Uniform[0,1], Q ~ c + Uniform[0,1]. The upper-tail
# trimmed mean of P at level alpha is (1+alpha)/2 (closed form, matches
# lem:trimmedmean); the lower-tail trimmed mean of Q at the same alpha is
# c + (1-alpha)/2. These meet at alpha* = c, at the point z* = (1+c)/2 --
# a genuine single contact, with the two regions strictly separated for
# alpha < alpha* exactly as thm:consistency requires.
#
# This is an *empirical* test of a plausible special case, not a proof that
# (B1)-(B3) hold here -- (B2), the marginal CLT for the boundary order
# statistic, is itself checked first (Stigler 1973-type trimmed-mean CLT,
# classical but not literally re-derived here) before being used to
# calibrate the joint prediction. Flagged as such throughout.
# --------------------------------------------------------------------------

def _z_star(c: float) -> float:
    return (1.0 + c) / 2.0


def w_boundary_marginal(args):
    """One replicate of sqrt(n) * (boundary order statistic - z*), P side."""
    n, alpha_star, seed = args
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, n)
    u = trimmed_mean(X, alpha_star, upper=True)
    return float(np.sqrt(n) * (u - _z_star(alpha_star)))


def w_boundary_marginal_Q(args):
    """Same, Q side (lower-tail trimmed mean of the shifted uniform)."""
    n, alpha_star, c, seed = args
    rng = np.random.default_rng(seed)
    Y = c + rng.uniform(0.0, 1.0, n)
    u = trimmed_mean(Y, alpha_star, upper=False)
    return float(np.sqrt(n) * (u - _z_star(alpha_star)))


def e9_boundary_marginal(env, alpha_star=0.4, n=20_000, reps=None):
    """Calibrate sigma_P, sigma_Q: does (B2) -- marginal CLT -- hold here?

    reps is deliberately large: each replicate is a sort of n uniforms, so
    this is one of the few places in the suite where "run it bigger" buys
    real statistical power (a tighter KS test in e9_boundary_joint) rather
    than hitting a structural floor -- unlike, e.g., the iterated-logarithm
    example the skill warns about.
    """
    reps = reps or env.n_mc_null
    jobs_p = [(n, alpha_star, 200_000 + i) for i in range(reps)]
    # c only shifts z* (already subtracted out), so any c reproduces the
    # same Q-side CLT; use c=alpha_star to match the joint experiment's own
    # construction exactly.
    jobs_q = [(n, alpha_star, alpha_star, 300_000 + i) for i in range(reps)]

    vp = np.array(_map(w_boundary_marginal, jobs_p, env.max_workers,
                       experiment="e9_marginal_P",
                       params=dict(n=n, alpha_star=alpha_star, reps=reps)))
    vq = np.array(_map(w_boundary_marginal_Q, jobs_q, env.max_workers,
                       experiment="e9_marginal_Q",
                       params=dict(n=n, alpha_star=alpha_star, reps=reps)))

    from scipy import stats
    sigma_p = float(vp.std(ddof=1))
    sigma_q = float(vq.std(ddof=1))
    ks_p = stats.kstest(vp, "norm", args=(float(vp.mean()), sigma_p))
    ks_q = stats.kstest(vq, "norm", args=(float(vq.mean()), sigma_q))
    row = dict(
        alpha_star=alpha_star, n=n, reps=reps,
        sigma_P=sigma_p, sigma_Q=sigma_q,
        mean_P=float(vp.mean()), mean_Q=float(vq.mean()),
        ks_stat_P=float(ks_p.statistic), ks_pvalue_P=float(ks_p.pvalue),
        ks_stat_Q=float(ks_q.statistic), ks_pvalue_Q=float(ks_q.pvalue),
    )
    return row, sigma_p, sigma_q


def w_boundary_joint(args):
    """One replicate of the boundary statistic D-hat_alpha* (r=1 case)."""
    n, m, alpha_star, c, seed = args
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, n)
    Y = c + rng.uniform(0.0, 1.0, m)
    amax_p = trimmed_mean(X, alpha_star, upper=True)
    amin_q = trimmed_mean(Y, alpha_star, upper=False)
    return float(max(0.0, amin_q - amax_p))


def e9_boundary_joint(env, sigma_p, sigma_q, alpha_star=0.4, pairs=None, reps=None):
    """Test sqrt(min(n,m)) * D-hat_alpha* against the conjectured limit.

    IMPORTANT and non-obvious: the limit is NOT a plain half-normal. D-hat
    = max(0, -Z) for Z ~ N(0, s^2) (the raw signed separation), which has a
    point mass of exactly 1/2 at zero plus, conditional on being positive, a
    genuine half-normal(scale=s) density -- this IS the chi-bar-square
    structure the conjecture names (a mixture with an atom, not a pure
    chi-square/half-normal). A naive single KS test against `halfnorm`
    compares an atomic distribution to a continuous one and is spuriously
    rejected regardless of whether the theory is right (verified: it fails
    with KS~0.5 even on a synthetic sample drawn EXACTLY from the correct
    null). The valid test is two-part: a binomial test that the atom mass is
    1/2, and a KS test restricted to the strictly-positive values against
    the continuous half-normal conditional law.
    """
    from scipy import stats
    pairs = pairs or [(2000, 2000), (2000, 8000), (8000, 2000), (4000, 4000)]
    reps = reps or env.n_mc_null
    rows = []
    for n, m in pairs:
        jobs = [(n, m, alpha_star, alpha_star, 400_000 + i) for i in range(reps)]
        d = np.array(_map(w_boundary_joint, jobs, env.max_workers,
                          experiment="e9_joint",
                          params=dict(n=n, m=m, alpha_star=alpha_star, reps=reps)))
        kappa = min(n, m)
        scaled = np.sqrt(kappa) * d
        s2 = (kappa / m) * sigma_q ** 2 + (kappa / n) * sigma_p ** 2
        s = float(np.sqrt(s2))

        n_atom = int(np.sum(scaled == 0.0))
        bt = stats.binomtest(n_atom, len(scaled), 0.5)
        pos = scaled[scaled > 0.0]
        ks = stats.kstest(pos, "halfnorm", args=(0.0, s))
        rows.append(dict(
            n=n, m=m, alpha_star=alpha_star, reps=reps, kappa=kappa,
            predicted_scale=s, sigma_P=sigma_p, sigma_Q=sigma_q,
            atom_fraction=float(n_atom / len(scaled)),
            atom_binom_pvalue=float(bt.pvalue),
            mean_conditional=float(pos.mean()) if pos.size else float("nan"),
            predicted_conditional_mean=float(s * np.sqrt(2.0 / np.pi)),
            ks_stat_conditional=float(ks.statistic),
            ks_pvalue_conditional=float(ks.pvalue),
        ))
    return rows


def e9_boundary_conjecture(env, alpha_star=0.4):
    """CLAIMS.md n18 (conj:boundary), r=1 tractable special case. Driver."""
    marg_row, sigma_p, sigma_q = e9_boundary_marginal(env, alpha_star=alpha_star)
    joint_rows = e9_boundary_joint(env, sigma_p, sigma_q, alpha_star=alpha_star)
    return marg_row, joint_rows


# --------------------------------------------------------------------------
# E10: RFF approximation rate (lem:rffkernel, thm:rffjoint) -- n15/n16
# --------------------------------------------------------------------------

def _rff_max_error(X, sigma, n_features, rng):
    """sup over the sample of |z_r(x)'z_r(y) - k(x,y)|, a sample-based proxy
    for the lemma's sup over the whole compact domain (a lower bound on the
    true sup, not the sup itself -- noted in the report)."""
    from ..kernels import RFF
    rff = RFF(dim=X.shape[1], n_features=n_features, sigma=sigma, rng=rng)
    Z = rff.transform(X)
    approx = Z @ Z.T
    exact = gaussian_kernel(X, X, sigma)
    return float(np.abs(approx - exact).max())


def e10_rff_rate(env, n=1500, sigma=1.0, d=3, reps=None, r_grid=None):
    """Fit the empirical decay exponent of the RFF kernel-approximation
    error in r (lem:rffkernel claims exactly r^{-1/2}, no log r), and run the
    Remark's own falsifying construction: a near-degenerate domain, where a
    sigma_Lambda*diam(X)-type prefactor would wrongly predict vanishing
    error, must still show nonvanishing O(r^{-1/2}) error.
    """
    reps = reps or max(20, env.n_reps // 40)
    r_grid = r_grid or sorted({64, 128, 256, 512, 1024, 2048,
                               min(8192, env.n_features * 4)})
    rng = np.random.default_rng(11)
    X = rng.standard_normal((n, d))

    rows = []
    for r in r_grid:
        errs = [_rff_max_error(X, sigma, r, np.random.default_rng(1000 + r + i))
                for i in range(reps)]
        rows.append(dict(regime="generic", n=n, d=d, r=r, reps=reps,
                         mean_max_err=float(np.mean(errs)),
                         std_max_err=float(np.std(errs))))

    # Degenerate-domain negative control: collapse X to (near-)one point.
    X_deg = np.tile(rng.standard_normal((1, d)), (n, 1)) + 1e-9 * rng.standard_normal((n, d))
    rows_deg = []
    for r in r_grid:
        errs = [_rff_max_error(X_deg, sigma, r, np.random.default_rng(2000 + r + i))
                for i in range(reps)]
        rows_deg.append(dict(regime="degenerate_domain", n=n, d=d, r=r, reps=reps,
                             mean_max_err=float(np.mean(errs)),
                             std_max_err=float(np.std(errs))))

    def fit_rate(rows_):
        log_r = np.log([row["r"] for row in rows_])
        log_e = np.log([max(row["mean_max_err"], 1e-300) for row in rows_])
        slope, intercept = np.polyfit(log_r, log_e, 1)
        return float(slope), float(np.exp(intercept))

    slope_generic, c_generic = fit_rate(rows)
    slope_deg, c_deg = fit_rate(rows_deg)
    fit = dict(
        generic_rate_exponent=slope_generic, generic_const=c_generic,
        degenerate_rate_exponent=slope_deg, degenerate_const=c_deg,
        theory_predicts_exponent=-0.5,
        # A diam-multiplicative bound would predict the degenerate error
        # collapses to ~0 (diam(X)->0); the actual claim (log-dependence
        # only) predicts it stays comparable to the generic case.
        degenerate_error_nonvanishing=bool(rows_deg[-1]["mean_max_err"] > 0.05),
    )
    return rows + rows_deg, fit


# --------------------------------------------------------------------------
# E11: trimmed Sinkhorn divergence -- prop:sinkmono, n11 continuity proxy
#
# New solver (trimmedmmd.sinkhorn), verified against POT for the inner
# entropic-OT computation (tests/test_sinkhorn.py). The OUTER trimming
# optimization is a genuinely nonconvex bilevel problem (unlike the convex
# QP of the MMD case), so a single-restart run can find a worse local
# optimum at one alpha than at a neighboring one purely from solver
# suboptimality -- this was observed directly while building this block
# (see CLAIMS.md's triage log). The fix used here, not a unit-test
# workaround: several random restarts per (instance, alpha) in addition to
# the warm start, keeping the best value found, before ever asking whether
# the *sequence* is monotone.
# --------------------------------------------------------------------------

def w_sinkhorn_path(args):
    """One instance: best-of-restarts trimmed Sinkhorn value at every alpha
    on the grid, warm-started along the path AND multi-restarted at each
    level. Returns the value sequence (for monotonicity) and consecutive
    gaps (for the n11 continuity proxy)."""
    from ..sinkhorn import trimmed_sinkhorn
    n, m, d, eps, grid, n_restarts, max_iter, sink_iter, seed = args
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    Y = rng.standard_normal((m, d)) + 0.8

    def cost_fn(A, B):
        return ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)

    vals = []
    w0 = v0 = None
    for alpha in grid:
        best = None
        # warm start from the previous (smaller) alpha
        if w0 is not None:
            r = trimmed_sinkhorn(X, Y, alpha, eps, cost_fn=cost_fn,
                                 max_iter=max_iter, sinkhorn_iter=sink_iter,
                                 w0=w0, v0=v0, lr=0.3)
            best = r
        for rs in range(n_restarts):
            r = trimmed_sinkhorn(X, Y, alpha, eps, cost_fn=cost_fn,
                                 max_iter=max_iter, sinkhorn_iter=sink_iter,
                                 lr=0.3, seed=seed + 1000 * rs + 7)
            if best is None or r["value"] < best["value"]:
                best = r
        w0, v0 = best["w"], best["v"]
        vals.append(best["value"])
    vals = np.array(vals)
    gaps = np.abs(np.diff(vals))
    return dict(values=vals.tolist(), gaps=gaps.tolist())


def e11_sinkhorn(env, n=25, m=20, d=2, eps=0.3, n_restarts=3,
                 max_iter=250, sink_iter=100, instances=None, grid=None):
    """CLAIMS.md n17 (prop:sinkmono) and n11 (continuity proxy for
    prop:sinkexist's hypothesis (C)). CPU-parallel (one process per
    instance) rather than GPU-batched: this box's 48-core process pool
    already gives the restart-parallelism that matters here, and a batched
    GPU Sinkhorn was judged not worth the added implementation risk in
    this pass given the correctness work already required for the inner
    solver -- left as a follow-up if this block turns out to be a
    bottleneck at real scale.
    """
    grid = grid or [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    instances = instances or max(20, env.max_workers)
    jobs = [(n, m, d, eps, grid, n_restarts, max_iter, sink_iter, 500_000 + i)
            for i in range(instances)]
    results = _map(w_sinkhorn_path, jobs, env.max_workers,
                   experiment="e11_sinkhorn",
                   params=dict(n=n, m=m, d=d, eps=eps, grid=grid,
                               n_restarts=n_restarts, max_iter=max_iter,
                               sink_iter=sink_iter, instances=instances))

    all_vals = np.array([r["values"] for r in results])  # instances x |grid|
    diffs = np.diff(all_vals, axis=1)  # negative = correctly decreasing
    violation_mask = diffs > 1e-4       # a real increase beyond solver noise
    n_violations = int(violation_mask.sum())
    n_checks = violation_mask.size
    all_gaps = np.concatenate([r["gaps"] for r in results])

    summary = dict(
        instances=instances, grid=grid, eps=eps, n=n, m=m, d=d,
        n_restarts=n_restarts,
        monotonicity_violations=n_violations,
        monotonicity_checks=n_checks,
        violation_rate=float(n_violations / n_checks) if n_checks else 0.0,
        max_violation_size=float(diffs[violation_mask].max()) if n_violations else 0.0,
        max_consecutive_gap=float(all_gaps.max()) if all_gaps.size else 0.0,
        mean_consecutive_gap=float(all_gaps.mean()) if all_gaps.size else 0.0,
        # a genuine discontinuity (proxy for hypothesis (C) failing) would
        # show as an outlier gap far above the typical consecutive gap
        # for the SAME instance, not just a generically larger one
        max_gap_to_mean_gap_ratio=float(
            (all_gaps.max() / all_gaps.mean())) if all_gaps.size and all_gaps.mean() > 0 else float("nan"),
    )
    per_instance_rows = [
        dict(instance=i, **{f"alpha_{a}": float(v) for a, v in zip(grid, vals)})
        for i, vals in enumerate(all_vals)
    ]
    return per_instance_rows, summary


# --------------------------------------------------------------------------
# E12: nestedness of the empirical rejection set (cor:lower's caveat, n8)
# --------------------------------------------------------------------------

def w_nestedness(args):
    """One replicate: rejection indicator across a fine alpha-grid, single
    draw. cor:lower assumes the empirical rejection set is nested (an
    interval) in finite samples; the paper states this only as a condition,
    since the empirical statistic uses a different witness at each alpha and
    "need not be exactly monotone." Count sign changes to see how often."""
    spec, n, m, alphas, seed, delta = args
    _, X, Y, rng = _draw(spec, n, m, seed)
    sigma = median_heuristic(X, Y, rng=rng)
    Kxx = gaussian_kernel(X, X, sigma)
    Kyy = gaussian_kernel(Y, Y, sigma)
    Kxy = gaussian_kernel(X, Y, sigma)

    def solve_one(a, w0, v0):
        return solve_gram(Kxx, Kyy, Kxy, a, w0=w0, v0=v0, rng=rng,
                          max_iter=4000, tol=1e-11)

    res = solve_path(solve_one, np.asarray(alphas))
    d = np.array([r.discrepancy for r in res])
    thr = np.array([tau(n, m, a, delta) for a in alphas])
    return (d > thr).astype(int).tolist()


def e12_nestedness(env, scenario="disjoint_d2_a20", n=300, delta=0.05,
                   n_grid=25, reps=None):
    """CLAIMS.md n8: how often is the empirical rejection set non-nested?"""
    sc = dgp.make(scenario)
    alphas = np.linspace(0.0, min(sc.alpha_star * 2.0, 0.9), n_grid)
    reps = reps or env.n_reps
    jobs = [(scenario, n, n, alphas.tolist(), 1_100_000 + i, delta)
            for i in range(reps)]
    results = _map(w_nestedness, jobs, env.max_workers,
                   experiment="e12_nestedness",
                   params=dict(scenario=scenario, n=n, alphas=alphas.tolist(),
                               delta=delta, reps=reps), seed_index=4)
    transitions = [int(np.sum(np.diff(r) != 0)) for r in results]
    non_nested = int(np.sum(np.array(transitions) > 1))
    return dict(scenario=scenario, n=n, reps=reps, n_grid=n_grid, delta=delta,
               alpha_star=sc.alpha_star,
               non_nested_count=non_nested,
               non_nested_rate=float(non_nested / reps),
               max_transitions=int(max(transitions)) if transitions else 0,
               mean_transitions=float(np.mean(transitions)) if transitions else 0.0)


# --------------------------------------------------------------------------
# E13: simultaneous validity across alpha (cor:uniform, n10)
# --------------------------------------------------------------------------

def w_uniform_path(args):
    """One replicate: does ANY point on a fine alpha-grid, from the SAME
    draw, violate its own tau(alpha)? Existing blocks resample per alpha
    (e1/e2), which cannot see a uniformity failure -- a per-alpha-resampled
    design trivially has each alpha's own ~delta rate and tells us nothing
    about whether ONE event controls the whole path."""
    spec, n, m, alphas, seed, delta = args
    _, X, Y, rng = _draw(spec, n, m, seed)
    sigma = median_heuristic(X, Y, rng=rng)
    Kxx = gaussian_kernel(X, X, sigma)
    Kyy = gaussian_kernel(Y, Y, sigma)
    Kxy = gaussian_kernel(X, Y, sigma)

    def solve_one(a, w0, v0):
        return solve_gram(Kxx, Kyy, Kxy, a, w0=w0, v0=v0, rng=rng,
                          max_iter=4000, tol=1e-11)

    res = solve_path(solve_one, np.asarray(alphas))
    d = np.array([r.discrepancy for r in res])
    thr = np.array([tau(n, m, a, delta) for a in alphas])
    return bool(np.any(d > thr))


def e13_uniform_validity(env, scenario="disjoint_d2_a20", n=300, delta=0.05,
                         eta=0.1, n_grid=30, reps=None):
    """CLAIMS.md n10: the "any-alpha violates" rate across [alpha*, 1-eta]
    should stay near delta (cor:uniform's single-event claim), not grow
    with the grid size the way a naive per-alpha union bound would predict.
    """
    sc = dgp.make(scenario)
    alphas = np.linspace(sc.alpha_star, 1.0 - eta, n_grid)
    reps = reps or env.n_mc_null
    jobs = [(scenario, n, n, alphas.tolist(), 1_200_000 + i, delta)
            for i in range(reps)]
    violations = _map(w_uniform_path, jobs, env.max_workers,
                      experiment="e13_uniform",
                      params=dict(scenario=scenario, n=n, alphas=alphas.tolist(),
                                  delta=delta, reps=reps), seed_index=4)
    rate = float(np.mean(violations))
    return dict(scenario=scenario, n=n, reps=reps, delta=delta, n_grid=n_grid,
               eta=eta, alpha_star=sc.alpha_star,
               any_alpha_violation_rate=rate,
               naive_independent_union_bound_estimate=float(min(1.0, delta * n_grid)),
               consistent_with_single_controlling_event=bool(rate <= 1.5 * delta))


# --------------------------------------------------------------------------
# E14: empirical consequence of admissible vs inadmissible delta-schedules
# (cor:consistency, n12 -- the exact tau-formula half is in
# tests/test_theory_claims.py::TestConsistencySchedule)
# --------------------------------------------------------------------------

def w_alphahat_at_delta(args):
    spec, n, m, grid, delta, seed = args
    _, X, Y, rng = _draw(spec, n, m, seed)
    sigma = median_heuristic(X, Y, rng=rng)
    Kxx = gaussian_kernel(X, X, sigma)
    Kyy = gaussian_kernel(Y, Y, sigma)
    Kxy = gaussian_kernel(X, Y, sigma)

    def solve_one(a, w0, v0):
        return solve_gram(Kxx, Kyy, Kxy, a, w0=w0, v0=v0, rng=rng,
                          max_iter=4000, tol=1e-11)

    res = solve_path(solve_one, np.asarray(grid))
    d = np.array([r.discrepancy for r in res])
    return alpha_hat_from_path(np.asarray(grid), d, n, m, delta)


def e14_consistency_schedule(env, scenario="disjoint_d2_a20", ns=None,
                             reps=None, n_grid=13):
    """CLAIMS.md n12: does alpha-hat's error actually shrink under an
    admissible schedule and fail to under the paper's own flagged
    counterexample? n capped modestly (<=700): delta_n=exp(-n) underflows
    to a literal 0 in float64 beyond that, which is a floating-point
    artifact, not part of what's being tested.
    """
    sc = dgp.make(scenario)
    ns = ns or [50, 100, 200, 400, 700]
    reps = reps or max(30, env.n_reps // 20)
    grid = np.linspace(0.0, 0.6, n_grid)
    rows = []
    for n in ns:
        for sched, delta in [("admissible_1_over_n", 1.0 / n),
                             ("inadmissible_exp_neg_n", max(np.exp(-n), 1e-300))]:
            jobs = [(scenario, n, n, grid.tolist(), delta, 1_300_000 + i)
                    for i in range(reps)]
            ah = np.array(_map(w_alphahat_at_delta, jobs, env.max_workers,
                               experiment="e14_schedule",
                               params=dict(scenario=scenario, n=int(n),
                                           schedule=sched, delta=float(delta),
                                           reps=reps, grid=grid.tolist()),
                               seed_index=5))
            err = np.abs(ah - sc.alpha_star)
            rows.append(dict(scenario=scenario, n=int(n), schedule=sched,
                             delta=float(delta), reps=reps,
                             mean_alpha_hat=float(ah.mean()),
                             mean_abs_error=float(err.mean()),
                             rmse=float(np.sqrt(np.mean(err ** 2)))))
    return rows


# --------------------------------------------------------------------------
# E15: coverage of thm:rate's own localization interval (n13, extends e3)
# --------------------------------------------------------------------------

def w_path_rff(args):
    """Whole alpha-path via random features (n too large for a dense Gram)."""
    spec, n, m, alphas, seed, n_features = args
    _, X, Y, rng = _draw(spec, n, m, seed)
    _, d, res = trimmed_mmd_path(X, Y, alphas, rng=rng, use_rff=True,
                                 n_features=n_features, max_iter=4000, tol=1e-11)
    if any(not r.converged for r in res):
        import warnings
        warnings.warn(f"RFF path did not converge at "
                      f"{sum(not r.converged for r in res)} of {len(res)} levels",
                      RuntimeWarning)
    return d


def gpu_path_rff(spec, n, m, alphas, seed, n_features, device="cuda",
                 max_iter=4000, tol=1e-11):
    """Same as w_path_rff, but the whole alpha-path runs on one GPU tensor
    transfer: this n (tens of thousands) is exactly the regime the CPU path
    was measured to be slow at, and exactly what gpu.py's solve_features_gpu
    was built and numerically verified for. Run sequentially across
    replicates (not process-parallel) to avoid this box's own workers
    fighting each other for the shared, contended GPU.
    """
    from .. import gpu as gpu_mod
    from ..kernels import RFF
    _, X, Y, rng = _draw(spec, n, m, seed)
    from ..kernels import median_heuristic as _mh
    sigma = _mh(X, Y, rng=rng)
    rff = RFF(dim=X.shape[1], n_features=n_features, sigma=sigma, rng=rng)
    Zx, Zy = rff.transform(X), rff.transform(Y)
    w0 = v0 = None
    vals = []
    for a in sorted(alphas):
        r = gpu_mod.solve_features_gpu(Zx, Zy, a, w0=w0, v0=v0,
                                       max_iter=max_iter, tol=tol, device=device)
        w0, v0 = r.w, r.v
        vals.append(r.discrepancy)
    order = np.argsort(np.argsort(alphas))  # vals is sorted-alpha order; unsort
    return np.array(vals)[order]


def e15_rate_coverage(env, scenario="disjoint_d2_a20", n=None, delta=0.05,
                      c0=1.49, gamma=1.03, n_grid=13, reps=None):
    """CLAIMS.md n13: not just alpha_hat <= alpha* (already tested in e2),
    but the THEOREM'S OWN two-sided claim: alpha* - epsilon_{n,m} <=
    alpha_hat <= alpha*, using its own explicit epsilon_{n,m} formula under
    the polynomial margin case. c0, gamma default to the same constants
    already used elsewhere in this codebase (e7/e8), for comparability.

    thm:rate only APPLIES once n, m are large enough that
    2*tau(alpha*,delta) <= omega(alpha*) = c0*alpha*^gamma -- otherwise
    epsilon_{n,m} is not even well-defined in [0, alpha*] and a "coverage"
    number would be meaningless. For this scenario and these margin
    constants that threshold works out to n ~ 20,000. That n was tried via
    both the CPU RFF path and a GPU path (gpu_path_rff, kept in this module)
    and both were impractically slow for a replicate sweep -- the CPU cost
    was measured directly, and the GPU version turned out to be dominated
    by per-iteration host-GPU sync overhead in the FISTA loop at this size,
    not by the matmuls GPU actually helps with. Rather than spend
    disproportionate engineering effort forcing that regime, this block
    reports the precondition check honestly at a reachable n (the envelope's
    own dense-Gram budget) -- "the theorem's own applicability threshold is
    impractically large for this scenario's margin constants" is itself a
    legitimate, informative finding, not a gap to paper over.
    """
    sc = dgp.make(scenario)
    n = n or env.max_n_gram
    omega_alpha_star = c0 * sc.alpha_star ** gamma
    t = tau(n, n, sc.alpha_star, delta)
    precondition_holds = bool(2.0 * t <= omega_alpha_star)
    grid = np.linspace(0.0, 0.6, n_grid)
    reps = reps or env.n_reps
    jobs = [(scenario, n, n, grid.tolist(), 1_400_000 + i, "gaussian")
            for i in range(reps)]
    paths = [np.asarray(p) for p in _map(
        w_path, jobs, env.max_workers, experiment="e15_paths",
        params=dict(scenario=scenario, n=n, reps=reps, grid=grid.tolist()),
        seed_index=4)]
    ah = np.array([alpha_hat_from_path(grid, d, n, n, delta) for d in paths])
    if precondition_holds:
        eps_nm = (2.0 * t / c0) ** (1.0 / gamma)
        lower_bound = sc.alpha_star - eps_nm
        coverage = float(np.mean((ah >= lower_bound - 1e-9) & (ah <= sc.alpha_star + 1e-9)))
    else:
        eps_nm = lower_bound = coverage = float("nan")
    return dict(scenario=scenario, n=n, reps=reps, delta=delta, c0=c0, gamma=gamma,
               alpha_star=sc.alpha_star, tau_at_alpha_star=float(t),
               omega_alpha_star=float(omega_alpha_star),
               precondition_2tau_le_omega_holds=precondition_holds,
               epsilon_nm=float(eps_nm), predicted_lower_bound=float(lower_bound),
               coverage=coverage, nominal=1.0 - delta,
               consistent_with_theorem=(bool(coverage >= 1.0 - delta - 0.02)
                                        if precondition_holds else None))


# --------------------------------------------------------------------------
# E16: ass:split negative control -- degenerate density at the cut quantile
# (n9)
# --------------------------------------------------------------------------

def _sample_split_regularity_kind(kind, n, z_alpha, p0, rng):
    if kind == "smooth":
        return rng.standard_normal(n)
    is_atom = rng.random(n) < p0
    return np.where(is_atom, z_alpha, rng.standard_normal(n))


def _split_regularity_reference(kind, alpha, z_alpha, p0, N=2_000_000, seed=999):
    rng = np.random.default_rng(seed)
    X = _sample_split_regularity_kind(kind, N, z_alpha, p0, rng)
    return float(trimmed_mean(X, alpha, upper=True))


def w_split_regularity(args):
    kind, n, alpha, z_alpha, p0, delta, n_boot, seed = args
    rng = np.random.default_rng(seed)
    X = _sample_split_regularity_kind(kind, n, z_alpha, p0, rng)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = trimmed_mean(X[idx], alpha, upper=True)
    return float(np.quantile(boots, delta / 2)), float(np.quantile(boots, 1 - delta / 2))


def e16_split_regularity(env, alpha=0.3, p0=0.2, delta=0.05, ns=None,
                         reps=None, n_boot=500):
    """CLAIMS.md n9: ass:split requires the witness-projected law to have
    positive density near its cut quantile. Positive control: standard
    normal (smooth everywhere). Negative control: a mixture with a point
    mass of probability p0 exactly AT the true (1-alpha)-quantile, directly
    violating the assumption. Ground truth for coverage comes from a single
    large (N=2e6) reference sample per kind, not a closed form, to avoid
    introducing a hand-derived formula that could itself be wrong.
    """
    from scipy.stats import norm
    z_alpha = float(norm.ppf(alpha))
    ns = ns or [200, 500, 1000, 2000]
    reps = reps or max(100, env.n_reps // 8)
    rows = []
    for kind in ("smooth", "atom_at_cutoff"):
        ref = _split_regularity_reference(kind, alpha, z_alpha, p0)
        for n in ns:
            jobs = [(kind, n, alpha, z_alpha, p0, delta, n_boot, 1_500_000 + i)
                    for i in range(reps)]
            res = _map(w_split_regularity, jobs, env.max_workers,
                       experiment="e16_split_reg",
                       params=dict(kind=kind, n=n, alpha=alpha, p0=p0,
                                   delta=delta, n_boot=n_boot, reps=reps),
                       seed_index=7)
            lo = np.array([r[0] for r in res])
            hi = np.array([r[1] for r in res])
            coverage = float(np.mean((lo <= ref) & (hi >= ref)))
            rows.append(dict(kind=kind, n=n, reps=reps, alpha=alpha,
                             reference_value=ref, mean_width=float((hi - lo).mean()),
                             coverage=coverage, nominal=1.0 - delta))
    return rows


# --------------------------------------------------------------------------
# E17: genuinely joint (n, r) RFF sweep, completing thm:rffjoint (n16)
# --------------------------------------------------------------------------

def e17_rffjoint_grid(env, sigma=1.0, d=3, reps=None, n_grid=None, r_grid=None):
    """CLAIMS.md n16 completion. e10_rff_rate already confirmed the
    r-exponent at one fixed n; this checks the claim that, AWAY from the
    null boundary, that O(r^{-1/2}) rate holds with an exponent that does
    NOT depend on n (the sampling and RFF error terms are additive and
    separately controlled, per thm:rffjoint) -- a genuinely joint (n,r)
    statement the single-n sweep cannot see.
    """
    from ..kernels import RFF
    from ..solver import solve_features
    reps = reps or max(8, env.n_reps // 100)
    n_grid = n_grid or [200, 800, 3200]
    r_grid = r_grid or [64, 128, 256, 512, 1024, 2048]
    rng = np.random.default_rng(21)
    rows = []
    for n in n_grid:
        X = rng.standard_normal((n, d))
        Y = rng.standard_normal((n, d)) + 0.8  # well separated: away from the boundary
        Kxx, Kyy, Kxy = (gaussian_kernel(X, X, sigma), gaussian_kernel(Y, Y, sigma),
                        gaussian_kernel(X, Y, sigma))
        exact = solve_gram(Kxx, Kyy, Kxy, 0.0, max_iter=6000, tol=1e-12).discrepancy
        for r in r_grid:
            errs = []
            for rep in range(reps):
                rff = RFF(dim=d, n_features=r, sigma=sigma,
                         rng=np.random.default_rng(2_000_000 + n + r + rep))
                Zx, Zy = rff.transform(X), rff.transform(Y)
                approx = solve_features(Zx, Zy, 0.0, max_iter=6000, tol=1e-12).discrepancy
                errs.append(abs(approx - exact))
            rows.append(dict(n=int(n), r=r, reps=reps, exact_dhat=float(exact),
                             mean_abs_diff=float(np.mean(errs))))
    fits = {}
    for n in n_grid:
        sub = [row for row in rows if row["n"] == n]
        logr = np.log([row["r"] for row in sub])
        loge = np.log([max(row["mean_abs_diff"], 1e-300) for row in sub])
        slope, intercept = np.polyfit(logr, loge, 1)
        fits[str(n)] = dict(exponent=float(slope), const=float(np.exp(intercept)))
    return rows, dict(fits=fits, theory_predicts_exponent_away_from_boundary=-0.5)


# --------------------------------------------------------------------------
# E18: cor:marginderived -- does the empirically-fit margin exponent gamma
# match the Sobolev-smoothness-implied closed form gamma=(s+t)/t? (n14)
# --------------------------------------------------------------------------

def w_sobolev_margin(args):
    """One replicate: D_alpha,k (Matern kernel) at a grid of alpha near the
    boundary, for the Sobolev bump-pair DGP."""
    m_spline, eps, rho, nu, n, grid, seed = args
    from ..dgp import sobolev_bump_pair
    from ..kernels import matern_kernel
    sp = sobolev_bump_pair(m_spline, eps=eps)
    rng = np.random.default_rng(seed)
    X = sp.sample(n, rng, "P")
    Y = sp.sample(n, rng, "Q")

    def solve_one(a, w0, v0):
        Kxx = matern_kernel(X, X, rho, nu)
        Kyy = matern_kernel(Y, Y, rho, nu)
        Kxy = matern_kernel(X, Y, rho, nu)
        return solve_gram(Kxx, Kyy, Kxy, a, w0=w0, v0=v0, rng=rng,
                          max_iter=4000, tol=1e-11)

    res = solve_path(solve_one, np.asarray(grid))
    return [r.discrepancy for r in res]


def e18_marginderived(env, m_splines=None, eps=0.05, rho=2.0, nu=1.5,
                      n=None, reps=None, n_eps_points=6):
    """CLAIMS.md n14. Caveat stated plainly (also in CLAIMS.md): Corollary
    "the margin condition, derived" requires the Sobolev-smoothness
    assumption to hold for the OPTIMAL TRIMMED PAIR at every alpha <=
    TV(P,Q), with a constant not depending on alpha -- not re-verified here,
    only P and Q themselves are built with a verified, known smoothness.
    This is therefore an indicative check of the predicted exponent, not an
    end-to-end verification of the corollary's full hypothesis set -- the
    same honest limitation e3's use of an off-the-shelf DGP family already
    carries for Assumption "Margin" itself.
    """
    from ..dgp import sobolev_bump_pair
    m_splines = m_splines or [2, 3, 4, 5]
    n = n or min(2000, env.max_n_gram)
    reps = reps or max(8, env.n_reps // 100)
    s = nu + 0.5  # Matern spectral exponent in 1D

    rows = []
    fits = []
    for m_spline in m_splines:
        sp = sobolev_bump_pair(m_spline, eps=eps)
        t = sp.t_used
        theta = t / (s + t)
        gamma_predicted = 1.0 / theta
        eps_grid_fracs = np.array([0.02, 0.04, 0.06, 0.09, 0.12, 0.16])[:n_eps_points]
        eps_grid = eps_grid_fracs * sp.alpha_star
        alpha_grid = np.sort(sp.alpha_star - eps_grid)

        jobs = [(m_spline, eps, rho, nu, n, alpha_grid.tolist(), 1_600_000 + i)
                for i in range(reps)]
        paths = np.array(_map(w_sobolev_margin, jobs, env.max_workers,
                              experiment="e18_sobolev_margin",
                              params=dict(m_spline=m_spline, eps=eps, rho=rho,
                                          nu=nu, n=n, reps=reps,
                                          alpha_grid=alpha_grid.tolist()),
                              seed_index=6))
        dbar = paths.mean(axis=0)
        e_sorted = sp.alpha_star - alpha_grid

        # floor at the boundary itself, same device as e3_margin
        floor_jobs = [(m_spline, eps, rho, nu, n, [float(sp.alpha_star)], 1_650_000 + i)
                      for i in range(reps)]
        floor = float(np.mean([r[0] for r in _map(
            w_sobolev_margin, floor_jobs, env.max_workers,
            experiment="e18_floor",
            params=dict(m_spline=m_spline, eps=eps, rho=rho, nu=nu, n=n,
                        reps=reps), seed_index=6)]))

        fit = margin_fits(e_sorted, dbar, floor, f"sobolev_m{m_spline}", n, reps)
        fit.update(m_spline=m_spline, t_used=t, matern_s=s, matern_nu=nu,
                  theta=theta, gamma_predicted=gamma_predicted,
                  alpha_star=sp.alpha_star, eps_dgp=eps)
        fits.append(fit)
        for e_, v_, a_ in zip(e_sorted, dbar, alpha_grid):
            rows.append(dict(m_spline=m_spline, t_used=t, n=n, reps=reps,
                             eps=float(e_), D_alpha=float(v_), alpha=float(a_),
                             floor=floor))
    return rows, fits


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

RUN_ALL_IDS = ("E1", "E1B", "E2", "E3", "E4", "E5", "E8", "E7", "E6")


def run_all(profile=None, quick=False, threshold="paper", certificate="bootstrap",
            outdir=None, only=None, workers=None, e7_reps=None):
    """Run E1-E8 and write their tables.

    Revision v2 keywords (defaults reproduce v1 exactly): ``threshold``
    ("paper" | "plugin"), ``certificate`` ("bootstrap" | "betting" |
    "both"), ``outdir`` (tables, figures and manifest.json go under it, with
    v1's layout; checkpoints stay in :data:`CHECKPOINTS` so that the
    threshold-independent D-hat replicates of v1 are reused),
    ``only`` (comma-separated ids among E1, E1b, E2, E3, E4, E5, E6, E7, E8),
    ``workers`` (worker processes), ``e7_reps`` (default
    :data:`E7_REPS` = 100, or the environment variable
    ``TRIMMEDMMD_E7_REPS``; the published e7_split_bounds.csv has 200).
    """
    env = detect(profile="smoke" if quick else profile)
    if workers is not None:
        env.max_workers = int(workers)
    if e7_reps is None and os.environ.get("TRIMMEDMMD_E7_REPS"):
        # the CLI has no flag for it: TRIMMEDMMD_E7_REPS=200 reproduces the
        # replicate count of the committed e7_split_bounds.csv
        e7_reps = int(os.environ["TRIMMEDMMD_E7_REPS"])
    if outdir is None:
        root, tables, figures = RESULTS, TABLES, FIGURES
    else:
        from .v2 import out_root
        root = out_root(outdir)
        tables, figures = root / "tables", root / "figures"
    if only is None:
        todo = set(RUN_ALL_IDS)
    else:
        items = only.split(",") if isinstance(only, str) else list(only)
        todo = {x.strip().upper() for x in items if x.strip()}
        bad = todo - set(RUN_ALL_IDS)
        if bad:
            raise ValueError(f"unknown experiment ids {sorted(bad)}; choose from {RUN_ALL_IDS}")
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    manifest = dict(envelope=json.loads(env.to_json()),
                    started=time.strftime("%Y-%m-%d %H:%M:%S"))
    if threshold != "paper" or certificate != "bootstrap" or outdir is not None or only is not None:
        manifest["v2"] = dict(threshold=threshold, certificate=certificate,
                              outdir=None if outdir is None else str(outdir),
                              only=sorted(todo), workers=env.max_workers,
                              e7_reps=e7_reps, checkpoints=str(CHECKPOINTS))
    t0 = time.perf_counter()
    n_small = 200 if quick else 300
    v2kw = dict(threshold=threshold) if threshold != "paper" else {}
    ckw = dict(certificate=certificate) if certificate != "bootstrap" else {}

    if "E1" in todo:
        print("E1 null calibration ...", flush=True)
        r = e1_calibration(env, n=n_small, **v2kw)
        pd.DataFrame(r).to_csv(tables / "e1_calibration.csv", index=False)

    if "E1B" in todo:
        print("E1b power vs n ...", flush=True)
        ns = [100, 200, 400] if quick else [100, 200, 400, 800, 1600,
                                            min(3200, env.max_n_gram)]
        r = e1_power_vs_n(env, ns=ns, **v2kw)
        pd.DataFrame(r).to_csv(tables / "e1_power_vs_n.csv", index=False)

    if "E2" in todo:
        print("E2 alpha-hat ...", flush=True)
        rows, detail = e2_alpha_hat(env, n=n_small, **v2kw)
        pd.DataFrame(rows).to_csv(tables / "e2_alpha_hat.csv", index=False)
        (tables / "e2_detail.json").write_text(json.dumps(detail, indent=2))

    if "E3" in todo:
        print("E3 margin exponent ...", flush=True)
        # reps pinned to the published design (paper.tex Appendix A.4 quotes
        # fitted values, bootstrap SEs and local slopes computed over exactly
        # 24 replicates; the function's own default of 8 is a different, much
        # noisier experiment).
        rows, fit = e3_margin(env, reps=4 if quick else 24)
        pd.DataFrame(rows).to_csv(tables / "e3_margin.csv", index=False)
        (tables / "e3_margin_fit.json").write_text(json.dumps(fit, indent=2))

    if "E4" in todo:
        print("E4 assumption violation ...", flush=True)
        rows, detail = e4_assumption_violation(env, n=200 if quick else 400)
        pd.DataFrame(rows).to_csv(tables / "e4_assumption.csv", index=False)
        (tables / "e4_detail.json").write_text(json.dumps(detail, indent=2))

    if "E5" in todo:
        print("E5 scaling ...", flush=True)
        rows = e5_scaling(env)
        pd.DataFrame(rows).to_csv(tables / "e5_scaling.csv", index=False)

    if "E8" in todo:
        print("E8 sample spaces / dimension ...", flush=True)
        rows = e8_sample_spaces(env, reps=6 if quick else 50, **v2kw, **ckw)
        pd.DataFrame(rows).to_csv(tables / "e8_sample_spaces.csv", index=False)

    if "E7" in todo:
        print("E7 split-sample bounds ...", flush=True)
        # reps pinned to the published design (paper.tex: "100 replicates at
        # each size"); both functions' own default is min(N, env.n_reps),
        # which silently drifts to 200 on any tier with n_reps >= 200 and
        # quietly changes the experiment rather than reproducing it.
        # (The committed results/tables/e7_split_bounds.csv has reps = 200;
        # pass e7_reps=200 to match it.)
        rows = e7_split_bounds(env, ns=[200] if quick else [200, 400, 800],
                               reps=20 if quick else (e7_reps or E7_REPS),
                               **v2kw, **ckw)
        pd.DataFrame(rows).to_csv(tables / "e7_split_bounds.csv", index=False)

    if "E6" in todo:
        print("E6 alternative calibrations ...", flush=True)
        rows = e6_alternative_calibrations(
            env, ns=[200] if quick else [200, 400], reps=20 if quick else 100,
            **v2kw, **ckw)
        pd.DataFrame(rows).to_csv(tables / "e6_calibrations.csv", index=False)

    manifest["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
    manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"done in {manifest['elapsed_seconds']}s -> {root}")
    return env

"""E19: kernel selection and the analytic test (E1-E18 already exist in run.py).

Remark ``rem:fixedkernel`` of the paper says the threshold is valid for a
kernel fixed before the data are seen, that every reported number uses the
pooled median heuristic instead, and that two remedies restore validity. This
experiment measures what that costs and what it buys, on scenarios with
exactly known TV.

Per replicate, the whole trimmed-MMD path is computed for

  * K fixed bandwidths sigma_k = s_k * sigma_pop, where sigma_pop is the median
    heuristic of one large independent pooled sample (so the sigma_k are fixed
    in advance, not functions of the replicate), on the full sample, on a pilot
    part (30%) and on the remaining test part (70%);
  * the pooled median heuristic of the replicate itself (the practice the
    paper's remark flags), on the full sample.

Policies are then read off the stored paths, offline:

  median        sigma = pooled median of the sample, level delta
  fixed s       sigma_k, level delta (each is valid on its own)
  best fixed    max over k of the mean certified level: an oracle ceiling that
                no one can implement, the envelope the policies chase
  union         every sigma_k at level delta/K; the largest certified level
  tune-then-test  sigma_k* chosen by the largest certified level on the pilot;
                the certified level is then read on the independent test part

Certified level L: the largest grid level at which H_0: TV <= alpha is rejected
(so TV > L is certified), or -1 if not even alpha = 0 is rejected.
Violation: L >= alpha*, a rejection of a true null. Under P = Q this is the
type-I error at alpha = 0.

Revision v2: ``--threshold plugin`` runs every policy with the plug-in
threshold tau_new (exact Gaussian Gram, so the coupled Gaussian constants;
the union policy uses tau_new at delta/K). Paths then stop at tau_new, which
is below v1's tau, so they are recomputed under a separate checkpoint name
("e11_kernel_selection_plugin") and extend at least as far as v1's; the
v1-threshold levels are read off the same paths into ``*_paper`` columns.
"""

from __future__ import annotations

import numpy as np

from .. import dgp
from ..kernels import gaussian_kernel, median_heuristic
from ..solver import solve_gram, solve_path
from ..threshold import tau
from .run import _build, _map

SCALES = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
GRID = tuple(np.round(np.arange(0.0, 0.801, 0.05), 3))
PILOT_FRAC = 0.3


def sigma_pop(spec, seed=777, n=3000):
    """Median heuristic of a large independent pooled sample: a fixed number,
    computed once per scenario before any replicate is drawn."""
    sc = _build(spec)
    rng = np.random.default_rng(seed)
    return float(median_heuristic(sc.sample(n, rng, "P"),
                                  sc.sample(n, rng, "Q"), rng=rng))


def _thr(threshold, n, m, a, delta):
    if threshold == "paper":
        return float(tau(n, m, a, delta))
    from .v2 import threshold_for
    return float(threshold_for(threshold, n, m, delta, backend="gram")(a))


def _path(X, Y, sigma, grid, rng, delta=0.05, threshold="paper"):
    """Lower Frank-Wolfe bound on D along the grid, stopping at the first level
    where the test accepts at the *smallest* threshold any policy uses
    (level delta). Rejections are downward closed, so every later level is
    accepted by every policy and is left at zero without being solved.
    Under ``threshold="plugin"`` the smallest threshold is tau_new at level
    delta (tau_new <= tau, so the path also serves v1's policies)."""
    Kxx = gaussian_kernel(X, X, sigma)
    Kyy = gaussian_kernel(Y, Y, sigma)
    Kxy = gaussian_kernel(X, Y, sigma)
    n, m = len(X), len(Y)
    out = np.zeros(len(grid))
    w0 = v0 = None
    for i, a in enumerate(grid):
        r = solve_gram(Kxx, Kyy, Kxy, a, w0=w0, v0=v0, rng=rng,
                       max_iter=1500, tol=1e-9)
        w0, v0 = r.w, r.v
        out[i] = r.discrepancy_lower
        if out[i] <= (tau(n, m, a, delta) if threshold == "paper"
                      else min(_thr(threshold, n, m, a, delta), float(tau(n, m, a, delta)))):
            break
    return out


def w_kernel_selection(args):
    spec, n, scales, grid, base_sigma, seed = args[:6]
    threshold = args[6] if len(args) > 6 else "paper"
    rng = np.random.default_rng(seed)
    sc = _build(spec)
    X, Y = sc.sample(n, rng, "P"), sc.sample(n, rng, "Q")
    ip = rng.permutation(n)
    npil = int(PILOT_FRAC * n)
    pil, tst = ip[:npil], ip[npil:]
    out = {"full": [], "pilot": [], "test": []}
    for s in scales:
        sg = s * base_sigma
        out["full"].append(_path(X, Y, sg, grid, rng, threshold=threshold))
        out["pilot"].append(_path(X[pil], Y[pil], sg, grid, rng, threshold=threshold))
        out["test"].append(_path(X[tst], Y[tst], sg, grid, rng, threshold=threshold))
    med = float(median_heuristic(X, Y, rng=rng))
    out["median"] = _path(X, Y, med, grid, rng, threshold=threshold)
    out = {k: np.asarray(v) for k, v in out.items()}
    out["median_sigma"] = med
    return out


def certified_level(grid, d, n, m, delta, threshold="paper"):
    """Largest rejected grid level, -1 if none (rejections are downward
    closed because D decreases and tau increases in alpha)."""
    best = -1.0
    for a, dv in zip(grid, d):
        if dv > (tau(n, m, a, delta) if threshold == "paper"
                 else _thr(threshold, n, m, a, delta)):
            best = float(a)
        else:
            break
    return best


def summarize(reps, spec, n, scales, grid, delta=0.05, threshold="paper"):
    rows = _summarize(reps, spec, n, scales, grid, delta, threshold)
    if threshold == "paper":
        return rows
    # revision v2: the v1 threshold's policies on the same (longer) paths
    ref = _summarize(reps, spec, n, scales, grid, delta, "paper")
    for r, r1 in zip(rows, ref):
        assert r["policy"].split(" (")[0] == r1["policy"].split(" (")[0] or \
            r["policy"].startswith("best fixed")
        r["threshold"] = threshold
        for k in ("mean_level", "frac_certified", "ratio_to_truth", "violation"):
            r[f"{k}_paper"] = r1[k]
    return rows


def _summarize(reps, spec, n, scales, grid, delta=0.05, threshold="paper"):
    cl = lambda g, d, n_, m_, dl: certified_level(g, d, n_, m_, dl, threshold)  # noqa: E731
    sc = _build(spec)
    astar = sc.alpha_star
    K = len(scales)
    npil = int(PILOT_FRAC * n)
    ntest = n - npil
    L = {"median": [], "union": [], "tune_then_test": []}
    Lk = [[] for _ in range(K)]
    chosen = []
    for r in reps:
        L["median"].append(cl(grid, r["median"], n, n, delta))
        ks = [cl(grid, r["full"][k], n, n, delta) for k in range(K)]
        for k in range(K):
            Lk[k].append(ks[k])
        L["union"].append(max(cl(grid, r["full"][k], n, n,
                                 delta / K) for k in range(K)))
        piv = [cl(grid, r["pilot"][k], npil, npil, delta)
               for k in range(K)]
        # ties broken toward the larger bandwidth multiple nearest 1
        kstar = max(range(K), key=lambda k: (piv[k], -abs(np.log(scales[k]))))
        chosen.append(scales[kstar])
        L["tune_then_test"].append(
            cl(grid, r["test"][kstar], ntest, ntest, delta))
    rows = []

    def row(policy, x):
        x = np.asarray(x, float)
        return dict(scenario=sc.name, dim=sc.dim, n=n, alpha_star=astar,
                    policy=policy, reps=len(x), mean_level=float(x.mean()),
                    mean_level_if_certified=float(x[x >= 0].mean())
                    if (x >= 0).any() else float("nan"),
                    frac_certified=float(np.mean(x >= 0)),
                    ratio_to_truth=float(np.mean(np.maximum(x, 0)) / astar)
                    if astar > 0 else float("nan"),
                    violation=float(np.mean(x >= astar - 1e-12)))
    rows.append(row("median heuristic (pooled)", L["median"]))
    for k, s in enumerate(scales):
        rows.append(row(f"fixed {s:g} x sigma_pop", Lk[k]))
    best = max(range(K), key=lambda k: np.mean(np.maximum(Lk[k], 0)))
    rows.append(row(f"best fixed (oracle, {scales[best]:g}x)", Lk[best]))
    rows.append(row(f"union bound over {K} bandwidths", L["union"]))
    rows.append(row("tune on pilot, test on rest", L["tune_then_test"]))
    rows[-1]["chosen_scales"] = {str(s): int(chosen.count(s)) for s in scales}
    return rows


def e19_kernel_selection(env, specs=None, n=1000, reps=None, delta=0.05,
                         scales=SCALES, grid=GRID, threshold="paper"):
    # Contamination is large on purpose: at n = 1000 the threshold at
    # alpha = 0 is 0.44, and a kernel discrepancy cannot exceed 2*alpha*kappa,
    # so smaller alpha* cannot be certified at all at this n.
    specs = specs or [("disjoint", 2, 0.5), ("gaussian", 2, 0.7, 1.5),
                      ("disjoint", 10, 0.5), ("gaussian", 10, 0.7, 1.5),
                      ("identical", 2)]
    reps = reps or max(30, env.n_reps // 2)
    rows = []
    for spec in specs:
        base = sigma_pop(spec)
        if threshold == "paper":
            jobs = [(spec, n, tuple(scales), tuple(grid), base, 50_000 + i)
                    for i in range(reps)]
            name = "e11_kernel_selection"
        else:
            jobs = [(spec, n, tuple(scales), tuple(grid), base, 50_000 + i, threshold)
                    for i in range(reps)]
            name = f"e11_kernel_selection_{threshold}"
        res = _map(w_kernel_selection, jobs, env.max_workers,
                   experiment=name,
                   params=dict(spec=list(spec), n=n, reps=reps,
                               scales=list(scales), grid=list(grid),
                               base=base, pilot=PILOT_FRAC),
                   seed_index=5)
        rows += summarize(res, spec, n, scales, grid, delta, threshold)
    return rows


if __name__ == "__main__":
    import argparse
    import json
    import pandas as pd
    from ..resources import detect
    from .run import TABLES

    from .v2 import add_v2_args, sub

    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=None,
                    help="replicates per scenario (the published run used 24)")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=None)
    add_v2_args(ap, certificate=False)
    a = ap.parse_args()
    env = detect()
    if a.workers:
        env.max_workers = a.workers
    kw = dict(threshold=a.threshold) if a.threshold != "paper" else {}
    rows = e19_kernel_selection(env, n=a.n, reps=a.reps, **kw)
    tables = TABLES if a.outdir is None else sub(a.outdir, "tables")
    tables.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(tables / "e19_kernel_selection.csv", index=False)
    pd.set_option("display.width", 220)
    print(df.drop(columns=["dim", "reps"], errors="ignore").round(3)
            .to_string(index=False))

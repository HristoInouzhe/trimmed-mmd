"""Tuning the kernel bandwidth for the analytic estimator, validly.

The threshold of the paper is valid for a kernel fixed before the data are
seen, so choosing the bandwidth by looking at the same data and then testing
on them is selection that the threshold does not pay for. Two honest routes:

  A. tune-then-test   choose the bandwidth on a tuning split (30% of each
                      group) by the largest alpha-hat there, then compute
                      alpha-hat on the remaining 70% with that bandwidth
                      fixed. No correction is needed.
  B. union bound      compute alpha-hat for every bandwidth of a fixed grid on
                      ALL the data at level delta/K and keep the largest: the
                      certificate TV > max_k last-rejected_k holds at level
                      delta. The price is log(2/(delta/K)) in the threshold,
                      small next to the 4/sqrt(n) term.

(The split-sample certificate needs neither: a kernel chosen on fold one gives
a valid fold-two bound, which is a large part of why it is sharper.)

Each alpha-hat is found by bisection on D_alpha - tau_alpha, which is
decreasing in alpha, with the Frank-Wolfe lower bound as the statistic.
Bandwidths run in parallel, each process holding its own threads to a few
cores.

    python scripts/fair_bandwidth_tune.py adult --workers 6

Revision v2 (defaults reproduce v1): ``--threshold plugin`` bisects against the
plug-in threshold tau_new with bounded-kernel constants (cos/sin features:
k(x,x) = 1, k possibly negative; see fair_case_study.threshold_fn); the union
route still runs each bandwidth at delta/K. ``--outdir D`` writes to D/fair.

    FAIR_THREADS=3 python scripts/fair_bandwidth_tune.py adult --workers 7 \
        --threshold plugin --outdir results/v2
"""

from __future__ import annotations

import os
import sys

_T = os.environ.get("FAIR_THREADS", "4")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = _T

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import fair_case_study as fc                                   # noqa: E402
from trimmedmmd.kernels import median_heuristic                # noqa: E402
from trimmedmmd.solver import solve_features                   # noqa: E402
from trimmedmmd.threshold import tau                           # noqa: E402

OUT = ROOT / "results" / "fair"
SCALES = (0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0)


def alpha_hat_bisect(Sa, Sb, sigma, delta, n_freq, seed, tol=0.0125,
                     max_iter=2500, hi=0.7, threshold="paper"):
    """(last_rejected, first_accepted) by bisection, kappa = 1 features."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((Sa.shape[1], n_freq)) / sigma
    Za, Zb = fc.cossin_features(Sa, W), fc.cossin_features(Sb, W)
    n, m = len(Za), len(Zb)
    thr = ((lambda a: tau(n, m, a, delta)) if threshold == "paper"
           else fc.threshold_fn(threshold, n, m, delta))

    def rejects(a):
        res = solve_features(Za, Zb, a, tol=1e-10, max_iter=max_iter,
                             rng=np.random.default_rng(0))
        return bool(res.discrepancy_lower > thr(a)), res

    rej0, res0 = rejects(0.0)
    if not rej0:
        return 0.0, 0.0, res0.discrepancy
    lo, hi_ = 0.0, hi
    while hi_ - lo > tol:
        mid = 0.5 * (lo + hi_)
        r, _ = rejects(mid)
        if r:
            lo = mid
        else:
            hi_ = mid
    return lo, hi_, res0.discrepancy


def _job(args):
    tag, Sa, Sb, sigma, delta, n_freq, seed = args[:7]
    threshold = args[7] if len(args) > 7 else "paper"
    t = time.time()
    lo, hi, d0 = alpha_hat_bisect(Sa, Sb, sigma, delta, n_freq, seed,
                                  threshold=threshold)
    return tag, lo, hi, d0, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["adult", "acs"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n-freq", type=int, default=512)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--tune-frac", type=float, default=0.3)
    ap.add_argument("--scales", type=float, nargs="*", default=list(SCALES))
    from trimmedmmd.experiments.v2 import add_v2_args
    add_v2_args(ap, certificate=False)
    a = ap.parse_args()
    if a.threshold != "paper" and a.outdir is None:
        raise SystemExit("v2 runs need --outdir (e.g. results/v2); refusing to overwrite "
                         "the published results/fair")
    outp = fc.out_dir(a.outdir)
    thx = () if a.threshold == "paper" else (a.threshold,)

    t0 = time.time()
    F, g, y, meta = fc.load(a.dataset, a.max_rows, 0)
    Fv = F.values.astype(float)
    sd = Fv.std(0)
    sd[sd == 0] = 1.0
    S = (Fv - Fv.mean(0)) / sd
    rng = np.random.default_rng(2026)
    tune = np.zeros(len(g), bool)
    for k in (0, 1):
        ix = np.nonzero(g == k)[0]
        tune[rng.choice(ix, int(a.tune_frac * len(ix)), replace=False)] = True
    sig_med = float(median_heuristic(S[tune & (g == 0)], S[tune & (g == 1)],
                                     rng=rng))
    K = len(a.scales)
    print(f"{a.dataset}: sigma_med(tune)={sig_med:.3f}; scales={a.scales}; "
          f"{a.workers} workers x {_T} threads", flush=True)

    def sets(mask):
        return S[mask & (g == 0)], S[mask & (g == 1)]

    jobs = []
    for s in a.scales:
        Ta, Tb = sets(tune)
        jobs.append((("tune", s), Ta, Tb, s * sig_med, 0.05, a.n_freq, 1) + thx)
        Fa, Fb = sets(np.ones(len(g), bool))
        jobs.append((("union", s), Fa, Fb, s * sig_med, 0.05 / K, a.n_freq, 2) + thx)
    res = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for tag, lo, hi, d0, dt in ex.map(_job, jobs):
            res[tag] = (lo, hi, d0)
            print(f"  {tag[0]:5s} scale={tag[1]:.2f}  certified TV > {lo:.4f} "
                  f"(alpha-hat {hi:.4f})  D0={d0:.4f}  ({dt:.0f}s)", flush=True)

    best = max(a.scales, key=lambda s: (res[("tune", s)][0], -abs(s - 0.5)))
    # A: test on the held-out 70% with the selected bandwidth, level delta
    Ea, Eb = sets(~tune)
    _, lo, hi, d0, dt = _job((("test", best), Ea, Eb, best * sig_med, 0.05,
                              a.n_freq, 3) + thx)
    union_best = max(a.scales, key=lambda s: res[("union", s)][0])
    out = {"dataset": a.dataset, "sigma_median_tune": sig_med,
           "n_freq": a.n_freq, "scales": a.scales,
           "tune": {str(s): res[("tune", s)] for s in a.scales},
           "union_delta_over_K": {str(s): res[("union", s)] for s in a.scales},
           "selected_scale": best,
           "tune_then_test": {"scale": best, "last_rejected": lo,
                              "alpha_hat": hi, "n_test": [len(Ea), len(Eb)]},
           "union_best": {"scale": union_best,
                          "last_rejected": res[("union", union_best)][0]},
           "seconds": time.time() - t0}
    out.update(fc.threshold_meta(a.threshold))       # nothing for the v1 threshold
    (outp / f"{a.dataset}_bandwidth_tune.json").write_text(
        json.dumps(out, indent=1))
    print(f"\nselected scale (tune split): {best}")
    print(f"tune-then-test on held-out {len(Ea)}+{len(Eb)}: certified TV > "
          f"{lo:.4f}  (alpha-hat {hi:.4f})")
    print(f"union bound over {K} scales, all data: best scale {union_best}, "
          f"certified TV > {res[('union', union_best)][0]:.4f}")


if __name__ == "__main__":
    main()

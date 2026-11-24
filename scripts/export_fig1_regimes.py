"""Which route is usable at which sample size, measured on real data.

The three deliverables of this paper do not become available at the same
time. On subsamples of the RHC arms of increasing size we record, for each:

* what the **analytic test** returns -- the threshold-crossing estimator
  alpha-hat, which is 0 until the closed-form threshold drops below the
  statistic;
* what the **split-sample certificate** returns -- a finite-sample lower
  bound on TV(P,Q), which needs far less data because it never takes a
  supremum over the RKHS unit ball;
* how stable the **recovered trimming** is -- fit the trimming on two
  disjoint halves of the subsample, push each witness through to the *same*
  reference patients, and take the Jaccard similarity of the two discarded
  sets. This is the empirical counterpart of the consistency theorem, and it
  is expected to converge slowly: the rate there is (n^m)^{-1/4}, degrading
  further as alpha approaches the contact level.

The point of the panel is honesty about regimes, not a horse race. Each
route is the right tool somewhere.

    PYTHONPATH=src python scripts/export_fig1_regimes.py

Revision v2 (defaults reproduce v1 exactly; outputs go to <outdir>/figures):

    PYTHONPATH=src python scripts/export_fig1_regimes.py --threshold plugin \
        --certificate all --alpha-read 0.30 --outdir results/v2

``--threshold plugin``: alpha-hat with tau_new (exact Gram, Gaussian constants);
``--certificate sequential|all``: ``certified`` is the sequential betting
certificate (the paper's main split result, batches (0, 1/4, 1/2, 1), seed
500 + r, bandwidth = median on batch 0); ``all`` also keeps the single-split
betting certificate in ``certified_betting`` and the bootstrap in
``certified_bootstrap`` (both on v1's split and fixed full-RHC bandwidth, so
the bootstrap equals v1's ``certified``). ``--certificate betting|both``: the
single-split betting certificate reported (+ bootstrap);
``--alpha-read``: the level of the two half-trimmings (v1: 0.25, the
certified RHC level; v2: the new certified level). In v2 mode each row also
stores its D-hat path, so alpha-hat can be recomputed for another threshold
without re-solving.

AoS (2026-11-20; defaults unchanged): ``--std batch0`` gives the sequential certificate the raw covariates of each
subsample and standardizes them by the mean and standard deviation of its batch 0, both arms pooled, after the
certificate's random order and batch cut (``splitting.sequential_order(standardize_batch0=True)``), so the fitting
rule depends on batch 0 only; the default ``full`` uses the full-study moments of ``load_rhc``. The subsamples, seeds
and everything else are unchanged. ``--skip-analytic`` / ``--skip-stability`` leave out alpha-hat and the Jaccard
part (their keys are then absent from the rows); ``--out`` names the output file.

    PYTHONPATH=src python scripts/export_fig1_regimes.py --threshold plugin --certificate sequential \
        --std batch0 --skip-analytic --skip-stability --outdir results/v3 --out fig1_regimes_std0.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trimmedmmd.discrepancy import trimmed_mmd_path                # noqa: E402
from trimmedmmd.estimate import alpha_hat_from_path                # noqa: E402
from trimmedmmd.kernels import median_heuristic                    # noqa: E402
from trimmedmmd.realdata import load_rhc                           # noqa: E402
from trimmedmmd.experiments import v2 as V2                        # noqa: E402
from trimmedmmd.splitting import lower_bound_betting, lower_bound_tv  # noqa: E402
from trimmedmmd.trimmings import optimal_trimming_mmd, witness_at  # noqa: E402

OUT = ROOT / "results" / "figures"
GRID = np.round(np.arange(0.0, 0.751, 0.05), 3)
DELTA = 0.05
SIZES = [100, 200, 400, 800, 1600]
REPS = 5
ALPHA_READ = 0.25
N_REF = 600          # reference patients the two half-witnesses both score


def jaccard(a, b):
    a, b = set(map(int, a)), set(map(int, b))
    return len(a & b) / max(1, len(a | b))


def main(argv=None):
    ap = argparse.ArgumentParser()
    V2.add_v2_args(ap)
    ap.add_argument("--alpha-read", type=float, default=ALPHA_READ,
                    help="level of the half-trimmings (v1: 0.25)")
    ap.add_argument("--sizes", type=int, nargs="*", default=None,
                    help="subsample sizes (default: the published 100..1600; smoke runs only)")
    ap.add_argument("--reps", type=int, default=None,
                    help="replicates per size (default: the published 5; smoke runs only)")
    ap.add_argument("--std", choices=("full", "batch0"), default="full",
                    help="standardization seen by the sequential certificate: full-study moments "
                         "(default) or the moments of its batch 0 (raw covariates in)")
    ap.add_argument("--skip-analytic", action="store_true", help="do not compute alpha-hat")
    ap.add_argument("--skip-stability", action="store_true", help="do not compute the Jaccard part")
    ap.add_argument("--out", default=None, help="output file name (default fig1_regimes.json)")
    args = ap.parse_args(argv)
    v2_mode = (args.threshold != "paper" or args.certificate != "bootstrap"
               or args.outdir is not None or args.alpha_read != ALPHA_READ
               or args.std != "full" or args.skip_analytic or args.skip_stability)
    if args.std != "full" and args.certificate == "bootstrap":
        raise SystemExit("--std batch0 applies to the sequential certificate (--certificate sequential|all)")
    out_name = args.out or "fig1_regimes.json"
    alpha_read = args.alpha_read
    sizes = SIZES if args.sizes is None else list(args.sizes)
    reps = REPS if args.reps is None else args.reps
    out_dir = V2.sub(args.outdir, "figures") if args.outdir is not None else OUT

    t0 = time.time()
    X, Y, names, _ = load_rhc()
    n, m = len(X), len(Y)
    if args.std == "batch0":
        Xr, Yr, _, _ = load_rhc(standardize=False)    # same rows, raw covariates
        assert Xr.shape == X.shape and Yr.shape == Y.shape
    rng = np.random.default_rng(11)
    sigma = float(median_heuristic(X, Y, rng=np.random.default_rng(0)))
    ref = np.sort(rng.choice(n, N_REF, replace=False))
    Xref = X[ref]
    k_ref = int(np.floor(N_REF * (1 - alpha_read)))   # kept on the reference

    rows = []
    for ns in sizes:
        for r in range(reps):
            ix = rng.choice(n, ns, replace=False)
            iy = rng.choice(m, ns, replace=False)
            Xs, Ys = X[ix], Y[iy]

            extra = {}
            if not args.skip_analytic:
                _, dh, _ = trimmed_mmd_path(Xs, Ys, GRID, sigma=sigma,
                                            rng=np.random.default_rng(r))
                a_hat = float(alpha_hat_from_path(GRID, dh, ns, ns, DELTA,
                                                  threshold=args.threshold))
                extra["alpha_hat"] = a_hat
            if args.certificate == "bootstrap":
                lo, _ = lower_bound_tv(Xs, Ys, GRID, delta=DELTA, sigma=sigma,
                                       rng=np.random.default_rng(500 + r))
            else:
                methods = V2.certificate_methods(args.certificate)
                single = tuple(mth for mth in methods if mth != "sequential")
                bounds = {}
                if "sequential" in methods:
                    # the paper's main certificate: batches (0, 1/4, 1/2, 1); its
                    # bandwidth is the median on batch 0 (sigma=None), so that
                    # nothing bet on enters the witness -- not the full-RHC sigma
                    if args.std == "batch0":
                        # raw covariates; sequential_order standardizes by batch 0 (both arms)
                        b_seq, _ = lower_bound_betting(
                            Xr[ix], Yr[iy], GRID, delta=DELTA, sigma=None,
                            rng=np.random.default_rng(500 + r), methods=("sequential",),
                            standardize_batch0=True)
                    else:
                        b_seq, _ = lower_bound_betting(
                            Xs, Ys, GRID, delta=DELTA, sigma=None,
                            rng=np.random.default_rng(500 + r), methods=("sequential",))
                    bounds.update(b_seq)
                if single:
                    # single split on v1's stream (fixed sigma, seed 500 + r), so the
                    # bootstrap column reproduces v1's ``certified``
                    b_one, _ = lower_bound_betting(
                        Xs, Ys, GRID, delta=DELTA, sigma=sigma,
                        rng=np.random.default_rng(500 + r), methods=single,
                        n_boot=1000 if "bootstrap" in single else 0)
                    bounds.update(b_one)
                lo = bounds[methods[0]]
                if "bootstrap" in bounds:
                    extra["certified_bootstrap"] = float(bounds["bootstrap"])
                if methods[0] != "betting" and "betting" in bounds:
                    extra["certified_betting"] = float(bounds["betting"])
            if v2_mode and not args.skip_analytic:
                extra["d_hat"] = [float(v) for v in dh]
                extra["alpha_hat_paper"] = float(alpha_hat_from_path(GRID, dh, ns, ns, DELTA))
            if args.skip_stability:
                rows.append({"n": ns, "rep": r, "certified": float(lo), **extra})
                print(f"  n={ns:5d} rep={r}: alpha_hat={extra.get('alpha_hat', float('nan')):.2f} "
                      f"certified={lo:.2f} ({time.time() - t0:.0f}s)", flush=True)
                continue

            # two disjoint halves -> two witnesses -> two discarded sets on
            # the SAME reference patients, so the sets are comparable.
            h = ns // 2
            drops = []
            for sl in (slice(0, h), slice(h, 2 * h)):
                t = optimal_trimming_mmd(Xs[sl], Ys[sl], alpha_read,
                                         sigma=sigma, select="none",
                                         rng=np.random.default_rng(0))
                w = witness_at(t, Xs[sl], Ys[sl], Xref, sigma=sigma)
                drops.append(np.argsort(w)[k_ref:])     # largest = discarded
            jac = jaccard(drops[0], drops[1])

            if args.skip_analytic:
                rows.append({"n": ns, "rep": r, "certified": float(lo), "jaccard": jac, **extra})
            else:
                rows.append({"n": ns, "rep": r, "alpha_hat": extra.pop("alpha_hat"),
                             "certified": float(lo), "jaccard": jac, **extra})
            a_hat = rows[-1].get("alpha_hat", float("nan"))
            print(f"  n={ns:5d} rep={r}: alpha_hat={a_hat:.2f} "
                  f"certified={lo:.2f} jaccard={jac:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    meta = {"sizes": sizes, "reps": reps, "alpha_read": alpha_read,
            "delta": DELTA, "n_ref": N_REF, "rows": rows,
            "seconds": time.time() - t0}
    if v2_mode:
        meta.update({"threshold": args.threshold, "certificate": args.certificate,
                     "certificate_reported": (V2.main_certificate(args.certificate)
                                              if args.certificate != "bootstrap"
                                              else "bootstrap")})
    if args.std != "full" or args.skip_analytic or args.skip_stability:
        meta.update({"std": args.std, "skip_analytic": args.skip_analytic,
                     "skip_stability": args.skip_stability})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / out_name).write_text(json.dumps(meta, indent=2))
    print(f"wrote {out_dir / out_name} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

"""Figure 2: the same pass, at a sample size where the analytic route works.

Figure 1 is honest about the RHC study being in the wrong regime for the
analytic test: it would need roughly 15,600 patients per arm. That is a
statement about the study, not about the method, and the way to show the
difference is to keep the covariate structure and vary what we control.

The design keeps RHC's geometry and plants a contamination whose level is
known exactly:

    core    ~ smoothed resample of the pooled RHC covariates (33 dims)
    P = (1 - a) core + a * Uniform(box at -sep)
    Q = (1 - a) core + a * Uniform(box at +sep)

with ``sep`` in pooled standard deviations. At ``sep = 3`` the contaminating
boxes sit inside the range the real covariates reach, so the contamination is
not trivially visible; the identity below does not care.

The two contaminating laws are mutually singular, so TV(P, Q) = a exactly at
any separation -- the ground-truth identity already used in
``trimmedmmd.dgp``. The core carries the real correlation structure, so this
is RHC-shaped synthetic data rather than a Gaussian toy.

    PYTHONPATH=src python scripts/export_fig2_data.py --pilot
    PYTHONPATH=src python scripts/export_fig2_data.py
    PYTHONPATH=src python scripts/export_fig2_data.py --levels 0 --sizes 500 1000 2000 4000 8000 \
        --out fig2_null.json                      # the null cells (alpha* = 0) of panel 3

Revision v2 (defaults reproduce v1 exactly):

    --threshold plugin       alpha-hat, the firing rule and the per-level decisions use tau_new
                             (Gaussian constants on the exact Gram, n < 2000; bounded constants with the
                             realized kappa of the phased random features, n >= 2000);
    --certificate all        the paper's certificate (author decision Q7): SEQUENTIAL betting
                             (Theorem thm:betting, batches e = (0, 1/4, 1/2, 1), per-level witnesses refit
                             on the earlier batches, batch-0 median bandwidth, random features from
                             n >= 2000 as v1) in ``certified`` / ``cert_reject``; the single-split betting
                             certificate and the v1 bootstrap (v1's split seed and bandwidth, so the
                             bootstrap reproduces v1) in ``certified_betting`` / ``certified_bootstrap``
                             (and ``cert_reject_*``). ``sequential``: the sequential column only;
                             ``betting``: single-split betting reported; ``both``: betting + bootstrap;
    --outdir results/v2      write <outdir>/figures/<name> instead of results/figures/<name>;
    --from-rows FILE         do not re-solve the paths: the D-hat paths stored in FILE (a v1
                             fig2_data.json / fig2_null.json) are threshold-free, so alpha-hat, fires,
                             reject and covers are recomputed from them; the data, the bandwidth and the
                             certificate's split seed are replayed from the row's seed (the path solver's
                             draws are replayed without solving), so the certificate is recomputed on the
                             same split as v1, and with ``both`` the bootstrap column must equal v1's
                             ``certified`` (checked, reported in the output's ``replay_check``).

    PYTHONPATH=src python scripts/export_fig2_data.py --from-rows results/figures/fig2_data.json \
        --threshold plugin --certificate all --outdir results/v2 --workers 6

AoS (2026-11-20; default unchanged): ``--rff-map cossin`` replaces, from n >= 2000, the phased map
sqrt(2/r) cos(W'x + b) by the cos/sin map sqrt(1/r')[cos(W'x), sin(W'x)] with r' = 256 frequencies (512 features,
the same dimension), whose diagonal is exactly 1: the path, every certificate column and the recall trimming use it,
and the plug-in threshold uses the bounded constants with kappa = 1 exactly (``kappa_rff`` = 1; the realized value,
1 up to rounding, is kept in ``kappa_realized``). It needs a full re-solve (no ``--from-rows``) and ``--outdir``;
the default output name gets the suffix ``_cossin``. The data, the bandwidth and the seeds are those of the phased
run; the random stream differs after the bandwidth (the paired map draws no phases).

    PYTHONPATH=src python scripts/export_fig2_data.py --rff-map cossin --sizes 2000 4000 8000 16000 \
        --threshold plugin --certificate all --outdir results/v3 --workers 6
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trimmedmmd.discrepancy import plain_mmd, trimmed_mmd_path      # noqa: E402
from trimmedmmd.estimate import alpha_hat_from_path                 # noqa: E402
from trimmedmmd.experiments import v2 as V2                         # noqa: E402
from trimmedmmd.kernels import RFF, RFF_MAPS, make_rff, median_heuristic, realized_kappa  # noqa: E402
from trimmedmmd.realdata import load_rhc                            # noqa: E402
from trimmedmmd.splitting import lower_bound_betting, lower_bound_tv  # noqa: E402
from trimmedmmd.threshold import tau                                # noqa: E402
from trimmedmmd.trimmings import optimal_trimming_mmd, witness_at   # noqa: E402

OUT = ROOT / "results" / "figures"
DELTA = 0.05
# Above this, everything -- the path, the certificate and the trimming behind
# the recall -- runs on random features, at O((n+m)r) instead of O((n+m)^2)
# in both time and memory. Below it the exact kernel is cheaper, because the
# feature count would exceed the sample size.
RFF_FROM_N = 2000
N_FEATURES = 512


def rhc_scenario(a, sep=3.0, width=1.0, jitter=0.25, n_contam_dims=4,
                 seed=0):
    """RHC-shaped core with a planted, mutually singular contamination.

    ``n_contam_dims`` keeps the contaminating boxes inside a few coordinates
    so the median bandwidth is not blown up by a handful of far-away points,
    which would flatten the kernel on the core and make the problem easier
    than it is.
    """
    X, Y, names, _ = load_rhc()
    Z = np.vstack([X, Y])
    d = Z.shape[1]
    rng0 = np.random.default_rng(seed)

    def core(k, rng):
        idx = rng.integers(0, Z.shape[0], size=k)
        return Z[idx] + jitter * rng.standard_normal((k, d))

    def box(k, rng, sign):
        out = core(k, rng)
        out[:, :n_contam_dims] = sign * (
            sep + rng.uniform(0.0, width, size=(k, n_contam_dims)))
        return out

    def sample(n, rng, which):
        k = rng.binomial(n, a) if a > 0 else 0
        base = core(n - k, rng)
        if k == 0:
            return base
        contam = box(k, rng, -1.0 if which == "P" else +1.0)
        out = np.vstack([base, contam])
        rng.shuffle(out)
        return out

    return sample, d, names


def _rff_kappa(X, Y, sigma, rng_state, n_features, rff_map="phased"):
    """Realized kappa of the random features the path used (phased by default).

    The path draws its RFF from ``rng`` right after the bandwidth; replaying
    that state rebuilds the same features without touching ``rng``.
    """
    g = np.random.default_rng()
    g.bit_generator.state = copy.deepcopy(rng_state)
    rff = make_rff(X.shape[1], n_features, sigma, rng=g, rff_map=rff_map)
    return float(realized_kappa(rff.transform(np.vstack([X, Y]))))


def _threshold_fn(threshold, n, kappa_rff, rff_map="phased"):
    """alpha -> threshold for one replicate (n = m).

    ``paper`` is v1's tau with kappa = 1 on both backends (as published);
    ``plugin`` uses the Gaussian constants on the exact Gram matrix and the
    bounded constants with the realized kappa on the phased random features
    (with ``rff_map="cossin"``: the bounded constants with kappa = 1).
    """
    if threshold == "paper":
        return lambda g: float(tau(n, n, g, DELTA))
    if kappa_rff is None:
        return V2.threshold_for(threshold, n, n, DELTA, backend="gram")
    if rff_map == "cossin":
        assert kappa_rff == 1.0
        return V2.threshold_for(threshold, n, n, DELTA, backend="cossin", kappa=1.0)
    return V2.threshold_for(threshold, n, n, DELTA, backend="rff", kappa=kappa_rff)


def _decisions(dh, grid, n, a, threshold, kappa_rff, rff_map="phased"):
    """alpha-hat, the firing rule and the per-level decisions from a path."""
    fn = _threshold_fn(threshold, n, kappa_rff, rff_map)
    thr = np.array([fn(g) for g in grid])
    a_hat = float(alpha_hat_from_path(grid, dh, n, n, DELTA, threshold_fn=fn))
    return {"alpha_hat": a_hat, "tau_0": float(thr[0]),
            "fires": bool(float(dh[0]) > thr[0]),
            "covers": bool(a_hat <= a + 1e-12),
            "reject": [bool(v) for v in (np.asarray(dh) > thr)]}


def _certificate_v2(X, Y, grid, sigma, rng, use_rff, n_features, certificate, rff_map="phased"):
    """v2 certificate(s) on the split seed of ``lower_bound_tv(..., rng=rng)``
    (the same single draw from ``rng``), per-level witnesses.

    ``betting`` / ``both``: single-split betting (+ the bootstrap, which then
    reproduces v1's ``certified``), with v1's bandwidth ``sigma``.
    ``sequential`` / ``all``: the paper's main certificate, sequential betting
    on batches (0, 1/4, 1/2, 1) in ``certified`` / ``cert_reject``. Its
    bandwidth is the median heuristic on batch 0 only (``sigma=None``), so
    every ramp is a function of the earlier batches as Theorem thm:betting
    requires; v1's full-sample ``sigma`` would also see the batches bet on.
    With ``all`` the single-split betting and the bootstrap (v1 bandwidth,
    same split seed) are kept in ``*_betting`` / ``*_bootstrap``.
    """
    methods = V2.certificate_methods(certificate)
    split_seed = int(rng.integers(0, 2**31 - 1))          # lower_bound_tv's draw
    single = tuple(mth for mth in methods if mth != "sequential")
    res = {}
    if "sequential" in methods:
        b, lv = lower_bound_betting(X, Y, grid, delta=DELTA, sigma=None, split_seed=split_seed,
                                    methods=("sequential",), use_rff=use_rff,
                                    n_features=n_features, rff_map=rff_map)
        res["sequential"] = (b["sequential"], lv)
    if single:
        b, lv = lower_bound_betting(X, Y, grid, delta=DELTA, sigma=sigma, split_seed=split_seed,
                                    methods=single, n_boot=1000 if "bootstrap" in single else 0,
                                    use_rff=use_rff, n_features=n_features, rff_map=rff_map)
        for mth in single:
            res[mth] = (b[mth], lv)
    out = {}
    for mth in methods:
        tag = "" if mth == methods[0] else f"_{mth}"
        bound, lv = res[mth]
        pad = [None] * (len(grid) - len(lv))
        out["certified" + tag] = float(bound)
        out["cert_reject" + tag] = [bool(r.get(mth, False)) for r in lv] + pad
        if mth == methods[0]:
            out["cert_levels_evaluated"] = len(lv)
    if "sequential" in methods:
        lv = res["sequential"][1]
        out["cert_seq_horizon"] = int(lv[0]["seq_horizon"]) if lv and "seq_horizon" in lv[0] else None
        out["cert_seq_skipped"] = [r.get("seq_skipped", []) for r in lv]
    return out


def one_rep(sample, a, n, grid, rng, use_rff, n_features, do_cert, do_trim,
            sep=3.0, threshold="paper", certificate="bootstrap", rff_map="phased"):
    if threshold != "paper" or certificate != "bootstrap" or rff_map != "phased":
        return _one_rep_v2(sample, a, n, grid, rng, use_rff, n_features, do_cert,
                           do_trim, sep, threshold, certificate, rff_map)
    X = sample(n, rng, "P")
    Y = sample(n, rng, "Q")
    sigma = float(median_heuristic(X, Y, rng=rng))
    t0 = time.time()
    _, dh, _ = trimmed_mmd_path(X, Y, grid, sigma=sigma, rng=rng,
                                use_rff=use_rff, n_features=n_features)
    secs = time.time() - t0
    a_hat = float(alpha_hat_from_path(grid, dh, n, n, DELTA))
    d0 = float(dh[0])
    thr = np.array([tau(n, n, g, DELTA) for g in grid])
    # The per-level decision vector IS the test of H_{0,alpha} at every alpha
    # on the grid. Averaged over replicates it gives the power curve below
    # alpha* and the type-I rate at and above it -- both free, because the
    # path has already been solved.
    row = {"n": n, "alpha_star": a, "alpha_hat": a_hat, "d_hat_0": d0,
           "tau_0": float(tau(n, n, 0.0, DELTA)),
           "fires": bool(d0 > tau(n, n, 0.0, DELTA)),
           "covers": bool(a_hat <= a + 1e-12),
           "d_hat": [float(v) for v in dh],
           "reject": [bool(v) for v in (np.asarray(dh) > thr)],
           "path_seconds": secs, "sigma": sigma}
    if do_cert:
        lo, out = lower_bound_tv(X, Y, grid, delta=DELTA, sigma=sigma, rng=rng,
                                 use_rff=use_rff, n_features=n_features)
        row["certified"] = float(lo)
        row["cert_reject"] = [bool(o.certified) for o in out]
    if do_trim and a > 0:
        # recall of the planted contamination at the true level
        h = n // 2
        t = optimal_trimming_mmd(X[:h], Y[:h], a, sigma=sigma, select="none",
                                 rng=np.random.default_rng(0),
                                 use_rff=use_rff, n_features=n_features)
        w = witness_at(t, X[:h], Y[:h], X[h:], sigma=sigma)
        k_keep = int(np.floor((n - h) * (1 - a)))
        dropped = np.argsort(w)[k_keep:]
        planted = np.abs(X[h:][:, 0]) > 0.5 * sep      # inside a contaminating box
        row["recall"] = float(planted[dropped].mean()) if dropped.size else np.nan
        row["planted_frac"] = float(planted.mean())
    return row


def _recall(X, Y, a, n, sigma, use_rff, n_features, sep, rff_map="phased"):
    """The trimming behind the recall panel (threshold-free: select="none")."""
    h = n // 2
    t = optimal_trimming_mmd(X[:h], Y[:h], a, sigma=sigma, select="none",
                             rng=np.random.default_rng(0),
                             use_rff=use_rff, n_features=n_features, rff_map=rff_map)
    w = witness_at(t, X[:h], Y[:h], X[h:], sigma=sigma)
    k_keep = int(np.floor((n - h) * (1 - a)))
    dropped = np.argsort(w)[k_keep:]
    planted = np.abs(X[h:][:, 0]) > 0.5 * sep
    return (float(planted[dropped].mean()) if dropped.size else np.nan,
            float(planted.mean()))


def _one_rep_v2(sample, a, n, grid, rng, use_rff, n_features, do_cert, do_trim,
                sep, threshold, certificate, rff_map="phased"):
    """``one_rep`` with the v2 threshold / certificate; same random stream
    (with ``rff_map="cossin"``: the same data and bandwidth, then the paired features)."""
    X = sample(n, rng, "P")
    Y = sample(n, rng, "Q")
    sigma = float(median_heuristic(X, Y, rng=rng))
    state = copy.deepcopy(rng.bit_generator.state)
    t0 = time.time()
    _, dh, _ = trimmed_mmd_path(X, Y, grid, sigma=sigma, rng=rng,
                                use_rff=use_rff, n_features=n_features, rff_map=rff_map)
    secs = time.time() - t0
    kap = (_rff_kappa(X, Y, sigma, state, n_features, rff_map)
           if (use_rff and threshold != "paper") else None)
    extra = {}
    if rff_map != "phased":
        # cos/sin map: k_hat(x, x) = 1 exactly; the threshold uses kappa = 1, the realized value is kept
        extra = {"rff_map": rff_map, "kappa_realized": kap}
        if kap is not None:
            assert abs(kap - 1.0) < 1e-9, kap
            kap = 1.0
    row = {"n": n, "alpha_star": a, "d_hat_0": float(dh[0]),
           "d_hat": [float(v) for v in dh], "path_seconds": secs, "sigma": sigma,
           "threshold": threshold, "kappa_rff": kap, **extra}
    row.update(_decisions(dh, grid, n, a, threshold, kap, rff_map))
    paper = _decisions(dh, grid, n, a, "paper", None)
    row.update({"alpha_hat_paper": paper["alpha_hat"], "tau_0_paper": paper["tau_0"],
                "fires_paper": paper["fires"]})
    if do_cert:
        if certificate == "bootstrap":
            lo, out = lower_bound_tv(X, Y, grid, delta=DELTA, sigma=sigma, rng=rng,
                                     use_rff=use_rff, n_features=n_features, rff_map=rff_map)
            row["certified"] = float(lo)
            row["cert_reject"] = [bool(o.certified) for o in out]
        else:
            row.update(_certificate_v2(X, Y, grid, sigma, rng, use_rff, n_features,
                                       certificate, rff_map))
        row["certificate"] = certificate
    if do_trim and a > 0:
        row["recall"], row["planted_frac"] = _recall(X, Y, a, n, sigma, use_rff,
                                                     n_features, sep, rff_map)
    return row


def _replay(sample, row, grid, n_features):
    """Replay a v1 replicate's random stream up to the certificate.

    Draws the data and the bandwidth exactly as ``one_rep``, then advances the
    generator past ``trimmed_mmd_path`` without solving it: the path consumes
    the random features (n >= RFF_FROM_N) and one power-iteration start
    ``standard_normal(n + m)`` per level, nothing else. The next draw is the
    certificate's split seed, as in v1.
    """
    a, n, r = row["alpha_star"], row["n"], row["rep"]
    rng = np.random.default_rng(10_000 + 97 * r + n + int(1e5 * a))
    X = sample(n, rng, "P")
    Y = sample(n, rng, "Q")
    sigma = float(median_heuristic(X, Y, rng=rng))
    if abs(sigma - row["sigma"]) > 1e-9 * max(1.0, abs(sigma)):
        raise RuntimeError(f"replay mismatch (sigma) at a*={a} n={n} rep={r}")
    state = copy.deepcopy(rng.bit_generator.state)
    if row["rff"]:
        RFF(dim=X.shape[1], n_features=n_features, sigma=sigma, rng=rng)
    for _ in range(len(grid)):
        rng.standard_normal(2 * n)
    return X, Y, sigma, rng, state


def _rethreshold_job(spec):
    """--from-rows: new decisions from the stored path, certificate on the replayed split."""
    row, grid, sep, threshold, certificate, n_features = spec
    grid = np.asarray(grid)
    a, n = row["alpha_star"], row["n"]
    key = (a, sep)
    if key not in _SCEN:
        _SCEN[key] = rhc_scenario(a, sep=sep)[0]
    t0 = time.time()
    X, Y, sigma, rng, state = _replay(_SCEN[key], row, grid, n_features)
    kap = (_rff_kappa(X, Y, sigma, state, n_features)
           if (row["rff"] and threshold != "paper") else None)
    out = {k: row[k] for k in ("n", "alpha_star", "d_hat_0", "d_hat", "path_seconds",
                               "sigma", "rep", "rff") if k in row}
    for k in ("recall", "planted_frac"):
        if k in row:
            out[k] = row[k]
    out["threshold"] = threshold
    out["kappa_rff"] = kap
    out.update(_decisions(np.asarray(row["d_hat"]), grid, n, a, threshold, kap))
    paper = _decisions(np.asarray(row["d_hat"]), grid, n, a, "paper", None)
    out.update({"alpha_hat_paper": paper["alpha_hat"], "tau_0_paper": paper["tau_0"],
                "fires_paper": paper["fires"]})
    out["replay_paper_matches_v1"] = bool(
        paper["alpha_hat"] == row["alpha_hat"] and paper["fires"] == row["fires"]
        and paper["reject"] == row["reject"])
    if "certified" in row:
        out["certified_v1"] = row["certified"]
        if certificate == "bootstrap":
            out["certified"], out["cert_reject"] = row["certified"], row["cert_reject"]
        else:
            out.update(_certificate_v2(X, Y, grid, sigma, rng, row["rff"], n_features,
                                       certificate))
            if "certified_bootstrap" in out:
                out["replay_bootstrap_matches_v1"] = bool(
                    out["certified_bootstrap"] == row["certified"])
        out["certificate"] = certificate
    out["v2_seconds"] = time.time() - t0
    return out


def from_rows(args, outdir):
    """Re-threshold a stored run (and recompute its certificate on the same splits)."""
    src = Path(args.from_rows)
    D = json.loads(src.read_text())
    rows = D["rows"]
    if args.sizes is not None:
        rows = [r for r in rows if r["n"] in set(args.sizes)]
    if args.levels is not None:
        rows = [r for r in rows if r["alpha_star"] in set(args.levels)]
    if args.max_rows is not None:
        rows = rows[:args.max_rows]
    sep = D.get("sep", 3.0)
    jobs = [(r, D["grid"], sep, args.threshold, args.certificate, N_FEATURES)
            for r in sorted(rows, key=lambda r: -r["n"])]
    print(f"{len(jobs)} stored rows from {src}, {args.workers} workers", flush=True)
    t0 = time.time()
    out_rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for row in ex.map(_job_rethreshold_safe, jobs):
            out_rows.append(row)
            print(f"  [{len(out_rows):3d}/{len(jobs)}] a*={row['alpha_star']:.2f} "
                  f"n={row['n']:5d} r={row['rep']}: tau0={row['tau_0']:.3f} "
                  f"fires={row['fires']} a_hat={row['alpha_hat']:.3f} "
                  f"cert={row.get('certified', float('nan')):.3f} "
                  f"(1-split {row.get('certified_betting', float('nan')):.3f}, "
                  f"boot {row.get('certified_bootstrap', float('nan')):.3f}, "
                  f"v1 {row.get('certified_v1', float('nan')):.3f}) "
                  f"({time.time() - t0:.0f}s)", flush=True)
    meta = {k: v for k, v in D.items() if k != "rows"}
    meta.update({"threshold": args.threshold, "certificate": args.certificate,
                 "from_rows": str(src), "v1_seconds": D.get("seconds"),
                 "workers": args.workers, "seconds": time.time() - t0,
                 "kernel_constants": ("gaussian (exact Gram, n < %d); bounded with the realized "
                                      "kappa of the phased RFF (n >= %d)" % (RFF_FROM_N, RFF_FROM_N))
                 if args.threshold != "paper" else "v1 tau, kappa = 1"})
    chk = {"rows": len(out_rows),
           "paper_decisions_match_v1": int(sum(r["replay_paper_matches_v1"] for r in out_rows))}
    if any("replay_bootstrap_matches_v1" in r for r in out_rows):
        chk["bootstrap_matches_v1"] = int(sum(r.get("replay_bootstrap_matches_v1", False)
                                              for r in out_rows))
        chk["bootstrap_compared"] = int(sum("replay_bootstrap_matches_v1" in r for r in out_rows))
    meta["replay_check"] = chk
    name = args.out or src.name
    dest = V2.sub(outdir, "figures") / name
    dest.write_text(json.dumps({**meta, "rows": out_rows}, indent=2))
    print(f"replay check: {chk}")
    print(f"wrote {dest} in {time.time() - t0:.0f}s")


def _job_rethreshold_safe(spec):
    return _rethreshold_job(spec)


_SCEN: dict = {}


def _job(spec):
    """Top-level so it pickles. Scenarios are cached per worker process."""
    a, n, r, grid, use_rff, nf, do_cert, do_trim, sep = spec[:9]
    threshold, certificate = spec[9:11] if len(spec) > 9 else ("paper", "bootstrap")
    rff_map = spec[11] if len(spec) > 11 else "phased"
    key = (a, sep)
    if key not in _SCEN:
        _SCEN[key] = rhc_scenario(a, sep=sep)[0]
    rng = np.random.default_rng(10_000 + 97 * r + n + int(1e5 * a))
    row = one_rep(_SCEN[key], a, n, np.asarray(grid), rng, use_rff, nf,
                  do_cert, do_trim, sep=sep, threshold=threshold,
                  certificate=certificate, rff_map=rff_map)
    row["rep"] = r
    row["rff"] = use_rff
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--sep", type=float, default=3.0,
                    help="contamination offset in pooled standard deviations")
    ap.add_argument("--levels", type=float, nargs="*", default=None,
                    help="override the contamination levels")
    ap.add_argument("--sizes", type=int, nargs="*", default=None)
    ap.add_argument("--out", type=str, default=None)
    V2.add_v2_args(ap)
    ap.add_argument("--rff-map", choices=RFF_MAPS, default="phased",
                    help="random-feature map from n >= %d: phased (default, v1/v2) or cossin "
                         "(diagonal exactly 1, threshold with kappa = 1; needs --outdir, no "
                         "--from-rows)" % RFF_FROM_N)
    ap.add_argument("--from-rows", type=str, default=None,
                    help="v2: re-threshold the stored paths of this JSON (no QP); the "
                         "certificate is recomputed on the replayed split unless "
                         "--certificate bootstrap")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="with --from-rows: only the first K rows (smoke runs)")
    ap.add_argument("--max-reps", type=int, default=None,
                    help="cap the replicates per cell (smoke runs only)")
    args = ap.parse_args()
    workers = args.workers
    sep = args.sep
    v2_mode = (args.threshold != "paper" or args.certificate != "bootstrap"
               or args.outdir is not None)
    if args.rff_map != "phased":
        if args.from_rows is not None:
            raise SystemExit("--rff-map cossin needs a full re-solve (the stored paths are phased)")
        if args.outdir is None:
            raise SystemExit("--rff-map cossin needs --outdir")
        v2_mode = True
    if args.from_rows is not None:
        if args.outdir is None:
            raise SystemExit("--from-rows needs --outdir (it would overwrite the v1 rows otherwise)")
        from_rows(args, args.outdir)
        return

    if args.pilot:
        levels, sizes, reps = [0.10, 0.20], [500, 1000, 2000], 2
        reps_by_n = {}
        grid = np.round(np.arange(0.0, 0.41, 0.05), 3)
        do_cert, do_trim = False, False
        out_name = "fig2_pilot.json"
    else:
        # Chosen so the four levels put their firing thresholds -- n >
        # (2C/D_0)^2, with D_0 ~ 0.99 alpha* at sep = 3 -- at roughly 1.2k,
        # 2.2k, 4.9k and 8.8k, one inside each gap of the size grid. The
        # transition is then visible four times over, not once.
        levels = [0.15, 0.20, 0.30, 0.40]
        sizes = [500, 1000, 2000, 4000, 8000, 16000]
        # More replicates where they are cheap: a power curve read off five
        # replicates is quantized to multiples of 0.2, which is useless.
        reps_by_n = {500: 20, 1000: 15, 2000: 15, 4000: 8, 8000: 5, 16000: 3}
        reps = max(reps_by_n.values())
        grid = np.round(np.arange(0.0, 0.46, 0.025), 4)
        do_cert, do_trim = True, True
        out_name = "fig2_data.json"
    if args.levels is not None:
        levels = list(args.levels)
    if args.sizes is not None:
        sizes = list(args.sizes)
        reps_by_n = {n: reps_by_n.get(n, 8) for n in sizes}
    if args.out is not None:
        out_name = args.out
    elif args.rff_map != "phased":
        out_name = out_name.replace(".json", "_cossin.json")
    if args.max_reps is not None:
        reps_by_n = {n: min(v, args.max_reps) for n, v in reps_by_n.items()}
        reps = min(reps, args.max_reps)

    t0 = time.time()
    jobs = []
    for a in levels:
        for n in sizes:
            pass
            nrep = reps if args.pilot else reps_by_n[n]
            if args.max_reps is not None:
                nrep = min(nrep, args.max_reps)
            # The certificate's witness_direction always forms exact Gram
            # matrices (there is no random-feature path through
            # splitting.py), so above a few thousand it costs O(n^2) time
            # and gigabytes per worker. Its point -- that it is informative
            # where the analytic route is silent -- lives at small n, so we
            # cap it rather than pay for it where it adds nothing.
            for r in range(nrep):
                spec = (a, n, r, grid.tolist(), n >= RFF_FROM_N,
                        N_FEATURES, do_cert, do_trim, sep)
                if v2_mode:
                    spec = spec + (args.threshold, args.certificate)
                if args.rff_map != "phased":
                    spec = spec + (args.rff_map,)
                jobs.append(spec)
    # Biggest jobs first so the tail does not straggle.
    jobs.sort(key=lambda j: -j[1])
    print(f"{len(jobs)} cells, {workers} workers", flush=True)

    rows = []
    # --rff-map cossin (a long full re-solve): rows are appended to <out>.partial.jsonl as they finish, in any
    # order, and sorted back into the job order at the end; the default path is unchanged.
    partial = (V2.sub(args.outdir, "figures") / (out_name + ".partial.jsonl")
               if args.rff_map != "phased" else None)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        if partial is None:
            results = ex.map(_job, jobs)
        else:
            from concurrent.futures import as_completed
            futs = {ex.submit(_job, j): i for i, j in enumerate(jobs)}
            done_idx = []

            def _stream():
                with open(partial, "a") as fh:
                    for f in as_completed(futs):
                        row = f.result()
                        done_idx.append(futs[f])
                        fh.write(json.dumps(row) + "\n")
                        fh.flush()
                        yield row
            results = _stream()
        for row in results:
            rows.append(row)
            print(f"  [{len(rows):3d}/{len(jobs)}] a*={row['alpha_star']:.2f} "
                  f"n={row['n']:5d} r={row['rep']}: d0={row['d_hat_0']:.3f} "
                  f"tau0={row['tau_0']:.3f} fires={row['fires']} "
                  f"a_hat={row['alpha_hat']:.3f} "
                  f"cert={row.get('certified', float('nan')):.2f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    if partial is not None:
        rows = [r for _, r in sorted(zip(done_idx, rows), key=lambda t: t[0])]

    meta = {"levels": levels, "sizes": sizes, "reps": reps, "workers": workers,
            "reps_by_n": reps_by_n,
            "grid": grid.tolist(), "delta": DELTA, "sep": sep, "rows": rows,
            "seconds": time.time() - t0}
    if v2_mode:
        meta.update({"threshold": args.threshold, "certificate": args.certificate})
        if args.rff_map != "phased":
            meta.update({"rff_map": args.rff_map, "n_freq": N_FEATURES // 2,
                         "kernel_constants": ("gaussian (exact Gram, n < %d); bounded with kappa = 1 "
                                              "(cos/sin RFF, %d frequencies, %d features; n >= %d)"
                                              % (RFF_FROM_N, N_FEATURES // 2, N_FEATURES, RFF_FROM_N))})
        dest_dir = V2.sub(args.outdir, "figures")
    else:
        dest_dir = OUT
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / out_name).write_text(json.dumps(meta, indent=2))
    if partial is not None:
        partial.unlink()
    print(f"wrote {dest_dir / out_name} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

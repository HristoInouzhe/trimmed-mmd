"""Valid-bandwidth reruns of the analytic test: NHANES child contrast and ACS.

The analytic threshold covers a kernel fixed before the data are seen
(Remark rem:fixedkernel); the published NHANES and ACS numbers use the median
bandwidth of the pooled sample, which is not. This script applies the two
remedies that scripts/fair_bandwidth_tune.py applied to Adult, with the same
bandwidth grid and the same splits:

  pilot     tune-then-test. A pilot of 30% of each group (numpy
            default_rng(2026), as for Adult) picks the bandwidth: for each
            scale s in SCALES, sigma = s * median(pilot), the certified level
            is computed on the pilot at level delta, and the scale with the
            largest certified level wins (ties broken towards 0.5, as for
            Adult). The analytic test is then run on the other 70% at level
            delta with that bandwidth fixed.
  union     every scale of the same grid on ALL the data at level delta/J,
            J = 7; the largest certified level over the scales is certified at
            level delta.

Exactly as in the Adult run, the union grid is anchored at the pilot median,
which is computed from a subset of the data the union test then uses, and on
ACS the covariates are standardized by the moments of the whole sample. Both
are data-dependent choices that the theorem does not formally cover. The
script therefore also runs strict variants, in which nothing the test uses
depends on the test data:

  union_heldout   the union over the 7 scales at delta/7 on the 70% held-out
                  part only (grid anchored at the pilot median; on ACS the
                  covariates are standardized with pilot moments);
  pilot_strict    (ACS only) tune-then-test with pilot-moment standardization.
                  On NHANES nothing is standardized, so `pilot` is already
                  strict.

The pooled-median route of the paper is recomputed as the reference.

Statistic and decision. For NHANES the kernel is the exact Gaussian Gram
matrix (as in the paper's NHANES analysis); for ACS it is the 512-frequency
cos-sin feature map with kappa = 1 exactly (as in the paper's ACS analysis
and the Adult bandwidth run, with feature seeds 1/2/3 for tune/union/test as
there). A level alpha is rejected only when the Frank-Wolfe lower bound on the
minimum of the QP exceeds tau(alpha)^2, so an unconverged iterate cannot help
the test reject. The FISTA loop is solver._fista; it additionally stops as
soon as the decision is settled -- the lower bound already exceeds tau^2
(reject) or the objective, an upper bound on the minimum, is already below
tau^2 (accept) -- which gives the decision of the exact minimizer. If neither
happens within max_iter, the level is accepted unless the lower bound clears
tau (conservative); such levels are flagged `unresolved`.

Levels: alpha is scanned on the paper's grid {0, 0.05, ..., 0.75} upward until
the first non-rejected level (alpha-hat on that grid; the largest rejected
grid level is alpha_last), and then bisected inside [alpha_last, alpha-hat] to
width 0.0125 (`certified_fine`, the resolution of the Adult run). Rejections
are valid simultaneously over alpha, so every rejected level is certified.

    PYTHONPATH=src python3 -I scripts/exp_valid_bandwidth.py --workers 4 --threads 4

Revision v2 (defaults unchanged): ``--threshold plugin`` decides every level
against tau_new instead of tau (exact Gram on NHANES: Gaussian coupled
constants; cos/sin features on ACS: the bounded-kernel constants, since the
feature kernel can be negative), with delta/J for the union routes as
before; ``--outdir D`` writes r12_* to D/tables, keeps the job cache in
D/checkpoints/r12_valid_bandwidth (v2 job keys also carry the threshold),
and takes the published reference (pooled-median alpha-hat, alpha_last) from
the v2 D/tables/r1_* and D/fair/acs_info.json.

    PYTHONPATH=src python3 -I scripts/exp_valid_bandwidth.py --workers 2 --threads 4 \
        --threshold plugin --outdir results/v2

JMLR v3 (2026-11-12, defaults unchanged): the other three contrasts of the paper's Table 2 -- ``rhc`` (exact Gram;
covariates standardized, so it also gets the strict pilot route with pilot-moment standardization, as ACS),
``nhanes_age`` and ``nhanes_bmi`` (exact Gram, no standardization, as nhanes_child) -- so that every analytic number
of that table has a bandwidth fixed before the test data are seen. ``--pilot-only`` skips the union routes for the
data sets it runs (cached union jobs are still summarized).

    PYTHONPATH=src python3 -I scripts/exp_valid_bandwidth.py --workers 8 --threads 4 --threshold plugin \
        --outdir results/v2 --pilot-only --datasets nhanes_child acs rhc nhanes_age nhanes_bmi

AoS referee runs (2026-11-22, defaults unchanged): data set ``rhc_full``, RHC with the 45 covariates of
realdata.load_rhc(covariates="full") (the 33 of ``rhc`` plus sex, dnr1 and the ten admission-diagnosis indicators,
coded 0/1), handled exactly as ``rhc`` (exact Gram, standardized, strict pilot route). Run it with its own --outdir,
since write_outputs summarizes only the data sets of the invocation.
"""

from __future__ import annotations

import os
import sys


def _threads_from_argv(default="4"):
    for i, a in enumerate(sys.argv):
        if a == "--threads" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith("--threads="):
            return a.split("=", 1)[1]
    return default


_T = _threads_from_argv()
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _T

import argparse                                                # noqa: E402
import json                                                    # noqa: E402
import platform                                                # noqa: E402
import time                                                    # noqa: E402
from concurrent.futures import ProcessPoolExecutor, wait       # noqa: E402
from pathlib import Path                                       # noqa: E402

import numpy as np                                             # noqa: E402
import pandas as pd                                            # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import fair_case_study as fc                                   # noqa: E402
from trimmedmmd.kernels import gaussian_kernel, median_heuristic  # noqa: E402
from trimmedmmd.realdata import load_nhanes_activity, load_rhc, nhanes_groups  # noqa: E402
from trimmedmmd.simplex import project_capped_simplex          # noqa: E402
from trimmedmmd.solver import (_linear_min_over_capped_simplex,  # noqa: E402
                               _power_iteration)
from trimmedmmd.threshold import tau                           # noqa: E402
from trimmedmmd.experiments.v2 import threshold_for            # noqa: E402

TABLES = ROOT / "results" / "tables"
CACHE = ROOT / "results" / "checkpoints" / "r12_valid_bandwidth"
FAIR = ROOT / "results" / "fair"
THRESHOLD = "paper"      # set by --threshold; carried in each v2 job spec
PILOT_ONLY = False       # set by --pilot-only (v3): no union routes for the data sets run
GRAM_NHANES = ("nhanes_child", "nhanes_age", "nhanes_bmi")
GRAM_DATASETS = GRAM_NHANES + ("rhc", "rhc_full")
STANDARDIZED = ("acs", "rhc", "rhc_full")   # rhc_full: AoS referee runs (2026-11-22), see the docstring


def threshold_value(qp, alpha, delta, threshold="paper"):
    """The test threshold for this QP's kernel (v1: tau with kappa = 1)."""
    if threshold == "paper":
        return tau(qp.n, qp.m, alpha, delta)
    backend = "gram" if qp.kind == "gram" else "cossin"
    return threshold_for(threshold, qp.n, qp.m, delta, backend=backend)(alpha)

SCALES = (0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0)   # fair_bandwidth_tune.SCALES
J = len(SCALES)
TUNE_FRAC = 0.3                                   # as for Adult
SPLIT_SEED = 2026                                 # as for Adult
W_SEED = {"tune": 1, "union": 2, "test": 3}       # as for Adult
STRICT_MEDIAN_SEED = 2027
DELTA = 0.05
GRID = np.round(np.arange(0.0, 0.751, 0.05), 3)   # realdata_run.ALPHA_GRID
REFINE_TOL = 0.0125
N_FREQ = 512
ACS_ROWS, ACS_SEED = 120_000, 0                   # the paper's ACS subsample
MAX_ITER = {"gram": 4000, "features": 2500}
TOL = 1e-10


# ---------------------------------------------------------------------------
# the two quadratic programs
# ---------------------------------------------------------------------------

class GramQP:
    kind = "gram"

    def __init__(self, A, B, sigma):
        self.Kxx = gaussian_kernel(A, A, sigma)
        self.Kyy = gaussian_kernel(B, B, sigma)
        self.Kxy = gaussian_kernel(A, B, sigma)
        self.n, self.m = len(A), len(B)

    def grad(self, w, v):
        return (2.0 * (self.Kxx @ w - self.Kxy @ v),
                2.0 * (self.Kyy @ v - self.Kxy.T @ w))

    def grad_value(self, w, v):
        kw, kv = self.Kxx @ w, self.Kxy @ v
        lv, lw = self.Kyy @ v, self.Kxy.T @ w
        val = float(w @ kw + v @ lv - 2.0 * (w @ kv))
        return 2.0 * (kw - kv), 2.0 * (lv - lw), val

    def matvec(self, z):
        w, v = z[:self.n], z[self.n:]
        return np.concatenate([self.Kxx @ w - self.Kxy @ v,
                               self.Kyy @ v - self.Kxy.T @ w])


class FeatureQP:
    kind = "features"

    def __init__(self, Zx, Zy):
        self.Zx, self.Zy = Zx, Zy
        self.n, self.m = len(Zx), len(Zy)

    def grad(self, w, v):
        d = self.Zx.T @ w - self.Zy.T @ v
        return 2.0 * (self.Zx @ d), -2.0 * (self.Zy @ d)

    def grad_value(self, w, v):
        d = self.Zx.T @ w - self.Zy.T @ v
        return 2.0 * (self.Zx @ d), -2.0 * (self.Zy @ d), float(d @ d)

    def matvec(self, z):
        w, v = z[:self.n], z[self.n:]
        d = self.Zx.T @ w - self.Zy.T @ v
        return np.concatenate([self.Zx @ d, -(self.Zy @ d)])


def lipschitz(qp):
    """As in solver.solve_*: power iteration with default_rng(0)."""
    lam = _power_iteration(qp.matvec, qp.n + qp.m, np.random.default_rng(0))
    return max(2.0 * lam, 1e-12)


def decide(qp, alpha, thr, lip, w0=None, v0=None):
    """FISTA of solver._fista with a settled-decision stop (see docstring)."""
    n, m = qp.n, qp.m
    cap_w, cap_v = 1.0 / (n * (1.0 - alpha)), 1.0 / (m * (1.0 - alpha))
    w = np.full(n, 1.0 / n) if w0 is None else np.clip(w0, 0, cap_w)
    v = np.full(m, 1.0 / m) if v0 is None else np.clip(v0, 0, cap_v)
    w, v = project_capped_simplex(w, cap_w), project_capped_simplex(v, cap_v)
    thr2 = thr * thr
    max_iter = MAX_ITER[qp.kind]

    def certificate(w, v):
        gw, gv, val = qp.grad_value(w, v)
        gap = (float(gw @ w) - _linear_min_over_capped_simplex(gw, cap_w)
               + float(gv @ v) - _linear_min_over_capped_simplex(gv, cap_v))
        return max(val, 0.0), max(val - max(gap, 0.0), 0.0)

    val, low = certificate(w, v)
    reason = None
    if low > thr2:
        reason = "lower>tau"
    elif val < thr2:
        reason = "upper<tau"
    yw, yv, t, step = w.copy(), v.copy(), 1.0, 1.0 / lip
    prev, it = None, 0
    while reason is None and it < max_iter:
        it += 1
        gw, gv = qp.grad(yw, yv)
        w_new = project_capped_simplex(yw - step * gw, cap_w)
        v_new = project_capped_simplex(yv - step * gv, cap_v)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        beta = (t - 1.0) / t_new
        yw = w_new + beta * (w_new - w)
        yv = v_new + beta * (v_new - v)
        w, v, t = w_new, v_new, t_new
        if it % 10 == 0 or it == max_iter:
            val, low = certificate(w, v)
            if prev is not None and val > prev:
                yw, yv, t = w.copy(), v.copy(), 1.0
            if low > thr2:
                reason = "lower>tau"
            elif val < thr2:
                reason = "upper<tau"
            elif prev is not None and abs(prev - val) <= TOL * max(abs(prev), 1e-300):
                reason = "converged"
            prev = val
    if reason is None:
        reason = "max_iter"
    reject = bool(low > thr2)
    return {"alpha": float(alpha), "tau": float(thr),
            "d_upper": float(np.sqrt(val)), "d_lower": float(np.sqrt(low)),
            "reject": reject, "reason": reason, "n_iter": it,
            "unresolved": reason in ("converged", "max_iter")}, w, v


def certify(qp, delta, log=None, threshold="paper"):
    """Scan GRID upward to the first acceptance, then bisect to REFINE_TOL."""
    lip = lipschitz(qp)
    steps, w0, v0 = [], None, None
    last, first_acc, w_last, v_last = None, None, None, None
    for a in GRID:
        r, w, v = decide(qp, a, threshold_value(qp, a, delta, threshold), lip, w0, v0)
        steps.append(r)
        if log:
            log(r)
        if r["reject"]:
            last, w_last, v_last, w0, v0 = float(a), w, v, w, v
        else:
            first_acc = float(a)
            break
    fine = 0.0 if last is None else last
    if last is not None and first_acc is not None:
        lo, hi = last, first_acc
        while hi - lo > REFINE_TOL + 1e-12:
            mid = 0.5 * (lo + hi)
            r, w, v = decide(qp, mid, threshold_value(qp, mid, delta, threshold), lip,
                             w_last, v_last)
            r["refine"] = True
            steps.append(r)
            if log:
                log(r)
            if r["reject"]:
                lo, w_last, v_last = mid, w, v
            else:
                hi = mid
        fine = lo
    return {
        "H00_rejected": bool(steps[0]["reject"]),
        "alpha_hat": first_acc if first_acc is not None else float(GRID[-1]),
        "alpha_last": last,
        "certified_fine": float(fine),
        "D0": steps[0]["d_upper"], "tau0": steps[0]["tau"],
        "unresolved_levels": [s["alpha"] for s in steps if s["unresolved"]],
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# data, splits and bandwidths (deterministic; recomputed in every worker)
# ---------------------------------------------------------------------------

_PREP: dict = {}


def prepare(dataset):
    if dataset in _PREP:
        return _PREP[dataset]
    if dataset in GRAM_NHANES:
        curves, cov, _ = load_nhanes_activity()
        X, Y, _, _ = nhanes_groups(curves, cov, by=dataset.split("_", 1)[1])
        Fv = np.vstack([X, Y]).astype(float)
        g = np.r_[np.zeros(len(X), int), np.ones(len(Y), int)]
        standardize = False
    elif dataset in ("rhc", "rhc_full"):
        # raw covariates; the paper standardizes them by the pooled moments (load_rhc), which std(ones) reproduces.
        # rhc_full: the 45 covariates of load_rhc(covariates="full") (33 + the 12 string-coded binaries)
        X, Y, _, _ = load_rhc(standardize=False, covariates="full" if dataset == "rhc_full" else "paper")
        Fv = np.vstack([X, Y]).astype(float)
        g = np.r_[np.zeros(len(X), int), np.ones(len(Y), int)]
        standardize = True
    elif dataset == "acs":
        F, g, _, _ = fc.load("acs", ACS_ROWS, ACS_SEED)
        Fv = F.values.astype(float)
        g = np.asarray(g, int)
        standardize = True
    else:
        raise ValueError(dataset)

    rng = np.random.default_rng(SPLIT_SEED)      # as fair_bandwidth_tune
    tune = np.zeros(len(g), bool)
    for k in (0, 1):
        ix = np.nonzero(g == k)[0]
        tune[rng.choice(ix, int(TUNE_FRAC * len(ix)), replace=False)] = True

    def std(mask):
        if not standardize:
            return Fv
        mu, sd = Fv[mask].mean(0), Fv[mask].std(0)
        sd[sd == 0] = 1.0
        return (Fv - mu) / sd

    S_full = std(np.ones(len(g), bool))
    sig_tune = float(median_heuristic(S_full[tune & (g == 0)],
                                      S_full[tune & (g == 1)], rng=rng))
    S_pil = std(tune)
    sig_tune_strict = float(median_heuristic(
        S_pil[tune & (g == 0)], S_pil[tune & (g == 1)],
        rng=np.random.default_rng(STRICT_MEDIAN_SEED)))
    if dataset in GRAM_NHANES:   # realdata_run.discrepancy_path (profiles not standardized)
        sig_pooled = float(median_heuristic(Fv[g == 0], Fv[g == 1],
                                            rng=np.random.default_rng(0)))
    else:                            # fair_case_study.featurize, seed 0
        sig_pooled = float(median_heuristic(S_full[g == 0], S_full[g == 1],
                                            rng=np.random.default_rng(0)))
    out = dict(dataset=dataset, Fv=Fv, g=g, tune=tune, S_full=S_full,
               S_pil=S_pil, sig_tune=sig_tune, sig_tune_strict=sig_tune_strict,
               sig_pooled=sig_pooled, standardize=standardize)
    _PREP[dataset] = out
    return out


def build_qp(P, S, mask, sigma, w_seed):
    A, B = S[mask & (P["g"] == 0)], S[mask & (P["g"] == 1)]
    if P["dataset"] in GRAM_DATASETS:
        return GramQP(A, B, sigma)
    if w_seed == "pooled":
        # identical features to the paper's ACS run (fair_case_study.featurize)
        Z, sig, _ = fc.featurize(pd.DataFrame(P["Fv"]), P["g"], N_FREQ, 0, 1.0)
        assert abs(sig - sigma) < 1e-12
        return FeatureQP(Z[P["g"] == 0], Z[P["g"] == 1])
    W = np.random.default_rng(w_seed).standard_normal((S.shape[1], N_FREQ)) / sigma
    return FeatureQP(fc.cossin_features(A, W), fc.cossin_features(B, W))


def job_key(spec):
    return "|".join(f"{k}={spec[k]}" for k in sorted(spec))


def run_job(spec):
    t0 = time.time()
    P = prepare(spec["dataset"])
    ones = np.ones(len(P["g"]), bool)
    mask = {"all": ones, "tune": P["tune"], "test": ~P["tune"]}[spec["subset"]]
    S = P["S_pil"] if spec["strict"] else P["S_full"]
    if spec["route"] == "pooled_median":
        sigma, w_seed = P["sig_pooled"], "pooled"
    else:
        base = P["sig_tune_strict"] if spec["strict"] else P["sig_tune"]
        sigma = spec["scale"] * base
        w_seed = W_SEED[spec["wseed"]]
    qp = build_qp(P, S, mask, sigma, w_seed)
    tag = f"[{spec['dataset']} {spec['route']} s={spec['scale']} {spec['subset']}]"

    def log(r):
        print(f"  {tag} a={r['alpha']:.4f} Dlow={r['d_lower']:.4f} "
              f"Dup={r['d_upper']:.4f} tau={r['tau']:.4f} "
              f"{'REJECT' if r['reject'] else 'accept'} ({r['reason']}, "
              f"it={r['n_iter']}) {time.time() - t0:.0f}s", flush=True)

    res = certify(qp, spec["delta"], log, threshold=spec.get("threshold", "paper"))
    res.update(spec)
    res.update(sigma=float(sigma), n=qp.n, m=qp.m, backend=qp.kind,
               seconds=round(time.time() - t0, 1), key=job_key(spec))
    return res


def make_jobs(dataset, strict=True):
    J_ = []

    def add(route, scale, subset, delta, wseed, phase, strict_=False):
        J_.append(_tag(dict(dataset=dataset, route=route, scale=scale,
                            subset=subset, delta=delta, wseed=wseed, phase=phase,
                            strict=strict_)))

    add("pooled_median", 1.0, "all", DELTA, "pooled", 0)
    if not PILOT_ONLY:
        for s in SCALES:
            add("union", s, "all", DELTA / J, "union", 0)
    for s in SCALES:
        add("pilot_tune", s, "tune", DELTA, "tune", 0)
    if strict:
        if not PILOT_ONLY:
            for s in SCALES:
                add("union_heldout", s, "test", DELTA / J, "union", 2, True)
        if dataset in STANDARDIZED:
            for s in SCALES:
                add("pilot_strict_tune", s, "tune", DELTA, "tune", 2, True)
    return J_


def _tag(spec):
    """v2 jobs carry the threshold (so their cache keys differ from v1's);
    v1 specs are left untouched, so their keys are unchanged."""
    if THRESHOLD != "paper":
        spec["threshold"] = THRESHOLD
    return spec


def select_scale(results, route):
    """Adult rule: largest certified level on the pilot, ties towards 0.5."""
    cand = [r for r in results if r["route"] == route]
    best = max(cand, key=lambda r: (r["certified_fine"], -abs(r["scale"] - 0.5)))
    return best["scale"]


def select_scale_d0(results, route):
    """Sensitivity: largest untrimmed discrepancy D0 on the pilot (a
    continuous criterion, which does not tie when the pilot certifies
    nothing at any bandwidth); ties towards 0.5."""
    cand = [r for r in results if r["route"] == route]
    best = max(cand, key=lambda r: (r["D0"], -abs(r["scale"] - 0.5)))
    return best["scale"]


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["nhanes_child", "acs"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-strict", action="store_true")
    # "aos" (AoS restructure, 2026-11-21): min(tau^P closed form, tau^A), threshold.tau_aos; default unchanged
    ap.add_argument("--threshold", choices=["paper", "plugin", "aos"], default="paper")
    ap.add_argument("--pilot-only", action="store_true",
                    help="v3: skip the union routes for the data sets run (cached ones are still summarized)")
    ap.add_argument("--outdir", default=None,
                    help="write r12_* to <outdir>/tables, cache in <outdir>/checkpoints")
    a = ap.parse_args()
    global THRESHOLD, TABLES, CACHE, FAIR, PILOT_ONLY
    THRESHOLD = a.threshold
    PILOT_ONLY = a.pilot_only
    if a.outdir is not None:
        od = Path(a.outdir)
        od = od if od.is_absolute() else ROOT / od
        TABLES, CACHE, FAIR = od / "tables", od / "checkpoints" / "r12_valid_bandwidth", od / "fair"
        TABLES.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    store = CACHE / "jobs.jsonl"
    done = {}
    if store.exists():
        for line in store.read_text().splitlines():
            r = json.loads(line)
            if r.get("threshold", "paper") != THRESHOLD:
                continue            # a run with the other threshold shares this file
            done[r["key"]] = r
        print(f"resuming: {len(done)} jobs on disk", flush=True)

    def record(r):
        done[r["key"]] = r
        with open(store, "a") as fh:
            fh.write(json.dumps(r) + "\n")
        print(f"DONE {r['dataset']} {r['route']} s={r['scale']}: "
              f"H00 {'rejected' if r['H00_rejected'] else 'not rejected'}, "
              f"alpha_hat={r['alpha_hat']}, alpha_last={r['alpha_last']}, "
              f"certified_fine={r['certified_fine']:.4f} ({r['seconds']}s)",
              flush=True)

    all_jobs = [j for d in a.datasets for j in make_jobs(d, not a.no_strict)]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        def submit(jobs):
            futs = []
            for j in jobs:
                if job_key(j) not in done:
                    futs.append(ex.submit(run_job, j))
            return futs

        # phase 0: pooled median, union on all data, pilot tuning
        futs = submit([j for j in all_jobs if j["phase"] == 0])
        pending = set(futs)
        test_submitted = set()
        while True:
            # phase 1: once a data set's pilot jobs are in, its test job
            for d in a.datasets:
                for route, troute, strict_ in (("pilot_tune", "pilot_test", False),
                                               ("pilot_strict_tune",
                                                "pilot_strict_test", True)):
                    if (d, troute) in test_submitted:
                        continue
                    if troute == "pilot_strict_test" and (a.no_strict or d not in STANDARDIZED):
                        continue
                    tunes = [r for r in done.values()
                             if r["dataset"] == d and r["route"] == route]
                    if len(tunes) == J:
                        s = select_scale(tunes, route)
                        spec = _tag(dict(dataset=d, route=troute, scale=s,
                                         subset="test", delta=DELTA, wseed="test",
                                         phase=1, strict=strict_))
                        pending |= set(submit([spec]))
                        test_submitted.add((d, troute))
                        if troute == "pilot_test":
                            s0 = select_scale_d0(tunes, route)
                            pending |= set(submit([dict(
                                spec, route="pilot_test_maxD0", scale=s0)]))
                            # strict jobs queue behind the main test job
                            pending |= set(submit([j for j in all_jobs
                                                   if j["dataset"] == d
                                                   and j["phase"] == 2]))
            if not pending:
                break
            fin, pending = wait(pending, return_when="FIRST_COMPLETED")
            for f in fin:
                record(f.result())

    results = [r for r in done.values() if r["dataset"] in a.datasets
               and r.get("threshold", "paper") == THRESHOLD]
    write_outputs(results, a, time.time() - t0)


def write_outputs(results, a, seconds):
    per_scale, summary = [], []
    for r in results:
        per_scale.append({k: r[k] for k in (
            "dataset", "route", "scale", "subset", "strict", "delta", "sigma",
            "n", "m", "backend", "H00_rejected", "alpha_hat", "alpha_last",
            "certified_fine", "D0", "tau0", "seconds")}
            | {"unresolved_levels": ";".join(map(str, r["unresolved_levels"]))})
    ps = pd.DataFrame(per_scale).sort_values(["dataset", "route", "scale"])
    ps.to_csv(TABLES / "r12_valid_bandwidth_per_scale.csv", index=False)

    paper = paper_reference()
    for d in sorted({r["dataset"] for r in results}):
        R = [r for r in results if r["dataset"] == d]

        def row(route_label, r, delta, scale_note, extra=None):
            out = dict(dataset=d, route=route_label, bandwidth=scale_note,
                       sigma=r["sigma"], delta_per_test=delta, n=r["n"],
                       m=r["m"], H00_rejected=r["H00_rejected"],
                       alpha_hat=r["alpha_hat"], alpha_last=r["alpha_last"],
                       certified_fine=r["certified_fine"], D0=r["D0"],
                       tau0=r["tau0"],
                       unresolved=";".join(map(str, r["unresolved_levels"])))
            out.update(extra or {})
            return out

        pm = [r for r in R if r["route"] == "pooled_median"]
        if pm:
            summary.append(row("pooled median (paper)", pm[0], DELTA,
                               "pooled median"))
        for troute, label in (("pilot_test", "pilot (tune-then-test)"),
                              ("pilot_strict_test",
                               "pilot, strict (pilot standardization)"),
                              ("pilot_test_maxD0",
                               "pilot, selection by largest pilot D0 "
                               "(sensitivity)")):
            t = [r for r in R if r["route"] == troute]
            if t:
                summary.append(row(label, t[0], DELTA,
                                   f"{t[0]['scale']} x pilot median"))
        for uroute, label in (("union", "union over 7 bandwidths, all data"),
                              ("union_heldout",
                               "union over 7 bandwidths, held-out 70% (strict)")):
            U = [r for r in R if r["route"] == uroute]
            if len(U) == J:
                U = sorted(U, key=lambda r: r["scale"])
                top = max(r["certified_fine"] for r in U)
                tied = [r for r in U if r["certified_fine"] == top]
                best = tied[0]
                # the union certificate: max over bandwidths, level delta
                summary.append(row(label, best, DELTA / J,
                                   "best = " + " and ".join(
                                       f"{r['scale']}" for r in tied)
                                   + " x pilot median",
                                   {"H00_rejected": any(u["H00_rejected"] for u in U),
                                    "alpha_last": max((u["alpha_last"] or 0.0)
                                                      for u in U)
                                    if any(u["H00_rejected"] for u in U) else None,
                                    "alpha_hat": max(u["alpha_hat"] for u in U),
                                    "certified_fine": max(u["certified_fine"]
                                                          for u in U)}))
    sm = pd.DataFrame(summary)
    if THRESHOLD != "paper":
        sm["threshold"] = THRESHOLD
    sm.to_csv(TABLES / "r12_valid_bandwidth.csv", index=False)

    meta = {
        "script": "scripts/exp_valid_bandwidth.py",
        "command": "PYTHONPATH=src python3 -I scripts/exp_valid_bandwidth.py "
                   f"--workers {a.workers} --threads {a.threads}"
                   + (" --no-strict" if a.no_strict else "")
                   + (f" --threshold {a.threshold}" if a.threshold != "paper" else "")
                   + (f" --outdir {a.outdir}" if a.outdir is not None else "")
                   + (" --pilot-only" if a.pilot_only else "")
                   + " --datasets " + " ".join(a.datasets),
        "datasets": a.datasets, "delta": DELTA, "scales": list(SCALES), "J": J,
        "tune_frac": TUNE_FRAC, "split_seed": SPLIT_SEED,
        "feature_seeds": W_SEED, "strict_median_seed": STRICT_MEDIAN_SEED,
        "n_freq_acs": N_FREQ, "acs_subsample": [ACS_ROWS, ACS_SEED],
        "alpha_grid": GRID.tolist(), "refine_tol": REFINE_TOL,
        "max_iter": MAX_ITER, "tol": TOL,
        "decision": "reject alpha iff Frank-Wolfe lower bound on min QP > "
                    "tau^2; FISTA stops early once the decision is settled",
        "paper_reference": paper,
        "jobs": [{k: v for k, v in r.items()} for r in results],
        "seconds_this_invocation": round(seconds, 1),
        "seconds_jobs_total": round(sum(r["seconds"] for r in results), 1),
        "numpy": np.__version__, "python": platform.python_version(),
        "workers": a.workers, "threads_per_worker": a.threads,
    }
    if THRESHOLD != "paper":
        meta["threshold"] = THRESHOLD
        meta["kernel_constants"] = (
            "nhanes_child exact Gram: Gaussian coupled (kappa 1, Dphi sqrt2, v2 1, rho sqrt2); "
            "acs cos/sin features: bounded (kappa 1, Dphi = rho = 2, v2 1, no coupling)")
    (TABLES / "r12_valid_bandwidth.json").write_text(json.dumps(meta, indent=1))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(sm.drop(columns=["sigma", "D0", "tau0"]).to_string(index=False))


def paper_reference():
    """The published pooled-median numbers, read from their result files."""
    out = {}
    try:
        s = pd.read_csv(TABLES / "r1_summary.csv").set_index("dataset")
        p = pd.read_csv(TABLES / "r1_paths.csv")
        p = p[p.dataset == "nhanes_child"]
        out["nhanes_child"] = {
            "source": (f"{TABLES.relative_to(ROOT)}/r1_summary.csv, r1_paths.csv"
                       if TABLES.is_relative_to(ROOT) else f"{TABLES}/r1_summary.csv"),
            "sigma": float(s.loc["nhanes_child", "sigma"]),
            "alpha_hat": float(s.loc["nhanes_child", "alpha_hat_analytic"]),
            "alpha_last": float(p[p.rejects].alpha.max()),
            "D0": float(p.d_hat.iloc[0]), "tau0": float(p.analytic_tau.iloc[0])}
    except Exception as exc:  # pragma: no cover
        out["nhanes_child"] = f"unavailable: {exc}"
    try:
        info = json.loads((FAIR / "acs_info.json").read_text())
        out["acs"] = {"source": str(FAIR.relative_to(ROOT) / "acs_info.json")
                      if FAIR.is_relative_to(ROOT) else str(FAIR / "acs_info.json"),
                      "sigma": info["sigma"], "alpha_hat": info["alpha_hat"],
                      "alpha_last": info["last_rejected"],
                      "D0": info["path"][0]["d_hat"],
                      "tau0": info["path"][0]["tau"]}
    except Exception as exc:  # pragma: no cover
        out["acs"] = f"unavailable: {exc}"
    return out


if __name__ == "__main__":
    main()

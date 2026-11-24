"""AoS revision: the baselines the referees asked for, on the replicates of the synthetic study.

The synthetic study (explore/harder_sims, read-only here) stored, for 19 designs and 4 P = Q nulls, 200 replicates per
design and size of 14 tests of H_{0,alpha}: TV(P,Q) <= alpha at alpha_j = j alpha*/8, j = 0..10, delta = 0.05. The
paper's "betting, ramp" is the study's seq_bet: batch 1 bets on the second quarter of each sample with a witness fitted
on the first quarter, batch 2 bets on the second half with the witness fitted on the first half, one wealth (about 3n/4
bets). Its classifiers (gbm_bet, logit_bet) are fitted once on the first half and bet hard labels on the second half.

Job kind "main" regenerates the SAME data, folds and classifier seeds as methods.run_replicate
(SeedSequence([crc32(design), n, rep])), takes the study's fold bandwidths from the stored replicate (recomputing them
agrees to rounding; field repro.sigma), and adds on every replicate:

* perm_pilot, perm_med: permutation test of P = Q with the V-statistic MMD^2 (the statistic of the analytic test at
  alpha = 0), Gaussian kernel at the analytic test's pilot bandwidth (perm_pilot) or at the median distance of the
  pooled sample (perm_med; a symmetric function of the pooled data, so the test stays exact), B = 999 random
  relabellings, p = (1 + #{T_b >= T}) / (B + 1), reject iff p <= delta. Tests H_{0,0} only.
* mmdagg: the aggregated MMD test of Schrab, Kim, Albert, Laurent, Guedj and Gretton (JMLR 2023, Algorithm 1),
  implemented here as their MMDAgg* (Sec. 5.2): Gaussian kernels exp(-|x-y|^2 / lambda^2) with the parameter-free
  collection of N = 10 bandwidths on [lambda_min / 2, 2 lambda_max] from the inter-sample distances, uniform weights,
  quadratic-time U-statistics, wild bootstrap (n = m) or permutations (n != m), quantile index
  ceil((B1 + 1)(1 - u w)) with the original statistic added, B3 = 50 bisection steps; B1 = B2 = 500 (the values of
  their Sec. 5.1 experiments; MMDAgg* uses 2000, too costly here). Tests H_{0,0} only.
* ramp0: the untrimmed ramp. Witness fitted at alpha = 0 (difference of the mean embeddings of the fit block) AND cut
  points at alpha = 0 (max of the self-excluded fit values on X, min on Y), the same ramp bet at every level, on the
  ramp's batch schedule. At j = 0 it is the paper's ramp (checked: field repro.seq0).
* ramp0c: the alpha = 0 witness with the cut points of the tested level (alpha-quantiles of its self-excluded fit
  values), i.e. the paper's ramp with only the witness QP left untrimmed.
* svm, svm05: the normal of a nu-SVM (scikit-learn NuSVC, i.e. libsvm, on the same Gaussian Gram matrix as the ramp:
  the batch's median bandwidth) as the witness, cut points of the tested level, same schedule and priors as the ramp.
  svm: nu chosen by 3-fold stratified CV on the fit block (balanced accuracy, nu in 0.1..0.9, feasible values only);
  svm05: nu = 0.5 (scikit-learn's default; the largest feasible grid value below it when 0.5 is infeasible).
  For n = m the nu-SVM normal is the trimmed witness at level 1 - nu, so svm differs from the ramp only in fitting
  the witness at one cross-validated level instead of at the tested level. svm_hard: its labels (decision > 0).
* {gbm,logit}_m_{hard,soft}: the paper's classifiers (scikit-learn defaults, as the study) refitted on the ramp's
  batch schedule (quarter 1 -> quarter 2, half 1 -> half 2, one wealth, horizon fixed in advance), betting either the
  hard label (X iff p_hat > training share of X, the study's rule) or the prior-corrected probability
  p~ = (p/pi)/((p/pi) + (1-p)/(1-pi)) in [0, 1] (posthoc_soft.balanced); neutral prior variance 0.5 as in the study.
* {gbm,logit}_mt_{hard,soft}: the same, tuned on each fit block by 3-fold stratified CV on log-loss (logistic: C in
  10^{-3..2} after standardizing; gradient boosting: max_leaf_nodes in {4, 31} x number of iterations 10..150 by
  staged predictions, learning rate 0.1), refitted on the whole fit block; the prior variance of the bets is computed
  from the out-of-fold predictions of the selected configuration (as the ramp's prior comes from its fit block).
* {gbm,logit}_ss_{hard,soft}: the single-split classifiers of the paper (fit on half 1, bet on half 2), recomputed
  (the hard bets reproduce the stored gbm_bet / logit_bet up to the scikit-learn version, field repro.hard_*).
* hplb_{bayes,adapt}_{gbm,logit}: the high-probability lower bounds of Michel, Naef and Meinshausen (arXiv:2005.06006)
  on the out-of-sample scores of the same single-split classifiers: lambda_bayes (their Prop. 3: in-class accuracies
  at the cutoff minus the normal quantile times the plug-in s.d.; asymptotic) and lambda_adapt (their Prop. 6-7: the
  counting process V_z over all cutoffs against the bounding function (17), with the simultaneous band of the
  hypergeometric process simulated, S = 2000, on a lambda grid of mesh 0.005 plus the test levels). H_{0,alpha_j} is
  rejected iff lambda_hat > alpha_j.
* ks (1-d designs only): sup_t |F_n - G_m| minus the two-sided DKW-Massart radii sqrt(log(4/delta)/(2n)) +
  sqrt(log(4/delta)/(2m)); H_{0,alpha_j} rejected iff this lower bound exceeds alpha_j.

Job kind "univ" runs the study's own methods.run_replicate (analytic tests, split-sample certificates incl. the
single-split empirical-Bernstein ramp_b2s, classifiers) on two new 1-d designs, U1_shift (the overlap design in d = 1:
{p > q} is a half-line, so KS = TV) and U1_t3scale (the t3 scale design in d = 1: {p > q} is an interval, so
KS = TV/2), at n = m = 100..800, plus the KS bound. Job kind "ks" adds the KS bound to the stored 1-d designs D8_N*
at the sizes below 800.

Every betting method is evaluated at all 11 levels (dec_all) and its lower bound is the down-closed jlo. Nothing here
changes a stored replicate; outputs go to results/aos/baselines/raw/<design>_n<n>[_kind].jsonl, appended and fsynced
after every replicate, so a killed run resumes ((n, rep) pairs already present are skipped).

    ~/.venvs/llm_book/bin/python -B scripts/aos_baselines_run.py calibrate --workers 6
    nohup ~/.venvs/llm_book/bin/python -B scripts/aos_baselines_run.py run --workers 6 > ... &
"""

from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True          # never write __pycache__ into explore/ (read-only)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
import zlib  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HS = ROOT / "explore" / "harder_sims"
for _p in (HS, ROOT / "src", ROOT / "explore" / "sharper_threshold", ROOT / "explore" / "split_adaptive"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
from scipy.stats import binom, ks_2samp, norm  # noqa: E402

import designs as DS  # noqa: E402
import methods as M  # noqa: E402
import sacert as S  # noqa: E402
from fold1 import cut_points, pair_var, priors  # noqa: E402
from trimmedmmd.kernels import median_heuristic, sqdist  # noqa: E402

OUT = ROOT / "results" / "aos" / "baselines"
# Working copy of the raw replicates. Appending to files on the pCloud FUSE mount corrupted them (blocks of NUL bytes
# replaced ~40 records per file at 09:1x on 2026-11-21), so runs append to a local directory (AOS_RAW) and `sync`
# copies every file atomically (write + rename, then re-read) to results/aos/baselines/raw.
RAW = Path(os.environ.get("AOS_RAW", str(OUT / "raw")))
RAW_OUT = OUT / "raw"
STORED = HS / "out"
PILOT = STORED / "pilot.json"
THR = STORED / "thr_cache.json"
PILOT_U = RAW / "pilot_univ.json"
DELTA = M.DELTA
B_PERM = 999
PERM_TAG = 20261121                       # extra entropy word: permutation stream, disjoint from the study's streams
AGG_TAG = 20261122                        # MMDAgg bootstrap stream
HPLB_TAG = 20261123                       # HPLB tie-breaking stream
AGG_L, AGG_B1, AGG_B2, AGG_B3 = 10, 500, 500, 50
HPLB_S = 2000
NU_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
LEAVES = (4, 31)
GBM_ITERS = 150
CLF = ("gbm", "logit")
DESIGNS = ["D9_rhc", "D1_d2_a10", "D1_d2_a20", "D1_d10_a10", "D1_d10_a20", "D2_d20", "D2_d50", "D2_d100", "D3a_loc",
           "D3b_corr", "D4_small", "D5b_mixed", "D6_t3", "D7_wide", "D7_narrow", "D8_N1", "D8_N4", "D8_N16", "D8_N64"]
NULLS = ["N0_D5b_mixed", "N0_D2_d100", "N0_D6_t3", "N0_D7_narrow"]
UNIV = ["U1_shift", "U1_t3scale"]
UNIV_NS = (100, 200, 400, 800)
KS_DESIGNS = ["D8_N1", "D8_N4", "D8_N16", "D8_N64"]
H00_NS = (100, 200, 400)           # job kind "h00": the study's other sizes (800 is in "main")
H00X_NS = (1600,)                  # job kind "h00x": beyond the grid, for designs whose power is below 0.8 at 800


def seed_key(key):
    return zlib.crc32(key.encode())


def catalogue():
    """The study's catalogue plus the two new 1-d designs (built with the study's classes, so their alpha* floats and
    level grids coincide with D1_d2_a20 and D6_t3 and the study's threshold cache applies)."""
    C = DS.catalogue()
    C["U1_shift"] = DS.GaussLoc("U1_shift", "U1 univariate", 1, 0.25, [DS.Z90], note="d = 1 overlap: KS = TV")
    C["U1_t3scale"] = DS.HeavyScale("U1_t3scale", "U1 univariate", 1, 0.5, 3.0, note="d = 1 t3 scale: KS = TV/2")
    return C


def main_n(key, C):
    """The size of Figure 2(a): n = 800, or the largest planned size (D5b: 400, m = 5n)."""
    dz = C[key]
    return 800 if 800 in dz.ns else max(dz.ns)


# ---------------------------------------------------------------------------
# scores on a fit block -> evaluation block
# ---------------------------------------------------------------------------

def balanced(p, pi1):
    """posthoc_soft.balanced: prior-corrected posterior, an estimate of p/(p+q)."""
    a = p / pi1
    b = (1.0 - p) / (1.0 - pi1)
    return a / np.maximum(a + b, 1e-300)


def witness0(Kfit, Kev, nf, mf):
    """alpha = 0 witness (mean-embedding difference of the fit block), normalized as methods.witness_values:
    self-excluded fit values (u on X, v on Y) and evaluation values."""
    z = np.r_[np.full(nf, 1.0 / nf), np.full(mf, -1.0 / mf)]
    ffit = Kfit @ z
    norm_ = math.sqrt(max(float(z @ ffit), 0.0))
    if norm_ <= 1e-12:
        return None
    ffit = ffit / norm_
    u = ffit[:nf] - (1.0 / nf) / norm_
    v = ffit[nf:] + (1.0 / mf) / norm_
    return u, v, (Kev @ z) / norm_


def nusvc_fit(K, y, nu):
    from sklearn.svm import NuSVC
    clf = NuSVC(nu=nu, kernel="precomputed")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(K, y)
    assert list(clf.classes_) == [0, 1]
    c = np.zeros(K.shape[0])
    c[clf.support_] = clf.dual_coef_[0]           # y_i alpha_i (libsvm scaling); decision > 0 <=> class 1 (X)
    return c, float(clf.intercept_[0])


def nu_feasible(y):
    n1 = int(y.sum())
    return 2.0 * min(n1, y.size - n1) / y.size


def nusvm(Kfit, Kev, nf, mf, seed, mode):
    """nu-SVM normal as witness. mode 'cv' (3-fold CV, balanced accuracy) or 'default' (nu = 0.5)."""
    from sklearn.model_selection import StratifiedKFold
    y = np.r_[np.ones(nf, int), np.zeros(mf, int)]
    folds = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(np.zeros(y.size), y))
    cap = min([nu_feasible(y)] + [nu_feasible(y[tr]) for tr, _ in folds]) * 0.99
    grid = [nu for nu in NU_GRID if nu < cap]
    cvs = None
    if mode == "default":
        nu = 0.5 if 0.5 in grid else max(grid)
    else:
        cvs = {}
        for nu in grid:
            acc = []
            try:
                for tr, te in folds:
                    c, b = nusvc_fit(Kfit[np.ix_(tr, tr)], y[tr], nu)
                    d = Kfit[np.ix_(te, tr)] @ c + b
                    yt = y[te]
                    acc.append(0.5 * (np.mean(d[yt == 1] > 0) + np.mean(d[yt == 0] <= 0)))
            except ValueError:
                continue
            cvs[nu] = float(np.mean(acc))
        nu = max(cvs, key=lambda k: (cvs[k], -k))      # best CV balanced accuracy, smaller nu on ties
    c, b = nusvc_fit(Kfit, y, nu)
    ffit = Kfit @ c
    u = ffit[:nf] - c[:nf]                              # self-excluded (k(x, x) = 1)
    v = ffit[nf:] - c[nf:]
    fev = Kev @ c
    return {"u": u, "v": v, "fev": fev, "lab_ev": (fev + b > 0).astype(float), "nu": nu, "cv": cvs}


def make_clf(name, seed, **kw):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name == "gbm":
        return HistGradientBoostingClassifier(random_state=seed, **kw)
    if name == "logit":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, **({"C": 1.0} | kw)))
    raise KeyError(name)


def fit_predict(clf, Z1, lab, Zev):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(Z1, lab)
        c = list(clf.classes_).index(1)
        return clf.predict_proba(Zev)[:, c]


def logloss(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def tuned_probs(name, Xf, Yf, Xe, Ye, seed):
    """3-fold CV on the fit block (log-loss); returns evaluation probabilities of the refit, out-of-fold
    probabilities of the selected configuration on the fit block, and the selection."""
    from sklearn.model_selection import StratifiedKFold
    nf = len(Xf)
    Z1 = np.vstack([Xf, Yf])
    y = np.r_[np.ones(nf, int), np.zeros(len(Yf), int)]
    folds = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(Z1, y))
    if name == "logit":
        oof = {C: np.zeros(y.size) for C in C_GRID}
        for tr, te in folds:
            for C in C_GRID:
                oof[C][te] = fit_predict(make_clf("logit", seed, C=C), Z1[tr], y[tr], Z1[te])
        loss = {C: logloss(oof[C], y) for C in C_GRID}
        best = min(loss, key=lambda k: (loss[k], k))
        p_ev = fit_predict(make_clf("logit", seed, C=best), Z1, y, np.vstack([Xe, Ye]))
        sel = {"C": best}
        p_oof = oof[best]
    else:
        oof = {L: np.zeros((GBM_ITERS, y.size)) for L in LEAVES}
        for tr, te in folds:
            for L in LEAVES:
                clf = make_clf("gbm", seed, max_leaf_nodes=L, max_iter=GBM_ITERS, learning_rate=0.1,
                               early_stopping=False)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf.fit(Z1[tr], y[tr])
                    c = list(clf.classes_).index(1)
                    for it, pp in enumerate(clf.staged_predict_proba(Z1[te])):
                        oof[L][it, te] = pp[:, c]
        best, bl = None, np.inf
        for L in LEAVES:
            for it in range(9, GBM_ITERS, 10):          # 10, 20, ..., 150 iterations
                ll = logloss(oof[L][it], y)
                if ll < bl - 1e-12:
                    best, bl = (L, it + 1), ll
        L, iters = best
        p_ev = fit_predict(make_clf("gbm", seed, max_leaf_nodes=L, max_iter=iters, learning_rate=0.1,
                                    early_stopping=False), Z1, y, np.vstack([Xe, Ye]))
        sel = {"leaves": L, "iters": iters}
        p_oof = oof[L][iters - 1]
    return p_ev[: len(Xe)], p_ev[len(Xe):], p_oof[:nf], p_oof[nf:], sel


# ---------------------------------------------------------------------------
# betting on a batch stream
# ---------------------------------------------------------------------------

def ramp_batch(u, v, fx, fy, a_cut, nbx, nby):
    """Ramp between the a_cut-quantile cut points of the self-excluded fit values; (G, H, prior var) or None."""
    s, sp = cut_points(u, v, a_cut)
    if not s > sp:
        return None
    G, H = S.ramp(fx, s, sp), S.ramp(fy, s, sp)
    pv = priors(u, v, s, sp, a_cut, nbx, nby)["ramp"]
    return G, H, pv


def stream_decisions(batch_fn, levels, N):
    """batch_fn(j, a) -> list of (G, H, prior var) or None per batch. Decisions at every level and the
    down-closed jlo."""
    dec = []
    for j, a in enumerate(levels):
        bs = batch_fn(j, a)
        Zs = [None if b is None else S.pair_up(b[0], -b[1]) - a for b in bs]
        pvs = [None if b is None else b[2] for b in bs]
        dec.append(int(M.bet_stream(Zs, pvs, a, N)[0]))
    return dec, jlo_of(dec)


def single_decisions(G, H, levels, pv=0.5):
    """The study's single-split bet (classifier_tests / posthoc_soft.soft_tests) at every level."""
    Zb = S.pair_up(G, -H)
    dec = [int(S.betting_test(Zb - a, -(1.0 + a), DELTA, c=0.5, prior_var=pv)[0]) for a in levels]
    return dec, jlo_of(dec)


def jlo_of(dec):
    j = -1
    for d in dec:
        if not d:
            break
        j += 1
    return j


def bound_decisions(lb, levels):
    """A lower bound lb on TV: H_{0,alpha_j} rejected iff lb > alpha_j."""
    dec = [int(lb > a) for a in levels]
    return dec, jlo_of(dec)


# ---------------------------------------------------------------------------
# MMDAgg (Schrab et al. 2023, Algorithm 1), Gaussian kernels, uniform weights
# ---------------------------------------------------------------------------

def agg_bandwidths(D2, n, L=AGG_L):
    """MMDAgg* collection (Schrab et al. 2023, Sec. 5.2): D = inter-sample distances |x - y|, x in X, y in Y;
    lambda_min = min D, or the 5% quantile of D if that is < 0.1, floored at 0.1; lambda_max = max D (at least
    0.3); N = 10 bandwidths spaced geometrically on [lambda_min / 2, 2 lambda_max]."""
    Dxy = np.sqrt(D2[:n, n:]).ravel()
    lmin = float(Dxy.min())
    if lmin < 0.1:
        lmin = max(float(np.quantile(Dxy, 0.05)), 0.1)
    lmax = max(float(Dxy.max()), 0.3)
    return np.geomspace(lmin / 2.0, 2.0 * lmax, L)


def mmdagg(D2, n, m, rng, alpha=DELTA, L=AGG_L, B1=AGG_B1, B2=AGG_B2, B3=AGG_B3):
    """MMDAgg (Algorithm 1) with the Gaussian kernel of the paper, exp(-|x - y|^2 / lambda^2), uniform weights."""
    lams = agg_bandwidths(D2, n, L)
    w = np.full(L, 1.0 / L)
    B = B1 + B2
    stats = np.empty((L, B + 1))                       # column 0: the original statistic
    if n == m:                                         # wild bootstrap on h_ij (U-statistic, n = m)
        E = rng.choice([-1.0, 1.0], size=(n, B))
        E = np.hstack([np.ones((n, 1)), E])
        for i, lam in enumerate(lams):
            K = np.exp(-D2 / lam ** 2)
            H = K[:n, :n] + K[n:, n:] - K[:n, n:] - K[n:, :n]
            np.fill_diagonal(H, 0.0)
            stats[i] = np.einsum("ib,ib->b", E, H @ E) / (n * (n - 1))
    else:                                              # permutations of the pooled sample
        N = n + m
        e0 = np.r_[np.ones(n), np.zeros(m)]
        E = np.hstack([e0[:, None], rng.permuted(np.tile(e0, (B, 1)), axis=1).T])
        for i, lam in enumerate(lams):
            K = np.exp(-D2 / lam ** 2)
            r = K.sum(1)
            tot = r.sum()
            sxx = np.einsum("ib,ib->b", E, K @ E)
            er = E.T @ r
            sxy = er - sxx
            syy = tot - 2.0 * er + sxx
            stats[i] = (sxx - n) / (n * (n - 1)) + (syy - m) / (m * (m - 1)) - 2.0 * sxy / (n * m)
    obs = stats[:, 0]
    q_set = np.sort(stats[:, : B1 + 1], axis=1)        # B1 bootstrap values plus the original
    second = stats[:, B1 + 1:]

    def quant(u):
        k = np.ceil((B1 + 1) * (1.0 - u * w)).astype(int)
        k = np.clip(k, 1, B1 + 1)
        return q_set[np.arange(L), k - 1]

    def prob(u):
        return float(np.mean(np.any(second > quant(u)[:, None], axis=0)))

    ulo, uhi = 0.0, float(np.min(1.0 / w))
    for _ in range(B3):
        mid = 0.5 * (ulo + uhi)
        if prob(mid) <= alpha:
            ulo = mid
        else:
            uhi = mid
    rej = bool(np.any(obs > quant(ulo)))
    return rej, {"u": ulo, "lams": [float(x) for x in lams]}


# ---------------------------------------------------------------------------
# HPLB (Michel, Naef, Meinshausen, arXiv:2005.06006)
# ---------------------------------------------------------------------------

_BAND = {}


def hg_band(mt, nt, alpha, S_=HPLB_S):
    """(1 - alpha) quantile of sup_k (V_k - k p) / w(k) for the hypergeometric process with mt circles and nt
    squares (simulated, deterministic seed); the simultaneous band of (17) is k p + c w(k)."""
    key = (mt, nt, alpha)
    if key not in _BAND:
        Nt = mt + nt
        rng = np.random.default_rng([mt, nt, HPLB_TAG])
        k = np.arange(1, Nt)
        p = mt / Nt
        wk = np.sqrt(mt * nt / Nt ** 2 * (Nt - k) / (Nt - 1) * k)
        lab = np.r_[np.ones(mt), np.zeros(nt)]
        sups = []
        for s0 in range(0, S_, 250):
            P_ = rng.permuted(np.tile(lab, (min(250, S_ - s0), 1)), axis=1)
            V = np.cumsum(P_, axis=1)[:, :-1]
            sups.append(np.max((V - k * p) / wk, axis=1))
        _BAND[key] = float(np.quantile(np.concatenate(sups), 1.0 - alpha, method="higher"))
    return _BAND[key]


def hplb_lambda_grid(levels):
    top = max(levels)
    g = set(np.round(np.arange(0.0, top + 1e-12, 0.005), 10).tolist()) | {float(a) for a in levels}
    return sorted(g)


def hplb_adapt(rhoX, rhoY, lam_grid, rng, alpha=DELTA):
    """lambda_adapt of Prop. 6 with the bounding function (17); rho = score of 'Q', small for P.
    Returns lambda_hat = inf{lam in grid : sup_z (V_z - Q(z, lam)) <= 0}."""
    m, n = rhoX.size, rhoY.size                        # m P-points, n Q-points (their notation)
    N = m + n
    key = np.r_[rhoX, rhoY]
    order = np.lexsort((rng.random(N), key))           # ascending rho, random tie-breaking
    V = np.cumsum(np.r_[np.ones(m), np.zeros(n)][order])[:-1]
    z = np.arange(1, N)
    for lam in lam_grid:
        qm = int(binom.ppf(1.0 - alpha / 3.0, m, lam)) if lam > 0 else 0
        qn = int(binom.ppf(1.0 - alpha / 3.0, n, lam)) if lam > 0 else 0
        mt, nt = m - qm, n - qn
        if mt < 1 or nt < 1:
            return float(lam)
        Nt = mt + nt
        Q = np.where(z <= qm, z.astype(float), float(m))
        mid = (z > qm) & (z < m + nt)
        k = z[mid] - qm
        wk = np.sqrt(mt * nt / Nt ** 2 * (Nt - k) / (Nt - 1) * k)
        Q[mid] = qm + k * mt / Nt + hg_band(mt, nt, alpha / 3.0) * wk
        if not np.any(V > Q + 1e-9):
            return float(lam)
    return float(lam_grid[-1])


def hplb_bayes(G, H, alpha=DELTA):
    """Prop. 3: in-class accuracies at the cutoff minus z_{1-alpha} times the plug-in s.d. (asymptotic)."""
    a0, a1 = float(G.mean()), float(1.0 - H.mean())
    sd = math.sqrt(a0 * (1 - a0) / G.size + a1 * (1 - a1) / H.size)
    return a0 + a1 - 1.0 - float(norm.ppf(1.0 - alpha)) * sd


def ks_bound(X, Y, delta=DELTA):
    n, m = len(X), len(Y)
    ks = float(ks_2samp(X[:, 0], Y[:, 0]).statistic)
    return ks, ks - math.sqrt(math.log(4.0 / delta) / (2 * n)) - math.sqrt(math.log(4.0 / delta) / (2 * m))


# ---------------------------------------------------------------------------
# one replicate
# ---------------------------------------------------------------------------

_C = {}


def _ctx():
    if "C" not in _C:
        _C["C"] = catalogue()
        _C["pil"] = json.loads(PILOT.read_text())
        if PILOT_U.exists():
            _C["pil"].update(json.loads(PILOT_U.read_text()))
    return _C["C"], _C["pil"]


def draw(key, n, rep):
    """Data, permutation and streams exactly as methods.run_replicate."""
    C, PIL = _ctx()
    dz = C[key]
    pil = PIL[key]
    m = dz.m_ratio * n
    ss = np.random.SeedSequence([seed_key(key), n, rep])
    rng = np.random.default_rng(ss)
    X = dz.sample(n, rng, "P")
    Y = dz.sample(m, rng, "Q")
    if pil["scale"] is not None:
        sc = np.array(pil["scale"])
        X, Y = X / sc, Y / sc
    px, py = rng.permutation(n), rng.permutation(m)
    return dz, pil, ss, X[px], Y[py]


def replicate(args):
    kind, key, n, rep, stored = args
    if kind == "univ":
        return replicate_univ(key, n, rep)
    if kind in ("h00", "h00x"):
        return replicate_h00(key, n, rep)
    if kind == "ks":
        dz, pil, ss, X, Y = draw(key, n, rep)
        ks, lb = ks_bound(X, Y)
        dec, jlo = bound_decisions(lb, dz.levels())
        return {"design": key, "n": n, "m": len(Y), "rep": rep, "alpha_star": dz.alpha_star, "levels": dz.levels(),
                "dec_all": {"ks": dec}, "jlo": {"ks": jlo}, "diag": {"ks_stat": ks, "ks_lb": lb}}
    T0 = time.perf_counter()
    dz, pil, ss, X, Y = draw(key, n, rep)
    m = len(Y)
    levels = dz.levels()
    Zall = np.vstack([X, Y])
    D2 = sqdist(Zall, Zall)
    tm = {}
    rec = {"design": key, "n": n, "m": m, "rep": rep, "alpha_star": dz.alpha_star, "levels": levels,
           "dec_all": {}, "jlo": {}, "pval": {}, "diag": {}, "repro": {}, "time": tm}
    N = n + m

    # ---------------- permutation MMD test and MMDAgg (H_{0,0}) ----------------
    t0 = time.perf_counter()
    zl = np.r_[np.full(n, 1.0 / n), np.full(m, -1.0 / m)]
    prng = np.random.default_rng(np.random.SeedSequence([seed_key(key), n, rep, PERM_TAG]))
    Zp = prng.permuted(np.tile(zl, (B_PERM, 1)), axis=1).T          # N x B relabellings
    dist = np.sqrt(D2[np.triu_indices(N, k=1)])
    dist = dist[dist > 0]
    sig_med = float(np.median(dist)) if dist.size else 1.0
    for name, sig in (("perm_pilot", pil["sigma_fix"]), ("perm_med", sig_med)):
        K = np.exp(-D2 / (2.0 * sig ** 2))
        T = float(zl @ (K @ zl))
        Tb = np.einsum("ib,ib->b", Zp, K @ Zp)
        p = (1.0 + float(np.sum(Tb >= T - 1e-12 * abs(T)))) / (B_PERM + 1.0)
        rec["pval"][name] = p
        rec["dec_all"][name] = [int(p <= DELTA)]
        rec["diag"][f"{name}_stat"] = T
        del K
    rec["diag"]["sigma_med"] = sig_med
    del Zp
    tm["perm"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    arng = np.random.default_rng(np.random.SeedSequence([seed_key(key), n, rep, AGG_TAG]))
    rej, info = mmdagg(D2, n, m, arng)
    rec["dec_all"]["mmdagg"] = [int(rej)]
    rec["diag"]["mmdagg"] = info
    del dist
    tm["mmdagg"] = time.perf_counter() - t0
    if dz.d == 1:
        ks, lb = ks_bound(X, Y)
        rec["dec_all"]["ks"], rec["jlo"]["ks"] = bound_decisions(lb, levels)
        rec["diag"].update({"ks_stat": ks, "ks_lb": lb})

    # ---------------- folds, batches and kernels exactly as methods.run_replicate ----------------
    n1, m1 = n // 2, m // 2
    n2, m2 = n - n1, m - m1
    iX1, iX2 = np.arange(n1), np.arange(n1, n)
    iY1, iY2 = n + np.arange(m1), n + np.arange(m1, m)
    f1, f2 = np.r_[iX1, iY1], np.r_[iX2, iY2]
    sig1 = median_heuristic(X[:n1], Y[:m1], rng=np.random.default_rng(ss.spawn(1)[0]))
    q_n, q_m = int(round(n / 4)), int(round(m / 4))
    b0 = np.r_[np.arange(q_n), n + np.arange(q_m)]
    b1 = np.r_[np.arange(q_n, n1), n + np.arange(q_m, m1)]
    nb1x, nb1y = n1 - q_n, m1 - q_m
    sig0 = median_heuristic(X[:q_n], Y[:q_m], rng=np.random.default_rng(ss.spawn(2)[1]))
    N_seq = M.n_pairs(nb1x, nb1y) + M.n_pairs(n2, m2)
    cseed = int(ss.generate_state(1)[0] % (2 ** 31 - 1))
    if stored is not None:
        # the study's own fold bandwidths; a recomputation differs only by floating-point rounding in the
        # distances (mixed-type data with tied points), which can move the median of D5b by ~1e-3
        rec["repro"]["sigma"] = bool(abs(sig1 - stored["sigma1"]) <= 1e-9 * sig1
                                     and abs(sig0 - stored["sigma0"]) <= 1e-9 * sig0)
        rec["repro"]["sigma_close"] = bool(abs(sig1 - stored["sigma1"]) <= 1e-2 * sig1
                                           and abs(sig0 - stored["sigma0"]) <= 1e-2 * sig0)
        sig1, sig0 = stored["sigma1"], stored["sigma0"]
    rec["diag"].update({"sigma1": sig1, "sigma0": sig0, "N_seq": N_seq})
    K11 = np.exp(-D2[np.ix_(f1, f1)] / (2 * sig1 ** 2))
    K21 = np.exp(-D2[np.ix_(f2, f1)] / (2 * sig1 ** 2))
    K00 = np.exp(-D2[np.ix_(b0, b0)] / (2 * sig0 ** 2))
    K10 = np.exp(-D2[np.ix_(b1, b0)] / (2 * sig0 ** 2))
    del D2
    # batch list: (Kfit, Kev, nf, mf, n_ev_x, n_ev_y) -- batch 1 = quarter 1 -> quarter 2, batch 2 = half 1 -> half 2
    KB = [(K00, K10, q_n, q_m, nb1x, nb1y), (K11, K21, n1, m1, n2, m2)]

    # ---------------- alpha = 0 witness ----------------
    t0 = time.perf_counter()
    W0 = []
    for Kf, Ke, nf, mf, nex, ney in KB:
        w = witness0(Kf, Ke, nf, mf)
        W0.append(None if w is None else (w[0], w[1], w[2][:nex], w[2][nex:], nex, ney))

    def rb(W, cut):
        def fn(j, a):
            return [None if w is None else ramp_batch(w[0], w[1], w[2], w[3], a if cut == "level" else 0.0, w[4], w[5])
                    for w in W]
        return fn

    for name, cut in (("ramp0", "zero"), ("ramp0c", "level")):
        rec["dec_all"][name], rec["jlo"][name] = stream_decisions(rb(W0, cut), levels, N_seq)
    if stored is not None:
        rec["repro"]["seq0"] = bool(rec["dec_all"]["ramp0"][0] == stored["dec"]["seq_bet"][0])
    tm["ramp0"] = time.perf_counter() - t0

    # ---------------- nu-SVM ----------------
    t0 = time.perf_counter()
    for name, mode in (("svm", "cv"), ("svm05", "default")):
        W, nus = [], []
        for Kf, Ke, nf, mf, nex, ney in KB:
            r = nusvm(Kf, Ke, nf, mf, cseed, mode)
            W.append((r["u"], r["v"], r["fev"][:nex], r["fev"][nex:], nex, ney, r["lab_ev"]))
            nus.append(r["nu"])
        rec["dec_all"][name], rec["jlo"][name] = stream_decisions(rb(W, "level"), levels, N_seq)
        rec["diag"][f"{name}_nu"] = nus
        if name == "svm":
            hb = [(w[6][:w[4]], w[6][w[4]:], 0.5) for w in W]
            rec["dec_all"]["svm_hard"], rec["jlo"]["svm_hard"] = stream_decisions(lambda j, a: hb, levels, N_seq)
    tm["svm"] = time.perf_counter() - t0

    # ---------------- classifiers ----------------
    blocks = [(X[:q_n], Y[:q_m], X[q_n:n1], Y[q_m:m1]), (X[:n1], Y[:m1], X[n1:], Y[m1:])]
    lam_grid = hplb_lambda_grid(levels)
    hrng = np.random.default_rng(np.random.SeedSequence([seed_key(key), n, rep, HPLB_TAG]))
    for c in CLF:
        t0 = time.perf_counter()
        # defaults on the matched schedule; batch 2 is the paper's single-split classifier
        bh, bs = [], []
        for Xf, Yf, Xe, Ye in blocks:
            nf, mf = len(Xf), len(Yf)
            thr = nf / (nf + mf)
            p = fit_predict(make_clf(c, cseed), np.vstack([Xf, Yf]), np.r_[np.ones(nf, int), np.zeros(mf, int)],
                            np.vstack([Xe, Ye]))
            px_, py_ = p[: len(Xe)], p[len(Xe):]
            bh.append(((px_ > thr).astype(float), (py_ > thr).astype(float), 0.5))
            bs.append((balanced(px_, thr), balanced(py_, thr), 0.5))
        rec["dec_all"][f"{c}_m_hard"], rec["jlo"][f"{c}_m_hard"] = stream_decisions(lambda j, a: bh, levels, N_seq)
        rec["dec_all"][f"{c}_m_soft"], rec["jlo"][f"{c}_m_soft"] = stream_decisions(lambda j, a: bs, levels, N_seq)
        rec["dec_all"][f"{c}_ss_hard"], rec["jlo"][f"{c}_ss_hard"] = single_decisions(bh[1][0], bh[1][1], levels)
        rec["dec_all"][f"{c}_ss_soft"], rec["jlo"][f"{c}_ss_soft"] = single_decisions(bs[1][0], bs[1][1], levels)
        rec["diag"][f"{c}_diff"] = float(bh[1][0].mean() - bh[1][1].mean())
        if stored is not None:
            sd = [rec["dec_all"][f"{c}_ss_hard"][j] for j in M.REPORT_J]
            rec["repro"][f"hard_{c}"] = bool(sd == stored["dec"][f"{c}_bet"]
                                             and rec["jlo"][f"{c}_ss_hard"] == stored["jlo"][f"{c}_bet"])
            rec["repro"][f"diff_{c}"] = bool(abs(rec["diag"][f"{c}_diff"] - stored[f"{c}_diff"]) < 1e-12)
        # HPLB on the single-split scores (rho = 1 - p~, the score of Y)
        lb_b = hplb_bayes(bh[1][0], bh[1][1])
        lb_a = hplb_adapt(1.0 - bs[1][0], 1.0 - bs[1][1], lam_grid, hrng)
        for nm, lb in (("bayes", lb_b), ("adapt", lb_a)):
            rec["dec_all"][f"hplb_{nm}_{c}"], rec["jlo"][f"hplb_{nm}_{c}"] = bound_decisions(lb, levels)
            rec["diag"][f"hplb_{nm}_{c}"] = lb
        tm[f"{c}_m"] = time.perf_counter() - t0
        # tuned on each fit block, prior variance from the out-of-fold predictions
        t0 = time.perf_counter()
        th, ts, sel = [], [], []
        for Xf, Yf, Xe, Ye in blocks:
            nf, mf = len(Xf), len(Yf)
            thr = nf / (nf + mf)
            px_, py_, ox, oy, s_ = tuned_probs(c, Xf, Yf, Xe, Ye, cseed)
            nex, ney = len(Xe), len(Ye)
            gh, hh = (ox > thr).astype(float), (oy > thr).astype(float)
            gs, hs = balanced(ox, thr), balanced(oy, thr)
            th.append(((px_ > thr).astype(float), (py_ > thr).astype(float),
                       max(pair_var(gh.var(), hh.var(), nex, ney), 1e-4)))
            ts.append((balanced(px_, thr), balanced(py_, thr), max(pair_var(gs.var(), hs.var(), nex, ney), 1e-4)))
            sel.append(s_)
        rec["dec_all"][f"{c}_mt_hard"], rec["jlo"][f"{c}_mt_hard"] = stream_decisions(lambda j, a: th, levels, N_seq)
        rec["dec_all"][f"{c}_mt_soft"], rec["jlo"][f"{c}_mt_soft"] = stream_decisions(lambda j, a: ts, levels, N_seq)
        rec["diag"][f"{c}_tuned"] = sel
        tm[f"{c}_mt"] = time.perf_counter() - t0
    tm["total"] = time.perf_counter() - T0
    return rec


def replicate_h00(key, n, rep):
    """Job kinds "h00" / "h00x": only the tests of H_{0,0} that need no split (permutation MMD at the pilot and at the
    pooled-median bandwidth, MMDAgg), at the other sizes of the study, so that Table 1's sample size for power 0.8 can
    be read for them. Same data and seed words as the main kind; "h00x" uses the replicate indices of the study's
    alpha = 0 extension (rep >= 200000), so it is paired with the stored *_n0ext replicates."""
    dz, pil, ss, X, Y = draw(key, n, rep)
    m = len(Y)
    N = n + m
    Zall = np.vstack([X, Y])
    D2 = sqdist(Zall, Zall)
    rec = {"design": key, "n": n, "m": m, "rep": rep, "alpha_star": dz.alpha_star, "dec_all": {}, "pval": {},
           "diag": {}, "time": {}}
    t0 = time.perf_counter()
    zl = np.r_[np.full(n, 1.0 / n), np.full(m, -1.0 / m)]
    prng = np.random.default_rng(np.random.SeedSequence([seed_key(key), n, rep, PERM_TAG]))
    Zp = prng.permuted(np.tile(zl, (B_PERM, 1)), axis=1).T
    iu = np.triu_indices(N, k=1)
    dist = np.sqrt(D2[iu])
    del iu
    dist = dist[dist > 0]
    sig_med = float(np.median(dist)) if dist.size else 1.0
    del dist
    for name, sig in (("perm_pilot", pil["sigma_fix"]), ("perm_med", sig_med)):
        K = np.exp(-D2 / (2.0 * sig ** 2))
        T = float(zl @ (K @ zl))
        Tb = np.einsum("ib,ib->b", Zp, K @ Zp)
        p = (1.0 + float(np.sum(Tb >= T - 1e-12 * abs(T)))) / (B_PERM + 1.0)
        rec["pval"][name] = p
        rec["dec_all"][name] = [int(p <= DELTA)]
        del K
    del Zp
    rec["time"]["perm"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    arng = np.random.default_rng(np.random.SeedSequence([seed_key(key), n, rep, AGG_TAG]))
    rej, info = mmdagg(D2, n, m, arng)
    rec["dec_all"]["mmdagg"] = [int(rej)]
    rec["diag"]["mmdagg"] = info
    rec["time"]["mmdagg"] = time.perf_counter() - t0
    return rec


def replicate_univ(key, n, rep):
    """The study's methods.run_replicate (unchanged) on a new 1-d design, plus the KS bound on the same data."""
    C, PIL = _ctx()
    dz = C[key]
    pil = PIL[key]
    thr = _C.setdefault("thr", json.loads(THR.read_text()))
    m = dz.m_ratio * n
    tab = {k: [thr[f"{n},{m},{a:.12g}"][k] for a in dz.levels()] for k in ("old", "newcf", "new")}
    t0 = time.perf_counter()
    rec = M.run_replicate(dz, n, rep, seed_key(key), pil["sigma_fix"], tab, scale=None)
    rec["time"]["total"] = time.perf_counter() - t0
    rec["alpha_star"] = dz.alpha_star
    rec["levels"] = dz.levels()
    _, _, _, X, Y = draw(key, n, rep)                  # the same draw (same stream order as run_replicate)
    ks, lb = ks_bound(X, Y)
    dec, jlo = bound_decisions(lb, dz.levels())
    rec["dec"]["ks"] = [dec[j] for j in M.REPORT_J]
    rec["jlo"]["ks"] = jlo
    rec["diag"].update({"ks_stat": ks, "ks_lb": lb})
    return rec


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def pilot_univ():
    """Pilot bandwidths of the new 1-d designs, by the study's protocol (run.pilot_one)."""
    C = catalogue()
    pil = json.loads(PILOT_U.read_text()) if PILOT_U.exists() else {}
    for key in UNIV:
        if key in pil:
            continue
        dz = C[key]
        rng = np.random.default_rng(np.random.SeedSequence([seed_key(key), 424242]))
        Xp, Yp = dz.sample(1000, rng, "P"), dz.sample(1000, rng, "Q")
        pil[key] = {"sigma_fix": float(median_heuristic(Xp, Yp, rng=np.random.default_rng(0))), "scale": None,
                    "n_pilot": 1000, "m_pilot": 1000}
    RAW.mkdir(parents=True, exist_ok=True)
    PILOT_U.write_text(json.dumps(pil, indent=1) + "\n")
    return pil


def stored_records(key, n):
    f = STORED / f"{key}.jsonl"
    out = {}
    if f.exists():
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["n"] == n:
                out[r["rep"]] = {"dec": r["dec"], "jlo": r["jlo"], "sigma1": r["diag"]["sigma1"],
                                 "sigma0": r["diag"]["sigma0"], "gbm_diff": r["diag"]["gbm_diff"],
                                 "logit_diff": r["diag"]["logit_diff"]}
    return out


def raw_path(kind, key, n):
    return RAW / (f"{key}_n{n}.jsonl" if kind == "main" else f"{key}_n{n}_{kind}.jsonl")


def done_reps(path):
    s = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                s.add(json.loads(line)["rep"])
            except Exception:
                pass
    return s


def read_valid(path):
    """Valid JSON lines of a jsonl file (corrupted or partial lines dropped), deduplicated by (design, n, rep)."""
    recs, seen = [], set()
    for line in path.read_bytes().split(b"\n"):
        if not line.strip() or b"\x00" in line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        k = (r.get("design"), r.get("n"), r.get("rep"))
        if k not in seen:
            seen.add(k)
            recs.append(r)
    return recs


def sync(log=print):
    """Copy the working raw files to results/aos/baselines/raw atomically and verify them."""
    if RAW.resolve() == RAW_OUT.resolve():
        return
    RAW_OUT.mkdir(parents=True, exist_ok=True)
    n_files = n_recs = 0
    for f in sorted(RAW.iterdir()):
        if not f.is_file() or f.suffix not in (".jsonl", ".json"):
            continue
        if f.suffix == ".jsonl":
            recs = read_valid(f)
            data = "".join(json.dumps(r) + "\n" for r in recs)
        else:
            data, recs = f.read_text(), []
        tmp = RAW_OUT / (f.name + ".tmp")
        tmp.write_text(data)
        os.replace(tmp, RAW_OUT / f.name)
        back = (RAW_OUT / f.name).read_text()
        if back != data:
            log(f"SYNC MISMATCH {f.name}")
        n_files += 1
        n_recs += len(recs)
    log(f"synced {n_files} files ({n_recs} records) to {RAW_OUT}")


def build_jobs(kinds, keys, reps, C):
    jobs = []
    if "main" in kinds:
        for key in keys:
            n = main_n(key, C)
            st = stored_records(key, n)
            done = done_reps(raw_path("main", key, n))
            jobs += [("main", key, n, r, st.get(r)) for r in range(reps) if r not in done]
    if "univ" in kinds:
        for key in UNIV:
            for n in UNIV_NS:
                done = done_reps(raw_path("univ", key, n))
                jobs += [("univ", key, n, r, None) for r in range(reps) if r not in done]
    if "h00" in kinds:
        for key in keys:
            for n in H00_NS:
                if n not in C[key].ns or n == main_n(key, C):
                    continue
                done = done_reps(raw_path("h00", key, n))
                jobs += [("h00", key, n, r, None) for r in range(reps) if r not in done]
    if "h00x" in kinds:
        for key in keys:
            for n in H00X_NS:
                done = done_reps(raw_path("h00x", key, n))
                jobs += [("h00x", key, n, r, None) for r in range(200000, 200000 + reps) if r not in done]
    if "ks" in kinds:
        for key in KS_DESIGNS:
            for n in (100, 200, 400):
                done = done_reps(raw_path("ks", key, n))
                jobs += [("ks", key, n, r, None) for r in range(reps) if r not in done]
    return jobs


def run(kinds, keys, reps, workers, log=print):
    C = catalogue()
    pilot_univ()
    jobs = build_jobs(kinds, keys, reps, C)
    # main jobs first (most expensive first within), then the 1-d jobs
    rank = {"main": 0, "univ": 1, "ks": 2, "h00": 3, "h00x": 4}
    # by replicate index within each kind, so that a killed run leaves every design with about the same number of
    # finished replicates (and every 1-d design with complete size ladders)
    jobs.sort(key=lambda j: (rank[j[0]], j[3], -(j[2] * (1 + C[j[1]].m_ratio)), j[1]))
    log(f"{len(jobs)} jobs: " + ", ".join(f"{k} {sum(j[0] == k for j in jobs)}" for k in rank))
    RAW.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    handles = {}
    done = 0
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(replicate, j): j for j in jobs}
        for fu in as_completed(futs):
            j = futs[fu]
            try:
                rec = fu.result()
            except Exception as e:
                log(f"FAILED {j[:4]} {e!r}")
                continue
            p = raw_path(j[0], j[1], j[2])
            f = handles.get(p)
            if f is None:
                f = handles[p] = open(p, "a")
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
            done += 1
            if done % 25 == 0 or done == len(jobs):
                el = time.time() - t0
                log(f"{done}/{len(jobs)} {el:.0f}s eta {el / done * (len(jobs) - done):.0f}s")
            if done % 250 == 0:
                try:
                    sync(log)
                except Exception as e:          # a cloud-drive hiccup must not stop the run
                    log(f"SYNC FAILED {e!r}")
    for f in handles.values():
        f.close()
    try:
        sync(log)
    except Exception as e:
        log(f"SYNC FAILED {e!r}")


def calibrate(workers):
    """One replicate per design (rep 0) at the main size, timed by component, plus one 1-d replicate per size;
    written to raw/calib.jsonl (not used by the tables)."""
    C = catalogue()
    pilot_univ()
    jobs = []
    for k in DESIGNS + NULLS:
        n = main_n(k, C)
        jobs.append(("main", k, n, 0, stored_records(k, n).get(0)))
    jobs += [("univ", k, n, 0, None) for k in UNIV for n in UNIV_NS]
    RAW.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        recs = list(ex.map(replicate, jobs))
    with open(RAW / "calib.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    tot = 0.0
    for r in recs:
        t = r["time"]
        tot += t["total"]
        print(f"{r['design']:13s} n={r['n']:4d} total {t['total']:6.2f}s | "
              + " ".join(f"{k} {v:.2f}" for k, v in t.items() if k != "total") + f" | repro {r.get('repro')}")
    print(f"sum of replicate times {tot:.0f}s; wall {time.time() - t0:.0f}s on {workers} workers")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calibrate", "run", "sync"])
    ap.add_argument("--kinds", nargs="+", default=["main", "univ", "ks"])
    ap.add_argument("--designs", nargs="+", default=["all"])
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    assert a.workers <= 6, "shared machine: at most 6 worker processes"
    if a.cmd == "calibrate":
        calibrate(a.workers)
        return
    if a.cmd == "sync":
        sync()
        return
    keys = DESIGNS + NULLS if a.designs == ["all"] else a.designs

    def log(s):
        print(time.strftime("%H:%M:%S"), s, flush=True)
    run(a.kinds, keys, a.reps, a.workers, log=log)


if __name__ == "__main__":
    main()

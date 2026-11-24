"""All tests of H_{0,alpha}: TV(P,Q) <= alpha on one replicate, on the same data.

Read-only imports: src/trimmedmmd (kernels, simplex, solver helper),
explore/sharper_threshold/thresholds.py (tau_old, tau_P_closed, tau_new),
explore/split_adaptive/sacert.py + fold1.py (ramp, betting, EB bound, priors).

Analytic tests (full sample, bandwidth fixed in advance from a pilot draw):
  the statistic D_hat_alpha is a convex QP. We never trust an unconverged
  iterate: every 10 FISTA steps we have a feasible point (value U >= D_hat^2)
  and its Frank-Wolfe gap (U - gap <= D_hat^2). A threshold t is ACCEPTED when
  U <= t^2 and REJECTED when U - gap > t^2; the solver stops as soon as every
  threshold of interest is decided. Undecided at max_iter => not rejected
  (conservative; counted). D_hat is nonincreasing in alpha and every tau is
  nondecreasing, so the rejection set is an initial segment of the grid and
  alpha_lo is found by bisection over the grid (decisions inferred by
  monotonicity are exact; inferences from an undecided level are conservative).

Split kernel tests (fold one = first half of a random permutation of each
sample; bandwidth = fold-one pooled median heuristic; witness = trimmed-MMD
witness at the tested level, warm-started along the grid):
  prop17   paper Prop. 17 (DKW, 2 kappa beta [e_n2 + e_m2])           finite-sample
  boot     paper Thm 16 percentile bootstrap, 1000 resamples           asymptotic
  ramp_b2s sacert Cor. 1, empirical-Bernstein closed form, delta1=0.8d finite-sample
  ramp_bet sacert Thm 1, betting, c = 1/2, WSR plug-in, w0 = 20        finite-sample
  seq_bet  sacert 2.4 sequential batches e = (0, 1/4, 1/2, 1)          finite-sample
           (batch 2 reuses the fold-one witness; batch 1 uses a witness on
           the first quarter with its own median bandwidth; the horizon N
           used in lambda_k is FIXED in advance as the total number of bets,
           so lambda_k is predictable even when a batch has no valid cut.)
Classifiers (fold one fit, fold two TPR - FPR, label X iff p_hat(X|z) >
n1/(n1+m1), i.e. the Bayes rule for the set {p > q} under the training mix):
  <clf>_hoef  Hoeffding, slack sqrt(log(2/d)/(2 n2)) + sqrt(log(2/d)/(2 m2))
  <clf>_bet   same betting test, neutral prior variance 0.5 (as split_adaptive)
  clf in {gbm (HistGradientBoosting defaults), logit (standardize + L2
  logistic, C = 1), knn (k = int(sqrt(n1+m1)) | 1)}.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import math
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT / "src", ROOT / "explore" / "sharper_threshold", ROOT / "explore" / "split_adaptive"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402

from trimmedmmd.kernels import median_heuristic, sqdist  # noqa: E402
from trimmedmmd.simplex import project_capped_simplex  # noqa: E402
from trimmedmmd.solver import _linear_min_over_capped_simplex as lmin  # noqa: E402

import sacert as S  # noqa: E402
from fold1 import cut_points, priors  # noqa: E402

DELTA = 0.05
REPORT_J = (0, 4, 6, 8)          # alpha = 0, a*/2, 0.75 a*, a* (boundary)
J_MAX = 10
AN_THR = ("old", "newcf", "new")
SPLIT_K = ("prop17", "boot", "ramp_b2s", "ramp_bet", "seq_bet")
CLFS = ("gbm", "logit", "knn")
METHODS = (tuple(f"an_{t}" for t in AN_THR) + SPLIT_K
           + tuple(f"{c}_{r}" for c in CLFS for r in ("hoef", "bet")))


# ---------------------------------------------------------------------------
# QP on a Gram matrix in the order [X; Y]
# ---------------------------------------------------------------------------

class GramQP:
    def __init__(self, K, n, m, seed=0):
        self.K, self.n, self.m = K, n, m
        r = np.random.default_rng(seed)
        x = r.standard_normal(n + m)
        x /= np.linalg.norm(x)
        lam = 1.0
        for _ in range(50):
            y = K @ x
            lam = float(np.linalg.norm(y))
            if lam <= 0:
                break
            x = y / lam
        self.lip = max(2.0 * lam * 1.02, 1e-12)   # Hessian = 2 D K D, D = diag(I, -I)

    def _z(self, w, v):
        return np.concatenate([w, -v])

    def value_grad(self, w, v):
        Kz = self.K @ self._z(w, v)
        val = float(w @ Kz[:self.n] - v @ Kz[self.n:])
        return max(val, 0.0), 2.0 * Kz[:self.n], -2.0 * Kz[self.n:]

    def solve(self, alpha, w0=None, v0=None, thr2=None, max_iter=1000, tol=1e-9, check=10):
        """FISTA with restart. thr2: squared thresholds to decide (early stop),
        or None for a plain solve (relative-change stopping).
        Returns (w, v, value, lower, decisions, iters)."""
        n, m = self.n, self.m
        cw, cv = 1.0 / (n * (1.0 - alpha)), 1.0 / (m * (1.0 - alpha))
        w = project_capped_simplex(np.full(n, 1.0 / n) if w0 is None else np.minimum(w0, cw), cw)
        v = project_capped_simplex(np.full(m, 1.0 / m) if v0 is None else np.minimum(v0, cv), cv)
        if thr2 is not None:
            thr2 = np.asarray(thr2, float)
            dec = np.zeros(thr2.size, int)
        else:
            dec = None
        yw, yv, t = w.copy(), v.copy(), 1.0
        step = 1.0 / self.lip
        prev = None
        val = lower = None
        it = 0
        for it in range(1, max_iter + 1):
            Kz = self.K @ self._z(yw, yv)
            w_new = project_capped_simplex(yw - step * 2.0 * Kz[:n], cw)
            v_new = project_capped_simplex(yv + step * 2.0 * Kz[n:], cv)
            t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
            beta = (t - 1.0) / t_new
            yw = w_new + beta * (w_new - w)
            yv = v_new + beta * (v_new - v)
            w, v, t = w_new, v_new, t_new
            if it % check == 0 or it == max_iter:
                val, gw, gv = self.value_grad(w, v)
                if prev is not None and val > prev:
                    yw, yv, t = w.copy(), v.copy(), 1.0
                if dec is not None:
                    gap = (float(gw @ w) - lmin(gw, cw) + float(gv @ v) - lmin(gv, cv))
                    lower = val - max(gap, 0.0)
                    dec[(dec == 0) & (val <= thr2)] = -1
                    dec[(dec == 0) & (lower > thr2)] = 1
                    if np.all(dec != 0):
                        break
                else:
                    if prev is not None and abs(prev - val) <= tol * max(abs(prev), 1e-300):
                        break
                prev = val
        if val is None:
            val, _, _ = self.value_grad(w, v)
        return w, v, val, lower, dec, it


# ---------------------------------------------------------------------------
# analytic tests
# ---------------------------------------------------------------------------

def analytic(K, n, m, levels, thr, max_iter=4000):
    """thr: dict name -> array of thresholds over the grid. Returns
    (jlo per threshold, decisions at REPORT_J per threshold, diag)."""
    qp = GramQP(K, n, m)
    J = len(levels)
    names = list(thr)
    cache = {}                     # j -> (decisions array over names, w, v, val, iters)

    def evaluate(j):
        if j in cache:
            return cache[j][0]
        lower_js = [i for i in cache if i < j]
        w0 = v0 = None
        if lower_js:
            i = max(lower_js)
            w0, v0 = cache[i][1], cache[i][2]
        thr2 = np.array([thr[k][j] ** 2 for k in names])
        w, v, val, lower, dec, it = qp.solve(levels[j], w0, v0, thr2=thr2, max_iter=max_iter)
        cache[j] = (dec, w, v, val, it, lower)
        return dec

    jlo, undecided = {}, 0
    for a, k in enumerate(names):
        lo, hi = -1, J            # largest known reject, smallest known accept
        # use what is already known
        for j, rec in cache.items():
            if rec[0][a] == 1:
                lo = max(lo, j)
            else:
                hi = min(hi, j)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            d = evaluate(mid)[a]
            if d == 1:
                lo = mid
            else:
                hi = mid
        jlo[k] = lo
    for rec in cache.values():
        undecided += int(np.sum(rec[0] == 0))
    decs = {k: [int(j <= jlo[k]) for j in REPORT_J] for k in names}
    diag = {"an_qps": len(cache), "an_iters": int(sum(r[4] for r in cache.values())),
            "an_undecided": undecided,
            "an_val": {str(j): [float(math.sqrt(max(r[3], 0))), float(math.sqrt(max(r[5], 0)))
                                if r[5] is not None else None] for j, r in cache.items()}}
    return jlo, decs, diag


# ---------------------------------------------------------------------------
# split helpers
# ---------------------------------------------------------------------------

def ltm_weights(n, alpha):
    cap = 1.0 / (n * (1.0 - alpha))
    k = min(int(np.floor(1.0 / cap + 1e-12)), n)
    wts = np.zeros(n)
    wts[:k] = cap
    if k < n:
        wts[k] = 1.0 - k * cap
    return wts


def boot_reject(u2, v2, alpha, rng, n_boot=1000, delta=DELTA):
    """Thm 16 percentile bootstrap: delta-quantile of T* > 0 (as splitting.split_test)."""
    n, m = u2.size, v2.size
    wu, wv = ltm_weights(n, alpha), ltm_weights(m, alpha)
    ia = rng.integers(0, n, size=(n_boot, n))
    ib = rng.integers(0, m, size=(n_boot, m))
    bu = np.sort(u2[ia], axis=1) @ wu
    bv = np.sort(-v2[ib], axis=1) @ wv
    return bool(np.quantile(bu + bv, delta) > 0)


def witness_values(qp, K_eval, n_fit, m_fit, alpha, warm, max_iter=1000):
    """Trimmed-MMD witness on the fit block (qp: GramQP of the fit Gram);
    returns self-excluded fit values, evaluation values, norm, and the (w, v)
    for warm starts."""
    K_fit = qp.K
    w0, v0 = warm if warm is not None else (None, None)
    w, v, val, _, _, it = qp.solve(alpha, w0, v0, thr2=None, max_iter=max_iter, tol=1e-9)
    norm = math.sqrt(max(val, 0.0))
    if norm <= 1e-12:
        return None, (w, v), it
    z = np.concatenate([w, -v])
    f_fit = (K_fit @ z) / norm
    u1 = f_fit[:n_fit] - w / norm
    v1 = f_fit[n_fit:] + v / norm
    f_ev = (K_eval @ z) / norm
    return (u1, v1, f_ev, norm), (w, v), it


def bet_stream(Zs, pvs, alpha, N, delta=DELTA, c=0.5, prior_w=20.0):
    """Sequential betting over several batches (seqcert.seq_ramp_bet) with the
    horizon N fixed in advance. Zs: list of arrays of Z_k - alpha (None = batch
    skipped, no bets), pvs: prior variances per batch."""
    ys, pv = [], []
    for z, p in zip(Zs, pvs):
        if z is None or z.size == 0:
            continue
        ys.append(z)
        pv.append(np.full(z.size, p))
    if not ys:
        return False, -np.inf
    Y = np.concatenate(ys)
    PV = np.concatenate(pv)
    L = math.log(1.0 / delta)
    lam_max = c / (1.0 + alpha)
    cnt = np.arange(Y.size, dtype=float)
    cs = np.concatenate([[0.0], np.cumsum(Y)[:-1]])
    mu = cs / (prior_w + cnt)
    cdev = np.concatenate([[0.0], np.cumsum((Y - mu) ** 2)[:-1]])
    sig2 = np.maximum((prior_w * PV + cdev) / (prior_w + cnt), 1e-8)
    lam = np.minimum(lam_max, np.sqrt(2.0 * L / (N * sig2)))
    logK = np.cumsum(np.log1p(lam * Y))
    return bool(logK.max() >= L), float(logK.max())


def n_pairs(a, b):
    return min(a, b)


# ---------------------------------------------------------------------------
# classifiers
# ---------------------------------------------------------------------------

def classifier_scores(name, X1, Y1, X2, Y2, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    n1, m1 = len(X1), len(Y1)
    if name == "gbm":
        clf = HistGradientBoostingClassifier(random_state=seed)
    elif name == "logit":
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    elif name == "knn":
        clf = KNeighborsClassifier(n_neighbors=int(math.sqrt(n1 + m1)) | 1)
    else:
        raise KeyError(name)
    Z1 = np.vstack([X1, Y1])
    lab = np.r_[np.ones(n1, int), np.zeros(m1, int)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(Z1, lab)
        cls = list(clf.classes_)
        px = clf.predict_proba(X2)[:, cls.index(1)]
        py = clf.predict_proba(Y2)[:, cls.index(1)]
    thr = n1 / (n1 + m1)
    return (px > thr).astype(float), (py > thr).astype(float)


def classifier_tests(G, H, levels, delta=DELTA):
    n2, m2 = G.size, H.size
    slack = math.sqrt(math.log(2 / delta) / (2 * n2)) + math.sqrt(math.log(2 / delta) / (2 * m2))
    diff = float(G.mean() - H.mean())
    L = diff - slack
    hoef = [L > a for a in levels]
    jlo_h = max([j for j in range(len(levels)) if all(hoef[: j + 1])], default=-1)
    Zb = S.pair_up(G, -H)
    bet, alive, jlo_b = {}, True, -1
    for j, a in enumerate(levels):
        if not alive and j not in REPORT_J:
            continue
        r = S.betting_test(Zb - a, -(1.0 + a), delta, c=0.5, prior_var=0.5)[0]
        bet[j] = r
        if alive and r:
            jlo_b = j
        elif alive:
            alive = False
    return ({"hoef": [int(hoef[j]) for j in REPORT_J], "bet": [int(bet[j]) for j in REPORT_J]},
            {"hoef": jlo_h, "bet": jlo_b}, {"diff": diff, "L": L})


# ---------------------------------------------------------------------------
# one replicate
# ---------------------------------------------------------------------------

def run_replicate(design, n, rep, seed_key, sigma_fix, thr_tab, scale=None,
                  do_analytic=True, n_boot=1000):
    """design: designs.Design; thr_tab: dict name -> list of thresholds over
    design.levels() for this (n, m)."""
    m = design.m_ratio * n
    ss = np.random.SeedSequence([seed_key, n, rep])
    rng = np.random.default_rng(ss)
    X = design.sample(n, rng, "P")
    Y = design.sample(m, rng, "Q")
    if scale is not None:
        X, Y = X / scale, Y / scale
    levels = design.levels()
    rec = {"design": design.key, "n": n, "m": m, "rep": rep, "sigma_fix": sigma_fix,
           "dec": {}, "jlo": {}, "time": {}, "diag": {}}

    # random fold assignment (same permutation used by every split method)
    px, py = rng.permutation(n), rng.permutation(m)
    X, Y = X[px], Y[py]
    Zall = np.vstack([X, Y])
    t0 = time.perf_counter()
    D2 = sqdist(Zall, Zall)
    t_d2 = time.perf_counter() - t0

    # ---------------- analytic ----------------
    if do_analytic:
        t0 = time.perf_counter()
        K = np.exp(-D2 / (2.0 * sigma_fix ** 2))
        jlo, decs, diag = analytic(K, n, m, levels, thr_tab)
        del K
        for k in AN_THR:
            rec["dec"][f"an_{k}"] = decs[k]
            rec["jlo"][f"an_{k}"] = jlo[k]
        rec["diag"].update(diag)
        rec["time"]["analytic"] = time.perf_counter() - t0 + t_d2

    # ---------------- split kernel ----------------
    t0 = time.perf_counter()
    n1, m1 = n // 2, m // 2
    n2, m2 = n - n1, m - m1
    iX1, iX2 = np.arange(n1), np.arange(n1, n)
    iY1, iY2 = n + np.arange(m1), n + np.arange(m1, m)
    f1 = np.r_[iX1, iY1]
    f2 = np.r_[iX2, iY2]
    sig1 = median_heuristic(X[:n1], Y[:m1], rng=np.random.default_rng(ss.spawn(1)[0]))
    K11 = np.exp(-D2[np.ix_(f1, f1)] / (2 * sig1 ** 2))
    K21 = np.exp(-D2[np.ix_(f2, f1)] / (2 * sig1 ** 2))
    # sequential batch 1: fit on the first quarter, bet on the second quarter
    q_n, q_m = int(round(n / 4)), int(round(m / 4))
    b0 = np.r_[np.arange(q_n), n + np.arange(q_m)]
    b1 = np.r_[np.arange(q_n, n1), n + np.arange(q_m, m1)]
    nb1x, nb1y = n1 - q_n, m1 - q_m
    sig0 = median_heuristic(X[:q_n], Y[:q_m], rng=np.random.default_rng(ss.spawn(2)[1]))
    K00 = np.exp(-D2[np.ix_(b0, b0)] / (2 * sig0 ** 2))
    K10 = np.exp(-D2[np.ix_(b1, b0)] / (2 * sig0 ** 2))
    N_seq = n_pairs(nb1x, nb1y) + n_pairs(n2, m2)
    qp11, qp00 = GramQP(K11, n1, m1), GramQP(K00, q_n, q_m)
    rec["diag"].update({"sigma1": sig1, "sigma0": sig0})
    brng = np.random.default_rng(ss.spawn(3)[2])
    dec_l = {k: {} for k in SPLIT_K}
    alive = {k: True for k in SPLIT_K}
    jlo_s = {k: -1 for k in SPLIT_K}
    warm1 = warm0 = None
    t_boot = t_seq = 0.0
    wit_iters = 0
    extra = {}
    for j, a in enumerate(levels):
        need = any(alive.values()) or j in REPORT_J
        if not need:
            continue
        beta = 1.0 / (1.0 - a)
        out, warm1, it = witness_values(qp11, K21, n1, m1, a, warm1)
        wit_iters += it
        res = {k: False for k in SPLIT_K}
        Z2, pv2 = None, None
        if out is not None:
            u1, v1, fev, norm = out
            u2, v2 = fev[:n2], fev[n2:]
            T_hat = S.trimmed_sep(u2, v2, a)
            e = lambda k: math.sqrt(math.log(4.0 / DELTA) / (2.0 * k))
            res["prop17"] = bool(T_hat > 2 * 1.0 * beta * (e(n2) + e(m2)))
            if alive["boot"] or j in REPORT_J:
                tb = time.perf_counter()
                res["boot"] = boot_reject(u2, v2, a, brng, n_boot=n_boot)
                t_boot += time.perf_counter() - tb
            s, sp = cut_points(u1, v1, a)
            if s > sp:
                G, H = S.ramp(u2, s, sp), S.ramp(v2, s, sp)
                res["ramp_b2s"] = bool(S.lcb_bern2s(G, 1.0, -H, 1.0, DELTA) > a)
                pv2 = priors(u1, v1, s, sp, a, n2, m2)["ramp"]
                Z2 = S.pair_up(G, -H) - a
                res["ramp_bet"] = S.betting_test(Z2, -(1.0 + a), DELTA, c=0.5, prior_var=pv2)[0]
                if j in REPORT_J:
                    extra[str(j)] = {"T_hat": T_hat, "Delta_hat": float(G.mean() - H.mean()),
                                     "norm": norm}
        # sequential (needs batch-1 witness)
        if alive["seq_bet"] or j in REPORT_J:
            ts = time.perf_counter()
            out0, warm0, it0 = witness_values(qp00, K10, q_n, q_m, a, warm0)
            wit_iters += it0
            Z1, pv1 = None, None
            if out0 is not None:
                u0, v0_, fev0, _ = out0
                s0, sp0 = cut_points(u0, v0_, a)
                if s0 > sp0:
                    G1, H1 = S.ramp(fev0[:nb1x], s0, sp0), S.ramp(fev0[nb1x:], s0, sp0)
                    Z1 = S.pair_up(G1, -H1) - a
                    pv1 = priors(u0, v0_, s0, sp0, a, nb1x, nb1y)["ramp"]
            res["seq_bet"] = bet_stream([Z1, Z2], [pv1, pv2], a, N_seq)[0]
            t_seq += time.perf_counter() - ts
        for k in SPLIT_K:
            if j in REPORT_J:
                dec_l[k][j] = int(res[k])
            if alive[k]:
                if res[k]:
                    jlo_s[k] = j
                else:
                    alive[k] = False
    for k in SPLIT_K:
        rec["dec"][k] = [dec_l[k][j] for j in REPORT_J]
        rec["jlo"][k] = jlo_s[k]
    rec["diag"]["split_extra"] = extra
    rec["diag"]["wit_iters"] = wit_iters
    tt = time.perf_counter() - t0
    rec["time"]["split_kernel"] = tt - t_boot - t_seq
    rec["time"]["boot_extra"] = t_boot
    rec["time"]["seq_extra"] = t_seq
    del D2, K11, K21, K00, K10, qp11, qp00

    # ---------------- classifiers ----------------
    X1, X2, Y1, Y2 = X[:n1], X[n1:], Y[:m1], Y[m1:]
    cseed = int(ss.generate_state(1)[0] % (2 ** 31 - 1))
    for c in CLFS:
        t0 = time.perf_counter()
        G, H = classifier_scores(c, X1, Y1, X2, Y2, cseed)
        dec, jlo, dg = classifier_tests(G, H, levels)
        for r in ("hoef", "bet"):
            rec["dec"][f"{c}_{r}"] = dec[r]
            rec["jlo"][f"{c}_{r}"] = jlo[r]
        rec["diag"][f"{c}_diff"] = dg["diff"]
        rec["time"][c] = time.perf_counter() - t0
    return rec

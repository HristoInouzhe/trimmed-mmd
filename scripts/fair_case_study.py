"""Group-fairness case study: repair a dataset by trimming, then audit it.

Question. Two demographic groups (sex) have different covariate laws P (men)
and Q (women). Any classifier f of the covariates has a demographic-parity gap
|E_P f - E_Q f| <= TV(P, Q), so a repair that makes the groups' covariate laws
coincide bounds the gap for *every* downstream classifier at once. The paper's
trimming is such a repair: it deletes (or down-weights) at most an alpha
fraction of each group, and TV(P,Q) <= alpha is exactly the condition under
which two trimmings can coincide.

What the paper's machinery supplies, and what this script does with it:

  1. the analytic estimator alpha-hat (threshold crossing of the trimmed MMD
     path against the closed-form threshold tau) -- a certified lower bound on
     TV(P, Q), so a certified *floor* on how much any deletion repair must
     remove, since TV(P', Q') >= (TV - alpha) / (1 - alpha) for trimmings;
  2. the trimming itself at a chosen level, i.e. the repaired dataset;
  3. an audit that does not use the kernel: a cross-fitted gradient-boosting
     classifier trained to recover the group from the repaired covariates.
     Its TPR - FPR is a lower bound on the repaired groups' TV, and is what a
     fairness practitioner would try first.

Baselines at the same retained fraction: random deletion (null), logistic
propensity-score trimming, cross-fitted gradient-boosting propensity trimming.

The kernel is the Gaussian kernel through exact-diagonal cos/sin random
features, so k(x, x) = 1 holds exactly (kappa = 1, no realized-kappa
correction), and the threshold of the paper applies to that feature kernel as
it stands. Only power depends on the random-feature dimension, not validity.

    PYTHONPATH=src python scripts/fair_case_study.py adult
    PYTHONPATH=src python scripts/fair_case_study.py acs

Revision v2 (defaults reproduce v1): ``--threshold plugin`` uses the plug-in
threshold tau_new with the constants of a BOUNDED kernel (kappa = 1,
Dphi = rho = 2, no coupling): the cos/sin feature kernel has k(x, x) = 1
exactly but can be negative, so the Gaussian coupled constants do not apply
to it. The Gaussian-coupled value is stored next to it (``tau_plugin_gaussian``)
for information only. ``--outdir D`` writes to ``D/fair``. ``--stage
rethreshold`` re-applies a threshold to the stored path of an existing run
(``--source-dir``, default results/fair) without re-solving: the path records
keep D-hat and its Frank-Wolfe lower bound, which do not depend on the
threshold; the path npz is copied alongside. ``--audit-alphas`` accepts the
token ``ahat`` for the run's own alpha-hat.

    PYTHONPATH=src python scripts/fair_case_study.py acs --stage rethreshold \
        --threshold plugin --outdir results/v2
    PYTHONPATH=src python scripts/fair_case_study.py acs --stage audit \
        --threshold plugin --outdir results/v2 --audit-alphas 0.15 ahat 0.35 0.40 0.50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trimmedmmd.kernels import median_heuristic              # noqa: E402
from trimmedmmd.solver import solve_features                  # noqa: E402
from trimmedmmd.threshold import tau                          # noqa: E402

DELTA = 0.05
OUT = ROOT / "results" / "fair"


# ---------------------------------------------------------------------------
# revision v2: threshold choice and output location
# ---------------------------------------------------------------------------

def out_dir(outdir=None):
    """``results/fair`` (v1) or ``<outdir>/fair``."""
    if outdir is None:
        return OUT
    from trimmedmmd.experiments.v2 import sub
    return sub(outdir, "fair")


def threshold_fn(threshold, n, m, delta=DELTA):
    """alpha -> threshold for the cos/sin feature statistic.

    ``"paper"``: v1's tau (kappa = 1, exact for these features). ``"plugin"``:
    tau_new with BOUNDED kernel constants (the feature kernel can be negative).
    """
    if threshold == "paper":
        return lambda a: float(tau(n, m, a, delta))
    from trimmedmmd.experiments.v2 import threshold_for
    return threshold_for(threshold, n, m, delta, backend="cossin")


def threshold_fields(threshold, n, m, a, delta=DELTA):
    """Extra per-level fields recorded under the v2 threshold (none for v1)."""
    if threshold == "paper":
        return {}
    from trimmedmmd.threshold import tau_new
    return {"tau_paper": float(tau(n, m, a, delta)),
            "tau_plugin_gaussian": float(tau_new(n, m, a, delta))}   # information only


def threshold_meta(threshold):
    if threshold == "paper":
        return {}
    from dataclasses import asdict
    from trimmedmmd.experiments.v2 import kernel_consts_for
    return {"threshold": threshold,
            "kernel_constants": asdict(kernel_consts_for("cossin")),
            "kernel_constants_note": "cos/sin random-feature kernel: k(x,x) = 1 exactly, "
                                     "k can be negative -> bounded constants (Dphi = rho = 2)"}


def summarize_path(recs):
    rejected = [r["alpha"] for r in recs if r["rejects"]]
    accepted = [r["alpha"] for r in recs if not r["rejects"]]
    alpha_hat = min(accepted) if accepted else float("nan")
    last_rej = max(rejected) if rejected else 0.0
    return alpha_hat, last_rej


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_adult():
    """UCI Adult / Census Income. Groups: men (X) and women (Y).

    ``relationship`` is dropped because Husband/Wife encodes the group
    directly; ``fnlwgt`` is a survey weight, not a person attribute."""
    cols = ["age", "workclass", "fnlwgt", "education", "educ_num", "marital",
            "occupation", "relationship", "race", "sex", "cap_gain",
            "cap_loss", "hours", "country", "income"]
    d = ROOT / "data_raw" / "adult"
    a = pd.concat([
        pd.read_csv(d / "adult.data", names=cols, skipinitialspace=True),
        pd.read_csv(d / "adult.test", names=cols, skipinitialspace=True,
                    skiprows=1)])
    a = a.replace("?", np.nan).dropna().reset_index(drop=True)
    F = pd.DataFrame({
        "age": a.age, "education_num": a.educ_num, "hours": a.hours,
        "log_capital_gain": np.log1p(a.cap_gain),
        "log_capital_loss": np.log1p(a.cap_loss),
        "married": a.marital.str.startswith("Married").astype(float),
        "never_married": (a.marital == "Never-married").astype(float),
        "private": (a.workclass == "Private").astype(float),
        "self_employed": a.workclass.str.startswith("Self").astype(float),
        "government": a.workclass.str.endswith("gov").astype(float),
        "white": (a.race == "White").astype(float),
        "us_born": (a.country == "United-States").astype(float)})
    g = (a.sex == "Female").astype(int).values          # 1 = women
    y = a.income.str.startswith(">50K").astype(int).values
    meta = {"source": "UCI Adult (Becker & Kohavi 1996), train + test",
            "group": "sex (0 = male, 1 = female)", "outcome": "income > 50K"}
    return F, g, y, meta


def _occupation_group(o):
    o = np.asarray(o)
    out = np.full(o.shape, 5)
    out[o < 3600] = 0
    out[(o >= 3600) & (o < 4700)] = 1
    out[(o >= 4700) & (o < 6000)] = 2
    out[(o >= 6000) & (o < 7700)] = 3
    out[(o >= 7700) & (o < 9800)] = 4
    return out


def load_acs(states=None, year="2018", max_rows=None, seed=0):
    """ACS PUMS via folktables, ACSIncome population. Groups: men, women."""
    from folktables import ACSDataSource, ACSIncome, state_list
    ds = ACSDataSource(survey_year=year, horizon="1-Year", survey="person",
                       root_dir=str(ROOT / "data_raw" / "acs"))
    states = states or state_list
    raw = ds.get_data(states=states, download=False)
    X, y, _ = ACSIncome.df_to_pandas(raw)
    mar = X.MAR.astype(int)
    cow = X.COW.astype(int)
    og = _occupation_group(X.OCCP.values)
    F = pd.DataFrame({
        "age": X.AGEP, "schooling": X.SCHL, "hours": X.WKHP,
        "married": (mar == 1).astype(float),
        "divorced": (mar == 3).astype(float),
        "never_married": (mar == 5).astype(float),
        "private": cow.isin([1, 2]).astype(float),
        "government": cow.isin([3, 4, 5]).astype(float),
        "self_employed": cow.isin([6, 7]).astype(float),
        **{f"occupation_{k}": (og == k).astype(float) for k in range(5)},
        "us_born": (X.POBP < 100).astype(float)})
    g = (X.SEX.values == 2).astype(int)
    yv = y.values[:, 0].astype(int)
    if max_rows is not None and len(F) > max_rows:
        ix = np.random.default_rng(seed).choice(len(F), max_rows, replace=False)
        F, g, yv = F.iloc[ix].reset_index(drop=True), g[ix], yv[ix]
    meta = {"source": f"ACS PUMS {year} 1-year, folktables ACSIncome, "
                      f"{len(states)} states",
            "group": "sex (0 = male, 1 = female)", "outcome": "income > $50k"}
    return F, g, yv, meta


# ---------------------------------------------------------------------------
# the trimming, on explicit features
# ---------------------------------------------------------------------------

def cossin_features(Z, W):
    P = Z @ W
    return np.hstack([np.cos(P), np.sin(P)]) / np.sqrt(W.shape[1])


def least_trimming(Za, Zb, alpha, base, n_ridge=4, tol=1e-12, max_iter=6000,
                   rng=None):
    """The minimizer closest to uniform weights, as in optimal_trimming_mmd
    (select="least"), for explicit features. Falls back to ``base`` if the
    continuation limit is no longer a minimizer."""
    value = base.value
    r = max(float(max(value, 1e-12)), 1e-8)
    ww, vv = base.w.copy(), base.v.copy()
    res = None
    for _ in range(n_ridge):
        res = solve_features(Za, Zb, alpha, w0=ww, v0=vv, ridge=r, tol=tol,
                             max_iter=max_iter, rng=rng)
        ww, vv = res.w, res.v
        r /= 10.0
    d = Za.T @ ww - Zb.T @ vv
    cand = float(d @ d)
    if cand <= value + 1e-6 * max(value, 1e-12) + 1e-9:
        return ww, vv, cand
    return base.w, base.v, value


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

def _balanced(wa, wb):
    """Weights with total mass 1 on each group; zero weights stay zero."""
    return wa / wa.sum(), wb / wb.sum()


def smd_max(F, g, w):
    """Largest absolute standardized mean difference, weighted by w (the
    pooled unweighted SD of the full data, as in the matching literature)."""
    Fv = F.values
    sd = Fv.std(0)
    sd[sd == 0] = 1.0
    wa, wb = w[g == 0], w[g == 1]
    ma = (Fv[g == 0] * wa[:, None]).sum(0) / wa.sum()
    mb = (Fv[g == 1] * wb[:, None]).sum(0) / wb.sum()
    d = (ma - mb) / sd
    return float(np.abs(d).max()), d


def clf_tv_lower_bound(F, g, w, n_folds=5, seed=0, max_iter=200, boot=0):
    """Lower bound on TV between the weighted group laws, from a cross-fitted
    gradient-boosting classifier of the group.

    Any classifier c satisfies TV(P', Q') >= P'(c=1) - Q'(c=1). Weights give
    each group total mass 1, so the classes are balanced; the threshold is
    1/2 on the predicted probability. Returns the weighted TPR - FPR on
    held-out folds, the weighted AUC, and (optionally) a bootstrap SE.
    """
    keep = w > 0
    Fv, gv, wv = F.values[keep], g[keep], w[keep]
    wv = wv.copy()
    for k in (0, 1):
        wv[gv == k] /= wv[gv == k].sum()
    p = np.zeros(len(gv))
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(Fv, gv):
        clf = HistGradientBoostingClassifier(max_iter=max_iter,
                                             random_state=seed)
        clf.fit(Fv[tr], gv[tr], sample_weight=wv[tr] / wv[tr].mean())
        p[te] = clf.predict_proba(Fv[te])[:, 1]

    def stat(idx):
        gg, ww, pp = gv[idx], wv[idx], p[idx]
        if ww[gg == 1].sum() == 0 or ww[gg == 0].sum() == 0:
            return np.nan, np.nan
        pred = pp > 0.5
        tpr = (ww[gg == 1] * pred[gg == 1]).sum() / ww[gg == 1].sum()
        fpr = (ww[gg == 0] * pred[gg == 0]).sum() / ww[gg == 0].sum()
        # weighted AUC by rank statistic
        o = np.argsort(pp)
        gg_o, ww_o = gg[o], ww[o]
        cum0 = np.cumsum(ww_o * (gg_o == 0))
        auc = (ww_o[gg_o == 1] * cum0[gg_o == 1]).sum() / (
            ww_o[gg_o == 1].sum() * ww_o[gg_o == 0].sum())
        return float(tpr - fpr), float(auc)

    tv, auc = stat(np.arange(len(gv)))
    out = {"tv_lb": tv, "auc": auc, "n_eff": float(1.0 / (wv ** 2).sum())}
    if boot:
        rb = np.random.default_rng(seed + 1)
        vals = [stat(rb.integers(0, len(gv), len(gv)))[0] for _ in range(boot)]
        out["tv_lb_se"] = float(np.nanstd(vals))
    return out, p


def crossfit_propensity(F, g, kind="gbm", n_folds=5, seed=0):
    """Out-of-fold P(group = 1 | x)."""
    Fv = F.values
    e = np.zeros(len(g))
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(Fv, g):
        if kind == "logit":
            mu, sd = Fv[tr].mean(0), Fv[tr].std(0)
            sd[sd == 0] = 1.0
            m = LogisticRegression(max_iter=2000, C=1e4)
            m.fit((Fv[tr] - mu) / sd, g[tr])
            e[te] = m.predict_proba((Fv[te] - mu) / sd)[:, 1]
        else:
            m = HistGradientBoostingClassifier(max_iter=150, random_state=seed)
            m.fit(Fv[tr], g[tr])
            e[te] = m.predict_proba(Fv[te])[:, 1]
    return e


def quantile_trim_weights(score, g, alpha):
    """Keep the (1-alpha) fraction of each group closest to the other group
    along ``score`` = P(group 1 | x): the largest scores of group 0, the
    smallest of group 1. Returns 0/1 weights (a hard subsample)."""
    w = np.zeros(len(g))
    for k, keep_low in ((0, False), (1, True)):
        ix = np.nonzero(g == k)[0]
        order = ix[np.argsort(score[ix])]
        if not keep_low:
            order = order[::-1]
        w[order[:int(round((1.0 - alpha) * len(ix)))]] = 1.0
    return w


def outcome_gap(y, g, w):
    """Weighted P(y=1 | group) for each group and the gap."""
    r0 = (w[g == 0] * y[g == 0]).sum() / w[g == 0].sum()
    r1 = (w[g == 1] * y[g == 1]).sum() / w[g == 1].sum()
    return float(r0), float(r1), float(r0 - r1)


def clf_dp_gap(F, g, y, w, n_folds=5, seed=0):
    """Demographic-parity gap of a cross-fitted outcome classifier trained on
    the weighted cohort, evaluated on the same weighted cohort."""
    keep = w > 0
    Fv, gv, yv, wv = F.values[keep], g[keep], y[keep], w[keep]
    pred = np.zeros(len(gv))
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(Fv, gv):
        clf = HistGradientBoostingClassifier(max_iter=150, random_state=seed)
        clf.fit(Fv[tr], yv[tr], sample_weight=wv[tr] / wv[tr].mean())
        pred[te] = clf.predict(Fv[te])
    r0 = (wv[gv == 0] * pred[gv == 0]).sum() / wv[gv == 0].sum()
    r1 = (wv[gv == 1] * pred[gv == 1]).sum() / wv[gv == 1].sum()
    acc = float((wv * (pred == yv)).sum() / wv.sum())
    return {"rate_g0": float(r0), "rate_g1": float(r1),
            "dp_gap": float(abs(r0 - r1)), "accuracy": acc}


# ---------------------------------------------------------------------------
# stage 1: the path and the analytic estimator
# ---------------------------------------------------------------------------

def load(name, max_rows=None, seed=0):
    if name == "adult":
        return load_adult()
    if name == "acs":
        cache = ROOT / "data_raw" / "acs" / "acsincome_2018_all.pkl"
        if cache.exists():
            F, g, y, meta = pd.read_pickle(cache)
        else:
            F, g, y, meta = load_acs()
            pd.to_pickle((F, g, y, meta), cache)
        if max_rows is not None and len(F) > max_rows:
            ix = np.random.default_rng(seed).choice(len(F), max_rows,
                                                    replace=False)
            F, g, y = F.iloc[ix].reset_index(drop=True), g[ix], y[ix]
        return F, g, y, meta
    raise SystemExit(f"unknown dataset {name}")


def featurize(F, g, n_freq, seed, bw_scale=1.0):
    rng = np.random.default_rng(seed)
    Fv = F.values.astype(float)
    mu, sd = Fv.mean(0), Fv.std(0)
    sd[sd == 0] = 1.0
    S = (Fv - mu) / sd
    sigma = float(median_heuristic(S[g == 0], S[g == 1], rng=rng)) * bw_scale
    W = rng.standard_normal((S.shape[1], n_freq)) / sigma
    Z = cossin_features(S, W)
    assert abs((Z ** 2).sum(1).max() - 1.0) < 1e-9        # kappa = 1 exactly
    return Z, sigma, rng


DEFAULT_GRID = np.round(np.r_[0.0, np.arange(0.025, 0.401, 0.025),
                              np.arange(0.45, 0.801, 0.05)], 4)


def compute_path(name, n_freq=512, grid=None, seed=0, max_rows=None,
                 max_iter=4000, tag="", bw_scale=1.0, threshold="paper",
                 outdir=None):
    """Solve the trimmed-MMD path with warm starts and cache everything."""
    t0 = time.time()
    out = out_dir(outdir)
    F, g, y, meta = load(name, max_rows, seed)
    n0, n1 = int((g == 0).sum()), int((g == 1).sum())
    print(f"{name}: {meta['source']}; n0(men)={n0} n1(women)={n1} "
          f"d={F.shape[1]}", flush=True)
    Z, sigma, rng = featurize(F, g, n_freq, seed, bw_scale)
    Za, Zb = Z[g == 0], Z[g == 1]
    n, m = len(Za), len(Zb)
    grid = DEFAULT_GRID if grid is None else np.asarray(grid)
    thr_fn = threshold_fn(threshold, n, m)

    recs, Wa, Wb = [], [], []
    w0 = v0 = None
    for a in grid:
        res = solve_features(Za, Zb, a, w0=w0, v0=v0, tol=1e-10,
                             max_iter=max_iter, rng=rng)
        w0, v0 = res.w, res.v
        th = float(thr_fn(a))
        # An unconverged iterate overstates D, which is anti-conservative for
        # a test that rejects when D is large; reject on the Frank-Wolfe lower
        # bound of the minimum instead.
        rej = bool(res.discrepancy_lower > th)
        recs.append({"alpha": float(a), "d_hat": res.discrepancy,
                     "d_lower": res.discrepancy_lower, "tau": th,
                     "rejects": rej, "converged": bool(res.converged),
                     "n_iter": int(res.n_iter), "fw_gap": float(res.gap),
                     **threshold_fields(threshold, n, m, a)})
        Wa.append(res.w.astype(np.float32))
        Wb.append(res.v.astype(np.float32))
        print(f"  alpha={a:.3f}  D={res.discrepancy:.4f}  "
              f"Dlow={res.discrepancy_lower:.4f}  tau={th:.4f}  "
              f"{'REJECT' if rej else 'accept'}  it={res.n_iter} "
              f"({time.time() - t0:.0f}s)", flush=True)
    alpha_hat, last_rej = summarize_path(recs)
    print(f"analytic alpha-hat = {alpha_hat}  (certified TV > {last_rej})",
          flush=True)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{name}{tag}_path.npz",
                        alphas=grid, Wa=np.array(Wa), Wb=np.array(Wb))
    info = {"dataset": name, **meta, "n0": n0, "n1": n1, "d": int(F.shape[1]),
            "features": list(F.columns), "sigma": sigma, "n_freq": n_freq,
            "delta": DELTA, "alpha_hat": alpha_hat, "last_rejected": last_rej,
            "max_rows": max_rows, "seed": seed, "bw_scale": bw_scale, "path": recs,
            "seconds": time.time() - t0, **threshold_meta(threshold)}
    (out / f"{name}{tag}_info.json").write_text(json.dumps(info, indent=1))
    return info


def rethreshold(name, tag="", threshold="plugin", outdir=None, source=None):
    """Re-apply a threshold to the stored path of an earlier run, no re-solving.

    Reads ``<source>/<name><tag>_info.json`` (default results/fair, the v1 run),
    recomputes tau, the decisions (on the Frank-Wolfe lower bound, as the path
    stage does), alpha-hat and the last rejected level, and writes the info to
    ``<outdir>/fair``; the path npz (warm-started weights, threshold-free) is
    copied next to it so that the audit stage can run from there. The audit
    summary ``orig`` of the source run (unrepaired data, threshold-free) is kept;
    the audit stage recomputes it.
    """
    import shutil
    src = OUT if source is None else Path(source)
    if not src.is_absolute():
        src = ROOT / src
    out = out_dir(outdir)
    if src.resolve() == out.resolve():
        raise SystemExit("rethreshold: --outdir must differ from the source directory")
    info = json.loads((src / f"{name}{tag}_info.json").read_text())
    n, m = int(info["n0"]), int(info["n1"])
    delta = float(info.get("delta", DELTA))
    thr_fn = threshold_fn(threshold, n, m, delta)
    recs = []
    for r in info["path"]:
        r = dict(r)
        r["tau"] = float(thr_fn(r["alpha"]))
        r["rejects"] = bool(r["d_lower"] > r["tau"])
        r.update(threshold_fields(threshold, n, m, r["alpha"], delta))
        recs.append(r)
        print(f"  alpha={r['alpha']:.3f}  Dlow={r['d_lower']:.4f}  tau={r['tau']:.4f}  "
              f"{'REJECT' if r['rejects'] else 'accept'}", flush=True)
    alpha_hat, last_rej = summarize_path(recs)
    info.update({"path": recs, "alpha_hat": alpha_hat, "last_rejected": last_rej,
                 "rethresholded_from": str(src / f"{name}{tag}_info.json"),
                 **threshold_meta(threshold)})
    if threshold == "paper":
        info["threshold"] = "paper"
    (out / f"{name}{tag}_info.json").write_text(json.dumps(info, indent=1))
    shutil.copy2(src / f"{name}{tag}_path.npz", out / f"{name}{tag}_path.npz")
    print(f"{name}{tag}: analytic alpha-hat = {alpha_hat}  (certified TV > {last_rej}) "
          f"-> {out}", flush=True)
    return info


# ---------------------------------------------------------------------------
# stage 2: the repaired datasets and the audit
# ---------------------------------------------------------------------------

AUDIT_ALPHAS = (0.05, 0.10, 0.175, 0.25, 0.35, 0.45, 0.50, 0.55, 0.60, 0.70)


def resolve_alphas(alphas, info):
    """Audit levels; the token ``"ahat"`` stands for the run's own alpha-hat."""
    out = []
    for a in alphas:
        if isinstance(a, str) and a.lower() == "ahat":
            ah = info.get("alpha_hat")
            if ah is None or ah != ah:          # nan: no accepted level
                continue
            a = ah
        a = round(float(a), 4)
        if a not in out:
            out.append(a)
    return out


def run_audit(name, alphas=AUDIT_ALPHAS, seed=0, max_rows=None, boot=0,
              tag="", n_freq=512, outdir=None):
    t0 = time.time()
    OUT = out_dir(outdir)                      # noqa: N806 (v1 name, v2 location)
    info = json.loads((OUT / f"{name}{tag}_info.json").read_text())
    P = np.load(OUT / f"{name}{tag}_path.npz")
    alphas = resolve_alphas(alphas, info)
    F, g, y, _ = load(name, info.get("max_rows"), info.get("seed", 0))
    Z, _, rng = featurize(F, g, info["n_freq"], info.get("seed", 0),
                          info.get("bw_scale", 1.0))
    Za, Zb = Z[g == 0], Z[g == 1]
    ia, ib = np.nonzero(g == 0)[0], np.nonzero(g == 1)[0]
    recs = {round(r["alpha"], 4): r for r in info["path"]}
    grid = [round(a, 4) for a in P["alphas"]]

    ones = np.ones(len(g))
    base_audit, _ = clf_tv_lower_bound(F, g, ones, seed=seed, boot=boot)
    r0, r1, gap0 = outcome_gap(y, g, ones)
    dp0 = clf_dp_gap(F, g, y, ones, seed=seed)
    orig = {"smd_max": smd_max(F, g, ones)[0], **base_audit,
            "rate_g0": r0, "rate_g1": r1, "gap": gap0, "clf_dp": dp0}
    print(f"original: SMDmax={orig['smd_max']:.3f} "
          f"TV_lb(clf)={base_audit['tv_lb']:.3f} AUC={base_audit['auc']:.3f}; "
          f"P(y=1|g)={r0:.3f}/{r1:.3f}; clf DP gap {dp0['dp_gap']:.3f}",
          flush=True)

    e_logit = crossfit_propensity(F, g, "logit", seed=seed)
    e_gbm = crossfit_propensity(F, g, "gbm", seed=seed)

    rows, weights = [], {}
    for a in alphas:
        a = round(a, 4)
        if a not in grid:
            continue
        k = grid.index(a)
        wa, wb = P["Wa"][k].astype(float), P["Wb"][k].astype(float)
        rec = recs[a]
        least = False
        if not rec["rejects"]:
            from trimmedmmd.solver import QPResult
            base = QPResult(value=rec["d_hat"] ** 2, w=wa, v=wb, n_iter=0,
                            converged=True, gap=rec["fw_gap"])
            wa, wb, _ = least_trimming(Za, Zb, a, base, rng=rng)
            least = True
        w_kernel = np.zeros(len(g))
        w_kernel[ia], w_kernel[ib] = wa, wb
        w_rand = np.zeros(len(g))
        for ix in (ia, ib):
            w_rand[rng.choice(ix, int(round((1 - a) * len(ix))),
                              replace=False)] = 1.0
        methods = [("kernel trimming", w_kernel),
                   ("random deletion", w_rand),
                   ("propensity (logit)", quantile_trim_weights(e_logit, g, a)),
                   ("propensity (GBM, cross-fit)",
                    quantile_trim_weights(e_gbm, g, a))]
        weights[a] = w_kernel
        for method, wm in methods:
            wmn = wm.copy()
            for kk in (0, 1):
                wmn[g == kk] /= wmn[g == kk].sum()
            au, _ = clf_tv_lower_bound(F, g, wmn, seed=seed, boot=boot)
            smd = smd_max(F, g, wmn)[0]
            r0a, r1a, gapa = outcome_gap(y, g, wmn)
            dp = clf_dp_gap(F, g, y, wmn, seed=seed)
            rows.append({
                "alpha": a, "method": method,
                "support_frac_g0": float((wmn[g == 0] > 1e-12).mean()),
                "support_frac_g1": float((wmn[g == 1] > 1e-12).mean()),
                "ess_frac_g0": float(1.0 / ((wmn[g == 0] ** 2).sum()
                                            * (g == 0).sum())),
                "ess_frac_g1": float(1.0 / ((wmn[g == 1] ** 2).sum()
                                            * (g == 1).sum())),
                "smd_max": smd, "tv_lb_clf": au["tv_lb"], "auc": au["auc"],
                "tv_lb_se": au.get("tv_lb_se", float("nan")),
                "rate_g0": r0a, "rate_g1": r1a, "outcome_gap": gapa,
                "clf_dp_gap": dp["dp_gap"], "clf_accuracy": dp["accuracy"],
                "least_trimming": least if method == "kernel trimming" else None,
                "d_hat_path": rec["d_hat"] if method == "kernel trimming" else None,
                "tau_path": rec["tau"] if method == "kernel trimming" else None})
        sub = rows[-4:]
        print(f"  alpha={a:.3f} " + " | ".join(
            f"{r['method'].split(' (')[0][:9]} SMD={r['smd_max']:.2f} "
            f"TVlb={r['tv_lb_clf']:.3f}" for r in sub)
            + f"  ({time.time() - t0:.0f}s)", flush=True)
        # persist after every level, so an interrupted audit keeps its work
        pd.DataFrame(rows).to_csv(OUT / f"{name}{tag}_audit.csv", index=False)
        np.savez_compressed(OUT / f"{name}{tag}_weights.npz", g=g, y=y,
                            alphas=np.array(list(weights)),
                            W=np.array([weights[x] for x in weights]))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"{name}{tag}_audit.csv", index=False)
    np.savez_compressed(OUT / f"{name}{tag}_weights.npz", g=g, y=y,
                        alphas=np.array(list(weights)),
                        W=np.array([weights[a] for a in weights]))
    info["orig"] = orig
    (OUT / f"{name}{tag}_info.json").write_text(json.dumps(info, indent=1))
    print(f"audit done in {time.time() - t0:.0f}s", flush=True)
    return df


def _alpha_token(x):
    return x if x.lower() == "ahat" else float(x)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["adult", "acs"])
    ap.add_argument("--stage", choices=["path", "audit", "both", "rethreshold"],
                    default="both")
    ap.add_argument("--n-freq", type=int, default=512)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--bw-scale", type=float, default=1.0)
    ap.add_argument("--max-iter", type=int, default=4000)
    ap.add_argument("--grid", type=float, nargs="*", default=None)
    ap.add_argument("--audit-alphas", type=_alpha_token, nargs="*",
                    default=AUDIT_ALPHAS,
                    help="audit levels; 'ahat' = this run's analytic alpha-hat")
    ap.add_argument("--source-dir", default=None,
                    help="rethreshold: directory of the run to re-threshold "
                         "(default results/fair)")
    from trimmedmmd.experiments.v2 import add_v2_args
    add_v2_args(ap, certificate=False)
    a = ap.parse_args()
    if a.threshold != "paper" and a.outdir is None:
        raise SystemExit("v2 runs need --outdir (e.g. results/v2); refusing to overwrite "
                         "the published results/fair")
    if a.stage == "rethreshold":
        if a.outdir is None:
            raise SystemExit("--stage rethreshold needs --outdir (never overwrite results/fair)")
        rethreshold(a.dataset, tag=a.tag, threshold=a.threshold, outdir=a.outdir,
                    source=a.source_dir)
    if a.stage in ("path", "both"):
        compute_path(a.dataset, n_freq=a.n_freq, grid=a.grid, seed=a.seed,
                     max_rows=a.max_rows, max_iter=a.max_iter, tag=a.tag,
                     bw_scale=a.bw_scale, threshold=a.threshold, outdir=a.outdir)
    if a.stage in ("audit", "both"):
        run_audit(a.dataset, alphas=a.audit_alphas, seed=a.seed, boot=a.boot,
                  tag=a.tag, outdir=a.outdir)

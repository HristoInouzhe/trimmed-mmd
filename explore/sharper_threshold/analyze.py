"""Tables for NOTE.md from the persisted replicates.

    python3 -I explore/sharper_threshold/analyze.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import common as C
import thresholds as T

DELTA = 0.05
G = T.GAUSS
NAMES = ["old", "A", "P_closed", "P", "P_coupled", "new"]


@lru_cache(maxsize=None)
def thr(name, n, m, alpha):
    if name == "old":
        return T.tau_old(n, m, alpha, DELTA)
    if name == "A":
        return T.tau_A(n, m, alpha, DELTA, **G)
    if name == "P_closed":
        return T.tau_P_closed(n, m, alpha, DELTA, **G)
    if name == "P":
        return T.tau_P(n, m, alpha, DELTA, **G)
    if name == "P_coupled":
        return T.tau_P_coupled(n, m, alpha, DELTA, kappa=1.0)
    if name == "new":
        return min(thr("A", n, m, alpha), thr("P", n, m, alpha), thr("P_coupled", n, m, alpha))
    raise KeyError(name)


def fmt(df):
    return df.to_markdown(index=False, floatfmt=".3f")


def table_thresholds():
    rows = []
    for n in (300,):
        for a in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
            r = dict(n=n, alpha=a)
            for k in NAMES:
                r[k] = thr(k, n, n, a)
            r["old/new"] = r["old"] / r["new"]
            r["n-factor"] = r["old/new"] ** 2
            rows.append(r)
    # the pieces of tau_P at n = 300
    pieces = []
    for a in (0.0, 0.2):
        val, det = T.tau_P(300, 300, a, DELTA, detail=True, **G)
        pieces.append(dict(alpha=a, tau_P=val, t1=det["t1"], rho_t2=G["rho"] * det["t2"],
                           delta1_share=det["split"]))
    return pd.DataFrame(rows), pd.DataFrame(pieces)


def table_validity():
    p = C.OUT / "validity_reps.jsonl"
    if not p.exists():
        return None, None
    df = pd.DataFrame([json.loads(l) for l in p.read_text().splitlines()])
    rows, ev = [], []
    for key, g in df.groupby("design", sort=False):
        n, a = int(g.n.iloc[0]), float(g.alpha.iloc[0])
        q = float(np.quantile(g.dhat, 1 - DELTA))
        r = dict(design=key, alpha=a, reps=len(g), mean_dhat=g.dhat.mean(), q95=q,
                 conv=g.converged.mean())
        for k in NAMES:
            t = thr(k, n, n, a)
            r[f"rej_{k}"] = float(np.mean(g.dhat > t))
            r[f"ratio_{k}"] = t / q if q > 0 else np.nan
        rows.append(r)
        if "M" in g and g["M"].notna().all():
            val, det = T.tau_P(n, n, a, DELTA, detail=True, **G)
            t1, t2 = det["t1"], det["t2"]
            bound = g.M + G["rho"] * (g.cX + g.cY)
            bound_emp = g.M + g.rho_emp * (g.cX + g.cY)
            ev.append(dict(
                design=key, alpha=a,
                max_dhatlo_minus_plugin=float((g.dhat_lower - g.plugin_value).max()),
                max_plugin_minus_bound=float((g.plugin_value - bound_emp).max()),
                plugin_all_feasible=bool(g.plugin_feasible.all()),
                q95_plugin=float(np.quantile(g.plugin_value, 1 - DELTA)),
                q95_M=float(np.quantile(g.M, 1 - DELTA)),
                q95_repair=float(np.quantile(G["rho"] * (g.cX + g.cY), 1 - DELTA)),
                t1=t1, rho_t2=G["rho"] * t2,
                freq_M_gt_t1=float(np.mean(g.M > t1)),
                freq_c_gt_t2=float(np.mean(g.cX + g.cY > t2)),
                freq_bound_gt_tauP=float(np.mean(bound > val)),
                max_bound_over_tauP=float((bound / val).max()),
            ))
    return pd.DataFrame(rows), pd.DataFrame(ev)


REPS_BY_N = {200: 100, 400: 100, 800: 100, 1600: 36, 3200: 14}


def _power_recs(p):
    """Deduplicated replicates, restricted to the planned seeds per size."""
    seen, out = set(), []
    for l in p.read_text().splitlines():
        r = json.loads(l)
        key = (r["n"], r["seed"])
        if key in seen or r["seed"] - 70_000 >= REPS_BY_N.get(r["n"], 10 ** 9):
            continue
        seen.add(key)
        out.append(r)
    return out


def table_power():
    p = C.OUT / "power_reps.jsonl"
    if not p.exists():
        return None, None
    recs = _power_recs(p)
    clf = pd.read_csv(C.ROOT / "results" / "tables" / "e6b_classifier_baseline_replicates.csv")
    from power import ALPHAS
    rows = []
    for n in sorted({r["n"] for r in recs}):
        R = [r for r in recs if r["n"] == n]
        D = np.array([r["dhat"] for r in R])
        DL = np.array([r["dhat_lower"] for r in R])
        n2 = R[0]["n2"]
        for j, a in enumerate(ALPHAS):
            row = dict(n=n, alpha=a, reps=len(R), mean_dhat=D[:, j].mean())
            for k in NAMES:
                t = thr(k, n, n, a)
                row[k] = float(np.mean(D[:, j] > t))
                if k == "new":
                    row["new_certified_lower"] = float(np.mean(DL[:, j] > t))
                    row["new_lower_same"] = bool(np.all((DL[:, j] > t) == (D[:, j] > t)))
                row[f"tau_{k}"] = t
            sp = [next(s for s in r["split"] if abs(s["alpha"] - a) < 1e-9) for r in R]
            beta = 1 / (1 - a)
            e = lambda k: math.sqrt(math.log(4 / DELTA) / (2 * k))
            crit17 = 2 * beta * (e(n2) + e(n2))
            row["prop17"] = float(np.mean([s["sep"] > crit17 for s in sp]))
            row["bootstrap"] = float(np.mean([s["certified"] for s in sp]))
            c = clf[clf.n == n]
            row["classifier_gbm"] = float(np.mean(c.gbm_L > a)) if len(c) else np.nan
            rows.append(row)
    df = pd.DataFrame(rows)
    # first n with power >= 0.8, per alpha < alpha* = 0.2
    first = []
    for a in ALPHAS[:-1]:
        g = df[np.isclose(df.alpha, a)].sort_values("n")
        r = dict(alpha=a)
        for k in NAMES + ["prop17", "bootstrap", "classifier_gbm"]:
            ok = g[g[k] >= 0.8]
            r[k] = int(ok.n.iloc[0]) if len(ok) else f">{int(g.n.max())}"
        # signal-threshold crossing estimate n*: mean D-hat at the largest n
        # against tau ~ c/sqrt(n) (the statistic's mean is nearly flat in n)
        gl = g.iloc[-1]
        for k in ("old", "new"):
            c0 = gl[f"tau_{k}"] * math.sqrt(gl.n)
            r[f"n_cross_{k}"] = int(round((c0 / gl.mean_dhat) ** 2))
        first.append(r)
    return df, pd.DataFrame(first)


def table_boundary_ratio():
    """tau / q95 at the boundary alpha = alpha* = 0.2 of the main design
    (median bandwidth), from the alpha = 0.2 point of the power paths."""
    p = C.OUT / "power_reps.jsonl"
    if not p.exists():
        return None
    recs = _power_recs(p)
    rows = []
    for n in sorted({r["n"] for r in recs}):
        d = np.array([r["dhat"][-1] for r in recs if r["n"] == n])
        q = float(np.quantile(d, 1 - DELTA))
        row = dict(n=n, reps=len(d), q95=q)
        for k in ("old", "A", "P", "P_coupled", "new"):
            row[f"ratio_{k}"] = thr(k, n, n, 0.2) / q
        rows.append(row)
    return pd.DataFrame(rows)


def table_slack():
    """tau / q95 on the published one-sided slack design (Table onesided_slack:
    disjoint, d = 2, alpha = alpha*, fixed wide bandwidth, 160 reps; the
    statistic there uses 256 cos/sin random features of the Gaussian kernel)."""
    d = pd.read_csv(C.ROOT / "results" / "tables" / "onesided_null.csv")
    rows = []
    for _, r in d.iterrows():
        n, a = int(r.n), float(r.alpha)
        row = dict(alpha=a, n=n, q95=r.q95_Dhat)
        for k in ("old", "A", "P", "P_coupled", "new"):
            row[f"ratio_{k}"] = thr(k, n, n, a) / r.q95_Dhat
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    out = []
    t, pieces = table_thresholds()
    out += ["## thresholds at n=m=300, delta=0.05", fmt(t), "", fmt(pieces), ""]
    v, ev = table_validity()
    if v is not None:
        out += ["## validity (boundary nulls, n=m=300)", fmt(v), "", fmt(ev), ""]
        fl = (v.sort_values("q95", ascending=False).groupby("alpha", as_index=False).first()
              [["alpha", "design", "q95", "reps", "ratio_new", "ratio_old"]])
        out += ["## data-free floor: largest null q95 found per alpha", fmt(fl), ""]
    p, first = table_power()
    if p is not None:
        out += ["## power, disjoint_d2_a20", fmt(p), "", "## first n with power >= 0.8", fmt(first), ""]
    b = table_boundary_ratio()
    if b is not None:
        out += ["## tau / q95 at the boundary of the main design (from the power paths)", fmt(b), ""]
    s = table_slack()
    out += ["## tau / q95 on the published slack design", fmt(s), ""]
    rd = C.OUT / "realdata_summary.csv"
    if rd.exists():
        out += ["## real data", fmt(pd.read_csv(rd)), ""]
    text = "\n".join(out)
    (C.OUT / "tables.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

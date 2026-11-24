"""Tables and figures from out/synth_<scenario>.jsonl."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

GRID = [0.05, 0.08, 0.11, 0.14, 0.17, 0.20, 0.23, 0.26]
METHODS = ["boot", "psi_clt", "ramp_clt",                       # asymptotic
           "prop17", "tlm", "psi_mpu", "psi_b2s", "psi_bet",    # finite-sample, psi
           "ramp_hoef", "ramp_mpu", "ramp_b2s", "ramp_mp_pair", "ramp_bet", "ramp_bet75",
           "dir0_ramp_bet", "cf_avg_bet", "cf_union_bet"]
GBM = ["gbm_hoef", "gbm_b2s", "gbm_bet"]


def load(scen="disjoint_d2_a20", tag=""):
    recs = [json.loads(l) for l in (HERE / "out" / f"synth_{scen}{tag}.jsonl").read_text().splitlines()]
    gfile = HERE / "out" / f"gbm_{scen}.jsonl"
    if gfile.exists() and not tag:
        # classifier columns recomputed with a predictable betting prior (gbm_rerun.py)
        gb = {(g["n"], g["rep"]): g["gbm"] for g in map(json.loads, gfile.read_text().splitlines())}
        for r in recs:
            r["gbm"] = gb.get((r["n"], r["rep"]), {})
    rows = []
    for r in recs:
        for a, d in r["levels"].items():
            row = {"n": r["n"], "rep": r["rep"], "alpha": float(a)}
            row.update({k: v for k, v in d.items()})
            for g in GBM:
                key = f"{g}@{float(a)}"
                if "gbm" in r and key in r["gbm"]:
                    row[g] = r["gbm"][key]
            rows.append(row)
    return pd.DataFrame(rows)


def rates(df, alphas=(0.0, 0.1, 0.15, 0.2)):
    cols = [c for c in METHODS + GBM if c in df.columns]
    out = (df[df.alpha.isin(alphas)].groupby(["alpha", "n"])[cols]
           .mean().reset_index())
    cnt = df[df.alpha.isin(alphas)].groupby(["alpha", "n"]).size().rename("reps").reset_index()
    return out.merge(cnt, on=["alpha", "n"])


def alpha_lo(df, method):
    """Down-closed lower bound over GRID (cor:lower), per replicate."""
    sub = df[df.alpha.isin(GRID)].pivot_table(index=["n", "rep"], columns="alpha", values=method)
    vals = []
    for idx, row in sub.iterrows():
        best = 0.0
        for a in GRID:
            if bool(row.get(a, False)):
                best = a
            else:
                break
        vals.append((idx[0], idx[1], best))
    return pd.DataFrame(vals, columns=["n", "rep", method])


def n_at_power(r, method, alpha, target=0.8):
    """Interpolate (log n) the smallest n with power >= target; nan if never."""
    d = r[np.isclose(r.alpha, alpha)].sort_values("n")
    ns, ps = d.n.values.astype(float), d[method].values
    if len(ps) == 0 or np.all(ps < target):
        return float("nan")
    if ps[0] >= target:
        return float(f"{ns[0]:.0f}") * 0 + -ns[0]   # flagged "<= smallest n" as negative
    k = int(np.argmax(ps >= target))
    x0, x1 = math.log(ns[k - 1]), math.log(ns[k])
    y0, y1 = ps[k - 1], ps[k]
    return float(math.exp(x0 + (target - y0) * (x1 - x0) / (y1 - y0)))


def main(scen="disjoint_d2_a20", tag=""):
    df = load(scen, tag)
    r = rates(df)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    r.to_csv(HERE / "out" / f"rates_{scen}{tag}_all.csv", index=False)
    print(r.round(3).to_string(index=False))
    # sample size for power 0.8
    rows = []
    for a in (0.0, 0.1, 0.15):
        row = {"alpha": a}
        for m in [c for c in METHODS + GBM if c in r.columns]:
            row[m] = n_at_power(r, m, a)
        rows.append(row)
    n80 = pd.DataFrame(rows)
    n80.to_csv(HERE / "out" / f"n80_{scen}{tag}.csv", index=False)
    print("\nn for power 0.8 (negative = already >= 0.8 at that smallest n; nan = not reached):")
    print(n80.set_index("alpha").round(0).to_string())
    # alpha_lo
    if df.alpha.isin(GRID).any():
        los = None
        for m in [c for c in METHODS + GBM if c in df.columns and c not in ("cf_avg_bet", "cf_union_bet")]:
            if df[df.alpha.isin(GRID)][m].isna().all():
                continue
            t = alpha_lo(df, m)
            los = t if los is None else los.merge(t, on=["n", "rep"])
        summ = los.groupby("n").mean().drop(columns="rep")
        cover = (los.drop(columns="rep").groupby("n").agg(lambda x: np.mean(x > 0.2 + 1e-12)))
        summ.to_csv(HERE / "out" / f"alphalo_mean_{scen}{tag}.csv")
        cover.to_csv(HERE / "out" / f"alphalo_exceed_{scen}{tag}.csv")
        print("\nmean alpha_lo (truth 0.2):")
        print(summ.round(3).to_string())
        print("\nP(alpha_lo > TV):")
        print(cover.round(3).to_string())
    # population diagnostics
    if "T_pop" in df.columns:
        d = df.dropna(subset=["T_pop"]).copy()
        d["slack_rel"] = (d.T_pop - d.Ts_pop) / d.T_pop.abs()
        d["clip_gain"] = d.Tclip_pop - d.Ts_pop
        d["sd_ratio"] = d.sd_ramp_T / d.sd_psi_q
        d["snr_boot"] = d.T_pop / d.sd_psi_q
        d["snr_ramp"] = d.Tclip_pop / d.sd_ramp_T
        g = d[d.alpha.isin([0.0, 0.1, 0.15, 0.2])].groupby(["alpha", "n"])[
            ["T_pop", "Ts_pop", "Tclip_pop", "slack_rel", "clip_gain", "sd_psi_q", "sd_ramp_T",
             "sd_ratio", "snr_boot", "snr_ramp", "Delta_pop"]].median()
        g.to_csv(HERE / "out" / f"popdiag_{scen}{tag}.csv")
        print("\npopulation diagnostics (medians over replicates):")
        print(g.round(4).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:])

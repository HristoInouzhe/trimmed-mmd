"""Markdown tables for NOTE.md, generated from out/ (no hand-typed numbers)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "out"
TAB = HERE / "tables"

LAB = {"boot": "bootstrap (Thm 16)", "ramp_clt": "ramp + CLT (asy.)", "ramp_bet": "**ramp + betting**",
       "ramp_bet75": "ramp + betting, c=3/4", "dir0_ramp_bet": "ramp + betting, α=0 witness",
       "cf_union_bet": "cross-fit (union, δ/2)", "cf_avg_bet": "cross-fit (e-value avg.)",
       "ramp_b2s": "ramp + 2-sample emp. Bernstein", "ramp_mp_pair": "ramp + MP Thm 4 on pairs",
       "ramp_mpu": "ramp + MP Thm 4 per sample", "ramp_hoef": "ramp + Hoeffding",
       "psi_bet": "unclipped ψ + betting", "psi_b2s": "unclipped ψ + emp. Bernstein",
       "psi_mpu": "unclipped ψ + MP per sample", "tlm": "DKW band (Thomas-Learned-Miller)",
       "prop17": "Prop 17 (DKW, 2κβ)", "gbm_hoef": "GBM classifier + Hoeffding (App. E.3)",
       "gbm_b2s": "GBM classifier + emp. Bernstein", "gbm_bet": "GBM classifier + betting",
       "psi_clt": "unclipped ψ + CLT (asy.)", "seq3": "sequential 3 batches + betting",
       "seq4": "sequential 4 batches + betting", "s30": "single split 30/70 + betting",
       "psi_bet75": "unclipped ψ + betting, c=3/4", "ramp_bet100": "ramp + betting, c=1"}


def md(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def power_tables():
    r = pd.read_csv(OUT / "rates_disjoint_d2_a20_all.csv")
    seq = OUT / "seq_rates_disjoint_d2_a20.csv"
    if seq.exists():
        s = pd.read_csv(seq)[["alpha", "n"] + [c for c in ("s30", "seq3", "seq4")
                                               if c in pd.read_csv(seq).columns]]
        r = r.merge(s, on=["alpha", "n"], how="left")
    meths = ["boot", "ramp_clt", "ramp_bet", "ramp_bet75", "dir0_ramp_bet", "seq3", "seq4", "s30",
             "cf_union_bet", "ramp_b2s", "ramp_mp_pair", "ramp_mpu", "ramp_hoef", "psi_bet", "tlm",
             "prop17", "gbm_hoef", "gbm_bet"]
    meths = [m for m in meths if m in r.columns]
    out = []
    for a in (0.0, 0.1, 0.15, 0.2):
        d = r[np.isclose(r.alpha, a)].sort_values("n")
        t = pd.DataFrame({"method": [LAB[m] for m in meths]})
        for n in d.n.values:
            t[f"n={int(n)}"] = [float(d.loc[d.n == n, m].iloc[0]) if m in d else np.nan for m in meths]
        title = (f"α = {a:g}: power (TV = 0.2 > α)" if a < 0.2 else
                 "α = 0.2 = TV: rejection rate under the null boundary (type-I, nominal 0.05)")
        out.append(f"**{title}** (200 replicates for n ≤ 800, 100 at n = 1600)\n\n" + md(t, "{:.2f}"))
    return "\n\n".join(out)


def n80_table():
    import analyze_synth as A
    r = pd.read_csv(OUT / "rates_disjoint_d2_a20_all.csv")
    seq = OUT / "seq_rates_disjoint_d2_a20.csv"
    if seq.exists():
        s = pd.read_csv(seq)
        r = r.merge(s[["alpha", "n"] + [c for c in ("s30", "seq3", "seq4") if c in s.columns]],
                    on=["alpha", "n"], how="left")
    meths = ["boot", "ramp_bet", "ramp_bet75", "dir0_ramp_bet", "seq3", "seq4", "cf_union_bet",
             "ramp_b2s", "psi_bet", "tlm", "prop17", "gbm_hoef", "gbm_bet"]
    meths = [m for m in meths if m in r.columns]
    rows = []
    for m in meths:
        row = {"method": LAB[m]}
        for a in (0.0, 0.1, 0.15):
            dd = r[~r[m].isna()]
            v = A.n_at_power(dd, m, a)
            nmax = int(dd.n.max())
            row[f"α={a:g}"] = ("≤100" if v < 0 else (f"> {nmax}" if np.isnan(v) else f"{v:.0f}"))
        rows.append(row)
    return md(pd.DataFrame(rows))


def alphalo_table():
    m = pd.read_csv(OUT / "alphalo_mean_disjoint_d2_a20.csv", index_col=0)
    e = pd.read_csv(OUT / "alphalo_exceed_disjoint_d2_a20.csv", index_col=0)
    meths = ["boot", "ramp_bet", "ramp_bet75", "dir0_ramp_bet", "ramp_b2s", "psi_bet", "tlm", "prop17",
             "gbm_hoef", "gbm_bet"]
    rows = []
    for k in meths:
        if k not in m.columns:
            continue
        row = {"method": LAB[k]}
        for n in m.index:
            row[f"n={n}"] = f"{m.loc[n, k]:.3f} ({e.loc[n, k]:.3f})"
        rows.append(row)
    return md(pd.DataFrame(rows))


def boundary_table():
    import analyze_synth as A
    rows = []
    meths = ["boot", "ramp_clt", "ramp_bet", "ramp_bet75", "ramp_b2s", "ramp_mp_pair", "psi_bet", "tlm",
             "prop17"]
    for f in sorted(OUT.glob("synth_*_boundary.jsonl")):
        scen = f.name[len("synth_"):-len("_boundary.jsonl")]
        df = A.load(scen, "_boundary")
        for (a, n), d in df.groupby(["alpha", "n"]):
            row = {"design": scen, "α=TV": f"{a:.3f}", "n": int(n), "reps": len(d)}
            for m in meths:
                if m in d:
                    row[LAB[m]] = float(d[m].mean())
            rows.append(row)
    return md(pd.DataFrame(rows), "{:.3f}")


def scalar_null_table():
    df = pd.read_csv(OUT / "null_scalar.csv")
    t = df.pivot_table(index=["family", "law"], columns="method", values="rate", aggfunc="max")
    t = t.reset_index()
    t.columns = [LAB.get(c, c) if c not in ("family", "law") else c for c in t.columns]
    return md(t, "{:.4f}")


def popdiag_table():
    g = pd.read_csv(OUT / "popdiag_disjoint_d2_a20.csv")
    g = g[g.alpha.isin([0.0, 0.1, 0.15, 0.2])]
    g = g[["alpha", "n", "T_pop", "Ts_pop", "Tclip_pop", "slack_rel", "sd_psi_q", "sd_ramp_T", "snr_boot",
           "snr_ramp"]]
    g.columns = ["α", "n", "T(f̂)", "θ(f̂;ŝ,ŝ')", "θ(clip f̂;ŝ,ŝ')", "rel. slack", "sd ψ_q (boot)",
                 "sd ramp (T units)", "SNR boot", "SNR ramp"]
    return md(g, "{:.4f}")


def real_tables():
    s = pd.read_csv(OUT / "real_summary.csv")
    meths = ["boot", "ramp_bet", "dir0_ramp_bet", "ramp_b2s", "ramp_mp_pair", "ramp_hoef", "ramp_clt", "tlm",
             "prop17"]
    names = {"rhc": "RHC", "nhanes_child": "NHANES 6–11 vs 60–79", "nhanes_age": "NHANES 20–39 vs 60–79",
             "nhanes_bmi": "NHANES BMI <25 vs ≥30", "adult": "Adult (men vs women)", "acs": "ACS income"}
    rows = []
    for d in names:
        dd = s[s.dataset == d]
        if dd.empty:
            continue
        row = {"contrast": names[d], "splits": int(dd["count"].max())}
        for m in meths:
            x = dd[dd.method == m]
            if len(x):
                mn, med, mx = x["min"].iloc[0], x["median"].iloc[0], x["max"].iloc[0]
                f = lambda v: "none" if v < 0 else f"{v:.2f}"
                row[LAB[m]] = f"{f(mn)} [{f(med)}, {f(mx)}]"
        rows.append(row)
    return md(pd.DataFrame(rows))


def real_fair_table():
    recs = [json.loads(l) for l in (OUT / "real_dir0.jsonl").read_text().splitlines()]
    root = HERE.parents[1]
    meths = ["boot_dir0", "ramp_bet", "ramp_b2s", "ramp_hoef", "ramp_clt", "tlm", "prop17"]
    lab = {"boot_dir0": "bootstrap (Thm 16), same α=0 witness", "ramp_bet": "**ramp + betting**, α=0 witness",
           "ramp_b2s": "ramp + 2-sample emp. Bernstein", "ramp_hoef": "ramp + Hoeffding",
           "ramp_clt": "ramp + CLT (asy.)", "tlm": "DKW band (T&LM), κ=√2", "prop17": "Prop 17, κ=√2"}
    rows = []
    f = lambda v: "none" if v < 0 else f"{v:.2f}"
    for d, name in (("adult", "Adult (men vs women)"), ("acs", "ACS income")):
        rr = [r for r in recs if r["dataset"] == d]
        if not rr:
            continue
        row = {"contrast": name, "splits": len(rr)}
        p = root / "results" / "fair" / f"{d}_split_cert.csv"
        if p.exists():
            q = pd.read_csv(p)
            q = q[q.method == "original"].certified_level
            row["published bootstrap, per-level witness (6 splits)"] = f"{f(q.min())} [{f(q.median())}, {f(q.max())}]"
        for m in meths:
            v = np.array([r["lower"][m] for r in rr])
            row[lab[m]] = f"{f(v.min())} [{f(np.median(v))}, {f(v.max())}]"
        rows.append(row)
    return md(pd.DataFrame(rows))


def seq_table():
    w = pd.read_csv(OUT / "seq_rates_disjoint_d2_a20.csv")
    w = w[w.alpha.isin([0.1, 0.15, 0.2])]
    cols = ["alpha", "n", "boot", "ramp_bet", "s30", "seq3", "seq4"]
    t = w[cols].copy()
    t.columns = ["α", "n", "bootstrap 50/50", "ramp+bet 50/50", "ramp+bet 30/70", "3 batches", "4 batches"]
    t["α"] = t["α"].map(lambda x: f"{x:g}")
    t["n"] = t["n"].astype(int).astype(str)
    return md(t, "{:.3f}")


if __name__ == "__main__":
    TAB.mkdir(exist_ok=True)
    jobs = {"power": power_tables, "n80": n80_table, "alphalo": alphalo_table,
            "boundary": boundary_table, "scalar": scalar_null_table, "popdiag": popdiag_table,
            "real": real_tables, "real_fair": real_fair_table, "seq": seq_table}
    for k, fn in jobs.items():
        try:
            (TAB / f"{k}.md").write_text(fn() + "\n")
            print("ok", k)
        except Exception as ex:  # noqa: BLE001
            print("skip", k, repr(ex))

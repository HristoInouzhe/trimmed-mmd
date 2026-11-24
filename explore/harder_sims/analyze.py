"""Tables and figures from out/<design>.jsonl (main runs: rep < 200).

    python3 -I explore/harder_sims/analyze.py
writes out/summary_long.csv, out/tables.md, figs/<design>.png, figs/overview.png
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as _beta, binom

HERE = Path(__file__).resolve().parent
OUT, FIGS = HERE / "out", HERE / "figs"
FIGS.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))

REPS = int(__import__("os").environ.get("HS_REPS", "200"))
JS = (0, 4, 6, 8)
LABEL_J = {0: "H0,0", 4: "H0,a*/2", 6: "H0,0.75a*", 8: "boundary"}
METHODS = ["an_old", "an_newcf", "an_new", "prop17", "boot", "ramp_b2s", "ramp_bet", "seq_bet",
           "gbm_hoef", "gbm_bet", "logit_hoef", "logit_bet", "knn_hoef", "knn_bet"]
FINITE = [m for m in METHODS if m != "boot"]
NAMES = {"an_old": "analytic, old tau", "an_newcf": "analytic, Thm P closed", "an_new": "analytic, tau_new",
         "prop17": "Prop 17 (DKW)", "boot": "bootstrap (Thm 16)", "ramp_b2s": "ramp, emp. Bernstein",
         "ramp_bet": "ramp, betting", "seq_bet": "ramp, betting 3/4 seq.",
         "gbm_hoef": "GBM, Hoeffding", "gbm_bet": "GBM, betting", "logit_hoef": "logistic, Hoeffding",
         "logit_bet": "logistic, betting", "knn_hoef": "kNN, Hoeffding", "knn_bet": "kNN, betting"}
# figure: 8 headline methods, categorical slots in fixed order (dataviz reference palette)
FIG_M = ["an_old", "an_new", "prop17", "boot", "ramp_bet", "seq_bet", "gbm_bet", "logit_bet"]
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STYLE = {"an_old": ("o", ":"), "an_new": ("o", "-"), "prop17": ("s", ":"), "boot": ("D", "--"),
         "ramp_bet": ("s", "-"), "seq_bet": ("^", "-"), "gbm_bet": ("v", "-"), "logit_bet": ("P", "-")}


def cp_upper(k, n, conf=0.95):
    return 1.0 if k >= n else float(_beta.ppf(1 - (1 - conf) / 2, k + 1, n - k))


def load():
    rows = []
    for f in sorted(OUT.glob("*.jsonl")):
        if "_" not in f.stem or f.stem.endswith("timing") or f.stem.endswith("_n0ext"):
            continue           # _n0ext: the post hoc alpha = 0 extension (rep >= 200000), summarized by ext_summary
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r["rep"] >= REPS:
                continue
            rows.append(r)
    return rows


def summarize(rows):
    import designs as DS
    C = DS.catalogue()
    by = {}
    for r in rows:
        by.setdefault((r["design"], r["n"]), {})[r["rep"]] = r
    recs, times = [], []
    for (d, n), reps in sorted(by.items()):
        R = len(reps)
        rs = list(reps.values())
        dz = C[d]
        lev = dz.levels()
        for mth in METHODS:
            if mth not in rs[0]["dec"]:
                continue
            dec = np.array([r["dec"][mth] for r in rs])
            jlo = np.array([r["jlo"][mth] for r in rs])
            alo = np.array([lev[j] if j >= 0 else 0.0 for j in jlo])
            k8 = int(dec[:, 3].sum())
            rec = dict(design=d, family=dz.family, n=n, m=rs[0]["m"], R=R, method=mth,
                       alpha_star=dz.alpha_star,
                       pow0=dec[:, 0].mean(), pow4=dec[:, 1].mean(), pow6=dec[:, 2].mean(),
                       typeI=k8 / R, typeI_k=k8, typeI_cp=cp_upper(k8, R),
                       typeI_p=float(binom.sf(k8 - 1, R, 0.05)) if k8 > 0 else 1.0,
                       alo_mean=alo.mean(), alo_gt=float(np.mean(jlo >= 9)),
                       alo_ge=float(np.mean(jlo >= 8)), alo_ge_k=int(np.sum(jlo >= 8)),
                       cert0=float(np.mean(jlo >= 0)))
            recs.append(rec)
        tk = set().union(*[r["time"].keys() for r in rs])
        t = {k: float(np.mean([r["time"].get(k, 0.0) for r in rs])) for k in tk}
        t.update(design=d, n=n, R=R,
                 an_undecided=int(sum(r["diag"].get("an_undecided", 0) for r in rs)),
                 an_qps=float(np.mean([r["diag"].get("an_qps", 0) for r in rs])),
                 an_iters=float(np.mean([r["diag"].get("an_iters", 0) for r in rs])))
        times.append(t)
    return pd.DataFrame(recs), pd.DataFrame(times)


def n80(df, design, method, col):
    s = df[(df.design == design) & (df.method == method) & (df.R >= REPS)].sort_values("n")
    hit = s[s[col] >= 0.8]
    if len(hit):
        return str(int(hit.n.iloc[0]))
    if len(s) == 0:
        return "-"
    return f">{int(s.n.max())}"


def fig_design(df, design, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = df[(df.design == design) & (df.R >= REPS)]
    if s.empty:
        return
    a = s.alpha_star.iloc[0]
    fig, axs = plt.subplots(1, 4, figsize=(13.5, 3.7), sharex=True)
    cols = [("pow0", "power, $H_{0,0}$"), ("pow4", r"power, $H_{0,\alpha^*/2}$"),
            ("pow6", r"power, $H_{0,0.75\alpha^*}$"), ("typeI", r"type-I at boundary $\alpha=\alpha^*$")]
    for ax, (c, lab) in zip(axs, cols):
        for i, mth in enumerate(FIG_M):
            t = s[s.method == mth].sort_values("n")
            if t.empty:
                continue
            mk, ls = STYLE[mth]
            ax.plot(t.n * 2 ** ((i - 3.5) * 0.035), t[c], color=PAL[i], marker=mk, ls=ls, lw=2, ms=6,
                    markeredgecolor="#fcfcfb", markeredgewidth=0.8, label=NAMES[mth])
        if c == "typeI":
            ax.axhline(0.05, color="#52514e", lw=1, ls="--")
            ax.text(s.n.min(), 0.055, r"$\delta$", color="#52514e", fontsize=9)
            ax.set_ylim(-0.01, max(0.2, float(s.typeI.max()) + 0.03))
        else:
            ax.axhline(0.8, color="#c3c2b7", lw=0.8, ls="-")
            ax.set_ylim(-0.03, 1.03)
        ax.set_xscale("log", base=2)
        ns = sorted(s.n.unique())
        ax.set_xticks(ns)
        ax.set_xticklabels([str(x) for x in ns])
        ax.set_title(lab, fontsize=10, color="#0b0b0b")
        ax.set_xlabel("n" + (" (m = 5n)" if s.m.iloc[0] != s.n.iloc[0] else " = m"), color="#52514e")
        ax.grid(True, color="#e6e5e0", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors="#52514e", labelsize=8)
    axs[0].set_ylabel("rejection rate", color="#52514e")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=8, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{design}: {title}  (TV = alpha* = {a:.4g}; 200 replicates per point)", fontsize=11,
                 color="#0b0b0b", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(FIGS / f"{design}.png", dpi=130, facecolor="#fcfcfb")
    plt.close(fig)


TITLES = {
    "D1_d2_a10": "overlapping Gaussian contamination, d=2", "D1_d2_a20": "overlapping Gaussian contamination, d=2",
    "D1_d10_a10": "overlapping Gaussian contamination, d=10", "D1_d10_a20": "overlapping Gaussian contamination, d=10",
    "D2_d20": "dense shift, d=20", "D2_d50": "dense shift, d=50", "D2_d100": "dense shift, d=100",
    "D3a_loc": "sparse shift (2 of 50 coords)", "D3b_corr": "sparse correlation flip (2 of 50 coords)",
    "D4_small": "small separated contamination", "D5b_mixed": "mixed types with ties, m = 5n",
    "D5a_imb": "D1 geometry, m = 5n", "D6_t3": "t3 core, scale contamination, d=5",
    "D7_wide": "Brownian curves + wide bump", "D7_narrow": "Brownian curves + narrow bump",
    "D8_N1": "oscillating density, N=1", "D8_N4": "oscillating density, N=4",
    "D8_N16": "oscillating density, N=16", "D8_N64": "oscillating density, N=64"}


def md_table(df, cols, fmt=".2f"):
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"{v:{fmt}}" if isinstance(v, (float, np.floating)) else str(v))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def main():
    rows = load()
    df, tm = summarize(rows)
    df.to_csv(OUT / "summary_long.csv", index=False)
    tm.to_csv(OUT / "runtime.csv", index=False)
    full = df[df.R >= REPS]
    designs = list(dict.fromkeys(full.design))
    for d in designs:
        if not d.startswith("N0"):
            fig_design(df, d, TITLES.get(d, ""))
    md = []
    # --- validity
    md.append("## Validity: boundary type-I (200 replicates per cell)\n")
    v = full.groupby(["design", "method"]).agg(max_typeI=("typeI", "max"), max_k=("typeI_k", "max"),
                                               max_cp=("typeI_cp", "max"), max_alo_ge=("alo_ge", "max"),
                                               min_p=("typeI_p", "min")).reset_index()
    flag = full[(full.typeI_k >= 19) | (full.alo_ge_k >= 19)]
    md.append(f"Cells: {len(full)} (design x n x method). Red flags (k >= 19 of 200, binomial p < 0.01): "
              f"{len(flag)}\n")
    if len(flag):
        md.append(md_table(flag[["design", "n", "method", "typeI_k", "typeI", "typeI_cp", "alo_ge"]],
                           ["design", "n", "method", "typeI_k", "typeI", "typeI_cp", "alo_ge"], ".3f"))
    over = full[(full.typeI > 0.05)]
    md.append(f"\nCells with type-I above 0.05 (any): {len(over)}\n")
    if len(over):
        md.append(md_table(over[["design", "n", "method", "typeI_k", "typeI", "typeI_cp", "typeI_p"]],
                           ["design", "n", "method", "typeI_k", "typeI", "typeI_cp", "typeI_p"], ".3f"))
    piv = v.pivot(index="design", columns="method", values="max_typeI")[[m for m in METHODS if m in v.method.unique()]]
    piv = piv.reindex([d for d in designs])
    md.append("\nMax over n of boundary type-I, per design and method:\n")
    md.append(md_table(piv.reset_index(), ["design"] + list(piv.columns), ".3f"))
    # --- n80 summary
    md.append("\n## Smallest n with power >= 0.8\n")
    for col, lab in (("pow0", "H_{0,0}"), ("pow4", "H_{0,a*/2}"), ("pow6", "H_{0,0.75a*}")):
        tab = []
        for d in designs:
            row = {"design": d}
            for mth in METHODS:
                row[mth] = n80(df, d, mth, col)
            tab.append(row)
        t = pd.DataFrame(tab)
        md.append(f"\n### {lab}\n")
        md.append(md_table(t, ["design"] + METHODS))
    # --- per design at largest n
    md.append("\n## Per design, all methods, at each n\n")
    for d in designs:
        s = full[full.design == d]
        md.append(f"\n### {d} ({TITLES.get(d, '')}; TV = {s.alpha_star.iloc[0]:.4g})\n")
        for n in sorted(s.n.unique()):
            t = s[s.n == n].copy()
            t["method"] = pd.Categorical(t.method, METHODS, ordered=True)
            t = t.sort_values("method")
            md.append(f"\nn = {n}, m = {int(t.m.iloc[0])}\n\n")
            md.append(md_table(t[["method", "pow0", "pow4", "pow6", "typeI", "typeI_cp", "alo_mean", "alo_ge",
                                  "alo_gt"]],
                               ["method", "pow0", "pow4", "pow6", "typeI", "typeI_cp", "alo_mean", "alo_ge",
                                "alo_gt"], ".3f"))
    # --- runtime
    md.append("\n## Mean runtime per replicate (s)\n")
    cols = [c for c in ["analytic", "split_kernel", "boot_extra", "seq_extra", "gbm", "logit", "knn", "total",
                        "an_qps", "an_iters", "an_undecided"] if c in tm.columns]
    tt = tm[tm.R >= REPS][["design", "n"] + cols]
    md.append(md_table(tt, ["design", "n"] + cols, ".2f"))
    (OUT / "tables.md").write_text("\n".join(md))
    print(f"{len(rows)} replicates; {len(full)} complete cells; {len(flag)} red flags; figures for {len(designs)}")


if __name__ == "__main__" and len(sys.argv) == 1:
    main()


# ---------------------------------------------------------------------------
# paired comparisons (same replicates): diff of rejection rates, paired s.e.
# ---------------------------------------------------------------------------

PAIRS = [("an_new", "an_old"), ("an_new", "an_newcf"), ("ramp_bet", "prop17"), ("ramp_bet", "ramp_b2s"),
         ("ramp_bet", "boot"), ("seq_bet", "boot"), ("seq_bet", "ramp_bet"), ("ramp_bet", "an_new"), ("seq_bet", "an_new"),
         ("ramp_bet", "gbm_bet"), ("ramp_bet", "logit_bet"), ("ramp_bet", "knn_bet"),
         ("seq_bet", "gbm_bet"), ("seq_bet", "logit_bet"), ("gbm_bet", "gbm_hoef"), ("logit_bet", "logit_hoef")]


def paired(rows, extra=None):
    """extra: dict (design, n, rep) -> {'dec': {...}} merged into rows (post hoc methods)."""
    by = {}
    for r in rows:
        dd = dict(r["dec"])
        if extra is not None and (r["design"], r["n"], r["rep"]) in extra:
            dd.update(extra[(r["design"], r["n"], r["rep"])])
        by.setdefault((r["design"], r["n"]), []).append(dd)
    out = []
    pairs = list(PAIRS)
    if extra:
        pairs += [("ramp_bet", "gbm_soft_bet"), ("ramp_bet", "logit_soft_bet"), ("gbm_soft_bet", "gbm_bet"),
                  ("logit_soft_bet", "logit_bet"), ("seq_bet", "gbm_soft_bet"), ("seq_bet", "logit_soft_bet")]
    for (d, n), decs in by.items():
        R = len(decs)
        if R < REPS or n > 800 or d.startswith("N0"):   # tier-1 grid, pre-registered designs
            continue
        for a, b in pairs:
            if not all(a in x and b in x for x in decs):
                continue
            A = np.array([x[a] for x in decs])[:, :3]
            B = np.array([x[b] for x in decs])[:, :3]
            for i, j in enumerate((0, 4, 6)):
                diff = A[:, i] - B[:, i]
                m = diff.mean()
                se = diff.std(ddof=1) / math.sqrt(R) if R > 1 else 0.0
                out.append(dict(design=d, n=n, level=j, a=a, b=b, pa=A[:, i].mean(), pb=B[:, i].mean(),
                                diff=m, se=se, res=("win" if m > 2 * se and m > 0 else
                                                    "loss" if m < -2 * se and m < 0 else "tie")))
    return pd.DataFrame(out)


def wtl(pdf):
    """Wins/ties/losses of a over b over (design, n, level) cells in which at
    least one of the two has rejection rate >= 0.1 (cells where both are ~0 are
    uninformative and counted separately)."""
    rows = []
    for (a, b), g in pdf.groupby(["a", "b"], sort=False):
        inf = g[(g.pa >= 0.1) | (g.pb >= 0.1)]
        rows.append(dict(a=a, b=b, cells=len(g), informative=len(inf), win=int((inf.res == "win").sum()),
                         tie=int((inf.res == "tie").sum()), loss=int((inf.res == "loss").sum()),
                         mean_diff=float(inf["diff"].mean()) if len(inf) else float("nan"),
                         max_diff=float(inf["diff"].max()) if len(inf) else float("nan"),
                         min_diff=float(inf["diff"].min()) if len(inf) else float("nan")))
    return pd.DataFrame(rows)


def load_posthoc():
    d = OUT / "posthoc_soft"
    ex = {}
    if d.exists():
        for f in d.glob("*.jsonl"):
            for line in f.read_text().splitlines():
                r = json.loads(line)
                ex[(r["design"], r["n"], r["rep"])] = r["dec"]
    return ex


def main2():
    rows = load()
    ex = load_posthoc()
    pdf = paired(rows, ex if ex else None)
    pdf.to_csv(OUT / "paired.csv", index=False)
    w = wtl(pdf)
    w.to_csv(OUT / "wtl.csv", index=False)
    print(w.round(3).to_string(index=False))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "paired":
    main2()


def overview(df, extra_methods=()):
    """Heatmaps: power at the largest complete n of each design (800; D5b 400;
    1600 where tier 2 finished is shown separately), H_{0,0} and H_{0,a*/2}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    full = df[df.R >= REPS]
    ms = ["an_old", "an_new", "prop17", "boot", "ramp_b2s", "ramp_bet", "seq_bet", "gbm_hoef", "gbm_bet",
          "logit_bet", "knn_bet"] + list(extra_methods)
    ms = [m for m in ms if m in full.method.unique()]
    order = [d for d in TITLES if d in full.design.unique()]
    cmap = LinearSegmentedColormap.from_list("blue", ["#f4f3ef", "#9ec5f0", "#2a78d6", "#0d3a73"])
    fig, axs = plt.subplots(1, 3, figsize=(17, 0.42 * len(order) + 2.2))
    for ax, (col, lab) in zip(axs, (("pow0", "power, $H_{0,0}$"), ("pow4", r"power, $H_{0,\alpha^*/2}$"),
                                     ("typeI", r"type-I at $\alpha^*$ (max over n)"))):
        M_ = np.full((len(order), len(ms)), np.nan)
        for i, d in enumerate(order):
            s = full[(full.design == d) & (full.n <= 800)]
            nmax = s.n.max()
            for j, m in enumerate(ms):
                t = s[s.method == m]
                if t.empty:
                    continue
                M_[i, j] = t.typeI.max() if col == "typeI" else t[t.n == nmax][col].iloc[0]
        vmax = 0.1 if col == "typeI" else 1.0
        ax.imshow(M_, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        for i in range(len(order)):
            for j in range(len(ms)):
                v = M_[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f"{v:.2f}".lstrip("0") if v < 1 else "1", ha="center", va="center", fontsize=7,
                        color="#0b0b0b" if v < 0.6 * vmax else "#fcfcfb")
        ax.set_xticks(range(len(ms)))
        ax.set_xticklabels(ms, rotation=60, ha="right", fontsize=8, color="#52514e")
        ax.set_yticks(range(len(order)))
        lab_y = [f"{d} (n={int(full[(full.design == d) & (full.n <= 800)].n.max())})" for d in order]
        ax.set_yticklabels(lab_y if ax is axs[0] else [""] * len(order), fontsize=8, color="#52514e")
        ax.set_title(lab, fontsize=10, color="#0b0b0b")
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle("All designs at the largest tier-1 n (200 replicates per cell; delta = 0.05)", fontsize=11,
                 x=0.01, ha="left", color="#0b0b0b")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIGS / "overview.png", dpi=130, facecolor="#fcfcfb")
    plt.close(fig)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "overview":
    d_ = pd.read_csv(OUT / "summary_long.csv")
    overview(d_)


def main_candidate(df):
    """Candidate main-text figure: 4 hard designs x (H_{0,0}, H_{0,a*/2})."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    full = df[df.R >= REPS]
    D = [("D1_d2_a20", "overlap, d = 2"), ("D2_d100", "dense shift, d = 100"),
         ("D7_narrow", "Brownian curves, narrow bump"), ("D8_N16", "oscillating density, N = 16")]
    ms = ["an_old", "an_new", "boot", "seq_bet", "gbm_bet", "logit_bet"]
    pal = {m: PAL[FIG_M.index(m)] for m in ms}
    fig, axs = plt.subplots(2, 4, figsize=(13.5, 6.0), sharex=False, sharey=True)
    for j, (d, t) in enumerate(D):
        s = full[full.design == d]
        for i, (col, lab) in enumerate((("pow0", r"$H_{0,0}$"), ("pow4", r"$H_{0,\alpha^*/2}$"))):
            ax = axs[i, j]
            for k, m in enumerate(ms):
                u = s[s.method == m].sort_values("n")
                mk, ls = STYLE[m]
                ax.plot(u.n * 2 ** ((k - 2.5) * 0.035), u[col], color=pal[m], marker=mk, ls=ls, lw=2, ms=6,
                        markeredgecolor="#fcfcfb", markeredgewidth=0.8, label=NAMES[m])
            ax.axhline(0.8, color="#c3c2b7", lw=0.8)
            ax.set_xscale("log", base=2)
            ns = sorted(s.n.unique())
            ax.set_xticks(ns)
            ax.set_xticklabels([str(x) for x in ns], fontsize=8, color="#52514e")
            ax.set_ylim(-0.03, 1.03)
            ax.grid(True, color="#e6e5e0", lw=0.6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(colors="#52514e", labelsize=8)
            if i == 0:
                ax.set_title(f"{t}\n(TV = {s.alpha_star.iloc[0]:.2g})", fontsize=10, color="#0b0b0b")
            if j == 0:
                ax.set_ylabel(f"power against {lab}", color="#52514e")
            if i == 1:
                ax.set_xlabel("n = m", color="#52514e")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIGS / "main_candidate.png", dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "maincand":
    main_candidate(pd.read_csv(OUT / "summary_long.csv"))


# ---------------------------------------------------------------------------
# POST HOC alpha = 0 extension (DESIGN.md section 9, 2026-11-10): out/<design>_n0ext.jsonl -> summary_n0ext.csv
# (rejection rate of H_{0,0} per design, n and method; complete cells only) and runtime_n0ext.csv. The pre-registered
# summaries above never read these files (their replicates have rep >= 200000 and load() keeps rep < REPS).
# ---------------------------------------------------------------------------

def ext_summary(rep0=200000, reps=None):
    import designs as DS
    C = DS.catalogue()
    reps = REPS if reps is None else reps
    by = {}
    for f in sorted(OUT.glob("*_n0ext.jsonl")):
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if rep0 <= r["rep"] < rep0 + reps:
                by.setdefault((r["design"], r["n"]), {})[r["rep"]] = r
    recs, times = [], []
    for (d, n), rr in sorted(by.items()):
        rs = list(rr.values())
        R = len(rs)
        if R < reps:
            print(f"ext_summary: {d} n={n} incomplete ({R}/{reps}), skipped")
            continue
        for mth in rs[0]["dec0"]:
            k = int(sum(r["dec0"][mth] for r in rs))
            recs.append(dict(design=d, family=C[d].family, n=n, m=rs[0]["m"], R=R, method=mth,
                             alpha_star=C[d].alpha_star, k0=k, pow0=k / R, cp0=cp_upper(k, R),
                             p0=float(binom.sf(k - 1, R, 0.05)) if k > 0 else 1.0))
        t = {c: float(np.mean([r["time"].get(c, 0.0) for r in rs])) for c in rs[0]["time"]}
        t.update(design=d, n=n, m=rs[0]["m"], R=R, mmd_mean=float(np.mean([r["stat"]["mmd"] for r in rs])),
                 tau_new=rs[0]["stat"]["tau"]["new"], tau_new_study=rs[0]["stat"]["tau"]["new_study"],
                 an_new_vs_study_diff=int(sum(r["dec0"]["an_new"] != r["dec0"]["an_new_study"] for r in rs)))
        times.append(t)
    df, tm = pd.DataFrame(recs), pd.DataFrame(times)
    df.to_csv(OUT / "summary_n0ext.csv", index=False)
    tm.to_csv(OUT / "runtime_n0ext.csv", index=False)
    print(f"summary_n0ext.csv: {len(tm)} complete cells, {int(tm.R.sum()) if len(tm) else 0} replicates")
    return df, tm


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "ext":
    ext_summary()

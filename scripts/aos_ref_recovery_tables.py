"""AoS referee 2, M5: recovery of alpha* against n with an oracle ceiling -- summaries, macros, the S6 table and
results/aos/main/RECOVERY_CURVES.md. No Monte Carlo here and no hand-typed number: everything is read from

    results/aos/main/raw/ref_ramp/<design>_n<n>.jsonl       Laplace ramp below the main size (aos_ref_recovery_run.py)
    results/aos/main/raw/synth_laplace/<design>_n<n>.jsonl  Laplace ramp at the main size (the main runs, reused)
    results/aos/main/raw/ref_oracle/...                     the betting test on psi = 1{p > q}
    results/aos/main/raw/ref_oracle_lr/...                  the betting test on the best likelihood-ratio level set
    results/aos/main/ref_recovery_lrsets.json               the level sets (population masses, thresholds)
    results/aos/main/ref_recovery_check.json                checks of the run script
    explore/harder_sims/out/<design>.jsonl                  Gaussian ramp (stored seq_bet) at the main size, read-only

Recovery of a replicate = alpha_lo / alpha* with alpha_lo the down-closed certified level on the grid j alpha*/8
(0 when nothing is certified), as scripts/aos_main_tables.py::frac. Every comparison is paired: at each (design, n)
the ramp, both oracles and the Gaussian ramp use the same replicates (0..R-1, R from the ramp records).

Writes
    results/aos/main/ref_recovery_summary.json      per design and n: means, s.e., 10th percentiles (figure data)
    results/aos/tables/aos_ref_recovery.tex         S6 table: ramp and oracle by n, Gaussian ramp at the main size
    results/aos/tables/aos_ref_recovery_macros.tex  macros (prefix \\rc)
    results/aos/main/RECOVERY_CURVES.md             the facts

    python -B scripts/aos_ref_recovery_tables.py
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

import aos_baselines_tables as T  # noqa: E402  (design labels, order of Table 1)

MAIN = ROOT / "results" / "aos" / "main"
RAW = MAIN / "raw"
TAB = ROOT / "results" / "aos" / "tables"
STORED = ROOT / "explore" / "harder_sims" / "out"
SIZES = (100, 200, 400, 800)
WORD = {100: "Hundred", 200: "TwoHundred", 400: "FourHundred", 800: "EightHundred"}
DESIGNS = T.DESIGNS
KEYS = [k for k, _, _ in DESIGNS]
LABEL = {k: lab for k, lab, _ in DESIGNS}
ASTAR = {k: a for k, _, a in DESIGNS}
EIGHTY = 0.8
MACROS, REPORT = [], []


def mac(name, val):
    assert name.isalpha() and name.startswith("rc"), name
    assert name not in {m for m, _ in MACROS}, name
    MACROS.append((name, str(val)))


def say(s=""):
    REPORT.append(s)


def r2(x):
    return str(Decimal(repr(float(x))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def r3(x):
    return str(Decimal(repr(float(x))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def jl(path):
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out.setdefault(r["rep"], r)
    return out


def frac(j, levels, ast):
    return 0.0 if j < 0 else levels[j] / ast


def q10(v):
    return float(np.quantile(np.asarray(v), 0.1, method="inverted_cdf"))


def main_n(key):
    return 400 if key == "D5b_mixed" else 800


def sizes_of(key):
    return tuple(n for n in SIZES if n <= main_n(key))


def load(key, n):
    """{method: array of recoveries on the common replicates}, R, seconds of the new records."""
    if n == main_n(key):
        ramp = jl(RAW / "synth_laplace" / f"{key}_n{n}.jsonl")
        new_sec = 0.0
    else:
        ramp = jl(RAW / "ref_ramp" / f"{key}_n{n}.jsonl")
        new_sec = sum(r["seconds"] for r in ramp.values())
    reps = sorted(ramp)
    assert reps == list(range(len(reps))), (key, n)
    orc = jl(RAW / "ref_oracle" / f"{key}_n{n}.jsonl")
    olr = jl(RAW / "ref_oracle_lr" / f"{key}_n{n}.jsonl")
    sec_or = sum(orc[r]["seconds"] for r in orc) + sum(olr[r]["seconds"] for r in olr)
    out = {"ramp": [], "ramp_linear": [], "psi": [], "lr": [], "psi_j8": [], "lr_j8": [], "ramp_j8": []}
    for rep in reps:
        a, o, l_ = ramp[rep], orc[rep], olr[rep]
        lv, ast = a["levels"], a["alpha_star"]
        assert o["levels"] == lv and l_["levels"] == lv
        out["ramp"].append(frac(a["cuts"]["order"]["jlo"], lv, ast))
        out["ramp_linear"].append(frac(a["cuts"]["linear"]["jlo"], lv, ast))
        out["psi"].append(frac(o["jlo"], lv, ast))
        out["lr"].append(frac(l_["jlo"], lv, ast))
        out["psi_j8"].append(o["dec"][8])
        out["lr_j8"].append(l_["dec"][8])
        out["ramp_j8"].append(a["cuts"]["order"]["dec"][8])
    if n == main_n(key):
        st = {}
        for line in (STORED / f"{key}.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r["n"] == n:
                    st[r["rep"]] = r
        out["gauss"] = [frac(st[rep]["jlo"]["seq_bet"], st[rep]["levels"], st[rep]["alpha_star"]) for rep in reps]
    return {k: np.asarray(v, float) for k, v in out.items()}, len(reps), new_sec, sec_or


def main():
    chk = json.loads((MAIN / "ref_recovery_check.json").read_text())
    lrs = json.loads((MAIN / "ref_recovery_lrsets.json").read_text())
    S, R_of, sec_ramp, sec_or = {}, {}, 0.0, 0.0
    for key in KEYS:
        for n in sizes_of(key):
            d, R, s1, s2 = load(key, n)
            S[(key, n)] = d
            R_of[(key, n)] = R
            sec_ramp += s1
            sec_or += s2
    R_small = sorted({R_of[c] for c in S if c[1] != main_n(c[0])})
    R_main = sorted({R_of[c] for c in S if c[1] == main_n(c[0])})
    assert len(R_small) == 1 and len(R_main) == 1, (R_small, R_main)
    R_small, R_main = R_small[0], R_main[0]

    # ------------------------------------------------------------------ per cell
    cell = {}
    for (key, n), d in S.items():
        R = len(d["ramp"])
        orc = np.maximum(d["psi"].mean(), d["lr"].mean())
        best = "lr" if d["lr"].mean() > d["psi"].mean() else "psi"
        diff = d["ramp"] - d[best]
        c = {"R": R, "ramp": float(d["ramp"].mean()), "ramp_se": float(d["ramp"].std(ddof=1) / np.sqrt(R)),
             "ramp_q10": q10(d["ramp"]), "ramp_linear": float(d["ramp_linear"].mean()),
             "psi": float(d["psi"].mean()), "psi_se": float(d["psi"].std(ddof=1) / np.sqrt(R)),
             "lr": float(d["lr"].mean()), "lr_se": float(d["lr"].std(ddof=1) / np.sqrt(R)),
             "oracle": float(orc), "oracle_is": best, "oracle_q10": q10(d[best]),
             "ramp_minus_oracle": float(diff.mean()), "ramp_minus_oracle_se": float(diff.std(ddof=1) / np.sqrt(R)),
             "lr_minus_psi": float((d["lr"] - d["psi"]).mean()),
             "lr_minus_psi_se": float((d["lr"] - d["psi"]).std(ddof=1) / np.sqrt(R)),
             "ratio": float(d["ramp"].mean() / orc) if orc > 0 else None,
             "ratio_psi": float(d["ramp"].mean() / d["psi"].mean()) if d["psi"].mean() > 0 else None,
             "psi_j8": int(d["psi_j8"].sum()), "lr_j8": int(d["lr_j8"].sum()), "ramp_j8": int(d["ramp_j8"].sum())}
        if "gauss" in d:
            c["gauss"] = float(d["gauss"].mean())
        cell[(key, n)] = c

    def col(n, f):
        """value of f over the designs at size n; n = 800 means the main size (D5b: 400)."""
        return [f(cell[(k, main_n(k) if n == 800 else n)]) for k in KEYS if (main_n(k) if n == 800 else n) in sizes_of(k)]

    # ------------------------------------------------------------------ macros
    mac("rcReps", R_small)
    mac("rcRepsMain", R_main)
    mac("rcNdesigns", len(KEYS))
    excluded = [k for k in KEYS if (k, main_n(k)) not in cell]
    mac("rcOracleDesigns", len(KEYS) - len(excluded))
    mac("rcOracleExcluded", len(excluded))
    mac("rcGridCeiling", "0.875")
    mac("rcLevelsetMC", f"{lrs['n_mc']:,}".replace(",", "{,}"))
    say("# Recovery of alpha* against n, with an oracle ceiling (referee 2, M5)")
    say()
    say("Generated by scripts/aos_ref_recovery_tables.py from the raw records (no hand-typed numbers). Run script: "
        "scripts/aos_ref_recovery_run.py; figure: scripts/aos_ref_recovery_fig.py -> aos/figs_aos/fig_recovery_curves.pdf.")
    say()
    say("## Protocol")
    say()
    say(f"- Designs: the 19 of Figure 2(a) (explore/harder_sims/designs.py), replicates regenerated exactly "
        f"(scripts/aos_baselines_run.py::draw), n = m in {{100, 200, 400, 800}} (mixed types: n in {{100, 200, 400}}, "
        f"m = 5n), levels j alpha*/8, j = 0..10, delta = 0.05. Recovery = alpha_lo / alpha*, alpha_lo the down-closed "
        f"certified level (0 when nothing is certified); the mean over replicates is reported.")
    say(f"- Replicates: {R_small} per size below the main size (replicates 0-{R_small - 1}; reduced from 100 to keep "
        f"the compute near 6 CPU-hours under the load of the machine, see Compute); at the main size (n = 800, mixed "
        f"n = 400) the {R_main} replicates of the main run (results/aos/main/raw/synth_laplace) are reused. At each "
        f"(design, n) every method uses the same replicates.")
    say("- Laplace ramp: scripts/aos_main_synth.py::replicate itself (stored fold bandwidths at that n, warm-started "
        "GramQP fits along the grid, own term removed, genuine quantile cuts); the linear-cut evaluation on the same "
        "fits is kept in the records.")
    say("- Oracle psi = 1{p > q}: the same betting test (batch 1: quarter 1 -> quarter 2, batch 2: half 1 -> half 2; "
        "pair_up; horizon N = n_pairs of both batches; bet_stream with c = 0.5 and prior weight 20; prior variance "
        "pair_var of psi on the fit block, floored at 1e-4) with psi in place of the fitted ramp; psi never skips a "
        "batch. E_P psi - E_Q psi = TV = alpha*, the largest drift any [0, 1]-valued witness can have.")
    say("- Oracle level set: psi maximizes the drift but not what this betting rule rewards. To first order the "
        "log-wealth after N bets is sqrt(2 L N) mu / sd - L (L = log(1/delta)), so a witness is judged by the "
        "signal-to-noise ratio mu/sd of its pair statistic. Among indicators 1_A this ratio, (P(A) - Q(A) - alpha) / "
        "sd_pair(A), is increasing in P(A) at fixed Q(A) whenever P(A) - Q(A) >= alpha, so by the Neyman-Pearson "
        "lemma its maximizer is a level set A_t = {p/q > t} (randomized on {p/q = t} where p/q has atoms, as in the "
        "mixed-types design; only deterministic level sets are used here). The second oracle bets at level alpha_j "
        "on A_{t_j}, t_j "
        f"maximizing the population ratio (masses by Monte Carlo, {lrs['n_mc']:,} draws per law, seed {lrs['seed']}, "
        "nothing from the replicates; results/aos/main/ref_recovery_lrsets.json), and on psi at alpha_j >= alpha*. "
        "Both oracles are fixed functions, so both tests are valid.")
    say("- Oracle (the reference line): at each (design, n) the better mean of the two oracles on the same replicates. "
        "Neither oracle is a proven upper bound for a fitted witness: the betting rule is not log-optimal, the "
        "fractions use estimated variances, and a non-indicator witness can have a better ratio than any indicator; "
        "the ramp can therefore exceed them (see below).")
    say("- Grid ceiling: the largest grid level below alpha* is 7 alpha*/8, so a valid test certifies at most "
        "0.875 alpha* on this grid except by a rejection of a true null at j >= 8.")
    say()

    say("## psi in closed form: no design excluded")
    say()
    say("Every design of designs.py gives the sign of p - q exactly (p - q = a (p0 - q0) in the contamination designs). "
        "psi on the raw coordinates (D5b_mixed: before its fixed scaling; the scaled copy equals draw() value for value):")
    say()
    for txt in ("overlap, dense shift, sparse shift, small contamination (GaussLoc): <mu, x> > 0",
                "correlation flip: x1 x2 > 0",
                "mixed types: rounded-normal and Bernoulli masses of (x1, x2); = {x2 = 1, x1 >= 0} or {x2 = 0, x1 >= 1}",
                "t3 tails: |x1| < x*, the single crossing of the s_P t3 and s_Q t3 densities (s_P < s_Q)",
                "curves (Brownian bump): <Db, Dx> > 0 (Cameron-Martin inner product)",
                "oscillating: frac(N x) >= 1/2 (p = 1, q = 0 there; q = 2 on A_N)",
                "RHC-shaped: x_{1..4} in [-4, -3]^4 (the P0 box; P0 and Q0 mutually singular)"):
        say(f"- {txt}")
    say()
    tv = chk["tv"]
    zs = [abs(v["z"]) for v in tv.values()]
    say(f"Check (Monte Carlo, {tv[KEYS[0]]['n_mc']:,} draws per law with the design samplers): E_P psi - E_Q psi "
        f"against alpha*: largest |z| = {max(zs):.2f} over the 19 designs; "
        + "; ".join(f"{k} {v['mc']:.4f} (alpha* {v['alpha_star']:.4f})" for k, v in tv.items()) + ".")
    osc = [k for k in KEYS if k.startswith("D8_")]
    yq = max(r["psi_y_mean"] for k in osc for n in sizes_of(k)
             for r in jl(RAW / "ref_oracle" / f"{k}_n{n}.jsonl").values())
    say(f"(With 19 comparisons one |z| near 3 is expected now and then. In the oscillating designs E_P psi = 1/2 "
        f"exactly, since psi = 1 on half of [0, 1], and psi = 0 on the support of Q: the largest mean of psi over the "
        f"Y's of any oscillating replicate is {yq:g}.)")
    say()
    box = lrs["designs"]["D9_rhc"]["box"]
    say(f"Level sets. Oscillating designs: p/q takes the values +inf and 1/2 only, so psi is the only level set with "
        f"a positive drift (the level-set oracle is psi). RHC-shaped: level sets with t >= 1 lie in the P0 box and those "
        f"with t < 1 contain the complement of the Q0 box; the core puts mass {box['R_box_P']:.1e} on the P0 box and "
        f"{box['R_box_Q']:.1e} on the Q0 box (exact Gaussian-jitter sums over the 5,735 pool points), and the largest "
        f"gain in the ratio any level set can have over psi is {box['snr_gap_max']:.1e}, so the level-set oracle is "
        f"psi. Population ratio at alpha = 0, best level set vs psi:")
    say()
    say("| design | t_0 | P(A) | Q(A) | ratio, level set | ratio, psi |")
    say("|---|---|---|---|---|---|")
    for k in KEYS:
        d = lrs["designs"][k]
        if d["rule"] == "lr":
            i = d["levels"][0]
            say(f"| {k} | {i['t']:+.3f} | {i['P']:.3f} | {i['Q']:.3f} | {i['snr']:.3f} | {i['snr_psi']:.3f} |")
        else:
            say(f"| {k} | psi | | | | |")
    say()
    mac("rcOracleExcludedList", "none" if not excluded else ", ".join(excluded))

    # ------------------------------------------------------------------ headline by n
    say("## Headline numbers by n (medians over the designs; n = 800 means the main size, mixed types n = 400)")
    say()
    say("| n | ramp | oracle | psi oracle | level-set oracle | ramp / oracle | ramp >= 0.8 oracle | ramp 10th pct |"
        " oracle 10th pct |")
    say("|---|---|---|---|---|---|---|---|---|")
    for n in SIZES:
        w = WORD[n]
        ramp = col(n, lambda c: c["ramp"])
        orc = col(n, lambda c: c["oracle"])
        psi = col(n, lambda c: c["psi"])
        lr = col(n, lambda c: c["lr"])
        rat = [x for x in col(n, lambda c: c["ratio"]) if x is not None]
        e80 = [k for k in KEYS if (main_n(k) if n == 800 else n) in sizes_of(k)
               and cell[(k, main_n(k) if n == 800 else n)]["ratio"] is not None
               and cell[(k, main_n(k) if n == 800 else n)]["ratio"] >= EIGHTY]
        rq = col(n, lambda c: c["ramp_q10"])
        oq = col(n, lambda c: c["oracle_q10"])
        nd = len(ramp)
        mac(f"rcRampMed{w}", r2(statistics.median(ramp)))
        mac(f"rcOracleMed{w}", r2(statistics.median(orc)))
        mac(f"rcOracleMin{w}", r2(min(orc)))
        mac(f"rcOracleMax{w}", r2(max(orc)))
        mac(f"rcPsiMed{w}", r2(statistics.median(psi)))
        mac(f"rcLevelsetMed{w}", r2(statistics.median(lr)))
        mac(f"rcRatioMed{w}", r2(statistics.median(rat)))
        mac(f"rcEighty{w}", len(e80))
        mac(f"rcDesigns{w}", nd)
        mac(f"rcRampTenthMed{w}", r2(statistics.median(rq)))
        mac(f"rcOracleTenthMed{w}", r2(statistics.median(oq)))
        say(f"| {n} | {r2(statistics.median(ramp))} | {r2(statistics.median(orc))} ({r2(min(orc))}-{r2(max(orc))}) | "
            f"{r2(statistics.median(psi))} | {r2(statistics.median(lr))} | {r2(statistics.median(rat))} "
            f"({len(rat)} designs) | {len(e80)} of {nd} | {r2(statistics.median(rq))} | {r2(statistics.median(oq))} |")
    say()
    e80m = [k for k in KEYS if cell[(k, main_n(k))]["ratio"] is not None and cell[(k, main_n(k))]["ratio"] >= EIGHTY]
    def short(k):          # a label that survives in a list: "overlap, $d=2$" -> "overlap $d=2$ ($\alpha^\star=0.2$)"
        lab = LABEL[k].replace(", ", " ")
        dup = [j for j in KEYS if LABEL[j] == LABEL[k]]
        return lab + (f" ($\\alpha^\\star={ASTAR[k]}$)" if len(dup) > 1 else "")
    mac("rcEightyDesignsEightHundred", "; ".join(short(k) for k in e80m))
    say("Designs where the ramp reaches 80% of the oracle, by n: "
        + "; ".join(f"n = {n}: " + (", ".join(k for k in KEYS if (main_n(k) if n == 800 else n) in sizes_of(k)
                                             and (cell[(k, main_n(k) if n == 800 else n)]["ratio"] or 0) >= EIGHTY)
                                   or "none") for n in SIZES) + ".")
    say()
    # ramp ratio trend: designs where ramp / oracle grows from the smallest to the main size
    grow = [k for k in KEYS if cell[(k, 100)]["ratio"] is not None and cell[(k, main_n(k))]["ratio"] is not None
            and cell[(k, main_n(k))]["ratio"] > cell[(k, 100)]["ratio"]]
    both = [k for k in KEYS if cell[(k, 100)]["ratio"] is not None and cell[(k, main_n(k))]["ratio"] is not None]
    mac("rcRatioGrows", len(grow))
    mac("rcRatioGrowsOf", len(both))
    say(f"ramp / oracle larger at the main size than at n = 100 in {len(grow)} of {len(both)} designs (oracle > 0 at "
        f"both): {', '.join(grow)}.")
    say()

    # ------------------------------------------------------------------ ramp above the oracle, psi vs level set
    above = [(k, n) for (k, n), c in cell.items() if c["ramp_minus_oracle"] > 2 * c["ramp_minus_oracle_se"]]
    above_psi = [(k, n) for (k, n), c in cell.items()
                 if (S[(k, n)]["ramp"] - S[(k, n)]["psi"]).mean()
                 > 2 * (S[(k, n)]["ramp"] - S[(k, n)]["psi"]).std(ddof=1) / np.sqrt(c["R"])]
    lr_wins = [(k, n) for (k, n), c in cell.items() if c["lr_minus_psi"] > 2 * c["lr_minus_psi_se"]]
    lr_loss = [(k, n) for (k, n), c in cell.items() if c["lr_minus_psi"] < -2 * c["lr_minus_psi_se"]]
    mac("rcAboveOracle", len(above))
    mac("rcAbovePsi", len(above_psi))
    mac("rcCells", len(cell))
    mac("rcLevelsetWins", len(lr_wins))
    mac("rcLevelsetLosses", len(lr_loss))
    c4 = cell[("D4_small", 800)]
    mac("rcSmallRamp", r2(c4["ramp"]))
    mac("rcSmallPsi", r2(c4["psi"]))
    mac("rcSmallLevelset", r2(c4["lr"]))
    say("## Where psi is not a ceiling")
    say()
    say(f"- Ramp above the psi oracle at two paired s.e.: {len(above_psi)} of {len(cell)} (design, n) cells: "
        + (", ".join(f"{k} n={n} ({r3(cell[(k, n)]['ramp'])} vs {r3(cell[(k, n)]['psi'])})" for k, n in above_psi)
           or "none") + ".")
    say(f"- Ramp above the oracle (better of the two) at two paired s.e.: {len(above)} cells"
        + (": " + ", ".join(f"{k} n={n} ({r3(cell[(k, n)]['ramp'])} vs {r3(cell[(k, n)]['oracle'])})" for k, n in above)
           if above else "") + ".")
    say(f"- Level-set oracle above psi at two paired s.e.: {len(lr_wins)} cells ("
        + ", ".join(f"{k} n={n} {r3(cell[(k, n)]['lr'])} vs {r3(cell[(k, n)]['psi'])}" for k, n in lr_wins)
        + f"); below psi: {len(lr_loss)} cells ("
        + ", ".join(f"{k} n={n} {r3(cell[(k, n)]['lr'])} vs {r3(cell[(k, n)]['psi'])}" for k, n in lr_loss) + ").")
    say(f"- Small contamination at n = 800: ramp {r2(c4['ramp'])}, psi {r2(c4['psi'])}, best level set "
        f"{r2(c4['lr'])}. psi = 1{{x1 > 0}} splits the core in half, so its pair statistic has variance near 1/2 for a "
        f"drift of 0.05; the level set (and the trimmed witness) keep the region where the contamination lives.")
    say()

    # ------------------------------------------------------------------ validity of the betting rule at the tight null
    pj8 = sum(c["psi_j8"] for c in cell.values())
    lj8 = sum(c["lr_j8"] for c in cell.values())
    rj8 = sum(c["ramp_j8"] for c in cell.values() if c["R"] == R_small)
    tot = sum(c["R"] for c in cell.values())
    tot_s = sum(c["R"] for c in cell.values() if c["R"] == R_small)
    mx = max(cell.items(), key=lambda kv: kv[1]["psi_j8"] / kv[1]["R"])
    mac("rcPsiBoundaryRej", pj8)
    mac("rcPsiBoundaryTot", f"{tot:,}".replace(",", "{,}"))
    mac("rcPsiBoundaryRate", f"{100 * pj8 / tot:.1f}")
    say("## Validity at the tight null")
    say()
    say(f"At j = 8 (alpha = alpha*) psi has drift exactly 0, the tight case of H_(0,alpha*). Rejections: psi oracle "
        f"{pj8} of {tot} replicates ({100 * pj8 / tot:.1f}%; largest cell {mx[0][0]} n={mx[0][1]} "
        f"{mx[1]['psi_j8']}/{mx[1]['R']}); level-set oracle {lj8} of {tot} (it bets on psi at j = 8); Laplace ramp at the sizes below the main "
        f"size {rj8} of {tot_s}.")
    say()

    # ------------------------------------------------------------------ per design table (markdown)
    say("## Per design (mean recovery; s.e. of the ramp in brackets; oracle = better of psi and the level set, "
        "marked L when the level set)")
    say()
    say("| design | alpha* | " + " | ".join(f"n={n}: ramp / psi / level set / ratio" for n in SIZES)
        + " | Gaussian ramp (main n) |")
    say("|---|---|" + "---|" * len(SIZES) + "---|")
    for k in KEYS:
        row = [k, ASTAR[k]]
        for n in SIZES:
            if n not in sizes_of(k):
                row.append("--")
                continue
            c = cell[(k, n)]
            row.append(f"{r3(c['ramp'])} ({r3(c['ramp_se'])}) / {r3(c['psi'])} / {r3(c['lr'])}"
                       f"{' L' if c['oracle_is'] == 'lr' else ''} / {r2(c['ratio']) if c['ratio'] is not None else '--'}")
        row.append(r3(cell[(k, main_n(k))]["gauss"]))
        say("| " + " | ".join(row) + " |")
    say()
    say("10th percentile of alpha-hat / alpha* (ramp | oracle), by n: ")
    say()
    say("| design | " + " | ".join(f"n={n}" for n in SIZES) + " |")
    say("|---|" + "---|" * len(SIZES))
    for k in KEYS:
        say("| " + k + " | " + " | ".join(
            (f"{r3(cell[(k, n)]['ramp_q10'])} / {r3(cell[(k, n)]['oracle_q10'])}" if n in sizes_of(k) else "--")
            for n in SIZES) + " |")
    say()
    gm = col(800, lambda c: c["gauss"])
    mac("rcGaussMedEightHundred", r2(statistics.median(gm)))
    lin = [abs(cell[(k, n)]["ramp"] - cell[(k, n)]["ramp_linear"]) for k, n in cell]
    say(f"Gaussian ramp at the main size (stored seq_bet, {R_main} replicates): median over designs "
        f"{r2(statistics.median(gm))}. Genuine quantile vs linear cuts on the same Laplace fits: largest difference of "
        f"the mean recovery over the {len(cell)} cells {r3(max(lin))}.")
    say()

    # ------------------------------------------------------------------ checks and compute
    say("## Checks (scripts/aos_ref_recovery_run.py check, results/aos/main/ref_recovery_check.json)")
    say()
    rp = chk["repro_main"]
    say(f"- The ramp of the run script reproduces the stored main run at the main size (decisions at all 11 levels, "
        f"both cut rules): " + ", ".join(f"{k} {sum(v)}/{len(v)}" for k, v in rp.items()) + ".")
    say(f"- psi records recomputed by the final code: {chk['oracle_recomputed_equal']} equal (decisions, log-wealth, "
        f"prior variances); oscillating and RHC-shaped level-set oracle = psi decisions in {chk['psi_only_lr_equal']}.")
    say(f"- D5b_mixed raw coordinates: the scaled copy equals scripts/aos_baselines_run.py::draw (asserted in every "
        f"oracle replicate of that design).")
    say(f"- Level-set Monte Carlo: sign(log p/q) = psi on every draw except "
        + ", ".join(f"{k} {lrs['designs'][k]['sign_rounding']}" for k in KEYS
                    if lrs['designs'][k].get('sign_rounding'))
        + " draws where |log p/q| < 1e-9, i.e. where both contamination densities are negligible against the core "
          "and p/q = 1 to machine precision (psi uses the exact sign there)." if any(lrs['designs'][k].get('sign_rounding') for k in KEYS)
        else "- Level-set Monte Carlo: sign(log p/q) = psi on every draw.")
    say()
    say("## Compute (sum of per-record seconds, single-threaded workers, 5 ramp workers next to another agent's job)")
    say()
    say(f"- Laplace ramp below the main size, {sum(1 for c in cell.values() if c['R'] == R_small)} cells x {R_small} "
        f"replicates: {sec_ramp / 3600:.2f} CPU-hours (the main size is reused: 0).")
    say(f"- Both oracles, every cell: {sec_or / 3600:.3f} CPU-hours; level sets (Monte Carlo) and checks: minutes.")
    mac("rcCpuHours", f"{(sec_ramp + sec_or) / 3600:.1f}")
    say()

    # ------------------------------------------------------------------ S6 table
    hdr = ["% Generated by scripts/aos_ref_recovery_tables.py -- do not edit. Mean certified level / alpha* by n: "
           f"Laplace ramp (alpha-hat) and oracle (the better of 1{{p>q}} and the best likelihood-ratio level set), "
           f"{R_small} replicates per size, {R_main} at n = 800 (mixed: n = 400, m = 5n); last column the Gaussian ramp "
           "at n = 800. Mixed types: the n = 800 columns repeat its main size n = 400 (marked with an asterisk).",
           r"\begin{tabular}{llccccccccc}", r"\toprule",
           r" & & \multicolumn{2}{c}{$n=100$} & \multicolumn{2}{c}{$n=200$} & \multicolumn{2}{c}{$n=400$} & "
           r"\multicolumn{3}{c}{$n=800$} \\",
           r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-11}",
           r"design & $\alpha^\star$ & $\widehat\alpha$ & oracle & $\widehat\alpha$ & oracle & $\widehat\alpha$ & oracle "
           r"& $\widehat\alpha$ & oracle & Gaussian \\", r"\midrule"]
    rows = []
    for k in KEYS:
        r = [LABEL[k], f"${ASTAR[k]}$"]
        for n in (100, 200, 400):
            c = cell[(k, n)]
            r += [f"${r2(c['ramp'])}$", f"${r2(c['oracle'])}$"]
        c = cell[(k, main_n(k))]
        r += [f"${r2(c['ramp'])}$", f"${r2(c['oracle'])}$", f"${r2(c['gauss'])}$"]
        if k == "D5b_mixed":           # its main size is n = 400: repeated under n = 800 with a mark, as Figure 2(a)
            r[-3] = f"${r2(c['ramp'])}^{{\\ast}}$"
        rows.append(" & ".join(r) + r" \\")
    foot = [r"\midrule",
            "median & & " + " & ".join(
                f"${r2(statistics.median(col(n, lambda c: c['ramp'])))}$ & "
                f"${r2(statistics.median(col(n, lambda c: c['oracle'])))}$" for n in SIZES)
            + f" & ${r2(statistics.median(gm))}$" + r" \\",
            r"\bottomrule", r"\end{tabular}"]
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / "aos_ref_recovery.tex").write_text("\n".join(hdr + rows + foot) + "\n")
    say("## Files")
    say()
    for f in ("results/aos/tables/aos_ref_recovery.tex (S6 table; mixed types: the n = 800 columns hold its main size "
              "n = 400, marked *)", "results/aos/tables/aos_ref_recovery_macros.tex",
              "results/aos/main/ref_recovery_summary.json (figure data)", "results/aos/main/ref_recovery_lrsets.json",
              "results/aos/main/ref_recovery_check.json", "results/aos/main/raw/ref_ramp, ref_oracle, ref_oracle_lr",
              "aos/figs_aos/fig_recovery_curves.pdf (copy: results/aos/main/figs/fig_recovery_curves.pdf)"):
        say(f"- {f}")
    say()
    say("## Macros")
    say()
    for m_, v in MACROS:
        say(f"- \\{m_} = {v}")

    # ------------------------------------------------------------------ write
    summ = {"R_small": R_small, "R_main": R_main, "designs": KEYS, "labels": LABEL, "alpha_star": ASTAR,
            "grid_ceiling": 0.875,
            "cells": {f"{k}|{n}": c for (k, n), c in cell.items()}}
    for p, txt in ((MAIN / "ref_recovery_summary.json", json.dumps(summ, indent=1)),
                   (TAB / "aos_ref_recovery_macros.tex",
                    "% Generated by scripts/aos_ref_recovery_tables.py -- do not edit.\n"
                    + "".join(f"\\newcommand{{\\{m_}}}{{{v}}}\n" for m_, v in MACROS)),
                   (MAIN / "RECOVERY_CURVES.md", "\n".join(REPORT) + "\n")):
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(txt)
        os.replace(tmp, p)
        print("wrote", p)


if __name__ == "__main__":
    main()

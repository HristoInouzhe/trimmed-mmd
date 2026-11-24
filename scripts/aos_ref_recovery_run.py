"""AoS referee 2, M5 (aos/reviews/referee2_methods.md): recovery of alpha* against n, with an oracle ceiling.

Two kinds of records, both on the replicates of the synthetic study (explore/harder_sims, read-only), regenerated
exactly by scripts/aos_baselines_run.py::draw, at n = m in {100, 200, 400, 800} (D5b_mixed: n in {100, 200, 400},
m = 5n, as stored), levels j alpha*/8, j = 0..10, delta = 0.05:

* ramp   -- the Laplace ramp of the main runs, computed by scripts/aos_main_synth.py::replicate itself (stored fold
            bandwidths at that n, GramQP witness fits warm-started along the grid, own term removed, cut rules "order"
            (the paper's method) and "linear" on the same fits). Only below the main size: at n = 800 (D5b: 400) the 200
            replicates of results/aos/main/raw/synth_laplace are reused (`check` recomputes a few of them).
* oracle -- the SAME betting test with the witness replaced by the true indicator psi = 1{p > q}: same batches (batch 1
            bets on the second quarter of each sample, batch 2 on the second half), same pairing (sacert.pair_up), same
            horizon N = n_pairs(batch 1) + n_pairs(batch 2), same betting fractions (methods.bet_stream: c = 0.5,
            prior weight 20) and the same prior variance rule (fold1.pair_var of psi on the batch's fit block, floored
            at 1e-4), delta = 0.05. psi has no cut points and never skips a batch. E_P psi - E_Q psi = TV(P, Q) =
            alpha*, the largest mean any [0, 1]-valued witness can bet with.

psi in closed form, on the raw (unscaled) coordinates, for every design of explore/harder_sims/designs.py (p - q =
a (p0 - q0) in the contamination designs, so the sign is that of p0 - q0):

    GaussLoc (D1, D2, D3a, D4)  <mu, x> > 0                            (p0/q0 = exp(2 <mu, x>/s^2))
    CorrFlip (D3b)              x1 x2 > 0                              (log p0/q0 = 2 rho x1 x2/(1 - rho^2))
    Mixed (D5b)                 pk(+C, x1) pi(x2) > pk(-C, x1) pi'(x2)  (exact rounded-normal and Bernoulli masses;
                                = {x2 = 1, x1 >= 0} or {x2 = 0, x1 >= 1}, asserted)
    HeavyScale (D6)             |x1| < x*                              (the single crossing of the s_P t_3, s_Q t_3
                                                                        densities, s_P < s_Q, root found by designs.py)
    BMBump (D7)                 <D b, D x> > 0                          (Cameron-Martin: log p0/q0 = 2 <Db, Dx>/dt)
    Oscillating (D8)            frac(N x) >= 1/2                        (p = 1 > q = 0 off A_N; q = 2 on A_N)
    RHCBox (D9)                 x_{1..4} in [-sep - width, -sep]^4      (P0, Q0 mutually singular; on the P0 box
                                                                        p - q = a p0 > 0, elsewhere p <= q)

`check` also estimates E_P psi - E_Q psi by Monte Carlo with the design samplers and compares with alpha*.

Records go to $AOS_REF_WORK/ref_<kind>/<design>_n<n>.jsonl (local disk, appended and fsynced by the parent process;
a killed run resumes: (design, n, rep) already present are skipped) and `sync` copies them atomically to
results/aos/main/raw/ref_<kind>/.

    python -B scripts/aos_ref_recovery_run.py time --kind ramp --n 400
    python -B scripts/aos_ref_recovery_run.py run --kind oracle --workers 4
    python -B scripts/aos_ref_recovery_run.py run --kind ramp --reps 100 --workers 5
    python -B scripts/aos_ref_recovery_run.py check
    python -B scripts/aos_ref_recovery_run.py sync
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
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

import aos_baselines_run as B  # noqa: E402  (puts explore/harder_sims, explore/split_adaptive, src on sys.path)
import aos_main_synth as AMS  # noqa: E402
import designs as DS  # noqa: E402
import methods as M  # noqa: E402
import sacert as S  # noqa: E402
from fold1 import pair_var  # noqa: E402

WORK = Path(os.environ.get("AOS_REF_WORK", "/home/hristouam/aos_main_work/raw"))
OUT = ROOT / "results" / "aos" / "main" / "raw"
MAINRAW = OUT / "synth_laplace"
DESIGNS = B.DESIGNS
SIZES = (100, 200, 400, 800)
KINDS = ("ramp", "oracle", "oracle_lr")


# ---------------------------------------------------------------------------------------------------------------------
# data and the oracle witness
# ---------------------------------------------------------------------------------------------------------------------

def draw_raw(key, n, rep):
    """B.draw without the design's fixed coordinate scaling (D5b_mixed only); the scaled copy is checked against
    B.draw value for value."""
    C, PIL = B._ctx()
    dz = C[key]
    pil = PIL[key]
    m = dz.m_ratio * n
    ss = np.random.SeedSequence([B.seed_key(key), n, rep])
    rng = np.random.default_rng(ss)
    X = dz.sample(n, rng, "P")
    Y = dz.sample(m, rng, "Q")
    px, py = rng.permutation(n), rng.permutation(m)
    Xr, Yr = X[px], Y[py]
    if pil["scale"] is not None:
        _, _, _, Xs, Ys = B.draw(key, n, rep)
        sc = np.array(pil["scale"])
        assert np.array_equal(Xr / sc, Xs) and np.array_equal(Yr / sc, Ys), (key, n, rep)
    return dz, Xr, Yr


def _log_pk(c, sd, k):
    """log P(round(c + sd Z) = k), accurate in both tails."""
    from scipy.stats import norm
    lo, hi = (k - 0.5 - c) / sd, (k + 0.5 - c) / sd
    right = lo > 0                                   # both ends in the right tail: use survival functions
    a = np.where(right, norm.logsf(lo), norm.logcdf(hi))
    b = np.where(right, norm.logsf(hi), norm.logcdf(lo))
    return a + np.log1p(-np.exp(b - a))


def psi(dz, Z):
    """1{p > q} at the rows of Z (raw coordinates), as a float array of 0/1."""
    if isinstance(dz, DS.GaussLoc):
        out = Z @ dz.mu > 0
    elif isinstance(dz, DS.CorrFlip):
        out = Z[:, 0] * Z[:, 1] > 0
    elif isinstance(dz, DS.Mixed):
        k, b = Z[:, 0], Z[:, 1]
        assert np.array_equal(k, np.round(k)) and np.all((b == 0) | (b == 1))
        lp = _log_pk(dz.C, dz.SD, k) + np.log(np.where(b == 1, 0.75, 0.25))
        lq = _log_pk(-dz.C, dz.SD, k) + np.log(np.where(b == 1, 0.25, 0.75))
        out = lp > lq
        rule = ((b == 1) & (k >= 0)) | ((b == 0) & (k >= 1))
        assert np.array_equal(out, rule)
    elif isinstance(dz, DS.HeavyScale):
        assert dz.sP < dz.sQ
        out = np.abs(Z[:, 0]) < dz.xstar
    elif isinstance(dz, DS.BMBump):
        db = np.diff(np.r_[0.0, dz.b])
        dx = np.diff(np.c_[np.zeros(len(Z)), Z], axis=1)
        out = dx @ db > 0
    elif isinstance(dz, DS.Oscillating):
        out = np.floor(2.0 * dz.N * Z[:, 0]) % 2 == 1          # frac(N x) >= 1/2
    elif isinstance(dz, DS.RHCBox):
        blk = Z[:, :dz.k]
        out = np.all((blk >= -dz.sep - dz.width) & (blk <= -dz.sep), axis=1)
    else:
        raise NotImplementedError(type(dz).__name__)
    return out.astype(float)


# ---------------------------------------------------------------------------------------------------------------------
# the second oracle: the best likelihood-ratio level set at each level
# ---------------------------------------------------------------------------------------------------------------------
# psi = 1{p > q} maximizes the drift E_P g - E_Q g over [0, 1]-valued g, but the betting rule rejects when the drift is
# large against the spread of the pair statistic (to first order the log-wealth grows like N (mu/sd) sqrt(2 L / N)).
# Among indicators the population signal-to-noise ratio (P(A) - Q(A) - alpha) / sd_pair(A), sd_pair^2 =
# pair_var(P(A)(1 - P(A)), Q(A)(1 - Q(A))), is increasing in P(A) at fixed Q(A) when P(A) - Q(A) >= alpha, so by the
# Neyman-Pearson lemma its maximizer is a level set A_t = {p/q > t}. "oracle_lr" bets on A_{t_j} with t_j maximizing
# this ratio at level alpha_j (population masses by Monte Carlo, 10^6 draws per law, a fixed seed, nothing from the
# replicates), and on psi at the levels alpha_j >= alpha* (the tight null at j = 8). D8: p/q takes the values +inf and
# 1/2 only, so psi is the only level set with a positive drift. D9: every level set with t >= 1 lies in the P0 box and
# every one with t < 1 contains the complement of the Q0 box; the ratio of any of them is at most that of a set with
# Q-mass 0 and the P-mass of the box, which `lrsets` bounds against psi with the exact box masses of the core.

LRSETS = ROOT / "results" / "aos" / "main" / "ref_recovery_lrsets.json"
LR_MC, LR_CHUNK, LR_SEED = 1_000_000, 50_000, 20261123
PSI_ONLY = ("D8_N1", "D8_N4", "D8_N16", "D8_N64", "D9_rhc")


def loglr(dz, Z):
    """log p(x)/q(x) of the mixtures P = (1 - a) R + a P0, Q = (1 - a) R + a Q0, exactly (common factors cancel)."""
    from scipy.stats import t as student_t
    if isinstance(dz, DS.GaussLoc):
        s, d = dz.s, dz.d
        lr = -0.5 * np.einsum("ij,ij->i", Z, Z)
        lp0 = -0.5 * np.einsum("ij,ij->i", Z - dz.mu, Z - dz.mu) / s ** 2 - d * math.log(s)
        lq0 = -0.5 * np.einsum("ij,ij->i", Z + dz.mu, Z + dz.mu) / s ** 2 - d * math.log(s)
    elif isinstance(dz, DS.CorrFlip):
        x1, x2, rho = Z[:, 0], Z[:, 1], dz.rho
        lr = -0.5 * (x1 ** 2 + x2 ** 2)
        c = -0.5 * math.log(1 - rho ** 2)
        lp0 = -(x1 ** 2 - 2 * rho * x1 * x2 + x2 ** 2) / (2 * (1 - rho ** 2)) + c
        lq0 = -(x1 ** 2 + 2 * rho * x1 * x2 + x2 ** 2) / (2 * (1 - rho ** 2)) + c
    elif isinstance(dz, DS.Mixed):
        k, b = Z[:, 0], Z[:, 1]
        lr = _log_pk(0.0, dz.SD, k) + math.log(0.5)
        lp0 = _log_pk(dz.C, dz.SD, k) + np.log(np.where(b == 1, 0.75, 0.25))
        lq0 = _log_pk(-dz.C, dz.SD, k) + np.log(np.where(b == 1, 0.25, 0.75))
    elif isinstance(dz, DS.HeavyScale):
        x = Z[:, 0]
        lr = student_t.logpdf(x, dz.NU)
        lp0 = student_t.logpdf(x / dz.sP, dz.NU) - math.log(dz.sP)
        lq0 = student_t.logpdf(x / dz.sQ, dz.NU) - math.log(dz.sQ)
    elif isinstance(dz, DS.BMBump):
        db = np.diff(np.r_[0.0, dz.b])
        dx = np.diff(np.c_[np.zeros(len(Z)), Z], axis=1)
        u = dx @ db / dz.dt                              # <b, x> in the Cameron-Martin inner product
        c = dz.cm ** 2
        lr = np.zeros(len(Z))
        lp0, lq0 = u - c / 2, -u - c / 2
    else:
        raise NotImplementedError(type(dz).__name__)
    la, l1a = math.log(dz.a), math.log1p(-dz.a)
    return np.logaddexp(l1a + lr, la + lp0) - np.logaddexp(l1a + lr, la + lq0)


def pair_ratio(dz, n=100):
    """n_b / m_b of the paired batches (1 for n = m, 1/5 for D5b), the weight of Var_Q in pair_var."""
    m = dz.m_ratio * n
    n2, m2 = n - n // 2, m - m // 2
    assert n2 <= m2
    return n2 / m2


def snr(P, Qm, alpha, ratio):
    return (P - Qm - alpha) / np.sqrt(np.maximum(P * (1 - P) + Qm * (1 - Qm) * ratio, 1e-300))


def lrsets(log):
    """Population masses of the level sets A_t by Monte Carlo; t_j per level; written to LRSETS."""
    from scipy.stats import norm
    C, _ = B._ctx()
    out = {"n_mc": LR_MC, "seed": LR_SEED, "designs": {}}
    for key in DESIGNS:
        dz = C[key]
        lv = dz.levels()
        ratio = pair_ratio(dz)
        rec = {"alpha_star": dz.alpha_star, "ratio": ratio}
        if key in PSI_ONLY:
            rec["rule"] = "psi"
            rec["t"] = [None] * len(lv)
            if key == "D9_rhc":
                Zc = DS.RHCBox._core_pool()[:, :dz.k]
                j = dz.jitter
                lo, hi = -dz.sep - dz.width, -dz.sep
                rP = float(np.mean(np.prod(norm.cdf((hi - Zc) / j) - norm.cdf((lo - Zc) / j), axis=1)))
                rQ = float(np.mean(np.prod(norm.cdf((-lo - Zc) / j) - norm.cdf((-hi - Zc) / j), axis=1)))
                a = dz.a
                PB, QB = a + (1 - a) * rP, (1 - a) * rP           # masses of the P0 box under P and Q
                PBq, QBq = (1 - a) * rQ, a + (1 - a) * rQ         # masses of the Q0 box under P and Q
                gaps = []
                for al in lv:
                    if al >= dz.alpha_star:
                        continue
                    s_psi = float(snr(PB, QB, al, ratio))
                    bound = max(float(snr(PB, 0.0, al, ratio)), float(snr(1.0, 1.0 - QBq, al, ratio)))
                    gaps.append(bound - s_psi)
                rec["box"] = {"R_box_P": rP, "R_box_Q": rQ, "snr_gap_max": max(gaps)}
                log(f"LR {key}: core mass of the P0 box {rP:.3e}, of the Q0 box {rQ:.3e}; "
                    f"largest SNR gain any level set can have over psi {max(gaps):.2e}")
            out["designs"][key] = rec
            continue
        rng = np.random.default_rng([LR_SEED, B.seed_key(key)])
        lP, lQ = [], []
        nbad = [0]
        for _ in range(LR_MC // LR_CHUNK):
            for which, acc in (("P", lP), ("Q", lQ)):
                Z = dz.sample(LR_CHUNK, rng, which)
                L = loglr(dz, Z)
                bad = (L > 0) != (psi(dz, Z) > 0)                       # sign(log p/q) = sign(p - q), up to
                assert np.all(np.abs(L[bad]) < 1e-9), key               # rounding where log p/q is ~0
                nbad[0] += int(bad.sum())
                acc.append(L)
        lP, lQ = np.sort(np.concatenate(lP)), np.sort(np.concatenate(lQ))
        pooled = np.concatenate([lP, lQ])
        uniq = np.unique(pooled)
        if uniq.size <= 5000:                                   # discrete log-ratio (D5b): every distinct level set
            cand = np.r_[(uniq[:-1] + uniq[1:]) / 2, 0.0]
        else:
            cand = np.r_[np.quantile(pooled, np.linspace(0.0005, 0.9995, 4000)), 0.0]
        cand = np.unique(cand)
        Pm = 1.0 - np.searchsorted(lP, cand, side="right") / lP.size
        Qm = 1.0 - np.searchsorted(lQ, cand, side="right") / lQ.size
        i0 = int(np.where(cand == 0.0)[0][0])
        ts, info = [], []
        for al in lv:
            if al >= dz.alpha_star:
                ts.append(0.0)
                info.append({"t": 0.0, "P": float(Pm[i0]), "Q": float(Qm[i0]), "snr": float(snr(Pm[i0], Qm[i0], al, ratio)),
                             "snr_psi": float(snr(Pm[i0], Qm[i0], al, ratio))})
                continue
            s = snr(Pm, Qm, al, ratio)
            i = int(np.argmax(s))
            ts.append(float(cand[i]))
            info.append({"t": float(cand[i]), "P": float(Pm[i]), "Q": float(Qm[i]), "snr": float(s[i]),
                         "snr_psi": float(s[i0])})
        rec.update({"rule": "lr", "t": ts, "levels": info, "psi_drift_mc": float(Pm[i0] - Qm[i0]),
                    "sign_rounding": nbad[0]})
        out["designs"][key] = rec
        log(f"LR {key:12s} psi drift {Pm[i0] - Qm[i0]:.4f} (alpha* {dz.alpha_star:.4f}); j=0: t {ts[0]:+.3f} "
            f"P {info[0]['P']:.3f} Q {info[0]['Q']:.3f} snr {info[0]['snr']:.3f} vs psi {info[0]['snr_psi']:.3f}; "
            f"j=4: snr {info[4]['snr']:.3f} vs psi {info[4]['snr_psi']:.3f}")
    tmp = LRSETS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=1))
    os.replace(tmp, LRSETS)
    log(f"wrote {LRSETS}")


_LR = {}


def lr_thresholds(key):
    if "d" not in _LR:
        _LR["d"] = json.loads(LRSETS.read_text())["designs"]
    return _LR["d"][key]


# ---------------------------------------------------------------------------------------------------------------------
# one replicate
# ---------------------------------------------------------------------------------------------------------------------

def _bet_fixed(gx, gy, a, n, m, N_seq):
    """The ramp's betting test at level a with the fixed witness values gx (on X), gy (on Y)."""
    n1, m1 = n // 2, m // 2
    n2, m2 = n - n1, m - m1
    q_n, q_m = int(round(n / 4)), int(round(m / 4))
    nb1x, nb1y = n1 - q_n, m1 - q_m
    # (fit block, evaluation block) of batch 1 (quarter 1 -> quarter 2) and batch 2 (half 1 -> half 2), as the ramp
    bat = (((slice(0, q_n), slice(0, q_m)), (slice(q_n, n1), slice(q_m, m1)), nb1x, nb1y),
           ((slice(0, n1), slice(0, m1)), (slice(n1, n), slice(m1, m)), n2, m2))
    pvs, Zs, means = [], [], []
    for (fx, fy), (ex, ey), nbx, nby in bat:
        pvs.append(max(pair_var(gx[fx].var(), gy[fy].var(), nbx, nby), 1e-4))
        Zs.append(S.pair_up(gx[ex], -gy[ey]) - a)
        means.append(float(gx[ex].mean() - gy[ey].mean()))
    d_, lm = M.bet_stream(Zs, pvs, a, N_seq)
    return int(d_), (None if not math.isfinite(lm) else round(float(lm), 6)), pvs, means


def oracle_replicate(key, n, rep, kind="oracle"):
    T0 = time.perf_counter()
    dz, X, Y = draw_raw(key, n, rep)
    m = len(Y)
    levels = dz.levels()
    n1, m1 = n // 2, m // 2
    n2, m2 = n - n1, m - m1
    q_n, q_m = int(round(n / 4)), int(round(m / 4))
    N_seq = M.n_pairs(n1 - q_n, m1 - q_m) + M.n_pairs(n2, m2)
    px, py = psi(dz, X), psi(dz, Y)
    if kind == "oracle_lr":
        th = lr_thresholds(key)
        if th["rule"] == "lr":
            lx, ly = loglr(dz, X), loglr(dz, Y)
    dec, logmax, pv, bm, mass = [], [], [], [], []
    for j, a in enumerate(levels):
        if kind == "oracle" or th["rule"] == "psi":
            gx, gy = px, py
        else:
            gx, gy = (lx > th["t"][j]).astype(float), (ly > th["t"][j]).astype(float)
        d_, lm, pvs, means = _bet_fixed(gx, gy, a, n, m, N_seq)
        dec.append(d_)
        logmax.append(lm)
        pv.append(pvs)
        bm.append(means)
        mass.append([float(gx.mean()), float(gy.mean())])
    rec = {"design": key, "n": n, "m": m, "rep": rep, "kind": kind, "alpha_star": dz.alpha_star, "levels": levels,
           "N_seq": N_seq, "dec": dec, "logmax": logmax, "jlo": AMS_jlo(dec)}
    if kind == "oracle":           # psi is the same at every level
        rec.update({"prior_var": pv[0], "batch_mean": bm[0], "psi_x_mean": mass[0][0], "psi_y_mean": mass[0][1]})
    else:
        rec.update({"prior_var": pv, "batch_mean": bm, "mass": mass, "rule": th["rule"]})
    rec["seconds"] = round(time.perf_counter() - T0, 4)
    return rec


def AMS_jlo(dec):
    j = -1
    for x in dec:
        if not x:
            break
        j += 1
    return j


def ramp_replicate(key, n, rep):
    rec = AMS.replicate((key, n, rep, "laplace"))
    rec["kind"] = "ramp"
    return rec


def job(args):
    kind, key, n, rep = args
    return ramp_replicate(key, n, rep) if kind == "ramp" else oracle_replicate(key, n, rep, kind)


# ---------------------------------------------------------------------------------------------------------------------
# bookkeeping
# ---------------------------------------------------------------------------------------------------------------------

def sizes_of(key, C):
    return tuple(n for n in SIZES if n in C[key].ns)


def work_path(kind, key, n):
    return WORK / f"ref_{kind}" / f"{key}_n{n}.jsonl"


def plan(kind, keys, C, reps, reps_main):
    """(job, path) still to do. ramp: n below the main size, replicates 0..reps-1. oracle: every size, replicates
    0..reps-1 below the main size and 0..reps_main-1 at it (the replicates of the reused main run)."""
    jobs = []
    for key in keys:
        nmain = B.main_n(key, C)
        for n in sizes_of(key, C):
            if kind == "ramp" and n == nmain:
                continue
            R = reps_main if n == nmain else reps
            p = work_path(kind, key, n)
            done = {r["rep"] for r in AMS.read_valid(p)}
            jobs += [((kind, key, n, r), p) for r in range(R) if r not in done]
    return jobs


def sync(log=print):
    nf = nr = 0
    for kind in KINDS:
        d = WORK / f"ref_{kind}"
        if not d.exists():
            continue
        od = OUT / d.name
        od.mkdir(parents=True, exist_ok=True)
        for f in sorted(d.glob("*.jsonl")):
            recs = sorted(AMS.read_valid(f), key=lambda r: r["rep"])
            data = "".join(json.dumps(r) + "\n" for r in recs)
            tmp = od / (f.name + ".tmp")
            tmp.write_text(data)
            os.replace(tmp, od / f.name)
            if (od / f.name).read_text() != data:
                log(f"SYNC MISMATCH {f}")
            nf += 1
            nr += len(recs)
    log(f"synced {nf} files ({nr} records) to {OUT}")


def check(log):
    """(1) E_P psi - E_Q psi by Monte Carlo with the samplers against alpha*; (2) the ramp of this script equals the
    stored main run (results/aos/main/raw/synth_laplace) at the main size on a few replicates; (3) the scaled copy of
    draw_raw equals B.draw (asserted inside draw_raw) for D5b_mixed."""
    C, _ = B._ctx()
    out = {"tv": {}, "repro_main": {}}
    rng = np.random.default_rng(20261122)
    NMC, CH = 400_000, 50_000
    for key in DESIGNS:
        dz = C[key]
        sx = sy = 0.0
        for _ in range(NMC // CH):
            sx += psi(dz, dz.sample(CH, rng, "P")).sum()
            sy += psi(dz, dz.sample(CH, rng, "Q")).sum()
        px, py = sx / NMC, sy / NMC
        se = math.sqrt((px * (1 - px) + py * (1 - py)) / NMC)
        est = px - py
        out["tv"][key] = {"alpha_star": dz.alpha_star, "mc": est, "se": se, "z": (est - dz.alpha_star) / max(se, 1e-12),
                          "Ppsi": px, "Qpsi": py, "n_mc": NMC}
        log(f"TV {key:12s} alpha* {dz.alpha_star:.6f} MC {est:.6f} +- {se:.6f} z {(est - dz.alpha_star) / max(se, 1e-12):+.2f}")
    for key in ("D1_d2_a10", "D5b_mixed", "D8_N4", "D9_rhc"):
        n = B.main_n(key, C)
        stored = {r["rep"]: r for r in AMS.read_valid(MAINRAW / f"{key}_n{n}.jsonl")}
        res = []
        for rep in (0, 1):
            r = ramp_replicate(key, n, rep)
            s = stored[rep]
            res.append(all(r["cuts"][c]["dec"] == s["cuts"][c]["dec"] for c in AMS.CUTS))
        out["repro_main"][key] = res
        log(f"ramp repro {key} n={n} reps 0,1: {res}")
    for rep in range(3):
        draw_raw("D5b_mixed", 100, rep)
    out["draw_raw_scaled_equal"] = True
    # the psi records (written before oracle_replicate was generalized to oracle_lr) recomputed by the current code
    tot = same = 0
    for key in DESIGNS:
        for n in sizes_of(key, C):
            stored = {r["rep"]: r for r in AMS.read_valid(work_path("oracle", key, n))}
            for rep in (0, 7, 42):
                r = oracle_replicate(key, n, rep)
                s = stored[rep]
                tot += 1
                same += int(r["dec"] == s["dec"] and r["logmax"] == s["logmax"] and r["prior_var"] == s["prior_var"])
    out["oracle_recomputed_equal"] = f"{same}/{tot}"
    log(f"psi oracle records recomputed: {same}/{tot} equal")
    # D8, D9: oracle_lr bets on psi, so its decisions must equal the psi records
    tot = same = 0
    for key in PSI_ONLY:
        for n in sizes_of(key, C):
            a_ = {r["rep"]: r for r in AMS.read_valid(work_path("oracle", key, n))}
            b_ = {r["rep"]: r for r in AMS.read_valid(work_path("oracle_lr", key, n))}
            for rep in set(a_) & set(b_):
                tot += 1
                same += int(a_[rep]["dec"] == b_[rep]["dec"])
    out["psi_only_lr_equal"] = f"{same}/{tot}"
    log(f"D8/D9 oracle_lr = psi decisions: {same}/{tot}")
    p = ROOT / "results" / "aos" / "main" / "ref_recovery_check.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=1))
    os.replace(tmp, p)
    log(f"wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "sync", "time", "check", "lrsets"])
    ap.add_argument("--kind", default="oracle", choices=KINDS)
    ap.add_argument("--designs", nargs="+", default=["all"])
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--reps-main", type=int, default=200)
    ap.add_argument("--n", type=int, default=None, help="time: the size to time")
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()

    def log(s):
        print(time.strftime("%H:%M:%S"), s, flush=True)
    if a.cmd == "sync":
        sync(log)
        return
    if a.cmd == "check":
        check(log)
        return
    if a.cmd == "lrsets":
        lrsets(log)
        return
    assert a.workers <= 6
    C, _ = B._ctx()
    keys = DESIGNS if a.designs == ["all"] else a.designs
    if a.cmd == "time":
        tot = 0.0
        for key in keys:
            n = a.n if a.n in C[key].ns else max(x for x in C[key].ns if x <= (a.n or 800))
            r = job((a.kind, key, n, 0))
            sec = r["seconds"]
            tot += sec
            jl = r["jlo"] if a.kind == "oracle" else [r["cuts"][c]["jlo"] for c in AMS.CUTS]
            log(f"{key:12s} n={n} {sec:.2f}s jlo {jl}")
        log(f"total {tot:.1f}s for one replicate of each design")
        return
    jobs = plan(a.kind, keys, C, a.reps, a.reps_main)
    jobs.sort(key=lambda j: (j[0][3], -j[0][2], j[0][1]))     # by replicate, larger n first: a killed run is balanced
    log(f"{len(jobs)} replicates, kind {a.kind}, workers {a.workers}")
    t0 = time.time()
    handles = {}
    done = 0
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(job, j): (j, p) for j, p in jobs}
        for fu in as_completed(futs):
            j, p = futs[fu]
            try:
                rec = fu.result()
            except Exception as e:
                log(f"FAILED {j} {e!r}")
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            f = handles.get(p)
            if f is None:
                f = handles[p] = open(p, "a")
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
            done += 1
            if done % 100 == 0 or done == len(jobs):
                el = time.time() - t0
                log(f"{done}/{len(jobs)} {el:.0f}s eta {el / done * (len(jobs) - done):.0f}s")
    for f in handles.values():
        f.close()
    sync(log)
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

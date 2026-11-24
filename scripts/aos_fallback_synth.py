"""Exploratory (AoS): cut-point rules OWN / FALLBACK / PLAIN (and the Gaussian vs Laplace kernel) for the paper's
sequential betting bound ("betting, ramp" = the study's seq_bet) on the synthetic designs of Figure 2(a).
Protocol: results/aos/fallback/NOTES.md.

Each replicate regenerates the stored data exactly (scripts/aos_baselines_run.py::draw: SeedSequence([crc32(design),
n, rep]), the study's pilot scaling and permutation), uses the stored fold bandwidths (sigma1 on fold one for the
batch-2 witness, sigma0 on batch 0 for the batch-1 witness), the study's GramQP witness fits
(explore/harder_sims/methods.py::witness_values, read-only) and its betting helpers (sacert.ramp / pair_up,
fold1.cut_points / priors, methods.bet_stream), so the OWN rule is the study's seq_bet. Every level j = 0..10 is
fitted for both batches (warm starts along the full grid) and the three rules are evaluated on the same fits.
``--chain stored`` instead fits only the levels the stored run fitted (from the stored jlo of the study's split
methods), which reproduces the stored warm starts; it is used to check the reproduction of seq_bet.

Kernel ``laplace``: K = exp(-|x - y| / h) with h = the same fold medians (h = sigma1, sigma0); ``--analytic`` adds the
analytic test (study column an_new: tau_new thresholds of thr_cache.json, valid for any kernel with k >= 0 and
k(x, x) = 1) on the Laplace Gram matrix at the pilot median h = sigma_fix.

Records are appended by the parent process to $AOS_FALLBACK_WORK/synth_<kernel>[_<chain>]/<design>_n<n>.jsonl
(local disk) and `sync` copies them atomically to results/aos/fallback/raw/. A killed run resumes.

    python -B scripts/aos_fallback_synth.py run --kernel gaussian --reps 50 --workers 4
    python -B scripts/aos_fallback_synth.py run --kernel gaussian --reps 50 --n 200 --levels zero \
        --designs D2_d20 D2_d50 D3a_loc D6_t3 D7_wide
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
for _p in (ROOT / "scripts",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

import aos_baselines_run as B  # noqa: E402  (puts explore/harder_sims, explore/split_adaptive, src on sys.path)
import methods as M  # noqa: E402
import sacert as S  # noqa: E402
from fold1 import cut_points, priors  # noqa: E402
from trimmedmmd.kernels import median_heuristic, sqdist  # noqa: E402

WORK = Path(os.environ.get("AOS_FALLBACK_WORK", "/home/hristouam/aos_fallback_work/raw"))
OUT = ROOT / "results" / "aos" / "fallback" / "raw"
RULES = ("own", "fallback", "plain")
DESIGNS = B.DESIGNS
SPLIT_K = M.SPLIT_K
_THR = {}


def thr_new(n, m, levels):
    if "t" not in _THR:
        _THR["t"] = json.loads(B.THR.read_text())
    return [_THR["t"][f"{n},{m},{a:.12g}"]["new"] for a in levels]


def stored_rec(key, n, rep):
    f = B.STORED / f"{key}.jsonl"
    cache = _THR.setdefault(("st", key, n), None)
    if cache is None:
        cache = {}
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r["n"] == n:
                    cache[r["rep"]] = {"dec": r["dec"], "jlo": r["jlo"], "sigma1": r["diag"]["sigma1"],
                                       "sigma0": r["diag"]["sigma0"]}
        _THR[("st", key, n)] = cache
    return cache.get(rep)


def pick(u, v, w, vv, norm, a, rule):
    """Cut points under a rule from the own-term values (u, v) and the weights: (s, sp, u_used, v_used, source)."""
    so, spo = cut_points(u, v, a)
    up, vp = u + w / norm, v - vv / norm                       # plain (in-sample) values, k(x, x) = 1
    sp_, spp = cut_points(up, vp, a)
    raw = (float(so), float(spo), float(sp_), float(spp))
    own_ok, pl_ok = so > spo, sp_ > spp
    if rule == "own":
        src = "own" if own_ok else "skip"
    elif rule == "plain":
        src = "plain" if pl_ok else "skip"
    else:
        src = "own" if own_ok else ("plain" if pl_ok else "skip")
    if src == "own":
        return so, spo, u, v, src, raw
    if src == "plain":
        return sp_, spp, up, vp, src, raw
    return None, None, None, None, src, raw


def replicate(args):
    key, n, rep, kernel, chain, level_mode, do_analytic = args
    T0 = time.perf_counter()
    dz, pil, ss, X, Y = B.draw(key, n, rep)
    m = len(Y)
    levels = dz.levels()
    st = stored_rec(key, n, rep)
    Zall = np.vstack([X, Y])
    D2 = sqdist(Zall, Zall)
    n1, m1 = n // 2, m // 2
    n2, m2 = n - n1, m - m1
    f1 = np.r_[np.arange(n1), n + np.arange(m1)]
    f2 = np.r_[np.arange(n1, n), n + np.arange(m1, m)]
    q_n, q_m = int(round(n / 4)), int(round(m / 4))
    b0 = np.r_[np.arange(q_n), n + np.arange(q_m)]
    b1 = np.r_[np.arange(q_n, n1), n + np.arange(q_m, m1)]
    nb1x, nb1y = n1 - q_n, m1 - q_m
    # fold bandwidths: recomputed as methods.run_replicate (same SeedSequence spawns), then the stored values used
    sig1 = median_heuristic(X[:n1], Y[:m1], rng=np.random.default_rng(ss.spawn(1)[0]))
    sig0 = median_heuristic(X[:q_n], Y[:q_m], rng=np.random.default_rng(ss.spawn(2)[1]))
    repro = {}
    if st is not None:
        repro["sigma"] = bool(abs(sig1 - st["sigma1"]) <= 1e-9 * sig1 and abs(sig0 - st["sigma0"]) <= 1e-9 * sig0)
        sig1, sig0 = st["sigma1"], st["sigma0"]
    N_seq = M.n_pairs(nb1x, nb1y) + M.n_pairs(n2, m2)
    if kernel == "gaussian":
        def Kb(r, c, s):
            return np.exp(-D2[np.ix_(r, c)] / (2 * s ** 2))          # exactly the study's expression
    elif kernel == "laplace":
        np.fill_diagonal(D2, 0.0)
        Dm = np.sqrt(D2)
        del D2

        def Kb(r, c, s):
            return np.exp(-Dm[np.ix_(r, c)] / s)
    else:
        raise ValueError(kernel)
    K11, K21 = Kb(f1, f1, sig1), Kb(f2, f1, sig1)
    K00, K10 = Kb(b0, b0, sig0), Kb(b1, b0, sig0)
    if kernel == "laplace":
        assert np.all(np.diag(K11) == 1.0) and np.all(np.diag(K00) == 1.0)
    qp11, qp00 = M.GramQP(K11, n1, m1), M.GramQP(K00, q_n, q_m)
    J = list(range(len(levels))) if level_mode == "all" else [0]
    if chain == "stored":
        if st is None:
            raise RuntimeError("no stored replicate for --chain stored")
        jmax_any = max(st["jlo"][k] for k in SPLIT_K)
        need11 = {j for j in J if j in M.REPORT_J or jmax_any >= j - 1}
        need00 = {j for j in J if j in M.REPORT_J or st["jlo"]["seq_bet"] >= j - 1}
    else:
        need11 = need00 = set(J)
    rec = {"design": key, "n": n, "m": m, "rep": rep, "kernel": kernel, "chain": chain, "alpha_star": dz.alpha_star,
           "levels": levels, "J": J, "N_seq": N_seq, "sigma1": sig1, "sigma0": sig0, "repro": repro,
           "rules": {r: {"dec": [], "logmax": [], "src": []} for r in RULES}, "batches": []}
    if st is not None:
        rec["stored"] = {"jlo": st["jlo"]["seq_bet"], "dec": st["dec"]["seq_bet"], "an_new_jlo": st["jlo"]["an_new"],
                         "an_new_dec": st["dec"]["an_new"]}
    warm1 = warm0 = None
    out1 = out0 = None
    iters = 0
    for j in J:
        a = levels[j]
        if j in need11:
            out1, warm1, it = M.witness_values(qp11, K21, n1, m1, a, warm1)
            iters += it
        if j in need00:
            out0, warm0, it0 = M.witness_values(qp00, K10, q_n, q_m, a, warm0)
            iters += it0
        if chain == "stored" and not (j in need11 and j in need00):
            for r in RULES:
                rec["rules"][r]["dec"].append(None)
                rec["rules"][r]["logmax"].append(None)
                rec["rules"][r]["src"].append(None)
            rec["batches"].append(None)
            continue
        bat = []
        # batch 1: quarter 1 -> quarter 2 (qp00); batch 2: half 1 -> half 2 (qp11)
        for out, warm, nbx, nby in ((out0, warm0, nb1x, nb1y), (out1, warm1, n2, m2)):
            if out is None:
                bat.append(None)
                continue
            u, v, fev, norm = out
            bat.append((u, v, warm[0], warm[1], norm, fev[:nbx], fev[nbx:], nbx, nby))
        binfo = []
        for b in bat:
            if b is None:
                binfo.append({"norm": 0.0})
                continue
            _, _, _, _, _, raw = pick(b[0], b[1], b[2], b[3], b[4], a, "own")
            binfo.append({"norm": float(b[4]), "own_gap": raw[0] - raw[1], "plain_gap": raw[2] - raw[3],
                          "own_crossed": bool(not raw[0] > raw[1]), "plain_crossed": bool(not raw[2] > raw[3])})
        rec["batches"].append(binfo)
        for r in RULES:
            Zs, pvs, srcs = [], [], []
            for b in bat:
                if b is None:
                    Zs.append(None)
                    pvs.append(None)
                    srcs.append("none")
                    continue
                u, v, w, vv, norm, fx, fy, nbx, nby = b
                s, sp, uu, vu, src, _ = pick(u, v, w, vv, norm, a, r)
                srcs.append(src)
                if src == "skip":
                    Zs.append(None)
                    pvs.append(None)
                    continue
                G, H = S.ramp(fx, s, sp), S.ramp(fy, s, sp)
                Zs.append(S.pair_up(G, -H) - a)
                pvs.append(priors(uu, vu, s, sp, a, nbx, nby)["ramp"])
            dec, lmax = M.bet_stream(Zs, pvs, a, N_seq)
            rec["rules"][r]["dec"].append(int(dec))
            rec["rules"][r]["logmax"].append(None if not math.isfinite(lmax) else float(lmax))
            rec["rules"][r]["src"].append(srcs)
    for r in RULES:
        d = rec["rules"][r]["dec"]
        jlo = -1
        for x in d:
            if not x:
                break
            jlo += 1
        rec["rules"][r]["jlo"] = jlo if level_mode == "all" else None
    if st is not None and level_mode == "all":
        own = rec["rules"]["own"]["dec"]
        rec["repro"]["seq_jlo"] = bool(rec["rules"]["own"]["jlo"] == st["jlo"]["seq_bet"]) if chain == "all" else None
        rec["repro"]["seq_dec"] = bool([own[j] for j in M.REPORT_J] == st["dec"]["seq_bet"])
        if chain == "stored":
            # the stored run fitted seq_bet's levels up to its first failure, so its jlo is checkable
            jj = -1
            for x in own:
                if not x:
                    break
                jj += 1
            rec["repro"]["seq_jlo"] = bool(jj == st["jlo"]["seq_bet"])
    elif st is not None:
        rec["repro"]["seq_dec0"] = bool(rec["rules"]["own"]["dec"][0] == st["dec"]["seq_bet"][0])
    rec["iters"] = iters
    del K11, K21, K00, K10, qp11, qp00
    if do_analytic:
        t0 = time.perf_counter()
        if kernel == "laplace":
            K = np.exp(-Dm / pil["sigma_fix"])
        else:
            K = np.exp(-D2 / (2.0 * pil["sigma_fix"] ** 2))
        jlo_a, decs, diag = M.analytic(K, n, m, levels, {"new": thr_new(n, m, levels)})
        rec["analytic"] = {"jlo": jlo_a["new"], "dec": decs["new"], "qps": diag["an_qps"],
                           "undecided": diag["an_undecided"], "h": pil["sigma_fix"], "seconds": time.perf_counter() - t0}
        del K
    rec["seconds"] = time.perf_counter() - T0
    return rec


def work_path(kernel, chain, key, n):
    d = WORK / (f"synth_{kernel}" + ("" if chain == "all" else f"_{chain}"))
    return d / f"{key}_n{n}.jsonl"


def read_valid(path):
    recs, seen = [], set()
    if path.exists():
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
    OUT.mkdir(parents=True, exist_ok=True)
    nf = nr = 0
    for d in sorted(WORK.glob("synth_*")):
        od = OUT / d.name
        od.mkdir(parents=True, exist_ok=True)
        for f in sorted(d.glob("*.jsonl")):
            recs = read_valid(f)
            data = "".join(json.dumps(r) + "\n" for r in recs)
            tmp = od / (f.name + ".tmp")
            tmp.write_text(data)
            os.replace(tmp, od / f.name)
            if (od / f.name).read_text() != data:
                log(f"SYNC MISMATCH {f}")
            nf += 1
            nr += len(recs)
    log(f"synced {nf} files ({nr} records) to {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "sync", "time"])
    ap.add_argument("--kernel", default="gaussian", choices=["gaussian", "laplace"])
    ap.add_argument("--chain", default="all", choices=["all", "stored"])
    ap.add_argument("--levels", default="all", choices=["all", "zero"])
    ap.add_argument("--analytic", action="store_true")
    ap.add_argument("--designs", nargs="+", default=["all"])
    ap.add_argument("--n", type=int, default=0, help="0 = the Figure 2(a) size (800; D5b 400)")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--rep0", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "sync":
        sync()
        return
    assert a.workers <= 6
    C = B.catalogue()
    keys = DESIGNS if a.designs == ["all"] else a.designs

    def log(s):
        print(time.strftime("%H:%M:%S"), s, flush=True)
    jobs = []
    for key in keys:
        n = a.n if a.n else B.main_n(key, C)
        if a.n and a.n not in C[key].ns:
            raise SystemExit(f"{key}: n = {a.n} not a study size")
        p = work_path(a.kernel, a.chain, key, n) if a.levels == "all" else \
            work_path(a.kernel, a.chain, key, n).with_name(f"{key}_n{n}_zero.jsonl")
        done = {r["rep"] for r in read_valid(p)}
        jobs += [((key, n, r, a.kernel, a.chain, a.levels, a.analytic), p)
                 for r in range(a.rep0, a.rep0 + a.reps) if r not in done]
    jobs.sort(key=lambda j: (j[0][2], j[0][0]))          # by replicate index, so a killed run is balanced
    if a.cmd == "time":
        for j, p in jobs[: a.workers]:
            r = replicate(j)
            log(f"{j[0][:3]} {r['seconds']:.1f}s iters {r['iters']} repro {r['repro']} "
                f"jlo {[r['rules'][x]['jlo'] for x in RULES]} an {r.get('analytic', {}).get('jlo')}")
        return
    log(f"{len(jobs)} replicates, kernel {a.kernel}, chain {a.chain}, levels {a.levels}, analytic {a.analytic}, "
        f"workers {a.workers}")
    t0 = time.time()
    handles = {}
    done = 0
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(replicate, j): (j, p) for j, p in jobs}
        for fu in as_completed(futs):
            j, p = futs[fu]
            try:
                rec = fu.result()
            except Exception as e:
                log(f"FAILED {j[:3]} {e!r}")
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
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
            if done % 200 == 0:
                try:
                    sync(log)
                except Exception as e:
                    log(f"SYNC FAILED {e!r}")
    for f in handles.values():
        f.close()
    sync(log)
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

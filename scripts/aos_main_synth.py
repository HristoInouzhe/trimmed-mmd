"""AoS restructure (plan section 4.1): the split-sample bound with the Laplace kernel and genuine empirical quantile cut
points on the 19 designs of Figure 2(a), every level j alpha*/8 (j = 0..10). Protocol: results/aos/main/NOTES.md.

Each replicate regenerates the stored data exactly (scripts/aos_baselines_run.py::draw), uses the stored fold bandwidths
(h = sigma1 on fold one for the batch-2 witness, h = sigma0 on batch 0 for the batch-1 witness), the study's GramQP
witness fits (explore/harder_sims/methods.py::witness_values, read-only) warm-started along the full grid, and its betting
helpers (sacert.ramp / pair_up, fold1.priors, methods.bet_stream), exactly as scripts/aos_fallback_synth.py --chain all.
On the same fits two cut rules are evaluated, both on the own-term-removed fit values, batch skipped if they cross:

* "order"  -- genuine empirical quantiles (trimmedmmd.splitting.ramp_cut_points(rule="order")), the paper's method;
* "linear" -- numpy.quantile with linear interpolation (the default; = fold1.cut_points), every earlier run.

Kernel "laplace": K = exp(-|x - y| / h) with an exact zero diagonal distance (k(x, x) = 1); "gaussian": the study's
exp(-|x - y|^2 / (2 h^2)) (with "linear" this is the stored seq_bet; used as the reproduction check).

Records go to $AOS_MAIN_WORK/synth_<kernel>/<design>_n<n>.jsonl (local disk; parent process appends, fsync) and
`sync` copies them atomically to results/aos/main/raw/. A killed run resumes.

    python -B scripts/aos_main_synth.py run --kernel laplace --rep0 0 --reps 200 --workers 5
    python -B scripts/aos_main_synth.py run --kernel gaussian --rep0 0 --reps 10 --workers 5
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
import methods as M  # noqa: E402
import sacert as S  # noqa: E402
from fold1 import priors  # noqa: E402
from trimmedmmd.kernels import median_heuristic, sqdist  # noqa: E402
from trimmedmmd.splitting import ramp_cut_points  # noqa: E402

WORK = Path(os.environ.get("AOS_MAIN_WORK", "/home/hristouam/aos_main_work/raw"))
# copies to the pCloud mount only at the end of a run (0) unless AOS_MAIN_SYNC_EVERY is set (20:40, NOTES.md)
SYNC_EVERY = int(os.environ.get("AOS_MAIN_SYNC_EVERY", "0"))
OUT = ROOT / "results" / "aos" / "main" / "raw"
CUTS = ("order", "linear")
DESIGNS = B.DESIGNS
_ST = {}


def stored_rec(key, n, rep):
    cache = _ST.get((key, n))
    if cache is None:
        cache = {}
        for line in (B.STORED / f"{key}.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r["n"] == n:
                    cache[r["rep"]] = {"dec": r["dec"], "jlo": r["jlo"], "sigma1": r["diag"]["sigma1"],
                                       "sigma0": r["diag"]["sigma0"]}
        _ST[(key, n)] = cache
    return cache.get(rep)


def replicate(args):
    key, n, rep, kernel = args
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
    # fold bandwidths recomputed as methods.run_replicate (same SeedSequence spawns, same order), stored values used
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
    rec = {"design": key, "n": n, "m": m, "rep": rep, "kernel": kernel, "alpha_star": dz.alpha_star,
           "levels": levels, "N_seq": N_seq, "sigma1": sig1, "sigma0": sig0, "repro": repro,
           "cuts": {c: {"dec": [], "logmax": [], "src": [], "gap": []} for c in CUTS}, "norms": []}
    if st is not None:
        rec["stored"] = {"jlo": st["jlo"]["seq_bet"], "dec": st["dec"]["seq_bet"]}
    warm1 = warm0 = None
    iters = 0
    for j, a in enumerate(levels):
        out1, warm1, it = M.witness_values(qp11, K21, n1, m1, a, warm1)
        out0, warm0, it0 = M.witness_values(qp00, K10, q_n, q_m, a, warm0)
        iters += it + it0
        # batch 1: quarter 1 -> quarter 2 (qp00); batch 2: half 1 -> half 2 (qp11)
        bat = []
        for out, nbx, nby in ((out0, nb1x, nb1y), (out1, n2, m2)):
            if out is None:
                bat.append(None)
                continue
            u, v, fev, norm = out
            bat.append((u, v, norm, fev[:nbx], fev[nbx:], nbx, nby))
        rec["norms"].append([None if b is None else float(b[2]) for b in bat])
        for c in CUTS:
            Zs, pvs, srcs, gaps = [], [], [], []
            for b in bat:
                if b is None:
                    Zs.append(None)
                    pvs.append(None)
                    srcs.append("none")
                    gaps.append(None)
                    continue
                u, v, norm, fx, fy, nbx, nby = b
                s, sp = ramp_cut_points(u, v, a, rule=c)
                gaps.append(float(s - sp))
                if not s > sp:
                    Zs.append(None)
                    pvs.append(None)
                    srcs.append("skip")
                    continue
                srcs.append("own")
                G, H = S.ramp(fx, s, sp), S.ramp(fy, s, sp)
                Zs.append(S.pair_up(G, -H) - a)
                pvs.append(priors(u, v, s, sp, a, nbx, nby)["ramp"])
            dec, lmax = M.bet_stream(Zs, pvs, a, N_seq)
            rc = rec["cuts"][c]
            rc["dec"].append(int(dec))
            rc["logmax"].append(None if not math.isfinite(lmax) else round(float(lmax), 6))
            rc["src"].append(srcs)
            rc["gap"].append(gaps)
    for c in CUTS:
        jlo = -1
        for x in rec["cuts"][c]["dec"]:
            if not x:
                break
            jlo += 1
        rec["cuts"][c]["jlo"] = jlo
    if st is not None:
        lin = rec["cuts"]["linear"]
        rec["repro"]["seq_jlo"] = bool(lin["jlo"] == st["jlo"]["seq_bet"])
        rec["repro"]["seq_dec"] = bool([lin["dec"][j] for j in M.REPORT_J] == st["dec"]["seq_bet"])
    rec["iters"] = iters
    rec["seconds"] = round(time.perf_counter() - T0, 3)
    return rec


def work_path(kernel, key, n):
    return WORK / f"synth_{kernel}" / f"{key}_n{n}.jsonl"


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


def sync(log=print, prefix="synth_"):
    OUT.mkdir(parents=True, exist_ok=True)
    nf = nr = 0
    for d in sorted(WORK.glob(prefix + "*")):
        od = OUT / d.name
        od.mkdir(parents=True, exist_ok=True)
        for f in sorted(d.glob("*.jsonl")):
            recs = sorted(read_valid(f), key=lambda r: r["rep"])
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
    ap.add_argument("--kernel", default="laplace", choices=["gaussian", "laplace"])
    ap.add_argument("--designs", nargs="+", default=["all"])
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--rep0", type=int, default=0)
    ap.add_argument("--workers", type=int, default=5)
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
        n = B.main_n(key, C)
        p = work_path(a.kernel, key, n)
        done = {r["rep"] for r in read_valid(p)}
        jobs += [((key, n, r, a.kernel), p) for r in range(a.rep0, a.rep0 + a.reps) if r not in done]
    jobs.sort(key=lambda j: (j[0][2], j[0][0]))          # by replicate index, so a killed run is balanced
    if a.cmd == "time":
        for j, p in jobs[: a.workers]:
            r = replicate(j)
            log(f"{j[0][:3]} {r['seconds']:.1f}s iters {r['iters']} repro {r['repro']} "
                f"jlo {[r['cuts'][c]['jlo'] for c in CUTS]} stored {r.get('stored', {}).get('jlo')}")
        return
    log(f"{len(jobs)} replicates, kernel {a.kernel}, workers {a.workers}")
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
            if done % 50 == 0 or done == len(jobs):
                el = time.time() - t0
                log(f"{done}/{len(jobs)} {el:.0f}s eta {el / done * (len(jobs) - done):.0f}s")
            if SYNC_EVERY and done % SYNC_EVERY == 0:
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

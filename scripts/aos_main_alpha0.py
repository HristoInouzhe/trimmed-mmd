"""AoS restructure (plan section 4.1, output (a)): power against H_{0,0} of the split-sample ramp ("betting, ramp", the
study's seq_bet) with the Laplace kernel, at every stored size of explore/harder_sims (n = 100..800, 1600 where stored,
replicates 0-199) and at its alpha = 0 extension sizes (*_n0ext.jsonl, replicates 200000-200199, up to n = 12800; mixed
types m = 5n), for the 19 designs and the four P = Q nulls. Protocol: results/aos/main/NOTES.md.

At alpha = 0 the trimmed-MMD witness of a fit block is the difference of its mean embeddings (the feasible set is one
point), so no QP is solved, and both cut rules give (max, min) of the own-term-removed fit values. As the stored
extension (explore/harder_sims/alpha0.py) the kernel sums are EXACT and computed in 512 x 512 blocks, so memory stays
bounded at n = 12800 (m = 64000): for every row the sums over the fold-one X and Y points (bandwidth sigma1, the batch-2
witness) and, for the fold-one rows, the sums over the batch-0 X and Y points (sigma0, the batch-1 witness). Data,
permutation and bandwidth streams are those of methods.run_replicate / alpha0.run_replicate
(scripts/aos_baselines_run.py::draw); the stored sigma1, sigma0 are used (recomputed values compared).

Kernel "laplace": exp(-|x - y| / h), the distance of a point to itself set to exactly 0; "gaussian": exp(-|x-y|^2/(2h^2))
as alpha0.py (diagonal as computed), used to check the reproduction of the stored seq_bet decisions.

Records go to $AOS_MAIN_WORK/alpha0_<kernel>/<design>_n<n>.jsonl and `sync` copies them to results/aos/main/raw/.

    python -B scripts/aos_main_alpha0.py run --kernel laplace --workers 5 [--max-n 6400] [--designs ...]
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
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

import aos_baselines_run as B  # noqa: E402
import methods as M  # noqa: E402
import sacert as S  # noqa: E402
from fold1 import priors  # noqa: E402
from trimmedmmd.kernels import median_heuristic  # noqa: E402
from trimmedmmd.splitting import ramp_cut_points  # noqa: E402

WORK = Path(os.environ.get("AOS_MAIN_WORK", "/home/hristouam/aos_main_work/raw"))
# copies to the pCloud mount only at the end of a run (0) unless AOS_MAIN_SYNC_EVERY is set (20:40, NOTES.md)
SYNC_EVERY = int(os.environ.get("AOS_MAIN_SYNC_EVERY", "0"))
OUT = ROOT / "results" / "aos" / "main" / "raw"
BLOCK = 512
DESIGNS = B.DESIGNS + B.NULLS


def stored_cells(key):
    """{(n, rep): (sigma1, sigma0, stored seq_bet decision at alpha = 0)} of the study and its extension."""
    out = {}
    for name in (f"{key}.jsonl", f"{key}_n0ext.jsonl"):
        f = B.STORED / name
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("mode") == "alpha0":
                out[(r["n"], r["rep"])] = (r["sigma1"], r["sigma0"], int(r["dec0"]["seq_bet"]))
            else:
                out[(r["n"], r["rep"])] = (r["diag"]["sigma1"], r["diag"]["sigma0"], int(r["dec"]["seq_bet"][0]))
    return out


def kernel_sums(Z, n1, m1, q_n, q_m, h1, h0, kernel, block=BLOCK):
    """Rows of Z: [X1, Y1, X2, Y2]. Returns A1, B1 (every row: sums over fold-one X resp. Y at h1) and A0, B0 (fold-one
    rows: sums over batch-0 X resp. Y at h0)."""
    N = Z.shape[0]
    N1 = n1 + m1
    sq = np.einsum("ij,ij->i", Z, Z)
    Zm = -2.0 * Z
    cols = [(0, q_n, 0, True), (q_n, n1, 0, False), (n1, n1 + q_m, 1, True), (n1 + q_m, N1, 1, False)]
    A1, B1 = np.zeros(N), np.zeros(N)
    A0, B0 = np.zeros(N1), np.zeros(N1)
    G = np.empty((block, block))
    E = np.empty((block, block))
    if kernel == "gaussian":
        g1, g0 = 1.0 / (2 * h1 ** 2), 1.0 / (2 * h0 ** 2)
    for cs, ce, ccls, cb0 in cols:
        for c0 in range(cs, ce, block):
            c1 = min(c0 + block, ce)
            for r0 in range(0, N, block):
                r1 = min(r0 + block, N)
                g = G[: r1 - r0, : c1 - c0]
                np.matmul(Zm[r0:r1], Z[c0:c1].T, out=g)
                g += sq[r0:r1, None]
                g += sq[None, c0:c1]
                np.maximum(g, 0.0, out=g)
                e = E[: r1 - r0, : c1 - c0]
                if kernel == "laplace":
                    lo, hi = max(r0, c0), min(r1, c1)
                    if lo < hi:                       # the distance of a point to itself is exactly 0
                        idx = np.arange(lo, hi)
                        g[idx - r0, idx - c0] = 0.0
                    np.sqrt(g, out=g)                 # g now holds distances
                    np.multiply(g, -1.0 / h1, out=e)
                else:
                    np.multiply(g, -g1, out=e)
                np.exp(e, out=e)
                (A1 if ccls == 0 else B1)[r0:r1] += e.sum(1)
                if cb0 and r0 < N1:
                    rr1 = min(r1, N1)
                    gg = g[: rr1 - r0]
                    ee = e[: rr1 - r0]
                    np.multiply(gg, -1.0 / h0 if kernel == "laplace" else -g0, out=ee)
                    np.exp(ee, out=ee)
                    (A0 if ccls == 0 else B0)[r0:rr1] += ee.sum(1)
    return A1, B1, A0, B0


def replicate(args):
    key, n, rep, kernel, sig_st = args
    T0 = time.perf_counter()
    dz, pil, ss, X, Y = B.draw(key, n, rep)
    m = len(Y)
    n1, m1 = n // 2, m // 2
    n2, m2 = n - n1, m - m1
    q_n, q_m = int(round(n / 4)), int(round(m / 4))
    nb1x, nb1y = n1 - q_n, m1 - q_m
    sig1 = median_heuristic(X[:n1], Y[:m1], rng=np.random.default_rng(ss.spawn(1)[0]))
    sig0 = median_heuristic(X[:q_n], Y[:q_m], rng=np.random.default_rng(ss.spawn(2)[1]))
    rec = {"design": key, "n": n, "m": m, "rep": rep, "kernel": kernel, "alpha_star": dz.alpha_star}
    if sig_st is not None:
        s1, s0, dec_st = sig_st
        rec["repro_sigma"] = bool(abs(sig1 - s1) <= 1e-9 * sig1 and abs(sig0 - s0) <= 1e-9 * sig0)
        rec["stored_seq0"] = dec_st
        sig1, sig0 = s1, s0
    rec.update(sigma1=sig1, sigma0=sig0)
    Z = np.ascontiguousarray(np.vstack([X[:n1], Y[:m1], X[n1:], Y[m1:]]).astype(float))
    del X, Y
    A1, B1, A0, B0 = kernel_sums(Z, n1, m1, q_n, q_m, sig1, sig0, kernel)
    del Z
    N1 = n1 + m1
    # batch 2: fold-one witness (mean-embedding difference at sigma1), evaluated on fold two
    Kz = A1 / n1 - B1 / m1
    val1 = max(float(Kz[:n1].sum() / n1 - Kz[n1:N1].sum() / m1), 0.0)
    norm = math.sqrt(val1)
    Z2 = pv2 = None
    if norm > 1e-12:
        u1 = Kz[:n1] / norm - (1.0 / n1) / norm
        v1 = Kz[n1:N1] / norm + (1.0 / m1) / norm
        u2, v2 = Kz[N1:N1 + n2] / norm, Kz[N1 + n2:] / norm
        s, sp = ramp_cut_points(u1, v1, 0.0, rule="order")
        assert (s, sp) == ramp_cut_points(u1, v1, 0.0, rule="linear")
        if s > sp:
            G, H = S.ramp(u2, s, sp), S.ramp(v2, s, sp)
            pv2 = priors(u1, v1, s, sp, 0.0, n2, m2)["ramp"]
            Z2 = S.pair_up(G, -H) - 0.0
    # batch 1: batch-0 witness at sigma0, evaluated on the second quarter
    Kz0 = A0 / q_n - B0 / q_m
    val0 = max(float(Kz0[:q_n].sum() / q_n - Kz0[n1:n1 + q_m].sum() / q_m), 0.0)
    norm0 = math.sqrt(val0)
    Z1 = pv1 = None
    if norm0 > 1e-12:
        u0 = Kz0[:q_n] / norm0 - (1.0 / q_n) / norm0
        v0 = Kz0[n1:n1 + q_m] / norm0 + (1.0 / q_m) / norm0
        s0, sp0 = ramp_cut_points(u0, v0, 0.0, rule="order")
        if s0 > sp0:
            G1, H1 = S.ramp(Kz0[q_n:n1] / norm0, s0, sp0), S.ramp(Kz0[n1 + q_m:N1] / norm0, s0, sp0)
            Z1 = S.pair_up(G1, -H1) - 0.0
            pv1 = priors(u0, v0, s0, sp0, 0.0, nb1x, nb1y)["ramp"]
    N_seq = M.n_pairs(nb1x, nb1y) + M.n_pairs(n2, m2)
    dec, lmax = M.bet_stream([Z1, Z2], [pv1, pv2], 0.0, N_seq)
    rec.update(dec0=int(dec), logmax=None if not math.isfinite(lmax) else round(float(lmax), 6),
               norm1=norm, norm0=norm0, seconds=round(time.perf_counter() - T0, 3))
    return rec


def work_path(kernel, key, n):
    return WORK / f"alpha0_{kernel}" / f"{key}_n{n}.jsonl"


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
    for d in sorted(WORK.glob("alpha0_*")):
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


_CAT = {}


def cost(key, n):
    if not _CAT:
        _CAT.update(B.catalogue())
    m = _CAT[key].m_ratio * n
    return (n + m) * (n + m) / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "sync", "time"])
    ap.add_argument("--kernel", default="laplace", choices=["gaussian", "laplace"])
    ap.add_argument("--designs", nargs="+", default=["all"])
    ap.add_argument("--min-n", type=int, default=0)
    ap.add_argument("--max-n", type=int, default=10 ** 9)
    ap.add_argument("--reps", type=int, default=200, help="replicates per cell (the first ones)")
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    if a.cmd == "sync":
        sync()
        return
    assert a.workers <= 6
    keys = DESIGNS if a.designs == ["all"] else a.designs

    def log(s):
        print(time.strftime("%H:%M:%S"), s, flush=True)
    jobs = []
    for key in keys:
        cells = stored_cells(key)
        by_n = {}
        for (n, rep), v in cells.items():
            by_n.setdefault(n, []).append((rep, v))
        for n, lst in sorted(by_n.items()):
            if not (a.min_n <= n <= a.max_n):
                continue
            lst = sorted(lst)[: a.reps]
            p = work_path(a.kernel, key, n)
            done = {r["rep"] for r in read_valid(p)}
            jobs += [((key, n, rep, a.kernel, v), p) for rep, v in lst if rep not in done]
    jobs.sort(key=lambda j: (cost(j[0][0], j[0][1]), j[0][2], j[0][0]))    # cheapest cells first
    if a.cmd == "time":
        for j, p in jobs[: a.workers]:
            r = replicate(j)
            log(f"{j[0][:3]} {r['seconds']:.2f}s dec0 {r['dec0']} stored {r.get('stored_seq0')} "
                f"sigma {r.get('repro_sigma')}")
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
                log(f"FAILED {j[0][:3]} {e!r}")
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            f = handles.get(p)
            if f is None:
                f = handles[p] = open(p, "a")
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
            done += 1
            if done % 500 == 0 or done == len(jobs):
                el = time.time() - t0
                log(f"{done}/{len(jobs)} {el:.0f}s")
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

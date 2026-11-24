"""AoS restructure (plan section 4.2): the analytic test with the threshold tau(alpha) = min{tau^P(alpha), tau^A(alpha,
delta)} (trimmedmmd.threshold.tau_aos; the sharper tau^{P'} is dropped) on the stored replicates of explore/harder_sims.

The study solved the analytic QP at a few levels per replicate (bisection) and stored, at each solved level j,
diag.an_val[j] = [sqrt(U), sqrt(U - gap)]: a feasible value U >= D-hat_j^2 and its Frank-Wolfe lower bound. A level is
rejected under the new threshold if U - gap > tau_j^2 and accepted if U <= tau_j^2 (the study's own rule); D-hat_alpha is
nonincreasing and tau nondecreasing in alpha, so a rejection at j implies rejection below j and an acceptance above.
When this leaves levels between the largest known rejection and the smallest known acceptance, the replicate is
regenerated (scripts/aos_baselines_run.py::draw), the Gaussian Gram matrix at the stored pilot bandwidth sigma_fix is
formed and the package solver (trimmedmmd.solver.solve_gram, ridge 0) is run at the middle level, in warm-started rounds
of 5,000 iterations until its Frank-Wolfe certificate decides (at most 20 rounds; still undecided = not rejected,
conservative, counted), bisecting until the rejection set is determined. At the alpha = 0 extension sizes
(*_n0ext.jsonl) the statistic is the stored MMD (stat.mmd2), compared with tau(0)^2.

Output: results/aos/main/raw/analytic/<design>.jsonl, one record per (n, rep): jlo, decisions at j = 0..10 (exact,
from the bounds and monotonicity), the source of the decision ("stored" or "resolved"), and for comparison the stored
an_new / an_newcf jlo. Resumable (finished (n, rep) are skipped). Memory: a dense Gram of (n + m) points (n = 1600:
82 MB).

    python -B scripts/aos_main_analytic.py dry        # count the levels that need a re-solve
    python -B scripts/aos_main_analytic.py run --workers 3
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from functools import lru_cache  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

import aos_baselines_run as B  # noqa: E402
from trimmedmmd import threshold as TH  # noqa: E402
from trimmedmmd.kernels import sqdist  # noqa: E402
from trimmedmmd.solver import solve_gram  # noqa: E402

OUT = ROOT / "results" / "aos" / "main" / "raw" / "analytic"
WORK = Path(os.environ.get("AOS_MAIN_WORK", "/home/hristouam/aos_main_work/raw")) / "analytic"
DELTA = 0.05
DESIGNS = B.DESIGNS + B.NULLS
ROUND_ITERS, MAX_ROUNDS = 5000, 20


@lru_cache(maxsize=None)
def tau_new_thr(n, m, alpha):
    return float(TH.tau_aos(n, m, alpha, DELTA, TH.GAUSSIAN))


def taus(n, m, levels):
    return [tau_new_thr(n, m, float(a)) for a in levels]


def known(rec, tau):
    """(lo, hi, per-level known decisions) from the stored certified bounds."""
    J = len(tau)
    lo, hi = -1, J
    dec = {}
    for js, (up, low) in rec["diag"]["an_val"].items():
        j = int(js)
        t2 = tau[j] ** 2
        if low is not None and low ** 2 > t2:
            dec[j] = 1
            lo = max(lo, j)
        elif up ** 2 <= t2:
            dec[j] = 0
            hi = min(hi, j)
    assert lo < hi, (rec["design"], rec["n"], rec["rep"], lo, hi)
    return lo, hi, dec


def solve_level(Kxx, Kyy, Kxy, alpha, t2, warm):
    """Rounds of the package solver until the certificate decides. Returns (decision, upper, lower, iters, w, v)."""
    w0, v0 = warm if warm is not None else (None, None)
    iters = 0
    rng = np.random.default_rng(0)
    for _ in range(MAX_ROUNDS):
        r = solve_gram(Kxx, Kyy, Kxy, alpha, w0=w0, v0=v0, max_iter=ROUND_ITERS, tol=0.0, rng=rng)
        iters += r.n_iter
        up, low = r.value, r.value_lower
        if low > t2:
            return 1, up, low, iters, r.w, r.v
        if up <= t2:
            return 0, up, low, iters, r.w, r.v
        w0, v0 = r.w, r.v
    return None, up, low, iters, r.w, r.v


def replicate(args):
    key, n, rep, rec = args
    T0 = time.perf_counter()
    levels = rec["levels"]
    m = rec["m"]
    tau = taus(n, m, levels)
    J = len(levels)
    lo, hi, dec = known(rec, tau)
    out = {"design": key, "n": n, "m": m, "rep": rep, "alpha_star": rec["alpha_star"], "tau": tau,
           "stored_jlo": {k: rec["jlo"][k] for k in ("an_new", "an_newcf", "an_old")}, "resolved": [],
           "undecided": 0}
    if hi - lo > 1:
        dz, pil, ss, X, Y = B.draw(key, n, rep)
        assert abs(pil["sigma_fix"] - rec["sigma_fix"]) < 1e-12
        Zall = np.vstack([X, Y])
        D2 = sqdist(Zall, Zall)
        K = np.exp(-D2 / (2.0 * rec["sigma_fix"] ** 2))
        del D2
        Kxx, Kyy, Kxy = K[:n, :n].copy(), K[n:, n:].copy(), K[:n, n:].copy()
        del K
        warm = None
        while hi - lo > 1:
            mid = (lo + hi) // 2
            d, up, low, it, w, v = solve_level(Kxx, Kyy, Kxy, float(levels[mid]), tau[mid] ** 2, warm)
            out["resolved"].append({"j": mid, "dec": d, "upper": float(np.sqrt(max(up, 0))),
                                    "lower": float(np.sqrt(max(low, 0))), "iters": it})
            if d == 1:
                lo, warm = mid, (w, v)
            else:
                if d is None:
                    out["undecided"] += 1          # conservative: not rejected
                hi = mid
    out["jlo"] = lo
    out["dec"] = [int(j <= lo) for j in range(J)]
    out["seconds"] = round(time.perf_counter() - T0, 3)
    return out


def ext_records(key):
    """alpha = 0 extension: decision mmd2 > tau(0)^2 with the stored MMD."""
    f = B.STORED / f"{key}_n0ext.jsonl"
    out = []
    if not f.exists():
        return out
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        t0 = tau_new_thr(r["n"], r["m"], 0.0)
        out.append({"design": key, "n": r["n"], "m": r["m"], "rep": r["rep"], "mode": "alpha0", "tau0": t0,
                    "mmd": r["stat"]["mmd"], "dec0": int(r["stat"]["mmd2"] > t0 ** 2),
                    "stored_dec0": {k: r["dec0"][k] for k in ("an_new", "an_newcf")}})
    return out


def stored(key):
    out = []
    for line in (B.STORED / f"{key}.jsonl").read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def read_done(path):
    done = {}
    if path.exists():
        for line in path.read_bytes().split(b"\n"):
            if not line.strip() or b"\x00" in line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            done[(r["n"], r["rep"])] = r
    return done


def sync(log=print):
    OUT.mkdir(parents=True, exist_ok=True)
    nr = 0
    for f in sorted(WORK.glob("*.jsonl")):
        recs = sorted(read_done(f).values(), key=lambda r: (r["n"], r["rep"]))
        data = "".join(json.dumps(r) + "\n" for r in recs)
        tmp = OUT / (f.name + ".tmp")
        tmp.write_text(data)
        os.replace(tmp, OUT / f.name)
        if (OUT / f.name).read_text() != data:
            log(f"SYNC MISMATCH {f}")
        nr += len(recs)
    log(f"synced {nr} analytic records to {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dry", "run", "sync"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--designs", nargs="+", default=["all"])
    a = ap.parse_args()
    keys = DESIGNS if a.designs == ["all"] else a.designs

    def log(s):
        print(time.strftime("%H:%M:%S"), s, flush=True)
    if a.cmd == "sync":
        sync(log)
        return
    if a.cmd == "dry":
        tot = need = 0
        per = {}
        for key in keys:
            for r in stored(key):
                tau = taus(r["n"], r["m"], r["levels"])
                lo, hi, _ = known(r, tau)
                tot += 1
                if hi - lo > 1:
                    need += 1
                    per[(key, r["n"])] = per.get((key, r["n"]), 0) + 1
        log(f"{need} of {tot} stored replicates need a re-solve")
        for k, v in sorted(per.items()):
            print(" ", k, v)
        return
    WORK.mkdir(parents=True, exist_ok=True)
    jobs = []
    for key in keys:
        p = WORK / f"{key}.jsonl"
        done = read_done(p)
        ext = [e for e in ext_records(key) if (e["n"], e["rep"]) not in done]
        if ext:
            with open(p, "a") as fh:
                for e in ext:
                    fh.write(json.dumps(e) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        for r in stored(key):
            if (r["n"], r["rep"]) not in done:
                jobs.append(((key, r["n"], r["rep"], r), p))
    jobs.sort(key=lambda j: (j[0][1], j[0][2]))
    log(f"{len(jobs)} replicates")
    t0 = time.time()
    handles, cnt = {}, 0
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(replicate, j): (j, p) for j, p in jobs}
        for fu in as_completed(futs):
            j, p = futs[fu]
            try:
                rec = fu.result()
            except Exception as e:
                log(f"FAILED {j[0][:3]} {e!r}")
                continue
            f = handles.get(p)
            if f is None:
                f = handles[p] = open(p, "a")
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
            cnt += 1
            if cnt % 2000 == 0:
                log(f"{cnt}/{len(jobs)} {time.time() - t0:.0f}s")
    for f in handles.values():
        f.close()
    sync(log)
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

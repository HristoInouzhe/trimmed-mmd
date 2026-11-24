"""Certified lower bounds on TV on the paper's real contrasts: the bootstrap
certificate (Theorem 16, exactly as the package computes it) versus the
finite-sample certificates, on the SAME splits and the SAME fold-1 witnesses.

RHC and NHANES: as realdata_run.w_lower -> splitting.lower_bound_tv
  (rng = default_rng(1000+i); split_seed = rng.integers(...); per level a fresh
  default_rng(split_seed) drives the split, the fold-1 median bandwidth, the QP and
  the bootstrap; exact Gram; n_boot = 1000; grid 0..0.75 by 0.05, child to 0.95).
Adult and ACS: as scripts/fair_split_certificate.certify_one on the original data
  (rng = default_rng(1000+s) per level; RFF backend, 512 features, n_boot = 500,
  grid 0..0.8 by 0.05).

Every level of the grid is evaluated until ALL methods have failed (the
down-closed rule stops each method at its own first failure).

    python3 -I explore/split_adaptive/real.py rhc --splits 8 --workers 8
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from trimmedmmd.kernels import RFF, median_heuristic  # noqa: E402
from trimmedmmd.splitting import evaluate_direction, trimmed_mean, witness_direction  # noqa: E402

import sacert as S  # noqa: E402
from fold1 import cut_points, pair_var  # noqa: E402

DELTA = 0.05
_CACHE = {}


def dataset(name):
    if name in _CACHE:
        return _CACHE[name]
    if name == "rhc":
        from trimmedmmd.realdata import load_rhc
        X, Y, _, _ = load_rhc()
    elif name.startswith("nhanes_"):
        from trimmedmmd.realdata import load_nhanes_activity, nhanes_groups
        curves, cov, _ = load_nhanes_activity()
        X, Y, _, _ = nhanes_groups(curves, cov, by=name.split("_", 1)[1])
    elif name in ("adult", "acs"):
        import fair_case_study as fc
        info = json.loads((ROOT / "results" / "fair" / f"{name}_info.json").read_text())
        F, g, _, _ = fc.load(name, info.get("max_rows"), info.get("seed", 0))
        Fv = F.values.astype(float)
        sd = Fv.std(0)
        sd[sd == 0] = 1.0
        Sx = (Fv - Fv.mean(0)) / sd
        X, Y = Sx[g == 0], Sx[g == 1]
    else:
        raise ValueError(name)
    _CACHE[name] = (np.asarray(X, float), np.asarray(Y, float))
    return _CACHE[name]


def grid_for(name):
    if name == "nhanes_child":
        return np.round(np.arange(0.0, 0.951, 0.05), 3)
    if name in ("adult", "acs"):
        return np.round(np.arange(0.0, 0.8 + 1e-9, 0.05), 4)
    return np.round(np.arange(0.0, 0.751, 0.05), 3)


def level(X, Y, a, split_seed, use_rff, n_features, n_boot):
    """One level: replicate split_test's RNG stream, then add the new tests."""
    rng = np.random.default_rng(split_seed)
    n, m = len(X), len(Y)
    ix, iy = rng.permutation(n), rng.permutation(m)
    n1, m1 = int(n * 0.5), int(m * 0.5)
    X1, X2 = X[ix[:n1]], X[ix[n1:]]
    Y1, Y2 = Y[iy[:m1]], Y[iy[m1:]]
    sigma = median_heuristic(X1, Y1, rng=rng)
    rff = RFF(dim=X.shape[1], n_features=n_features, sigma=sigma, rng=rng) if use_rff else None
    w, v, norm = witness_direction(X1, Y1, a, sigma, rng=rng, rff=rff)
    u2 = evaluate_direction(w, v, X1, Y1, X2, sigma, norm, rff=rff)
    v2 = evaluate_direction(w, v, X1, Y1, Y2, sigma, norm, rff=rff)

    def stat(av, bv):
        return trimmed_mean(av, a, upper=False) - trimmed_mean(bv, a, upper=True)

    sep = stat(u2, v2)
    idx_a = rng.integers(0, u2.size, size=(n_boot, u2.size))
    idx_b = rng.integers(0, v2.size, size=(n_boot, v2.size))
    boots = np.array([stat(u2[i], v2[j]) for i, j in zip(idx_a, idx_b)])
    out = {"alpha": float(a), "T_hat": float(sep), "boot": bool(np.quantile(boots, DELTA) > 0),
           "norm": float(norm)}
    if norm <= 0:
        for k in ("ramp_bet", "ramp_b2s", "ramp_mp_pair", "ramp_mpu", "ramp_hoef", "prop17", "tlm", "ramp_clt"):
            out[k] = False
        return out
    # fold-1 self-excluded values
    u1 = evaluate_direction(w, v, X1, Y1, X1, sigma, norm, rff=rff)
    v1 = evaluate_direction(w, v, X1, Y1, Y1, sigma, norm, rff=rff)
    if rff is None:
        u1, v1 = u1 - w / norm, v1 + v / norm
        kappa = 1.0
    else:
        zx, zy = rff.transform(X1), rff.transform(Y1)
        u1 = u1 - w * (zx ** 2).sum(1) / norm
        v1 = v1 + v * (zy ** 2).sum(1) / norm
        kappa = math.sqrt(2.0)   # phased RFF: k_hat(x,x) <= 2
    if a == 0.0:
        out["_arrays"] = (u1, v1, u2, v2)
    s, sp = cut_points(u1, v1, a)
    beta = 1.0 / (1.0 - a)
    e = lambda k: math.sqrt(math.log(4.0 / DELTA) / (2.0 * k))
    out["prop17"] = bool(sep > 2 * kappa * beta * (e(u2.size) + e(v2.size)))
    out["tlm"] = bool(S.dkw_lower_trimmed_lcb(u2, a, -kappa, DELTA / 2)
                      + S.dkw_lower_trimmed_lcb(-v2, a, -kappa, DELTA / 2) > 0)
    out["s"], out["sp"] = s, sp
    if s > sp:
        G, H = S.ramp(u2, s, sp), S.ramp(v2, s, sp)
        Gf, Hf = S.ramp(u1, s, sp), S.ramp(v1, s, sp)
        pv = max(pair_var(Gf.var(), Hf.var(), G.size, H.size), 1e-4)
        Z = S.pair_up(G, -H) - a
        out["Delta_hat"] = float(G.mean() - H.mean())
        out["ramp_bet"] = S.betting_test(Z, -(1 + a), DELTA, c=0.5, prior_var=pv)[0]
        out["ramp_b2s"] = bool(S.lcb_bern2s(G, 1.0, -H, 1.0, DELTA) > a)
        out["ramp_mpu"] = bool(S.lcb_mp_union(G, 1.0, -H, 1.0, DELTA) > a)
        out["ramp_mp_pair"] = bool(S.mp_pair_lcb(G, H, DELTA) > a)
        out["ramp_hoef"] = bool(S.hoeffding2s_lcb(G, 1.0, -H, 1.0, DELTA) > a)
        out["ramp_clt"] = bool(out["Delta_hat"] - 1.6448536269514722
                               * math.sqrt(G.var(ddof=1) / G.size + H.var(ddof=1) / H.size) > a)
        # lower confidence bound on TV from this single ramp (no grid): Delta LCB
        out["ramp_b2s_lcb"] = S.lcb_bern2s(G, 1.0, -H, 1.0, DELTA)
    else:
        for k in ("ramp_bet", "ramp_b2s", "ramp_mp_pair", "ramp_mpu", "ramp_hoef", "ramp_clt"):
            out[k] = False
    return out


METHODS = ["boot", "ramp_bet", "ramp_b2s", "ramp_mp_pair", "ramp_mpu", "ramp_hoef", "prop17", "tlm", "ramp_clt"]


def one_split(args):
    name, i, n_features, n_boot = args
    X, Y = dataset(name)
    fair = name in ("adult", "acs")
    if fair:
        split_seed = 1000 + i
    else:
        split_seed = int(np.random.default_rng(1000 + i).integers(0, 2**31 - 1))
    alive = set(METHODS)
    rows = []
    arrays = None
    for a in grid_for(name):
        r = level(X, Y, a, split_seed, fair, n_features, n_boot)
        if "_arrays" in r:
            arrays = r.pop("_arrays")
        rows.append(r)
        alive = {mth for mth in alive if r.get(mth, False)}
        if not alive:
            break
    # one direction for all levels: the alpha = 0 witness with level-specific cut points
    dir0 = {}
    if arrays is not None:
        u1, v1, u2, v2 = arrays
        for a in grid_for(name):
            s, sp = cut_points(u1, v1, a)
            if not s > sp:
                dir0[float(a)] = False
                continue
            G, H = S.ramp(u2, s, sp), S.ramp(v2, s, sp)
            Gf, Hf = S.ramp(u1, s, sp), S.ramp(v1, s, sp)
            pv = max(pair_var(Gf.var(), Hf.var(), G.size, H.size), 1e-4)
            dir0[float(a)] = bool(S.betting_test(S.pair_up(G, -H) - a, -(1 + a), DELTA, c=0.5,
                                                 prior_var=pv)[0])
    for r in rows:
        r["dir0_ramp_bet"] = dir0.get(r["alpha"], False)
    best = 0.0 if not fair else -1.0
    for a in grid_for(name):
        if dir0.get(float(a), False):
            best = float(a)
        else:
            break
    lo = {}
    for mth in METHODS:
        best = 0.0 if not fair else -1.0
        for r in rows:
            if r.get(mth, False):
                best = r["alpha"]
            else:
                break
        lo[mth] = best
    lo["dir0_ramp_bet"] = best
    return {"dataset": name, "split": i, "split_seed": split_seed, "lower": lo, "levels": rows,
            "n": len(X), "m": len(Y)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--splits", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = HERE / "out" / "real.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["dataset"], r["split"]))
    jobs = []
    for nm in a.names:
        nb = 500 if nm in ("adult", "acs") else 1000
        jobs += [(nm, i, 512, nb) for i in range(a.splits) if (nm, i) not in done]
    t0 = time.time()
    print(f"{len(jobs)} jobs", flush=True)
    with ProcessPoolExecutor(a.workers) as ex, open(out, "a") as fh:
        for rec in ex.map(one_split, jobs):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"  {rec['dataset']} split {rec['split']}: {rec['lower']}  ({time.time()-t0:.0f}s)",
                  flush=True)


if __name__ == "__main__":
    main()
